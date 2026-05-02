"""
AVG (Averaged-DQN Variance Scaling) Implementation with Ablation Study

This module implements three variants for thesis research:

# Variant A (baseline, faithful to paper) - 3 seeds
for seed in 1 2 3; do
  python -m algorithms.avg_baseline \
    --env_name HalfCheetah-v4 \
    --scaling_mode td_only \
    --seed $seed \
    --total_steps 500000
done

# Variant B (reward rescaling only) - 3 seeds
for seed in 1 2 3; do
  python -m algorithms.avg_baseline \
    --env_name HalfCheetah-v4 \
    --scaling_mode td_reward \
    --seed $seed \
    --total_steps 500000
done

# Variant C (reward + entropy rescaling) - 3 seeds
for seed in 1 2 3; do
  python -m algorithms.avg_baseline \
    --env_name HalfCheetah-v4 \
    --scaling_mode td_reward_entropy \
    --seed $seed \
    --total_steps 500000
done

With LayerNorm enabled -> saved as avg_A_ln, avg_B_ln, avg_C_ln  
    python -m algorithms.avg_baseline --scaling_mode td_only --seed 1 --use_layer_norm
"""

import argparse
import json
import os
import pickle
from pathlib import Path
#from env_factory import make_train_env, make_eval_env
from evaluation.avg_evaluator import evaluate_policy_avg
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation, ClipAction
from paths import ensure_run_dirs, train_csv_path, eval_csv_path, returns_pkl_path
from evaluation.fixed_evaluator import evaluate_policy
from logging_utils.csv_logger import CSVLogger


# helper functions for environment creation and wrapping, which are used both in training and evaluation to ensure consistent normalization and time info handling.
def _canonical_dmcontrol_name(env_name: str) -> str:
    if env_name.startswith("dm_control/"):
        return env_name
    return f"dm_control/{env_name}"


def make_avg_env(env_name: str, backend: str, render: bool = False):
    if backend == "dmcontrol":
        env_name = _canonical_dmcontrol_name(env_name)

    kwargs = {}
    if render:
        kwargs["render_mode"] = "human"

    env = gym.make(env_name, **kwargs)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.ClipAction(env)
    return env

#---------


def orthogonal_weight_init(m: nn.Module) -> None:
    """
    Faithful to uploaded AVG code on Github
    """
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.0)


def set_one_thread() -> None:
    """
    Match upstream AVG behavior to avoid PyTorch over-threading on CPU.
    """
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def save_run_config(results_root, backend, algo, env_name, seed, config):
    run_path = ensure_run_dirs(results_root, backend, algo, env_name, seed)
    config_path = Path(run_path) / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)


class RunningMeanVar:
    """
    Welford-style online sample mean/variance.
    """
    def __init__(self):
        self.mean = 0.0
        self.p = 0.0
        self.var = 1.0
        self.count = 0

    def update(self, x: float) -> None:
        x = float(x)

        if self.count == 0:
            self.mean = x
            self.p = 0.0
            self.var = 1.0
            self.count = 1
            return

        self.count += 1
        new_mean = self.mean + (x - self.mean) / self.count
        self.p = self.p + (x - self.mean) * (x - new_mean)
        self.mean = new_mean
        self.var = self.p / (self.count - 1) if self.count >= 2 else 1.0

    def get_scale(self) -> float:
        """Return sqrt(var) for reward/feature scaling."""
        return float(np.sqrt(max(self.var, 1e-8)))


class TDErrorScaler:
    """
    TD-error scaling faithful to AVG paper Algorithm 3.
    """
    def __init__(self, eps: float = 1e-8):
        self.reward_stats = RunningMeanVar()      # For r_ent (entropy-adjusted)
        self.gamma_stats = RunningMeanVar()
        self.g2_stats = RunningMeanVar()
        self.sigma = 1.0
        self.eps = eps

    def update(self, r_raw: float, gamma: float, G: float | None) -> None:
        """
        Update with entropy-adjusted reward (r - alpha*log_prob).
        """
        self.reward_stats.update(r_raw)
        self.gamma_stats.update(gamma)

        if G is not None:
            self.g2_stats.update(float(G) ** 2)

        if self.g2_stats.count > 1:
            sigma_sq = self.reward_stats.var + self.g2_stats.mean * self.gamma_stats.var
            self.sigma = float(np.sqrt(max(sigma_sq, self.eps)))
        else:
            self.sigma = 1.0


class RewardScaler:
    """
    scaler for internal reward rescaling (Variants B/C only).
    """
    def __init__(self, eps: float = 1e-8):
        self.reward_stats = RunningMeanVar()      # For raw reward only

    def update(self, raw_reward: float) -> None:
        """Update with raw environment reward (no entropy adjustment)."""
        self.reward_stats.update(raw_reward)

    def get_scale(self) -> float:
        """Return sqrt(Var[raw_reward]) for rescaling."""
        return self.reward_stats.get_scale()



class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, device: torch.device, n_hid: int, use_layer_norm: bool = False):
        super().__init__()
        self.device = device
        self.use_layer_norm = use_layer_norm
        self.LOG_STD_MAX = 2.0
        self.LOG_STD_MIN = -20.0

        self.fc1 = nn.Linear(obs_dim, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)

        if use_layer_norm:
            self.ln1 = nn.LayerNorm(n_hid)
            self.ln2 = nn.LayerNorm(n_hid)
        self.mu = nn.Linear(n_hid, action_dim)
        self.log_std = nn.Linear(n_hid, action_dim)

        self.apply(orthogonal_weight_init)
        self.to(device=self.device)

    def forward(self, obs: torch.Tensor):
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        x = obs.to(self.device)
        phi = self.fc1(x)

        if self.use_layer_norm:
            phi = F.leaky_relu(self.ln1(phi))
        else:
            phi = F.leaky_relu(phi)
        
        phi = self.fc2(phi)
        if self.use_layer_norm:
            phi = F.leaky_relu(self.ln2(phi))
        else:
            phi = F.leaky_relu(phi)

        # Penultimate normalization
        phi_norm = torch.norm(phi, dim=1, keepdim=True).clamp_min(1e-8)
        phi = phi / phi_norm

        mu = self.mu(phi)
        log_std = self.log_std(phi)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        dist = MultivariateNormal(mu, torch.diag_embed(log_std.exp()))

        action_pre = dist.rsample()
        lprob = dist.log_prob(action_pre)

        lprob -= (2 * (np.log(2) - action_pre - F.softplus(-2 * action_pre))).sum(axis=1)

        action = torch.tanh(action_pre)

        action_info = {
            "mu": mu,
            "log_std": log_std,
            "dist": dist,
            "lprob": lprob,
            "action_pre": action_pre,
        }
        return action, action_info

    def mean_action(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        x = obs.to(self.device)
        phi = self.fc1(x)
        if self.use_layer_norm:
            phi = F.leaky_relu(self.ln1(phi))
        else:
            phi = F.leaky_relu(phi)
        
        phi = self.fc2(phi)
        if self.use_layer_norm:
            phi = F.leaky_relu(self.ln2(phi))
        else:
            phi = F.leaky_relu(phi)
        
        phi_norm = torch.norm(phi, dim=1, keepdim=True).clamp_min(1e-8)
        phi = phi / phi_norm
        mu = self.mu(phi)
        return torch.tanh(mu)


class Q(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, device: torch.device, n_hid: int, use_layer_norm: bool = False):
        super().__init__()
        self.device = device
        self.use_layer_norm = use_layer_norm

        self.fc1 = nn.Linear(obs_dim + action_dim, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        if use_layer_norm:
            self.ln1 = nn.LayerNorm(n_hid)
            self.ln2 = nn.LayerNorm(n_hid)
        self.q = nn.Linear(n_hid, 1)

        self.apply(orthogonal_weight_init)
        self.to(device=self.device)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if obs.dim() == 1:
            obs = obs.unsqueeze(0)
        if action.dim() == 1:
            action = action.unsqueeze(0)

        x = torch.cat((obs, action), dim=-1).to(self.device)
        phi = self.fc1(x)
        if self.use_layer_norm:
            phi = F.leaky_relu(self.ln1(phi))
        else:
            phi = F.leaky_relu(phi)
        
        phi = self.fc2(phi)
        if self.use_layer_norm:
            phi = F.leaky_relu(self.ln2(phi))
        else:
            phi = F.leaky_relu(phi)

        # Penultimate normalization
        phi_norm = torch.norm(phi, dim=1, keepdim=True).clamp_min(1e-8)
        phi = phi / phi_norm

        return self.q(phi).view(-1)


class AVGBaseline(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.steps = 0

        self.actor = Actor(
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            device=cfg.device,
            n_hid=cfg.nhid_actor,
            use_layer_norm=cfg.use_layer_norm,
        )
        self.Q = Q(
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            device=cfg.device,
            n_hid=cfg.nhid_critic,
            use_layer_norm=cfg.use_layer_norm,
        )

        self.popt = torch.optim.Adam(
            self.actor.parameters(),
            lr=cfg.actor_lr,
            betas=cfg.betas,
        )
        self.qopt = torch.optim.Adam(
            self.Q.parameters(),
            lr=cfg.critic_lr,
            betas=cfg.betas,
        )

        self.alpha = cfg.alpha_lr
        self.gamma = cfg.gamma
        self.device = cfg.device

        
        self.td_error_scaler = TDErrorScaler()
        
        # Reward rescaling
        self.reward_scaler = RewardScaler()
        
        self.scaling_mode = cfg.scaling_mode
        self.G = 0.0

    def compute_action(self, obs_np: np.ndarray):
        obs = torch.tensor(obs_np.astype(np.float32), device=self.device).unsqueeze(0)
        action, action_info = self.actor(obs)
        return action, action_info

    def pi(self, obs: torch.Tensor):
        """
        Compatibility layer for evaluation.fixed_evaluator.
        Computes deterministic policy: tanh(mu) with std from learned log_std.
        """

        if obs.dim() == 1:
            obs = obs.unsqueeze(0)

        # Rebuild features using the actor's actual layers
        x = obs.to(self.device)
        phi = self.actor.fc1(x)
        if self.actor.use_layer_norm:
            phi = F.leaky_relu(self.actor.ln1(phi))
        else:
            phi = F.leaky_relu(phi)
        
        phi = self.actor.fc2(phi)
        if self.actor.use_layer_norm:
            phi = F.leaky_relu(self.actor.ln2(phi))
        else:
            phi = F.leaky_relu(phi)

        # Penultimate normalization
        phi_norm = torch.norm(phi, dim=1, keepdim=True).clamp_min(1e-8)
        phi = phi / phi_norm

        mu = self.actor.mu(phi)
        log_std = self.actor.log_std(phi)
        log_std = torch.clamp(log_std, self.actor.LOG_STD_MIN, self.actor.LOG_STD_MAX)
        std = log_std.exp()

        # Return deterministic action (tanh of mean) and std
        mu_squashed = torch.tanh(mu)

        if mu_squashed.shape[0] == 1:
            return mu_squashed.squeeze(0), std.squeeze(0)
        return mu_squashed, std
    

    def update(self, obs, action, next_obs, reward, done, **kwargs):
        obs = torch.tensor(obs.astype(np.float32), device=self.device).unsqueeze(0)
        next_obs = torch.tensor(next_obs.astype(np.float32), device=self.device).unsqueeze(0)

        action = action.to(self.device)
        lprob = kwargs["lprob"]

        # STEP 1: Update reward scaler (for variants B/C only)
        # Track raw environment reward variance SEPARATE from TD scaling
        self.reward_scaler.update(raw_reward=reward)
        
        # STEP 2: Compute entropy-adjusted reward (UNSCALED)
        r_ent = reward - self.alpha * lprob.detach().item()
        self.G += r_ent
        
        # STEP 3: Update TD-error scaler with unscaled entropy-adjusted reward
        # NOTE: Assumes paper uses r_ent for variance (awaiting supervisor verification)
        if done:
            self.td_error_scaler.update(r_raw=r_ent, gamma=0.0, G=self.G)
            self.G = 0.0
        else:
            self.td_error_scaler.update(r_raw=r_ent, gamma=self.gamma, G=None)
        
        # STEP 4: Compute scaling factors (identity for variant A)
        if self.scaling_mode == "td_only":
            reward_scale = 1.0
            alpha_scale = 1.0

        elif self.scaling_mode == "td_reward":
            reward_scale = self.reward_scaler.get_scale()
            alpha_scale = 1.0  # Test: does NOT scaling alpha break learning?

        else:  # td_reward_entropy
            reward_scale = self.reward_scaler.get_scale()
            alpha_scale = reward_scale  # Maintain balance: scale reward and alpha together
        
        # STEP 5: Scale reward and entropy coefficient for loss computation
        reward_scaled = reward / reward_scale if reward_scale > 1e-8 else reward
        alpha_scaled = self.alpha / alpha_scale if alpha_scale > 1e-8 else self.alpha

        # STEP 6: Critic and Actor losses (using scaled reward/entropy)
        q = self.Q(obs, action.detach())

        with torch.no_grad():
            next_action, next_action_info = self.actor(next_obs)
            next_lprob = next_action_info["lprob"]
            q2 = self.Q(next_obs, next_action)
            target_v = q2 - alpha_scaled * next_lprob

        delta = reward_scaled + (1 - float(done)) * self.gamma * target_v - q
        delta = delta / max(self.td_error_scaler.sigma, 1e-8)
        qloss = (delta ** 2).mean()

        ploss = (alpha_scaled * lprob - self.Q(obs, action)).mean()

        
        self.popt.zero_grad()
        ploss.backward()
        self.popt.step()

        self.qopt.zero_grad()
        qloss.backward()
        self.qopt.step()

        self.steps += 1


def main(args):
    set_one_thread()

    print(f"\n{'='*70}")
    print(f"AVG Algorithm - Faithful Implementation with Scaling Ablations")
    print(f"{'='*70}")
    print(f"Variant: {args.algo}")
    print(f"Scaling Mode: {args.scaling_mode}")

    if args.scaling_mode == "td_only":
        print(f"TD-error scaling only (baseline from paper)")
    elif args.scaling_mode == "td_reward":
        print(f"TD-error + internal reward rescaling (scale by Var[r]^0.5)")
    elif args.scaling_mode == "td_reward_entropy":
        print(f"TD-error + internal reward rescaling + entropy scaling")

    print(f"LayerNorm in hidden layers: {'YES' if args.use_layer_norm else 'NO (vanilla AVG)'}")
    print(f"Environment: {args.env_name} ({args.backend})")
    print(f"Total Steps: {args.total_steps}")
    print(f"Seed: {args.seed}")
    print(f"{'='*70}\n")

    ensure_run_dirs(args.results_root, args.backend, args.algo, args.env_name, args.seed)

    config = {
        "algo": args.algo,
        "env_name": args.env_name,
        "backend": args.backend,
        "seed": args.seed,
        "total_steps": args.total_steps,
        "actor_lr": args.actor_lr,
        "critic_lr": args.critic_lr,
        "beta1": args.beta1,
        "beta2": 0.999, # need to be changed if we want to tune hyperparamters
        "gamma": args.gamma,
        "alpha_lr": args.alpha_lr,
        "nhid_actor": args.nhid_actor,
        "nhid_critic": args.nhid_critic,
        "eval_interval": args.eval_interval,
        "eval_episodes": args.eval_episodes,
        "uses_avg": True,
        "uses_q_critic": True,
        "uses_reparam_actor": True,
        "uses_penult_norm": True,
        "uses_td_scaling": True,
        "uses_layer_norm": args.use_layer_norm,
        "uses_internal_reward_rescaling": args.scaling_mode != "td_only",
        "uses_internal_entropy_rescaling": args.scaling_mode == "td_reward_entropy",
        "scaling_mode": args.scaling_mode,
        "uses_obs_norm": True,
        "uses_replay_buffer": False,
        "uses_target_network": False,
    }
    save_run_config(args.results_root, args.backend, args.algo, args.env_name, args.seed, config)

    # Create environment with consistent normalization and time info handling for both training and evaluation
    env = make_avg_env(
        env_name=args.env_name,
        backend=args.backend,
        #gamma=args.gamma,
        render=args.render,
        #use_reward_scaling =False, 
        #use_time_info =False, 
    )

    # Reproducibility
    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.obs_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]

    agent = AVGBaseline(args)

    train_logger = CSVLogger(
        train_csv_path(args.results_root, args.backend, args.algo, args.env_name, args.seed),
        ["step", "episode_return"],
    )
    eval_logger = CSVLogger(
        eval_csv_path(args.results_root, args.backend, args.algo, args.env_name, args.seed),
        ["step", "eval_return_mean", "eval_return_std"],
    )

    returns = []
    term_steps = []

    obs, _ = env.reset()
    ret = 0.0

    for t in range(1, args.total_steps + 1):
        action, action_info = agent.compute_action(obs)
        sim_action = action.detach().cpu().view(-1).numpy()

        next_obs, reward, terminated, truncated, info = env.step(sim_action)
        done = terminated or truncated

        agent.update(
            obs,
            action,
            next_obs,
            reward,
            done,
            **action_info,
        )

        ret += reward
        obs = next_obs

        if args.eval_interval > 0 and t % args.eval_interval == 0:
            eval_mean, eval_std = evaluate_policy_avg(
                agent=agent,
                env_name=args.env_name,
                backend=args.backend,
                train_env=env,
                episodes=args.eval_episodes,
                seed=args.seed,
            )
            eval_logger.log(
                {
                    "step": t,
                    "eval_return_mean": eval_mean,
                    "eval_return_std": eval_std,
                }
            )
            print(f"[EVAL] step = {t} mean = {eval_mean:.2f} std = {eval_std:.2f}")

        if done:
            # Keep project's logging convention
            episode_return = float(ret)
            returns.append(episode_return)
            term_steps.append(t)

            train_logger.log(
                {
                    "step": t,
                    "episode_return": episode_return,
                }
            )

            obs, _ = env.reset()
            ret = 0.0

    env.close()
    train_logger.close()
    eval_logger.close()

    with open(
        returns_pkl_path(args.results_root, args.backend, args.algo, args.env_name, args.seed),
        "wb",
    ) as f:
        pickle.dump((returns, term_steps, args.env_name), f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env_name", type=str, default="HalfCheetah-v4")
    parser.add_argument("--backend", type=str, default="mujoco", choices=["mujoco", "dmcontrol"])
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--total_steps", type=int, default=1000000)

    # AVG defaults from uploaded avg.py
    parser.add_argument("--actor_lr", type=float, default=0.0063)
    parser.add_argument("--critic_lr", type=float, default=0.0087)
    parser.add_argument("--beta1", type=float, default=0.0) # change
    parser.add_argument("--gamma", type=float, default=0.99) # change
    parser.add_argument("--alpha_lr", type=float, default=0.07) # at first stick to it 
    parser.add_argument("--nhid_actor", type=int, default=256)
    parser.add_argument("--nhid_critic", type=int, default=256)

    # Scaling experiment mode
    parser.add_argument(
        "--scaling_mode",
        type=str,
        default="td_only",
        choices=["td_only", "td_reward", "td_reward_entropy"],
        help="(A) TD-error only | (B) +reward scaling | (C) +entropy scaling"
    )
    parser.add_argument(
        "--use_layer_norm",
        action="store_true",
        help="Enable LayerNorm in hidden layers (not in vanilla AVG)"
    )

    # Thesis shell settings
    parser.add_argument("--eval_interval", type=int, default=10000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    # AVG Adam setup
    args.betas = [args.beta1, 0.999]

    if torch.cuda.is_available() and "cuda" in args.device:
        args.device = torch.device(args.device)
    else:
        args.device = torch.device("cpu")

    # Naming convention for results: avg_A, avg_B, avg_C (+ _ln if LayerNorm)
    mode_names = {"td_only": "A", "td_reward": "B", "td_reward_entropy": "C"}
    base_algo = f"avg_{mode_names[args.scaling_mode]}"
    args.algo = f"{base_algo}_ln" if args.use_layer_norm else base_algo

    main(args)