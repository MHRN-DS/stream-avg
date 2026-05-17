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
from incremental_rl.envs.dm_control_wrapper import DMControl
from incremental_rl.td_error_scaler import TDErrorScaler as OriginalTDErrorScaler
from paths import ensure_run_dirs, train_csv_path, eval_csv_path, returns_pkl_path
from evaluation.fixed_evaluator import evaluate_policy
from logging_utils.csv_logger import CSVLogger


# helper functions for environment creation and wrapping, which are used both in training and evaluation to ensure consistent normalization and time info handling.
def _canonical_dmcontrol_name(env_name: str) -> str:
    if env_name.startswith("dm_control/"):
        return env_name
    return f"dm_control/{env_name}"


def _parse_dmcontrol_name(env_name: str) -> tuple[str, str]:
    task_name = env_name.replace("dm_control/", "").replace("dm_control__", "")
    task_name = task_name.removesuffix("-v0")
    domain, task = task_name.split("-", 1)
    return domain, task


def make_avg_env(env_name: str, backend: str, render: bool = False):
    if backend == "dmcontrol":
        domain, task = _parse_dmcontrol_name(env_name)
        env = DMControl(
            domain=domain,
            task=task,
            render_mode="human" if render else None,
        )
    else:
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


class CustomTDErrorScaler:
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
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: torch.device,
        n_hid: int,
        use_layer_norm: bool = False,
        action_distribution: str = "tanh_normal",
    ):
        super().__init__()
        self.device = device
        self.use_layer_norm = use_layer_norm
        self.action_distribution = action_distribution
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

        dist = MultivariateNormal(mu, scale_tril=torch.diag_embed(log_std.exp()))

        action_pre = dist.rsample()
        lprob = dist.log_prob(action_pre)

        if self.action_distribution == "tanh_normal":
            lprob -= (2 * (np.log(2) - action_pre - F.softplus(-2 * action_pre))).sum(axis=1)
            action = torch.tanh(action_pre)
        elif self.action_distribution == "clipped_normal":
            action = torch.clamp(action_pre, -1.0, 1.0)
        else:
            raise ValueError(f"Unknown action_distribution: {self.action_distribution}")

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
        if self.action_distribution == "tanh_normal":
            return torch.tanh(mu)
        if self.action_distribution == "clipped_normal":
            return torch.clamp(mu, -1.0, 1.0)
        raise ValueError(f"Unknown action_distribution: {self.action_distribution}")


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
            action_distribution=getattr(cfg, "action_distribution", "tanh_normal"),
        )
        self.Q = Q(
            obs_dim=cfg.obs_dim,
            action_dim=cfg.action_dim,
            device=cfg.device,
            n_hid=cfg.nhid_critic,
            use_layer_norm=cfg.use_layer_norm,
        )

        actor_adam_kwargs = {"lr": cfg.actor_lr}
        critic_adam_kwargs = {"lr": cfg.critic_lr}
        if cfg.betas is not None:
            actor_adam_kwargs["betas"] = tuple(cfg.betas)
            critic_adam_kwargs["betas"] = tuple(cfg.betas)

        self.popt = torch.optim.Adam(self.actor.parameters(), **actor_adam_kwargs)
        self.qopt = torch.optim.Adam(self.Q.parameters(), **critic_adam_kwargs)

        self.alpha = cfg.alpha_lr
        self.gamma = cfg.gamma
        self.device = cfg.device

        
        self.td_scaler_impl = getattr(cfg, "td_scaler_impl", "custom")
        if self.td_scaler_impl == "original":
            self.td_error_scaler = OriginalTDErrorScaler()
        elif self.td_scaler_impl == "custom":
            self.td_error_scaler = CustomTDErrorScaler()
        else:
            raise ValueError(f"Unknown td_scaler_impl: {self.td_scaler_impl}")
        self.td_scaler_reward_units = getattr(cfg, "td_scaler_reward_units", "unscaled")
        
        # Reward rescaling
        self.reward_scaler = RewardScaler()
        
        self.scaling_mode = cfg.scaling_mode
        self.G = 0.0
        self.last_update_stats = {}

    def _update_td_error_scaler(self, reward_for_scaler: float, gamma: float, G: float | None) -> None:
        if self.td_scaler_impl == "original":
            self.td_error_scaler.update(reward=reward_for_scaler, gamma=gamma, G=G)
        else:
            self.td_error_scaler.update(r_raw=reward_for_scaler, gamma=gamma, G=G)

    @property
    def td_sigma(self) -> float:
        return float(self.td_error_scaler.sigma)

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



        # decision if we use tanh or clipped normal for evaluation mean action, to match the distribution used during training
        if self.actor.action_distribution == "tanh_normal":
            mu_squashed = torch.tanh(mu)
            
        elif self.actor.action_distribution == "clipped_normal":
            mu_squashed = torch.clamp(mu, -1.0, 1.0)
        else:
            raise ValueError(f"Unknown action_distribution: {self.actor.action_distribution}")

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
        
        # STEP 2: Compute scaling factors (identity for variant A)
        if self.scaling_mode == "td_only":
            reward_scale = 1.0
            alpha_scale = 1.0

        elif self.scaling_mode == "td_reward":
            reward_scale = self.reward_scaler.get_scale()
            alpha_scale = 1.0  # Test: does NOT scaling alpha break learning?

        else:  # td_reward_entropy
            reward_scale = self.reward_scaler.get_scale()
            alpha_scale = reward_scale  # Maintain balance: scale reward and alpha together
        
        # STEP 3: Scale reward and entropy coefficient for loss computation
        reward_scaled = reward / reward_scale if reward_scale > 1e-8 else reward
        alpha_scaled = self.alpha / alpha_scale if alpha_scale > 1e-8 else self.alpha

        # STEP 4: Update TD-error scaler before the learning update.
        # For fixed ablations, the scaler sees the same reward/eta units used in the TD target.
        if self.td_scaler_reward_units == "loss":
            r_ent_for_scaler = reward_scaled - alpha_scaled * lprob.detach().item()
        elif self.td_scaler_reward_units == "unscaled":
            r_ent_for_scaler = reward - self.alpha * lprob.detach().item()
        else:
            raise ValueError(f"Unknown td_scaler_reward_units: {self.td_scaler_reward_units}")

        self.G += r_ent_for_scaler
        if done:
            self._update_td_error_scaler(reward_for_scaler=r_ent_for_scaler, gamma=0.0, G=self.G)
            self.G = 0.0
        else:
            self._update_td_error_scaler(reward_for_scaler=r_ent_for_scaler, gamma=self.gamma, G=None)
        td_sigma = max(self.td_sigma, 1e-8)

        # STEP 5: Critic and Actor losses (using scaled reward/entropy)
        q = self.Q(obs, action.detach())

        with torch.no_grad():
            next_action, next_action_info = self.actor(next_obs)
            next_lprob = next_action_info["lprob"]
            q2 = self.Q(next_obs, next_action)
            target_v = q2 - alpha_scaled * next_lprob

        delta = reward_scaled + (1 - float(done)) * self.gamma * target_v - q
        delta = delta / td_sigma
        qloss = (delta ** 2).mean()

        ploss = (alpha_scaled * lprob - self.Q(obs, action)).mean()

        self.last_update_stats = {
            "reward_scale": float(reward_scale),
            "alpha_scaled": float(alpha_scaled),
            "td_sigma": float(td_sigma),
            "q_loss": float(qloss.detach().cpu().item()),
            "policy_loss": float(ploss.detach().cpu().item()),
            "log_prob": float(lprob.detach().cpu().mean().item()),
            "r_ent_for_scaler": float(r_ent_for_scaler),
        }

        
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
        "beta2": args.beta2,
        "betas": [args.beta1, args.beta2],
        "uses_adam_default_betas": getattr(args, "use_adam_default_betas", False),
        "gamma": args.gamma,
        "alpha_lr": args.alpha_lr,
        "nhid_actor": args.nhid_actor,
        "nhid_critic": args.nhid_critic,
        "eval_interval": args.eval_interval,
        "eval_episodes": args.eval_episodes,
        "eval_action_mode": getattr(args, "eval_action_mode", "sample"),
        "uses_avg": True,
        "uses_q_critic": True,
        "uses_reparam_actor": True,
        "action_distribution": getattr(args, "action_distribution", "tanh_normal"),
        "uses_penult_norm": True,
        "uses_td_scaling": True,
        "td_scaler_impl": args.td_scaler_impl,
        "td_scaler_reward_units": args.td_scaler_reward_units,
        "bootstrap_on_truncation": args.bootstrap_on_truncation,
        "log_update_stats": args.log_update_stats,
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

    train_fields = ["step", "episode_return"]
    if args.log_update_stats:
        train_fields.extend(
            [
                "reward_scale",
                "alpha_scaled",
                "td_sigma",
                "q_loss",
                "policy_loss",
                "log_prob",
                "r_ent_for_scaler",
            ]
        )

    train_logger = CSVLogger(
        train_csv_path(args.results_root, args.backend, args.algo, args.env_name, args.seed),
        train_fields,
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
        update_done = terminated if args.bootstrap_on_truncation else done

        agent.update(
            obs,
            action,
            next_obs,
            reward,
            update_done,
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
                eval_action_mode=getattr(args, "eval_action_mode", "sample"),
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

            train_row = {
                "step": t,
                "episode_return": episode_return,
            }
            if args.log_update_stats:
                train_row.update(agent.last_update_stats)
            train_logger.log(train_row)

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
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument(
        "--use_adam_default_betas",
        action="store_true",
        help="Use PyTorch Adam default betas (0.9, 0.999) instead of passing beta1/beta2.",
    )
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
        "--td_scaler_impl",
        type=str,
        default="original",
        choices=["custom", "original"],
        help="TD-error scaler implementation. Use original to match upstream AVG.",
    )
    parser.add_argument(
        "--td_scaler_reward_units",
        type=str,
        default="loss",
        choices=["unscaled", "loss"],
        help="Reward/eta units used to update the TD-error scaler.",
    )
    parser.add_argument(
        "--bootstrap_on_truncation",
        action="store_true",
        help="Match original AVG: do not zero the bootstrap target on time-limit truncation.",
    )
    parser.add_argument(
        "--log_update_stats",
        action="store_true",
        help="Add reward/TD scale and loss diagnostics to train.csv.",
    )
    parser.add_argument(
        "--use_layer_norm",
        action="store_true",
        help="Enable LayerNorm in hidden layers (not in vanilla AVG)"
    )
    parser.add_argument(
        "--action_distribution",
        type=str,
        default="tanh_normal",
        choices=["tanh_normal", "clipped_normal"],
        help="AVG policy action transform. tanh_normal is the original path; clipped_normal improves sparse DMControl finger-spin exploration.",
    )

    # Thesis shell settings
    parser.add_argument("--eval_interval", type=int, default=10000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--eval_action_mode", type=str, default="sample", choices=["sample", "mean"])
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        help="Override the result-folder name. Use avg_default for Phase 2 Adam-default runs.",
    )
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    # AVG Adam setup
    if args.use_adam_default_betas:
        args.beta1 = 0.9
        args.beta2 = 0.999
        args.betas = None
    else:
        args.betas = [args.beta1, args.beta2]

    if torch.cuda.is_available() and "cuda" in args.device:
        args.device = torch.device(args.device)
    else:
        args.device = torch.device("cpu")

    # Naming convention for results: avg_A, avg_B, avg_C (+ _ln if LayerNorm)
    if args.algo is None:
        mode_names = {"td_only": "A", "td_reward": "B", "td_reward_entropy": "C"}
        base_algo = f"avg_{mode_names[args.scaling_mode]}"
        args.algo = f"{base_algo}_ln" if args.use_layer_norm else base_algo
        if args.action_distribution == "clipped_normal":
            args.algo = f"{args.algo}_clip"

    main(args)
