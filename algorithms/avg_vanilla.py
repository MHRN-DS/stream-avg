import argparse
import json
import os
import pickle
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from gymnasium.wrappers import NormalizeObservation, ClipAction

from incremental_rl.td_error_scaler import TDErrorScaler
from incremental_rl.envs.dm_control_wrapper import DMControl
from incremental_rl.envs.gymnasium_wrapper import GymnasiumWrapper
from evaluation.avg_evaluator import evaluate_policy_avg
from paths import ensure_run_dirs, train_csv_path, eval_csv_path, returns_pkl_path
from logging_utils.csv_logger import CSVLogger


def orthogonal_weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)


def set_one_thread():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def save_run_config(results_root, backend, algo, env_name, seed, config):
    run_path = ensure_run_dirs(results_root, backend, algo, env_name, seed)
    config_path = Path(run_path) / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def make_avg_env(env_name: str, backend: str, seed: int = 0, render: bool = False):
    if backend == "dmcontrol":
        task_name = env_name.replace("-v0", "")
        domain, task = task_name.split("-", 1)
        base_env = DMControl(
            domain=domain,
            task=task,
            render_mode="human" if render else None,
        )
    else:
        base_env = GymnasiumWrapper(env=env_name, seed=seed, time_limit=0)

    env = gym.wrappers.FlattenObservation(base_env)
    env = NormalizeObservation(env)
    env = ClipAction(env)
    return env


class Actor(nn.Module):
    def __init__(self, obs_dim, action_dim, device, n_hid):
        super().__init__()
        self.device = device
        self.LOG_STD_MAX = 2
        self.LOG_STD_MIN = -20

        self.phi = nn.Sequential(
            nn.Linear(obs_dim, n_hid),
            nn.LeakyReLU(),
            nn.Linear(n_hid, n_hid),
            nn.LeakyReLU(),
        )

        self.mu = nn.Linear(n_hid, action_dim)
        self.log_std = nn.Linear(n_hid, action_dim)

        self.apply(orthogonal_weight_init)
        self.to(device=device)

    def forward(self, obs):
        phi = self.phi(obs.to(self.device))
        phi = phi / torch.norm(phi, dim=1, keepdim=True).clamp_min(1e-8)
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

    def mean_action(self, obs):
        phi = self.phi(obs.to(self.device))
        phi = phi / torch.norm(phi, dim=1, keepdim=True).clamp_min(1e-8)
        mu = self.mu(phi)
        return torch.tanh(mu)


class Q(nn.Module):
    def __init__(self, obs_dim, action_dim, device, n_hid):
        super().__init__()
        self.device = device

        self.phi = nn.Sequential(
            nn.Linear(obs_dim + action_dim, n_hid),
            nn.LeakyReLU(),
            nn.Linear(n_hid, n_hid),
            nn.LeakyReLU(),
        )
        self.q = nn.Linear(n_hid, 1)

        self.apply(orthogonal_weight_init)
        self.to(device=device)

    def forward(self, obs, action):
        x = torch.cat((obs, action), -1).to(self.device)
        phi = self.phi(x)
        phi = phi / torch.norm(phi, dim=1, keepdim=True).clamp_min(1e-8)
        return self.q(phi).view(-1)


class AVG(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.steps = 0

        self.actor = Actor(cfg.obs_dim, cfg.action_dim, cfg.device, cfg.nhid_actor)
        self.Q = Q(cfg.obs_dim, cfg.action_dim, cfg.device, cfg.nhid_critic)

        self.popt = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr, betas=cfg.betas)
        self.qopt = torch.optim.Adam(self.Q.parameters(), lr=cfg.critic_lr, betas=cfg.betas)

        self.alpha = cfg.alpha_lr
        self.gamma = cfg.gamma
        self.device = cfg.device
        self.td_error_scaler = TDErrorScaler()
        self.G = 0.0

    def compute_action(self, obs):
        obs = torch.tensor(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        action, action_info = self.actor(obs)
        return action, action_info

    def update(self, obs, action, next_obs, reward, terminated, **kwargs):
        obs = torch.tensor(obs.astype(np.float32)).unsqueeze(0).to(self.device)
        next_obs = torch.tensor(next_obs.astype(np.float32)).unsqueeze(0).to(self.device)
        action = action.to(self.device)
        lprob = kwargs["lprob"]

        r_ent = reward - self.alpha * lprob.detach().item()
        self.G += r_ent
        if terminated:
            self.td_error_scaler.update(reward=r_ent, gamma=0, G=self.G)
            self.G = 0.0
        else:
            self.td_error_scaler.update(reward=r_ent, gamma=self.cfg.gamma, G=None)

        q = self.Q(obs, action.detach())
        with torch.no_grad():
            next_action, action_info = self.actor(next_obs)
            next_lprob = action_info["lprob"]
            q2 = self.Q(next_obs, next_action)
            target_V = q2 - self.alpha * next_lprob

        delta = reward + (1 - float(terminated)) * self.gamma * target_V - q
        delta /= self.td_error_scaler.sigma
        qloss = (delta ** 2).mean()

        ploss = (self.alpha * lprob - self.Q(obs, action)).mean()

        self.popt.zero_grad()
        ploss.backward()
        self.popt.step()

        self.qopt.zero_grad()
        qloss.backward()
        self.qopt.step()

        self.steps += 1


def main(args):
    set_one_thread()

    print("\n" + "=" * 70)
    print("AVG Vanilla Baseline")
    print("=" * 70)
    print(f"Environment: {args.env_name} ({args.backend})")
    print(f"Total Steps: {args.total_steps}")
    print(f"Seed: {args.seed}")
    print("=" * 70 + "\n")

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
        "beta2": 0.999,
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
        "uses_obs_norm": True,
        "uses_replay_buffer": False,
        "uses_target_network": False,
    }
    save_run_config(args.results_root, args.backend, args.algo, args.env_name, args.seed, config)

    env = make_avg_env(
        env_name=args.env_name,
        backend=args.backend,
        seed=args.seed,
        render=args.render,
    )

    env.reset(seed=args.seed)
    env.action_space.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.obs_dim = env.observation_space.shape[0]
    args.action_dim = env.action_space.shape[0]
    agent = AVG(args)

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

    obs, _ = env.reset(seed=args.seed)
    ret = 0.0

    for t in range(1, args.total_steps + 1):
        action, action_info = agent.compute_action(obs)
        sim_action = action.detach().cpu().view(-1).numpy()

        next_obs, reward, terminated, truncated, _ = env.step(sim_action)
        done = terminated or truncated

        # Repo-faithful update semantics with this DMC wrapper are fine because truncated=False there
        agent.update(obs, action, next_obs, reward, terminated, **action_info)

        ret += reward
        obs = next_obs

        if t % args.eval_interval == 0:
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
    parser.add_argument("--env_name", type=str, default="Humanoid-v4")
    parser.add_argument("--backend", type=str, default="mujoco", choices=["mujoco", "dmcontrol"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total_steps", type=int, default=1000000)

    parser.add_argument("--actor_lr", type=float, default=0.0063)
    parser.add_argument("--critic_lr", type=float, default=0.0087)
    parser.add_argument("--beta1", type=float, default=0.0)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha_lr", type=float, default=0.07)
    parser.add_argument("--nhid_actor", type=int, default=256)
    parser.add_argument("--nhid_critic", type=int, default=256)

    parser.add_argument("--eval_interval", type=int, default=10000)
    parser.add_argument("--eval_episodes", type=int, default=10)
    parser.add_argument("--results_root", type=str, default="results")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()

    args.betas = [args.beta1, 0.999]
    if torch.cuda.is_available() and "cuda" in args.device:
        args.device = torch.device(args.device)
    else:
        args.device = torch.device("cpu")

    args.algo = "avg_vanilla"
    main(args)