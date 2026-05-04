import copy
import json
import pickle
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro

from env_factory import make_train_env
from evaluation.fixed_evaluator import evaluate_policy
from logging_utils.csv_logger import CSVLogger
from paths import ensure_run_dirs, eval_csv_path, returns_pkl_path, train_csv_path

"""
TD3 baseline using the same thesis logging and evaluation pipeline as Stream AC.

MuJoCo:
python -m algorithms.td3_baseline --backend mujoco --env-name HalfCheetah-v4 --seed 1 --total-timesteps 1000000

DM Control:
python -m algorithms.td3_baseline --backend dmcontrol --env-name finger-spin-v0 --seed 1 --total-timesteps 1000000
"""


@dataclass
class Args:
    env_name: str = "HalfCheetah-v4"
    """Environment name, e.g. HalfCheetah-v4 or finger-spin-v0."""

    backend: Literal["mujoco", "dmcontrol"] = "mujoco"
    """Environment backend."""

    seed: int = 1
    """Random seed."""

    total_timesteps: int = 1_000_000
    """Total environment interaction steps."""

    learning_rate: float = 3e-4
    """Actor and critic Adam learning rate."""

    buffer_size: int = 1_000_000
    """Replay buffer capacity."""

    gamma: float = 0.99
    """Discount factor."""

    tau: float = 0.005
    """Target-network Polyak update coefficient."""

    batch_size: int = 256
    """Replay minibatch size."""

    policy_noise: float = 0.2
    """Target policy smoothing noise."""

    exploration_noise: float = 0.1
    """Exploration noise added to actor actions during training."""

    learning_starts: int = 25_000
    """Random-action warmup steps before TD3 updates."""

    policy_frequency: int = 2
    """Delayed actor/target update frequency."""

    noise_clip: float = 0.5
    """Target policy smoothing noise clip."""

    eval_interval: int = 10_000
    """Evaluate every N environment steps."""

    eval_episodes: int = 10
    """Number of episodes per evaluation checkpoint."""

    hidden_size: int = 256
    """Hidden width for actor and critics."""

    use_reward_scaling: bool = False
    """Apply training reward scaling. Evaluation never uses reward scaling."""

    use_time_info: bool = True
    """Append episode-time feature, matching the Stream AC env pipeline."""

    torch_deterministic: bool = True
    """Use deterministic cuDNN settings where available."""

    cuda: bool = True
    """Use CUDA when available."""

    save_model: bool = False
    """Save final TD3 weights under the run artifacts folder."""

    results_root: str = "results"
    """Root directory for thesis results."""

    algo: str = "td3"
    """Result folder name."""

    debug: bool = False
    """Print extra training diagnostics."""


class ReplayBuffer:
    def __init__(self, obs_dim: int, action_dim: int, capacity: int, device: torch.device):
        self.capacity = int(capacity)
        self.device = device
        self.obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, obs, next_obs, action, reward: float, done: bool) -> None:
        self.obs[self.pos] = np.asarray(obs, dtype=np.float32)
        self.next_obs[self.pos] = np.asarray(next_obs, dtype=np.float32)
        self.actions[self.pos] = np.asarray(action, dtype=np.float32)
        self.rewards[self.pos] = float(reward)
        self.dones[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        obs = torch.as_tensor(self.obs[idx], device=self.device)
        next_obs = torch.as_tensor(self.next_obs[idx], device=self.device)
        actions = torch.as_tensor(self.actions[idx], device=self.device)
        rewards = torch.as_tensor(self.rewards[idx], device=self.device)
        dones = torch.as_tensor(self.dones[idx], device=self.device)
        return obs, next_obs, actions, rewards, dones


class Actor(nn.Module):
    def __init__(self, obs_dim: int, action_space: gym.spaces.Box, hidden_size: int, device: torch.device):
        super().__init__()
        action_dim = int(np.prod(action_space.shape))
        self.device = device
        self.fc1 = nn.Linear(obs_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc_mu = nn.Linear(hidden_size, action_dim)

        action_low = torch.as_tensor(action_space.low, dtype=torch.float32, device=device)
        action_high = torch.as_tensor(action_space.high, dtype=torch.float32, device=device)
        if not (torch.isfinite(action_low).all() and torch.isfinite(action_high).all()):
            raise ValueError("TD3 requires finite Box action bounds.")

        self.register_buffer("action_low", action_low)
        self.register_buffer("action_high", action_high)
        self.register_buffer("action_scale", (action_high - action_low) / 2.0)
        self.register_buffer("action_bias", (action_high + action_low) / 2.0)

    def forward(self, x):
        x = x.to(self.device)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_size: int, device: torch.device):
        super().__init__()
        self.device = device
        self.fc1 = nn.Linear(obs_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, obs, action):
        x = torch.cat([obs.to(self.device), action.to(self.device)], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class TD3Agent(nn.Module):
    def __init__(self, obs_dim: int, action_space: gym.spaces.Box, hidden_size: int, device: torch.device):
        super().__init__()
        if not isinstance(action_space, gym.spaces.Box):
            raise TypeError("TD3 only supports continuous Box action spaces.")

        action_dim = int(np.prod(action_space.shape))
        self.device = device
        self.actor = Actor(obs_dim, action_space, hidden_size, device).to(device)
        self.qf1 = QNetwork(obs_dim, action_dim, hidden_size, device).to(device)
        self.qf2 = QNetwork(obs_dim, action_dim, hidden_size, device).to(device)
        self.target_actor = copy.deepcopy(self.actor).to(device)
        self.qf1_target = copy.deepcopy(self.qf1).to(device)
        self.qf2_target = copy.deepcopy(self.qf2).to(device)

    def pi(self, x):
        return self.actor(x), None

    @torch.no_grad()
    def select_action(self, obs: np.ndarray) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.actor(obs_t).squeeze(0).cpu().numpy()


def set_seed(seed: int, torch_deterministic: bool) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = torch_deterministic
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def soft_update(source: nn.Module, target: nn.Module, tau: float) -> None:
    for param, target_param in zip(source.parameters(), target.parameters()):
        target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)


def scalar(value) -> float:
    return float(np.asarray(value).item())


def find_finite_box_action_space(env: gym.Env) -> gym.spaces.Box:
    current = env
    while current is not None:
        action_space = getattr(current, "action_space", None)
        if isinstance(action_space, gym.spaces.Box):
            low = np.asarray(action_space.low)
            high = np.asarray(action_space.high)
            if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
                return action_space
        current = getattr(current, "env", None)
    raise ValueError("TD3 requires a finite Box action space inside the environment wrappers.")


def save_run_config(args: Args, device: torch.device) -> None:
    run_path = ensure_run_dirs(args.results_root, args.backend, args.algo, args.env_name, args.seed)
    config = asdict(args)
    config.update(
        {
            "device": str(device),
            "uses_td3": True,
            "uses_replay_buffer": True,
            "uses_target_networks": True,
            "uses_delayed_policy_updates": True,
            "uses_target_policy_smoothing": True,
            "strict_online_streaming": False,
            "uses_obs_norm": True,
            "uses_reward_scaling_train": args.use_reward_scaling,
            "eval_uses_reward_scaling": False,
            "eval_copies_obs_norm_stats": True,
            "eval_interval": args.eval_interval,
            "eval_episodes": args.eval_episodes,
        }
    )
    with open(Path(run_path) / "config.json", "w") as f:
        json.dump(config, f, indent=2, sort_keys=True)


def log_evaluation(agent: TD3Agent, train_env: gym.Env, args: Args, step: int, eval_logger: CSVLogger) -> None:
    eval_mean, eval_std = evaluate_policy(
        agent=agent,
        env_name=args.env_name,
        backend=args.backend,
        train_env=train_env,
        episodes=args.eval_episodes,
        seed=args.seed,
        use_time_info=args.use_time_info,
        eval_action_mode="mean",
    )
    eval_logger.log(
        {
            "step": step,
            "eval_return_mean": eval_mean,
            "eval_return_std": eval_std,
        }
    )
    print(f"[EVAL] step={step} mean={eval_mean:.2f} std={eval_std:.2f}")


def train(args: Args) -> None:
    torch.set_num_threads(1)
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")
    set_seed(args.seed, args.torch_deterministic)
    ensure_run_dirs(args.results_root, args.backend, args.algo, args.env_name, args.seed)
    save_run_config(args, device)

    train_logger = CSVLogger(
        train_csv_path(args.results_root, args.backend, args.algo, args.env_name, args.seed),
        ["step", "episode_return"],
    )
    eval_logger = CSVLogger(
        eval_csv_path(args.results_root, args.backend, args.algo, args.env_name, args.seed),
        ["step", "eval_return_mean", "eval_return_std"],
    )

    env = make_train_env(
        env_name=args.env_name,
        backend=args.backend,
        gamma=args.gamma,
        render=False,
        use_reward_scaling=args.use_reward_scaling,
        use_time_info=args.use_time_info,
    )
    finite_action_space = find_finite_box_action_space(env)
    finite_action_space.seed(args.seed)

    obs_dim = int(np.prod(env.observation_space.shape))
    action_dim = int(np.prod(finite_action_space.shape))
    agent = TD3Agent(obs_dim, finite_action_space, args.hidden_size, device)
    actor_optimizer = optim.Adam(agent.actor.parameters(), lr=args.learning_rate)
    q_optimizer = optim.Adam(list(agent.qf1.parameters()) + list(agent.qf2.parameters()), lr=args.learning_rate)
    replay = ReplayBuffer(obs_dim, action_dim, args.buffer_size, device)

    returns = []
    term_time_steps = []
    obs, _ = env.reset(seed=args.seed)
    episode_return_fallback = 0.0
    start_time = time.time()

    try:
        for global_step in range(1, args.total_timesteps + 1):
            if global_step <= args.learning_starts:
                action = finite_action_space.sample()
            else:
                action = agent.select_action(obs)
                noise = np.random.normal(
                    loc=0.0,
                    scale=agent.actor.action_scale.detach().cpu().numpy() * args.exploration_noise,
                    size=action_dim,
                )
                action = np.clip(action + noise, finite_action_space.low, finite_action_space.high)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            episode_return_fallback += scalar(reward)

            replay.add(obs, next_obs, action, reward, done)
            obs = next_obs

            if global_step > args.learning_starts and replay.size >= args.batch_size:
                b_obs, b_next_obs, b_actions, b_rewards, b_dones = replay.sample(args.batch_size)

                with torch.no_grad():
                    target_noise = torch.randn_like(b_actions) * args.policy_noise
                    target_noise = target_noise.clamp(-args.noise_clip, args.noise_clip)
                    target_noise = target_noise * agent.target_actor.action_scale
                    next_actions = agent.target_actor(b_next_obs) + target_noise
                    next_actions = torch.max(
                        torch.min(next_actions, agent.target_actor.action_high),
                        agent.target_actor.action_low,
                    )
                    qf1_next = agent.qf1_target(b_next_obs, next_actions)
                    qf2_next = agent.qf2_target(b_next_obs, next_actions)
                    min_qf_next = torch.min(qf1_next, qf2_next)
                    next_q = b_rewards + (1.0 - b_dones) * args.gamma * min_qf_next

                qf1_values = agent.qf1(b_obs, b_actions)
                qf2_values = agent.qf2(b_obs, b_actions)
                qf_loss = F.mse_loss(qf1_values, next_q) + F.mse_loss(qf2_values, next_q)

                q_optimizer.zero_grad()
                qf_loss.backward()
                q_optimizer.step()

                if global_step % args.policy_frequency == 0:
                    actor_loss = -agent.qf1(b_obs, agent.actor(b_obs)).mean()
                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    soft_update(agent.actor, agent.target_actor, args.tau)
                    soft_update(agent.qf1, agent.qf1_target, args.tau)
                    soft_update(agent.qf2, agent.qf2_target, args.tau)

            if done:
                episode_return = scalar(info["episode"]["r"]) if "episode" in info else episode_return_fallback
                returns.append(episode_return)
                term_time_steps.append(global_step)
                train_logger.log({"step": global_step, "episode_return": episode_return})

                if args.debug:
                    elapsed = max(time.time() - start_time, 1e-8)
                    print(
                        f"[TRAIN] step={global_step} return={episode_return:.2f} "
                        f"episodes={len(returns)} sps={int(global_step / elapsed)}"
                    )

                obs, _ = env.reset()
                episode_return_fallback = 0.0

            if global_step % args.eval_interval == 0:
                log_evaluation(agent, env, args, global_step, eval_logger)

    finally:
        env.close()
        train_logger.close()
        eval_logger.close()

    with open(returns_pkl_path(args.results_root, args.backend, args.algo, args.env_name, args.seed), "wb") as f:
        pickle.dump((returns, term_time_steps, args.env_name), f)

    if args.save_model:
        run_path = ensure_run_dirs(args.results_root, args.backend, args.algo, args.env_name, args.seed)
        model_path = Path(run_path) / "artifacts" / "td3_model.pt"
        torch.save(
            {
                "actor": agent.actor.state_dict(),
                "qf1": agent.qf1.state_dict(),
                "qf2": agent.qf2.state_dict(),
                "args": asdict(args),
            },
            model_path,
        )
        print(f"Saved model to {model_path}")


if __name__ == "__main__":
    train(tyro.cli(Args))
