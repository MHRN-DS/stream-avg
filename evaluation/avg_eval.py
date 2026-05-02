import numpy as np
import torch
import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation, ClipAction

from incremental_rl.envs.dm_control_wrapper import DMControl
from incremental_rl.envs.gymnasium_wrapper import GymnasiumWrapper


def make_avg_eval_env(env_name: str, backend: str, seed: int = 0, render: bool = False):
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


def find_gym_norm_wrapper(env):
    current = env
    while current is not None:
        if isinstance(current, NormalizeObservation):
            return current
        current = getattr(current, "env", None)
    raise ValueError("Gymnasium NormalizeObservation wrapper not found")


@torch.no_grad()
def evaluate_policy_avg(agent, env_name, backend, train_env, episodes=10, seed=0):
    returns = []
    eval_env = make_avg_eval_env(env_name=env_name, backend=backend, seed=seed, render=False)

    train_norm = find_gym_norm_wrapper(train_env)
    eval_norm = find_gym_norm_wrapper(eval_env)

    # copy train obs normalization stats into eval env
    eval_norm.obs_rms.mean = train_norm.obs_rms.mean.copy()
    eval_norm.obs_rms.var = train_norm.obs_rms.var.copy()
    eval_norm.obs_rms.count = train_norm.obs_rms.count

    # freeze eval normalization
    eval_norm.update_running_mean = False

    was_training = agent.training
    agent.eval()

    try:
        for ep in range(episodes):
            obs, _ = eval_env.reset(seed=seed + ep)
            done = False
            ep_return = 0.0

            while not done:
                x = torch.tensor(obs.astype(np.float32)).unsqueeze(0).to(agent.device)
                action = agent.actor.mean_action(x).detach().cpu().view(-1).numpy()

                low = eval_env.action_space.low
                high = eval_env.action_space.high
                if np.all(np.isfinite(low)) and np.all(np.isfinite(high)):
                    action = np.clip(action, low, high)

                obs, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                ep_return += float(reward)

            returns.append(ep_return)
    finally:
        if was_training:
            agent.train()
        eval_env.close()

    return float(np.mean(returns)), float(np.std(returns, ddof=0))