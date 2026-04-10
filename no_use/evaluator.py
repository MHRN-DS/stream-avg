import numpy as np
import torch
import gymnasium as gym

from time_wrapper import AddTimeInfo
from normalization_wrappers import NormalizeObservation, ScaleReward


@torch.no_grad()
def evaluate_policy(agent, env_name, gamma, episodes=10, seed=0):
    returns = []

    env = gym.make(env_name)
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    #env = ScaleReward(env, gamma=gamma)
    env = NormalizeObservation(env) 
    env = AddTimeInfo(env)
    # copy stats and modify wrapper normalization
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_return = 0.0

        while not done:
            x = torch.from_numpy(obs).float()
            mu, std = agent.pi(x)
            action = mu.detach().cpu().numpy()   # deterministische Evaluation

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward

        returns.append(ep_return)

    env.close()

    return float(np.mean(returns)), float(np.std(returns))