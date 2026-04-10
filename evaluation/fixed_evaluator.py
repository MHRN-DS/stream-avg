import numpy as np
import torch

from env_factory import make_eval_env
from fixed_normalization_wrappers import NormalizeObservation, ScaleReward

# helper functions to check for and find specific wrappers in the environment stack, which is useful for ensuring that the evaluation environment has the same normalization statistics as the training environment.
def has_wrapper(env, wrapper_type):
    current = env
    while current is not None:
        if isinstance(current, wrapper_type):
            return True
        current = getattr(current, "env", None)
    return False

def find_wrapper(env, wrapper_type):
    current = env
    while current is not None:
        if isinstance(current, wrapper_type):
            return current
        current = getattr(current, "env", None)
    raise ValueError(f"Wrapper {wrapper_type.__name__} not found")


@torch.no_grad()
def evaluate_policy(agent, env_name, backend, train_env, episodes=10, seed=0):
    returns = []
    eval_env = make_eval_env(env_name=env_name, backend=backend)

    if has_wrapper(eval_env, ScaleReward):
        raise RuntimeError("Evaluation environment should not have reward scaling wrapper")

    train_norm = find_wrapper(train_env, NormalizeObservation)
    eval_norm = find_wrapper(eval_env, NormalizeObservation)
    eval_norm.copy_stats_from(train_norm)
    eval_norm.set_update_stats(False)

    was_training = agent.training
    agent.eval()
    try:
        for ep in range(episodes):
            obs, _ = eval_env.reset(seed=seed + ep)
            done = False
            ep_return = 0.0

            while not done:
                x = torch.from_numpy(obs).float()
                mu, _ = agent.pi(x)
                action =  mu.detach().cpu().numpy() #np.clip(mu.detach().cpu().numpy(), -1.0, 1.0)
                obs, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                ep_return += reward

            returns.append(ep_return)
    finally:
        if was_training:
            agent.train()
        eval_env.close()

    return float(np.mean(returns)), float(np.std(returns, ddof=0))