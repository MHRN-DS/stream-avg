import numpy as np
import torch

from env_factory import make_eval_env
from fixed_normalization_wrappers import NormalizeObservation, ScaleReward


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
def evaluate_policy(
    agent,
    env_name,
    backend,
    train_env,
    episodes=10,
    seed=0,
    use_time_info=True,
    eval_action_mode="mean",   # "mean" for AC, "sample" for AVG
):
    returns = []
    eval_env = make_eval_env(
        env_name=env_name,
        backend=backend,
        use_time_info=use_time_info,
    )

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
                if eval_action_mode == "mean":
                    x = torch.from_numpy(obs).float().unsqueeze(0)
                    mu, _ = agent.pi(x)
                    action = mu.squeeze(0).detach().cpu().numpy()

                elif eval_action_mode == "sample":
                    action_t, _ = agent.compute_action(obs)
                    action = action_t.detach().cpu().view(-1).numpy()

                else:
                    raise ValueError(f"Unknown eval_action_mode: {eval_action_mode}")

                # Clip only if action space bounds are finite
                if hasattr(eval_env.action_space, "low") and hasattr(eval_env.action_space, "high"):
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