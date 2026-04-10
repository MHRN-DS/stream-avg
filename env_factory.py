import gymnasium as gym

from fixed_normalization_wrappers import NormalizeObservation, ScaleReward
from time_wrapper import AddTimeInfo

import gymnasium as gym

try:
    import shimmy  # needed to register DM Control envs
except ImportError:
    shimmy = None


def make_base_env(env_name: str, backend: str, render: bool = False):
    render_mode = "human" if render else None

    if backend not in {"mujoco", "dmcontrol"}:
        raise ValueError(f"Unsupported backend: {backend}")

    if backend == "dmcontrol" and shimmy is None:
        raise ImportError(
            "DM Control backend requires shimmy. Install with: pip install 'shimmy[dm-control]'"
        )

    kwargs = {}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode

    return gym.make(env_name, **kwargs)


def wrap_train_env(env, gamma: float, use_reward_scaling: bool = True):
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)

    if use_reward_scaling:
        env = ScaleReward(env, gamma=gamma)

    env = NormalizeObservation(env)
    env = AddTimeInfo(env)
    return env


def wrap_eval_env(env):
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = NormalizeObservation(env, update_stats=False)
    env = AddTimeInfo(env)
    return env


def make_train_env(
    env_name: str,
    backend: str,
    gamma: float,
    render: bool = False,
    use_reward_scaling: bool = True,
):
    env = make_base_env(env_name, backend, render=render)
    return wrap_train_env(env, gamma, use_reward_scaling)


def make_eval_env(env_name: str, backend: str):
    env = make_base_env(env_name, backend, render=False)
    return wrap_eval_env(env)






