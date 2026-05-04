from fixed_normalization_wrappers import NormalizeObservation, ScaleReward
from time_wrapper import AddTimeInfo

import gymnasium as gym

def _parse_dmcontrol_name(env_name: str) -> tuple[str, str]:
    # allow: finger-spin-v0, dm_control/finger-spin-v0, dm_control__finger-spin-v0
    task_name = env_name.replace("dm_control/", "").replace("dm_control__", "")
    task_name = task_name.removesuffix("-v0")
    domain, task = task_name.split("-", 1)
    return domain, task

def make_base_env(env_name: str, backend: str, render: bool = False):
    render_mode = "human" if render else None

    if backend not in {"mujoco", "dmcontrol"}:
        raise ValueError(f"Unsupported backend: {backend}")

    if backend == "dmcontrol":
        from incremental_rl.envs.dm_control_wrapper import DMControl

        domain, task = _parse_dmcontrol_name(env_name)
        return DMControl(domain=domain, task=task, render_mode=render_mode)

    kwargs = {}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode

    return gym.make(env_name, **kwargs)


def wrap_train_env(env, gamma: float, use_reward_scaling: bool = True, use_time_info: bool = True):
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)

    if use_reward_scaling:
        env = ScaleReward(env, gamma=gamma)

    env = NormalizeObservation(env)


    if use_time_info:
        env = AddTimeInfo(env)

    return env


def wrap_eval_env(env, use_time_info: bool = True):
    env = gym.wrappers.FlattenObservation(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.ClipAction(env)
    env = NormalizeObservation(env, update_stats=False)
    
    if use_time_info:
    
        env = AddTimeInfo(env)
    return env


def make_train_env(
    env_name: str,
    backend: str,
    gamma: float,
    render: bool = False,
    use_reward_scaling: bool = True,
    use_time_info: bool = True
):
    env = make_base_env(env_name, backend, render=render)
    return wrap_train_env(env, gamma, use_reward_scaling, use_time_info)


def make_eval_env(env_name: str, backend: str, use_time_info: bool = True):
    env = make_base_env(env_name, backend, render=False)
    return wrap_eval_env(env, use_time_info=use_time_info)




