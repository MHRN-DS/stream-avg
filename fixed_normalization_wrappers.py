import copy
import numpy as np
import gymnasium as gym


class SampleMeanStd:
    def __init__(self, shape=()):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.p = np.zeros(shape, dtype=np.float64)
        self.count = 0

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        if self.count == 0:
            self.mean = x.copy()
            self.p = np.zeros_like(x, dtype=np.float64)
            self.var = np.ones_like(x, dtype=np.float64)
            self.count = 1
            return
        self.mean, self.var, self.p, self.count = self.update_mean_var_count_from_moments(
            self.mean, self.p, self.count, x
        )

    def update_mean_var_count_from_moments(self, mean, p, count, sample):
        new_count = count + 1
        new_mean = mean + (sample - mean) / new_count
        p = p + (sample - mean) * (sample - new_mean)
        new_var = np.ones_like(sample, dtype=np.float64) if new_count < 2 else p / (new_count - 1)
        return new_mean, new_var, p, new_count

    def copy_from(self, other: "SampleMeanStd"):
        self.mean = np.array(other.mean, copy=True)
        self.var = np.array(other.var, copy=True)
        self.p = np.array(other.p, copy=True)
        self.count = int(other.count)

    def state_dict(self):
        return {
            "mean": np.array(self.mean, copy=True),
            "var": np.array(self.var, copy=True),
            "p": np.array(self.p, copy=True),
            "count": int(self.count),
        }

    def load_state_dict(self, state):
        self.mean = np.array(state["mean"], copy=True)
        self.var = np.array(state["var"], copy=True)
        self.p = np.array(state["p"], copy=True)
        self.count = int(state["count"])


class NormalizeObservation(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env, epsilon: float = 1e-8, update_stats: bool = True):
        gym.utils.RecordConstructorArgs.__init__(self, epsilon=epsilon, update_stats=update_stats)
        gym.Wrapper.__init__(self, env)
        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False

        if self.is_vector_env:
            self.obs_stats = SampleMeanStd(shape=self.single_observation_space.shape)
        else:
            self.obs_stats = SampleMeanStd(shape=self.observation_space.shape)
        self.epsilon = epsilon
        self.update_stats = update_stats

    def set_update_stats(self, enabled: bool):
        self.update_stats = bool(enabled)

    def copy_stats_from(self, other: "NormalizeObservation"):
        self.obs_stats.copy_from(other.obs_stats)

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if self.is_vector_env:
            obs = self.normalize(obs)
        else:
            obs = self.normalize(np.array([obs]))[0]
        return obs, rews, terminateds, truncateds, infos

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if self.is_vector_env:
            return self.normalize(obs), info
        return self.normalize(np.array([obs]))[0], info

    def normalize(self, obs):
        if self.update_stats:
            self.obs_stats.update(obs)
        return (obs - self.obs_stats.mean) / np.sqrt(self.obs_stats.var + self.epsilon)


class ScaleReward(gym.core.Wrapper, gym.utils.RecordConstructorArgs):

    def __init__(self, env: gym.Env, gamma: float = 0.99, epsilon: float = 1e-8):
        gym.utils.RecordConstructorArgs.__init__(self, gamma=gamma, epsilon=epsilon)
        gym.Wrapper.__init__(self, env)
        try:
            self.num_envs = self.get_wrapper_attr("num_envs")
            self.is_vector_env = self.get_wrapper_attr("is_vector_env")
        except AttributeError:
            self.num_envs = 1
            self.is_vector_env = False
        self.reward_stats = SampleMeanStd(shape=())
        self.reward_trace = np.zeros(self.num_envs)
        self.gamma = gamma
        self.epsilon = epsilon

    def reset(self, **kwargs):
        self.reward_trace = np.zeros(self.num_envs)
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, rews, terminateds, truncateds, infos = self.env.step(action)
        if not self.is_vector_env:
            rews = np.array([rews])
        term = np.asarray(terminateds) | np.asarray(truncateds)
        self.reward_trace = self.reward_trace * self.gamma * (1 - term) + rews
        rews = self.normalize(rews)
        if not self.is_vector_env:
            rews = rews[0]
        return obs, rews, terminateds, truncateds, infos

    def normalize(self, rews):
        self.reward_stats.update(self.reward_trace)
        return rews / np.sqrt(self.reward_stats.var + self.epsilon)
