import gymnasium as gym
import numpy as np


class GymnasiumWrapper:
    def __init__(self, env, seed, time_limit) -> None:
        self.env = gym.make(env)
        self.time_limit = time_limit
        self.steps = 0

        # Set seed once at construction
        self.env.reset(seed=seed)
        self.env.action_space.seed(seed)

    def reset(self, **kwargs):
        self.steps = 0
        observation, info = self.env.reset(**kwargs)
        return observation, info

    def step(self, action):
        self.steps += 1

        action = np.clip(action, self.env.action_space.low, self.env.action_space.high)

        next_observation, reward, terminated, truncated, info = self.env.step(action)
        terminated = terminated or truncated
        return next_observation, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    @property
    def observation_space(self):
        return self.env.observation_space

    @property
    def action_space(self):
        return self.env.action_space

    def close(self):
        self.env.close()