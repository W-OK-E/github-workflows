import gymnasium as gym
import numpy as np

from .base_wrapper import BaseWrapper


class NormAct(gym.ActionWrapper, BaseWrapper):
  """
  Normalized Action      => [ -1, 1 ]
  """

  def __init__(self, env):
    super(NormAct, self).__init__(env)
    ub = np.ones(self.env.action_space.shape)
    self.action_space = gym.spaces.Box(-1 * ub, ub)
    self.lb = self.env.action_space.low
    self.ub = self.env.action_space.high

  def action(self, action):
    action = np.tanh(action)
    scaled_action = self.lb + (action + 1.) * 0.5 * (self.ub - self.lb)
    return np.clip(scaled_action, self.lb, self.ub)

#TODO: Implement NormObs and NormRet wrappers along with the Normalizer class as required in /home/manas/Research/ResLocoTransformer/envs/builder.py line 10

class NormObs(gym.ObservationWrapper, BaseWrapper):
    def __init__(self, env):
        super(NormObs, self).__init__(env)
        self.obs_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=env.observation_space.shape, dtype=np.float32
        )
        self.lb = self.env.observation_space.low
        self.ub = self.env.observation_space.high

    def observation(self, observation):
        observation = np.tanh(observation)
        scaled_observation = self.lb + (observation + 1.) * 0.5 * (self.ub - self.lb)
        return np.clip(scaled_observation, self.lb, self.ub)

class NormRet(gym.RewardWrapper, BaseWrapper):
    def __init__(self, env):
        super(NormRet, self).__init__(env)
        self.ret_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=env.reward_space.shape, dtype=np.float32
        )
        self.lb = self.env.reward_space.low
        self.ub = self.env.reward_space.high

    def reward(self, reward):
        reward = np.tanh(reward)
        scaled_reward = self.lb + (reward + 1.) * 0.5 * (self.ub - self.lb)
        return np.clip(scaled_reward, self.lb, self.ub)

# A normalizer class that can behave as follows:

class Normalizer:
    def __init__(self, shape, epsilon=1e-4, clipob=10.0):
        self.epsilon = epsilon
        self.clipob = clipob
        self.mean = np.zeros(shape)
        self.var = np.ones(shape)
        self.count = 0

    def update_estimate(self, x):
        x = np.asarray(x)
        if x.ndim > 1:
            # Batch input (e.g. from a VecEnv): Welford-style chunk update so
            # mean/var stay shaped (obs_dim,) regardless of batch size.
            m = x.shape[0]
            batch_mean = x.mean(axis=0)
            batch_var = x.var(axis=0)
            n = self.count
            new_count = n + m
            delta = batch_mean - self.mean
            self.mean = self.mean + delta * m / new_count
            self.var = (n * self.var + m * batch_var + delta ** 2 * n * m / new_count) / new_count
            self.count = new_count
        else:
            self.count += 1
            alpha = 1.0 / self.count
            delta = x - self.mean
            self.mean = self.mean + alpha * delta
            self.var = (1 - alpha) * self.var + alpha * (x - self.mean) * delta

    def filt(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + self.epsilon), -self.clipob, self.clipob)

