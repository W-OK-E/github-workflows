"""
envs/wrappers.py — Gym observation wrappers.

ObservationDictionaryToArrayWrapper flattens a dict-of-arrays observation
into a single flat numpy array.  It is included for compatibility with
pybullet-based A1 environments; the MuJoCo UnitreeMujocoGymEnv already
returns a flat array and does NOT require this wrapper.
"""

import numpy as np
import gymnasium as gym
from gym import spaces


def _flatten_obs_spaces(obs_space: spaces.Dict, excluded=()):
    """Return a flat Box spanning all keys not in *excluded*."""
    low, high = [], []
    for key, space in obs_space.spaces.items():
        if key in excluded:
            continue
        low.append(space.low.flatten())
        high.append(space.high.flatten())
    return spaces.Box(
        low=np.concatenate(low),
        high=np.concatenate(high),
        dtype=np.float32,
    )


def _flatten_obs(obs_dict: dict, excluded=()):
    """Concatenate dict observation values into a flat array."""
    parts = [
        np.asarray(v).flatten()
        for k, v in obs_dict.items()
        if k not in excluded
    ]
    return np.concatenate(parts).astype(np.float32)


class ObservationDictionaryToArrayWrapper(gym.Env):
    """Flattens a dict-keyed observation space into a 1-D array."""

    def __init__(self, gym_env, observation_excluded=()):
        self.observation_excluded = observation_excluded
        self._gym_env = gym_env
        self.observation_space = _flatten_obs_spaces(
            gym_env.observation_space, observation_excluded
        )
        self.action_space = gym_env.action_space

    def __getattr__(self, attr):
        return getattr(self._gym_env, attr)

    def reset(self, **kwargs):
        obs = self._gym_env.reset(**kwargs)
        return _flatten_obs(obs, self.observation_excluded)

    def step(self, action):
        obs, reward, done, info = self._gym_env.step(action)
        return _flatten_obs(obs, self.observation_excluded), reward, done, info

    def render(self, mode="human"):
        return self._gym_env.render(mode)
