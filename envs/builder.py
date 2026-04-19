"""
envs/builder.py — Environment factory and vectorised-env builders.

keeping only the UnitreeMujocoGymEnv path.  All A1/pybullet code removed.
"""

import gymnasium as gym
import numpy as np

from torchrl.env.continuous_wrapper import NormAct, NormObs, NormRet, Normalizer
from torchrl.env.base_wrapper import BaseWrapper
from torchrl.env.vecenv import VecEnv
from torchrl.env.subproc_vecenv import SubProcVecEnv
from gymnasium.wrappers import TimeLimit


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

def build_unitree_mujoco_env(
    scene_path: str,
    get_image: bool = True,   # accepted for config compatibility; env always has depth
    **kwargs,
):
    """Factory for :class:`UnitreeMujocoGymEnv`.

    All keyword arguments beyond *scene_path* are forwarded to the
    environment constructor (``Kp``, ``Kd``, ``num_action_repeat``,
    ``sim_dt``, ``target_vel``, ``alive_reward``, ``fall_reward``,
    ``max_episode_steps``, ``depth_norm``, ``enable_rendering``).

    ``get_image`` is accepted so that training configs can include the key
    (needed by ``NormObsWithImg`` wrapper detection) without causing an
    unexpected-keyword-argument error.
    """
    from envs.mujoco_env import UnitreeMujocoGymEnv
    return UnitreeMujocoGymEnv(scene_path=scene_path, **kwargs)


ENV_DICT = {
    "UnitreeMujocoGymEnv": build_unitree_mujoco_env,
}

TIMELIMIT_DICT = {
    "UnitreeMujocoGymEnv": 1000,
}


# ---------------------------------------------------------------------------
# Observation normalizer with image pass-through
# ---------------------------------------------------------------------------

class NormObsWithImg(gym.ObservationWrapper, BaseWrapper):
    """Normalises only the state portion of [state | image] observations."""

    def __init__(self, env, epsilon=1e-4, clipob=10.0):
        super().__init__(env)
        self.clipob = clipob
        self._obs_normalizer = Normalizer(env.observation_space.shape)
        self.state_shape = int(np.prod(env.observation_space.shape))

    def observation(self, observation):
        if self.training:
            self._obs_normalizer.update_estimate(
                observation[..., :self.state_shape]
            )
        img_obs = observation[..., self.state_shape:]
        state_obs = self._obs_normalizer.filt(observation[..., :self.state_shape])
        return np.hstack([
            state_obs,
            img_obs,
        ])


# ---------------------------------------------------------------------------
# Single-env builder
# ---------------------------------------------------------------------------

def make_env(env_name, env_build_params):
    return ENV_DICT[env_name](**env_build_params)


def get_single_env(env_id, env_param):
    env = make_env(env_id, env_param["env_build"])
    env = BaseWrapper(env)

    if "rew_norm" in env_param:
        env = NormRet(env, **env_param["rew_norm"])

    if env_id in TIMELIMIT_DICT:
        max_steps = env_param.get("horizon", TIMELIMIT_DICT[env_id])
        env = TimeLimit(env=env, max_episode_steps=max_steps)

    if isinstance(env.action_space, gym.spaces.Box):
        env = NormAct(env)

    return env


# ---------------------------------------------------------------------------
# Vectorised-env builders
# ---------------------------------------------------------------------------

def get_env(env_id, env_param):
    env = get_single_env(env_id, env_param)
    if env_param.get("obs_norm"):
        if "get_image" in env_param.get("env_build", {}):
            env = NormObsWithImg(env)
        else:
            env = NormObs(env)
    return env


def get_vec_env(env_id, env_param, vec_env_nums):
    if isinstance(env_param, list):
        assert vec_env_nums % len(env_param) == 0
        env_args = [
            [env_id, sub] for sub in env_param
        ] * (vec_env_nums // len(env_param))
        vec_env = VecEnv(vec_env_nums, [get_single_env] * vec_env_nums, env_args)
        ref = env_param[0]
    else:
        vec_env = VecEnv(vec_env_nums, get_single_env, [env_id, env_param])
        ref = env_param

    if ref.get("obs_norm"):
        if "get_image" in ref.get("env_build", {}):
            vec_env = NormObsWithImg(vec_env)
        else:
            vec_env = NormObs(vec_env)
    return vec_env


def get_subprocvec_env(env_id, env_param, vec_env_nums, proc_nums):
    if isinstance(env_param, list):
        assert vec_env_nums % len(env_param) == 0
        env_args = [
            [env_id, sub] for sub in env_param
        ] * (vec_env_nums // len(env_param))
        vec_env = SubProcVecEnv(
            proc_nums, vec_env_nums, [get_single_env] * vec_env_nums, env_args
        )
        ref = env_param[0]
    else:
        vec_env = SubProcVecEnv(
            proc_nums, vec_env_nums, get_single_env, [env_id, env_param]
        )
        ref = env_param

    if ref.get("obs_norm"):
        if "get_image" in ref.get("env_build", {}):
            vec_env = NormObsWithImg(vec_env)
        else:
            vec_env = NormObs(vec_env)
    return vec_env
