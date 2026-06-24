import numpy as np
from .vecenv import VecEnv
import multiprocessing as mp
from toolz.dicttoolz import merge_with
import sys

mp.set_start_method('spawn', force=True)


def env_worker(env_funcs, env_args, child_pipe, parent_pipe):
  import os
  if "MUJOCO_GL" not in os.environ:
      os.environ["MUJOCO_GL"] = "egl"
  envs = [
    env_func(*env_arg)
    for env_func, env_arg in zip(env_funcs, env_args)
  ]
  sys.stderr = open("train_process.stderr", "a")
  sys.stdout = open("train_process.stdout", "a")

  try:
    while True:
      command, data = child_pipe.recv()

      if command == 'step':
        results = []
        for env, action in zip(envs, data):
          obs, rew, terminated, truncated, info = env.step(np.squeeze(action))
          done = bool(terminated) or bool(truncated)
          results.append((obs, rew, done, info))
        child_pipe.send(results)

      elif command == 'reset':
        # gymnasium reset() returns (obs, info) — extract obs only
        results = [env.reset(**data)[0] for env in envs]
        # print(f"DEBUG: Worker reset returned {len(results)} obs, first shape {results[0].shape if results else 'N/A'}")
        child_pipe.send(results)

      elif command == 'partial_reset':
        index_mask, kwargs = data
        indexs = np.argwhere(index_mask == 1).reshape((-1))
        results = [envs[index].reset(**kwargs)[0] for index in indexs]
        child_pipe.send(results)

      elif command == 'train':
        for env in envs:
          env.train()

      elif command == 'eval':
        for env in envs:
          env.eval()

      elif command == 'close':
        child_pipe.close()
        break

  except Exception as e:
    print(e)
  finally:
    for env in envs:
      env.close()


class SubProcVecEnv(VecEnv):
  def __init__(self, proc_nums, env_nums, env_funcs, env_args):
    self.proc_nums = proc_nums
    super().__init__(env_nums, env_funcs, env_args)

  def set_up_envs(self):
    self.example_env = self.env_funcs[0](*self.env_args[0])
    self.workers = []
    self.parent_pipes = []

    assert self.env_nums % self.proc_nums == 0
    self.env_nums_per_proc = self.env_nums // self.proc_nums

    self.ctx = mp.get_context()
    for i in range(self.proc_nums):
      env_idx_start = i * self.env_nums_per_proc
      env_idx_end   = (i + 1) * self.env_nums_per_proc
      parent_pipe, child_pipe = self.ctx.Pipe()
      p = self.ctx.Process(
        target=env_worker,
        args=(
          self.env_funcs[env_idx_start: env_idx_end],
          self.env_args[env_idx_start: env_idx_end],
          child_pipe,
          parent_pipe,
        )
      )
      p.start()
      self.workers.append(p)
      self.parent_pipes.append(parent_pipe)

  def train(self):
    for parent_pipe in self.parent_pipes:
      parent_pipe.send(('train', None))

  def eval(self):
    for parent_pipe in self.parent_pipes:
      parent_pipe.send(('eval', None))

  def close(self):
    for parent_pipe in self.parent_pipes:
      parent_pipe.send(('close', None))

  def reset(self, **kwargs):
    for parent_pipe in self.parent_pipes:
      parent_pipe.send(('reset', kwargs))

    obs = []
    for parent_pipe in self.parent_pipes:
      worker_obs = parent_pipe.recv()
      # print(f"DEBUG: Parent received {len(worker_obs)} obs from worker")
      obs += worker_obs   # each element is a plain obs array

    if not obs:
      print("CRITICAL ERROR: SubProcVecEnv.reset() received NO observations from workers!")
    else:
      print(f"DEBUG: SubProcVecEnv.reset() stacked obs shape {np.stack(obs).shape}")

    self._obs = np.stack(obs)
    return self._obs, {}          # (obs, info) for gymnasium wrapper compatibility

  def partial_reset(self, index_mask, **kwargs):
    index_mask_per_proc = np.split(index_mask, self.proc_nums)
    for index_mask_current, parent_pipe in zip(index_mask_per_proc, self.parent_pipes):
      parent_pipe.send(('partial_reset', (index_mask_current, kwargs)))

    partial_obs = []
    for parent_pipe in self.parent_pipes:
      partial_obs += parent_pipe.recv()
    self._obs[index_mask] = partial_obs
    return self._obs              # plain array — called directly, not via gymnasium wrapper

  def step(self, actions):
    actions = np.split(actions, self.proc_nums * self.env_nums_per_proc)
    for index, parent_pipe in enumerate(self.parent_pipes):
      parent_pipe.send((
        'step',
        actions[index * self.env_nums_per_proc: (index + 1) * self.env_nums_per_proc],
      ))

    results = []
    for parent_pipe in self.parent_pipes:
      results += parent_pipe.recv()

    # Workers send 4-tuples: (obs, rew, done, info)  (terminated|truncated merged)
    obs, rews, dones, infos = zip(*results)
    self._obs = np.stack(obs)
    terminated = np.stack(dones)[:, np.newaxis]     # (N, 1) bool
    truncated  = np.zeros_like(terminated)           # (N, 1) — already merged into terminated
    infos = merge_with(np.array, *infos)

    # Return 5-tuple for gymnasium ObservationWrapper compatibility
    return self._obs, np.stack(rews)[:, np.newaxis], terminated, truncated, infos

  def seed(self, seed):
    for idx, parent_pipe in enumerate(self.parent_pipes):
      parent_pipe.send(('seed', seed * self.env_nums + idx))

  @property
  def observation_space(self):
    return self.example_env.observation_space

  @property
  def action_space(self):
    return self.example_env.action_space

  def __getattr__(self, attr):
    return getattr(self.example_env, attr)
