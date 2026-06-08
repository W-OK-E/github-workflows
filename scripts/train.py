"""
scripts/train.py — PPO training entry point for ResLocoTransformer.

Usage:
    python3 scripts/train.py --config configs/go2_attnres_mujoco.json
    python3 scripts/train.py --config configs/go2_attnres_mujoco.json --resume
    python3 scripts/train.py --config configs/go2_attnres_mujoco.json --resume --checkpoint 1000
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import patch_mujoco
import copy
import glob
import re
import time
import sys
import pickle
import os.path as osp
import numpy as np
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from envs.builder import get_subprocvec_env
import torch
import gymnasium as gym

from torchrl.collector.on_policy import VecOnPolicyCollector
from torchrl.algo import PPO
import torchrl.networks as networks
import torchrl.policies as policies
from torchrl.utils import Logger
from torchrl.replay_buffers.on_policy import OnPolicyReplayBuffer
from torchrl.utils import get_params, get_args


args = get_args()
params = get_params(args.config)


def _find_latest_checkpoint_epoch(model_dir):
    """Return the highest numeric epoch found in model_dir, or None."""
    files = glob.glob(osp.join(model_dir, "model_pf_*.pth"))
    epochs = []
    for f in files:
        m = re.search(r"model_pf_(\d+)\.pth$", osp.basename(f))
        if m:
            epochs.append(int(m.group(1)))
    return max(epochs) if epochs else None


def experiment(args):

    device = torch.device(
        "cuda:{}".format(args.device) if args.cuda else "cpu"
    )

    env = get_subprocvec_env(
        params["env_name"],
        params["env"],
        args.vec_env_nums,
        args.proc_nums,
    )
    eval_env = get_subprocvec_env(
        params["env_name"],
        params["env"],
        max(2, args.vec_env_nums),
        max(2, args.proc_nums),
    )

    if hasattr(env, "_obs_normalizer"):
        eval_env._obs_normalizer = env._obs_normalizer
    #possible error here
    # env.seed(seed = args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    buffer_param = params["replay_buffer"]

    experiment_name = (
        os.path.split(os.path.splitext(args.config)[0])[-1]
        if args.id is None
        else args.id
    )

    # ------------------------------------------------------------------
    # Resolve resume checkpoint before creating the logger so we know
    # whether to keep or wipe the existing work directory.
    # ------------------------------------------------------------------
    resume_epoch = 0
    if args.resume:
        model_dir = osp.join(
            args.log_dir, experiment_name, params["env_name"],
            str(args.seed), "model",
        )
        if args.checkpoint is not None:
            ckpt_epoch = int(args.checkpoint)
        else:
            ckpt_epoch = _find_latest_checkpoint_epoch(model_dir)

        if ckpt_epoch is not None:
            resume_epoch = ckpt_epoch + 1
            print(f"Resuming from checkpoint epoch {ckpt_epoch} "
                  f"(training will start at epoch {resume_epoch})")
        else:
            print("No checkpoint found — starting from scratch")

    logger = Logger(
        experiment_name, params["env_name"],
        args.seed, params, args.log_dir,
        overwrite=args.overwrite,
        resume=args.resume,
    )
    params["general_setting"]["env"] = env

    replay_buffer = OnPolicyReplayBuffer(
        env_nums=args.vec_env_nums,
        max_replay_buffer_size=int(buffer_param["size"]),
        time_limit_filter=buffer_param["time_limit_filter"],
    )
    params["general_setting"]["replay_buffer"] = replay_buffer
    params["general_setting"]["logger"] = logger
    params["general_setting"]["device"] = device

    params["net"]["base_type"] = networks.MLPBase

    encoder = networks.LocoTransformerEncoder(
        in_channels=4,             #!NOTE - Remember depth image is being passed.
        state_input_dim=env.observation_space.shape[0],
        **params["encoder"],
    )
    # Independent copy so actor and critic have separate encoder parameters
    # and each optimizer only updates its own weights (no double-update).
    encoder_vf = copy.deepcopy(encoder)

    # Select LocoAttnResTransformer when 'attn_res_heads' is configured;
    # otherwise fall back to the standard LocoTransformer.
    _use_attn_res = "attn_res_heads" in params.get("net", {})

    if _use_attn_res:
        pf = policies.GaussianContPolicyLocoAttnResTransformer(
            encoder=encoder,
            state_input_shape=env.observation_space.shape[0],
            visual_input_shape=(4, 64, 64),
            output_shape=env.action_space.shape[0],
            **params["net"],
            **params["policy"],
        )
        vf = networks.LocoAttnResTransformer(
            encoder=encoder_vf,
            state_input_shape=env.observation_space.shape[0],
            visual_input_shape=(4, 64, 64),
            output_shape=1,
            **params["net"],
        )
    else:
        pf = policies.GaussianContPolicyLocoTransformer(
            encoder=encoder,
            state_input_shape=env.observation_space.shape[0],
            visual_input_shape=(4, 64, 64),
            output_shape=env.action_space.shape[0],
            **params["net"],
            **params["policy"],
        )
        vf = networks.LocoTransformer(
            encoder=encoder,
            state_input_shape=env.observation_space.shape[0],
            visual_input_shape=(4, 64, 64),
            output_shape=1,
            **params["net"],
        )

    print(pf)
    print(vf)

    params["general_setting"]["collector"] = VecOnPolicyCollector(
        vf, env=env, eval_env=eval_env, pf=pf,
        replay_buffer=replay_buffer, device=device,
        train_render=False,
        **params["collector"],
    )
    params["general_setting"]["save_dir"] = osp.join(
        logger.work_dir, "model"
    )

    # ------------------------------------------------------------------
    # Load checkpoint weights when resuming
    # ------------------------------------------------------------------
    if args.resume and resume_epoch > 0:
        ckpt_epoch = resume_epoch - 1
        model_dir = params["general_setting"]["save_dir"]
        pf_path = osp.join(model_dir, f"model_pf_{ckpt_epoch}.pth")
        vf_path = osp.join(model_dir, f"model_vf_{ckpt_epoch}.pth")
        norm_path = osp.join(model_dir, f"_obs_normalizer_{ckpt_epoch}.pkl")

        pf.load_state_dict(torch.load(pf_path, map_location=device))
        vf.load_state_dict(torch.load(vf_path, map_location=device))
        print(f"Loaded pf weights from {pf_path}")
        print(f"Loaded vf weights from {vf_path}")

        if osp.exists(norm_path) and hasattr(env, "_obs_normalizer"):
            with open(norm_path, "rb") as f:
                loaded_norm = pickle.load(f)
            env._obs_normalizer = loaded_norm
            eval_env._obs_normalizer = loaded_norm
            print(f"Loaded obs normalizer from {norm_path}")

    agent = PPO(
        pf=pf,
        vf=vf,
        **params["ppo"],
        **params["general_setting"],
        resume_epoch=resume_epoch,
    )
    agent.train()


if __name__ == "__main__":
    experiment(args)
