export MUJOCO_GL=egl

# Fresh training run (overwrites any existing log for this config/seed):
python3 scripts/train.py --config configs/curriculum_tf_obstacles_gait.json --overwrite --vec_env_nums 8 --proc_nums 8

# Resume training from the latest checkpoint (uncomment to use):
# python3 scripts/train.py --config configs/go2_attnres_mujoco.json --resume

# Resume from a specific checkpoint epoch (uncomment to use):
# python3 scripts/train.py --config configs/go2_attnres_mujoco.json --resume --checkpoint 1000
