export MUJOCO_GL=glfw
python3 scripts/play.py \
    --config /home/manas/Research/ResLocoTransformer/log/go2_attnres_mujoco/UnitreeMujocoGymEnv/0/params.json \
    --checkpoint best \
    --episodes 10 \
    --log_dir /home/manas/Research/ResLocoTransformer/log/go2_attnres_mujoco/UnitreeMujocoGymEnv/0 \
    --render
