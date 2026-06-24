export MUJOCO_GL=glfw
python3 scripts/play.py \
    --config /home/manas/Research/ResLocoTransformer/log_gait/gait_v8_tf_flat/UnitreeMujocoGymEnv/42/params.json \
    --checkpoint best \
    --episodes 20 \
    --log_dir /home/manas/Research/ResLocoTransformer/log_gait/gait_v8_tf_flat/UnitreeMujocoGymEnv/42 \
    --render
