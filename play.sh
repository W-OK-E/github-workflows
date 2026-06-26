export MUJOCO_GL=glfw
python3 scripts/play.py \
    --config /home/manas/Research/ResLocoTransformer/log_gait/natural_gait_mlp/UnitreeMujocoGymEnv/42/params.json \
    --checkpoint best \
    --episodes 20 \
    --log_dir /home/manas/Research/ResLocoTransformer/log_gait/natural_gait_mlp/UnitreeMujocoGymEnv/42 \
    --render \
    --use_mlp
