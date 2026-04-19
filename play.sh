export MUJOCO_GL=glfw
python3 scripts/play.py \
    --config configs/go2_attnres_mujoco.json \
    --checkpoint 1000 \
    --episodes 10 \
    --render
