# Project TODO

## Research & Analysis
- [x] Initial codebase exploration
- [x] Understanding of environment, model, and training pipeline
- [x] Preparation of `report.tex` baseline
- [ ] Review `scripts/compare_experiments.py` to understand how to analyze multiple runs
- [ ] Investigate the `reward_improvement_todo.md` in the `.venv` directory (if possible) or create a fresh overhaul plan

## Implementation Tasks
- [ ] Verify environment rendering locally (if needed)
- [x] Test the `play.py` script with the best checkpoint -- \textbf{Failed: Architecture Incompatible}
- [ ] Investigate if standard transformer is still supported or if checkpoint needs porting.
- [ ] Implement a small tweak to the reward function to verify the training pipeline works

## Experiments
- [ ] Run a baseline training session (Tier 2/3)
- [ ] Experiment with different `attn_res_heads` in `LocoAttnResTransformer`
- [ ] Abalate the effect of depth camera by training a state-only policy



Your goal is to iterate with changes until the quadruped learns a decent gait - which can be checked by monitoring it's height above the ground, how many times it touches the ground, body collisions etc. Feel free to lookup the web.