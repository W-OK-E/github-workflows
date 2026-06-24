 What changed

  ┌─────────────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────┐
  │              File               │                                         Change                                          │
  ├─────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
  │ envs/mujoco_env.py              │ Added r_ang_vel as always-on base term; 5 use_* enable flags gate optional penalties;   │
  │                                 │ all terms still computed + logged when disabled                                         │
  ├─────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
  │ configs/go2_attnres_mujoco.json │ All reward weights + enable flags in env_build (all disabled by default = clean         │
  │                                 │ baseline)                                                                               │
  ├─────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
  │ torchrl/collector/on_policy.py  │ Accumulates reward/* and diag/* means per training epoch; returns them from             │
  │                                 │ train_one_epoch()                                                                       │
  ├─────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
  │ torchrl/algo/rl_algo.py         │ Forwards those term means into logger.add_epoch_info() → TensorBoard + CSV              │
  ├─────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
  │ configs/reward_experiments/     │ 5 configs: 00_baseline + 4 action-rate weight variants (0.001, 0.01, 0.05, 0.1)         │
  ├─────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
  │ scripts/reward_sweep.py         │ Sequential sweep runner; --filter 01_action runs only the weight sweep                  │
  ├─────────────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────┤
  │ scripts/compare_experiments.py  │ Reads CSVs, prints ranked table with Δ vs baseline, saves 3-panel plot                  │
  └─────────────────────────────────┴───────────────────────────────────────────────