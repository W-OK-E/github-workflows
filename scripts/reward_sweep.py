"""
scripts/reward_sweep.py — Run reward experiments sequentially and print a comparison.

Usage:
    python3 scripts/reward_sweep.py
    python3 scripts/reward_sweep.py --configs-dir configs/reward_experiments --log-dir ./log
    python3 scripts/reward_sweep.py --filter 01_action  # only run configs matching substring
    python3 scripts/reward_sweep.py --dry-run
"""

import argparse
import csv
import glob
import os
import os.path as osp
import subprocess
import sys
import time

SCRIPT_DIR = osp.dirname(osp.abspath(__file__))
REPO_ROOT   = osp.dirname(SCRIPT_DIR)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--configs-dir", default=osp.join(REPO_ROOT, "configs", "reward_experiments"))
    p.add_argument("--log-dir",     default=osp.join(REPO_ROOT, "log"))
    p.add_argument("--seed",        type=int, default=0)
    p.add_argument("--vec-env-nums",type=int, default=4)
    p.add_argument("--proc-nums",   type=int, default=4)
    p.add_argument("--no-cuda",     action="store_true")
    p.add_argument("--filter",      default="", help="Only run configs whose filename contains this string")
    p.add_argument("--dry-run",     action="store_true")
    return p.parse_args()


def read_metric(log_dir, exp_id, env_name, seed, column):
    csv_path = osp.join(log_dir, exp_id, env_name, str(seed), "log.csv")
    if not osp.exists(csv_path):
        return None, None
    try:
        with open(csv_path) as f:
            rows = list(csv.DictReader(f))
        vals = [float(r[column]) for r in rows if r.get(column, "").strip()]
        return (vals[-1], max(vals)) if vals else (None, None)
    except Exception as e:
        print(f"  [warn] {csv_path}: {e}")
        return None, None


def run_experiment(config_path, exp_id, args):
    cmd = [
        sys.executable, osp.join(SCRIPT_DIR, "train.py"),
        "--config", config_path,
        "--id", exp_id,
        "--log_dir", args.log_dir,
        "--seed", str(args.seed),
        "--vec_env_nums", str(args.vec_env_nums),
        "--proc_nums", str(args.proc_nums),
        "--overwrite",
    ]
    if args.no_cuda:
        cmd.append("--no_cuda")

    print(f"\n{'='*60}\n  {exp_id}\n{'='*60}")
    if args.dry_run:
        print(f"  [dry-run] {' '.join(cmd)}")
        return True

    t0 = time.time()
    ok = subprocess.run(cmd, cwd=REPO_ROOT).returncode == 0
    print(f"  {'DONE' if ok else 'FAILED'} in {(time.time()-t0)/60:.1f} min")
    return ok


def print_table(results):
    print(f"\n{'='*72}\n  SWEEP RESULTS\n{'='*72}")
    fmt = f"{{:<38}} {{:>10}} {{:>10}} {{:>8}}"
    print(fmt.format("Experiment", "Final", "Peak", "Status"))
    print("-" * 72)

    baseline_peak = next((peak for eid, _, peak, ok in results if ok and peak is not None
                          and "baseline" in eid), None)
    for exp_id, final, peak, ok in results:
        fs = f"{final:.4f}" if final is not None else "n/a"
        ps = f"{peak:.4f}"  if peak  is not None else "n/a"
        delta = f" ({peak-baseline_peak:+.4f})" if (baseline_peak and peak) else ""
        print(fmt.format(exp_id[:38], fs, ps + delta, "ok" if ok else "FAIL"))
    print(f"{'='*72}")


def main():
    args = parse_args()
    configs = sorted(glob.glob(osp.join(args.configs_dir, "*.json")))
    if args.filter:
        configs = [c for c in configs if args.filter in osp.basename(c)]
    if not configs:
        print(f"No configs found in {args.configs_dir}" +
              (f" matching '{args.filter}'" if args.filter else ""))
        sys.exit(1)

    print(f"Running {len(configs)} experiment(s):")
    for c in configs:
        print(f"  {osp.basename(c)}")

    env_name = "UnitreeMujocoGymEnv"
    results  = []
    for cfg in configs:
        exp_id = osp.splitext(osp.basename(cfg))[0]
        ok     = run_experiment(cfg, exp_id, args)
        final, peak = read_metric(args.log_dir, exp_id, env_name,
                                  args.seed, "Running_Average_Rewards")
        results.append((exp_id, final, peak, ok))

    print_table(results)
    print(f"\nTensorBoard : tensorboard --logdir {args.log_dir}")
    print(f"Full report : python3 scripts/compare_experiments.py --log-dir {args.log_dir}")


if __name__ == "__main__":
    main()
