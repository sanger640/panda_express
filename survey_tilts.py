"""
Survey peak block tilt across many episodes to choose a defensible topple threshold.

Motivation: SIM.check_failure defaults to 15 deg, but a block leaning 15-30 deg is still
standing -- "toppled" should mean lying flat (~90 deg from vertical). This script replays
episodes and reports the distribution of peak tilt so the threshold can be set from the
data rather than assumed.

Reports two metrics side by side:
    quat  -- current implementation: geodesic angle from the post-reset reference quaternion.
             Includes yaw, so a block that slides/spins without tipping still registers.
    axis  -- angle between the block's own z-axis and world z: true tilt from vertical,
             immune to yaw. This is what "lying flat" actually means.

Usage:
    python survey_tilts.py --n 12 --speed 4
    python survey_tilts.py --episodes 1 2 3 --repeat 3
"""

import argparse
import glob
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import mujoco

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sim import SimRobotInterface, SimGripperInterface, SIM

TRACKED = ["block_left", "block_right"]

# Interpretation bands for tilt-from-vertical.
UPRIGHT_MAX   = 10.0    # below this: untouched
PERTURBED_MAX = 45.0    # 10-45: nudged but still standing;  >45: past balance, going flat


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", default="tasks/jenga_mujoco/episodes")
    p.add_argument("--episodes",   nargs="+", default=None)
    p.add_argument("--n",          type=int, default=10)
    p.add_argument("--repeat",     type=int, default=1)
    p.add_argument("--speed",      type=float, default=4.0)
    p.add_argument("--settle",     type=float, default=0.5)
    p.add_argument("--output-dir", default="results/tilt_survey")
    return p.parse_args()


def axis_tilt(name):
    """Angle between the block's own z-axis and world z (degrees). Yaw-immune."""
    bid = mujoco.mj_name2id(SIM.model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid == -1:
        return 0.0
    R = SIM.data.xmat[bid].reshape(3, 3)
    return float(np.degrees(np.arccos(np.clip(R[2, 2], -1.0, 1.0))))


def classify(tilt):
    if tilt < UPRIGHT_MAX:
        return "upright"
    if tilt < PERTURBED_MAX:
        return "perturbed"
    return "toppled"


def replay(traj_path, args):
    with open(traj_path) as f:
        waypoints = json.load(f)["waypoints"]
    if not waypoints:
        return None

    robot = SimRobotInterface()
    SimGripperInterface()
    robot.reset()
    time.sleep(args.settle)

    base = {b: axis_tilt(b) for b in TRACKED}
    peak_q = {b: 0.0 for b in TRACKED}
    peak_a = {b: 0.0 for b in TRACKED}

    t0, s0 = time.time(), waypoints[0]["timestamp"]
    for wp in waypoints:
        tt = (wp["timestamp"] - s0) / args.speed
        el = time.time() - t0
        if tt > el:
            time.sleep(tt - el)

        q = robot.target_quat
        robot.update_desired_ee_pose(
            torch.Tensor(wp["position"]),
            torch.tensor([q[1], q[2], q[3], q[0]]),
        )
        g = wp["gripper"]
        is_closed = g if isinstance(g, bool) else g > 0.5
        with SIM.lock:
            SIM.gripper_val = 0.0 if is_closed else 110

        with SIM.lock:
            for b in TRACKED:
                peak_q[b] = max(peak_q[b], float(SIM.get_block_tilt(b)))
                peak_a[b] = max(peak_a[b], abs(axis_tilt(b) - base[b]))

    return {
        "peak_quat": max(peak_q.values()),
        "peak_axis": max(peak_a.values()),
        "per_block_axis": {b: round(peak_a[b], 2) for b in TRACKED},
    }


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.episodes:
        cand = [str(e) for e in args.episodes]
    else:
        cand = sorted([d for d in os.listdir(args.source_dir) if d.isdigit()],
                      key=lambda x: int(x))

    rows = []
    for ep in cand:
        tj = sorted(glob.glob(os.path.join(args.source_dir, ep, "trajectory_*.json")))
        if not tj:
            continue
        for rep in range(args.repeat):
            r = replay(tj[0], args)
            if r is None:
                continue
            r["episode"] = ep
            r["replay"] = rep + 1
            r["class_axis"] = classify(r["peak_axis"])
            rows.append(r)
            print(f"ep {ep:>4} r{rep+1}  quat {r['peak_quat']:6.1f}d  "
                  f"axis {r['peak_axis']:6.1f}d  -> {r['class_axis']}")
        if not args.episodes and len({x['episode'] for x in rows}) >= args.n:
            break

    if not rows:
        print("nothing replayed")
        return

    axis = np.array([r["peak_axis"] for r in rows])
    quat = np.array([r["peak_quat"] for r in rows])

    counts = {c: sum(1 for r in rows if r["class_axis"] == c)
              for c in ["upright", "perturbed", "toppled"]}

    print("\n--- peak tilt-from-vertical (axis metric) ---")
    print(f"  n runs      : {len(rows)}")
    print(f"  upright   (<{UPRIGHT_MAX:.0f}d) : {counts['upright']}")
    print(f"  perturbed (<{PERTURBED_MAX:.0f}d) : {counts['perturbed']}")
    print(f"  toppled   (>{PERTURBED_MAX:.0f}d) : {counts['toppled']}")
    print(f"  median {np.median(axis):.1f}d   mean {axis.mean():.1f}d   max {axis.max():.1f}d")

    print("\n--- how the verdict changes with threshold (axis metric) ---")
    print(f"{'thresh':>8} {'failures':>9} {'rate':>7}")
    for th in [10, 15, 20, 30, 45, 60, 75]:
        n = int((axis > th).sum())
        print(f"{th:>7}d {n:>9} {n/len(axis):>6.0%}")

    plt.figure(figsize=(10, 4.5))
    bins = np.arange(0, 100, 5)
    plt.hist(axis, bins=bins, alpha=0.75, color="#3b7dd8", edgecolor="white",
             label="peak tilt from vertical (axis)")
    plt.hist(quat, bins=bins, alpha=0.4, color="#d8733b", edgecolor="white",
             label="peak quaternion deviation (current metric)")
    plt.axvline(15, color="red", linestyle="--", linewidth=1.5, label="current threshold (15d)")
    plt.axvline(PERTURBED_MAX, color="green", linestyle="--", linewidth=1.5,
                label=f"proposed topple boundary ({PERTURBED_MAX:.0f}d)")
    plt.xlabel("peak tilt (deg)")
    plt.ylabel("count")
    plt.title("Peak block tilt across episodes — where is a 'topple'?")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out / "tilt_distribution.png", dpi=120)
    plt.close()

    with open(out / "survey.json", "w") as f:
        json.dump({"bands": {"upright_max": UPRIGHT_MAX, "perturbed_max": PERTURBED_MAX},
                   "counts": counts, "runs": rows}, f, indent=2)
    print(f"\n-> {out}/tilt_distribution.png")


if __name__ == "__main__":
    main()
    import sys
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
