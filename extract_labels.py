"""
Build labels.json from the ground truth recorded during replay.

replay_noisy.py records what physically happened *as each rollout executes* and stores it in
the episode's trajectory JSON metadata. This script collects that into the labels.json format
compute_metrics.py expects.

This replaces generate_labels.py for any dataset produced by the instrumented replay.
generate_labels.py re-simulates the actions from a fresh random reset, which yields a
different rollout than the frames the monitor is scored on -- so its labels describe the
wrong trajectory. Here the label and the frames come from the same rollout by construction.

Index re-alignment
------------------
`failure_step` indexes the episode's raw waypoint list. The LMDB builder matches each waypoint
to the nearest camera frame and drops any whose closest frame is more than 0.1 s away (~0.3%),
so LMDB indices are shifted relative to raw waypoint indices. We therefore re-derive the index
from `failure_timestamp` against the retained frame timestamps, which is exact regardless of
how many waypoints were dropped.

Usage:
    python extract_labels.py --episodes-dir tasks/jenga_mujoco_noise/episodes \
                             --lmdb tasks/jenga_mujoco_noise/jenga_single.lmdb \
                             --output labels.json
"""

import argparse
import glob
import json
import os
import pickle
from pathlib import Path

import lmdb
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--episodes-dir", required=True,
                   help="Directory of episode folders written by replay_noisy.py")
    p.add_argument("--lmdb", default=None,
                   help="LMDB to align indices against. Without it, raw waypoint indices "
                        "are emitted unchanged (fine only if nothing was dropped).")
    p.add_argument("--output", default="labels.json")
    p.add_argument("--strict", action="store_true",
                   help="Fail instead of warning when an episode lacks recorded ground truth")
    return p.parse_args()


def load_frame_times(txn, ep_name, meta):
    """Timestamps (seconds) of the cam2 frames the LMDB actually retained, in order."""
    ep = meta["episodes"].get(ep_name)
    if ep is None:
        return None
    keys = ep["keys"]["cam2"]
    # key format: <ep>_cam2_<milliseconds>
    return np.array([int(k.split("_")[-1]) / 1000.0 for k in keys])


def main():
    args = parse_args()
    ep_dirs = sorted(
        [d for d in glob.glob(os.path.join(args.episodes_dir, "*")) if os.path.isdir(d)],
        key=lambda p: int(os.path.basename(p)) if os.path.basename(p).isdigit() else 1 << 30,
    )
    if not ep_dirs:
        print(f"No episode folders under {args.episodes_dir}")
        return

    lmdb_meta = None
    if args.lmdb:
        env = lmdb.open(args.lmdb, readonly=True, lock=False)
        with env.begin() as txn:
            lmdb_meta = pickle.loads(txn.get(b"__metadata__"))
        env.close()

    labels, skipped, realigned = {}, [], 0

    for d in ep_dirs:
        ep_name = os.path.basename(d)
        tj = sorted(glob.glob(os.path.join(d, "trajectory_*.json")))
        if not tj:
            skipped.append((ep_name, "no trajectory json"))
            continue

        with open(tj[0]) as f:
            data = json.load(f)
        m = data.get("metadata", {})

        if "outcome" not in m:
            msg = ("no recorded ground truth -- episode predates the instrumented "
                   "replay_noisy.py; regenerate it or fall back to generate_labels.py")
            if args.strict:
                raise SystemExit(f"{ep_name}: {msg}")
            skipped.append((ep_name, msg))
            continue

        fstep = m.get("failure_step")
        fts = m.get("failure_timestamp")

        # Re-derive the index against the frames the LMDB kept.
        if fstep is not None and fts is not None and lmdb_meta is not None:
            ftimes = load_frame_times(None, ep_name, lmdb_meta)
            if ftimes is not None and len(ftimes):
                idx = int(np.argmin(np.abs(ftimes - fts)))
                if idx != fstep:
                    realigned += 1
                fstep = idx

        labels[ep_name] = {
            "outcome": m["outcome"],
            "failure_step": fstep,
            "peak_tilt_deg": m.get("peak_tilt_deg"),
            "failure_block": m.get("failure_block"),
            "topple_threshold_deg": m.get("topple_threshold_deg"),
            "source": "recorded_during_replay",
        }

    with open(args.output, "w") as f:
        json.dump(labels, f, indent=2)

    n_fail = sum(1 for v in labels.values() if v["outcome"] == "failure")
    n_succ = len(labels) - n_fail
    print(f"episodes    : {len(labels)}")
    print(f"  failure   : {n_fail} ({n_fail/max(len(labels),1)*100:.1f}%)")
    print(f"  success   : {n_succ}")
    if lmdb_meta is not None:
        print(f"realigned   : {realigned} failure_step(s) shifted by LMDB frame matching")
    if skipped:
        print(f"skipped     : {len(skipped)}")
        for ep, why in skipped[:5]:
            print(f"    {ep}: {why}")
    peaks = [v["peak_tilt_deg"] for v in labels.values() if v.get("peak_tilt_deg") is not None]
    if peaks:
        peaks = np.array(peaks)
        print(f"peak tilt   : median {np.median(peaks):.1f}  "
              f"standing(<45) {int((peaks<45).sum())}  toppled(>45) {int((peaks>45).sum())}")
    print(f"\n-> {args.output}")


if __name__ == "__main__":
    main()
