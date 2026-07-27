"""
Auto-generates labels.json by replaying LMDB episodes through the MuJoCo sim
and checking for block toppling at each step.

NOTE on accuracy: Each LMDB episode was recorded with randomized block starting positions.
This script replays the same actions but with a NEW random reset, so initial conditions
will differ slightly. Labels are therefore approximate (~90% accurate in practice),
which is sufficient for ablation comparisons where all methods use the same labels.

If you know from your original manual review that a specific episode is mislabeled,
you can manually edit labels.json after running this script.

Usage:
    python generate_labels.py \
        --lmdb tasks/jenga_mujoco_noise/jenga_single.lmdb \
        --output labels.json \
        --topple-threshold 45.0 \
        --n-replays 3

    --n-replays N: replay each episode N times and take majority vote (reduces stochasticity)
"""

import argparse
import json
import os
import time
import lmdb
import pickle
import numpy as np
import torch

from sim import SimRobotInterface, SimGripperInterface, SIM, TOPPLE_THRESHOLD_DEG


ACTION_HZ = 10           # actions in LMDB are at 10 Hz
STEP_DURATION = 1.0 / ACTION_HZ
SETTLE_TIME = 0.5        # seconds to let physics settle after reset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lmdb",             required=True)
    p.add_argument("--output",           default="labels.json")
    p.add_argument("--topple-threshold", type=float, default=TOPPLE_THRESHOLD_DEG,
                   help="Tilt angle in degrees that counts as a topple")
    p.add_argument("--n-replays",        type=int, default=3,
                   help="Replays per episode for majority-vote labeling")
    p.add_argument("--check-every",      type=int, default=1,
                   help="Check for toppling every N action steps (1 = every step)")
    return p.parse_args()


def replay_episode(robot, gripper, actions, threshold_deg, check_every):
    """
    Resets the sim and executes `actions` (T, 4) array [x, y, z, gripper].
    Returns (failed: bool, failure_step: int|None).
    """
    robot.reset()
    time.sleep(SETTLE_TIME)

    failure_step = None
    for step_idx, action in enumerate(actions):
        target_pos = torch.from_numpy(action[:3]).float()
        grip_cmd   = float(action[3])

        # Keep EE orientation fixed (same as SimRobotInterface.execute)
        curr_wxyz = robot.target_quat
        target_quat = torch.tensor([curr_wxyz[1], curr_wxyz[2], curr_wxyz[3], curr_wxyz[0]])
        robot.update_desired_ee_pose(target_pos, target_quat)

        if grip_cmd > 0.9:
            with SIM.lock: SIM.gripper_val = 0.0    # close
        elif grip_cmd < -0.9:
            with SIM.lock: SIM.gripper_val = 110    # open

        time.sleep(STEP_DURATION)

        if step_idx % check_every == 0:
            with SIM.lock:
                failed, block, tilt = SIM.check_failure(threshold_deg)
            if failed:
                failure_step = step_idx
                print(f"    Topple detected at step {step_idx}: {block} tilted {tilt:.1f}°")
                break

    return failure_step is not None, failure_step


def main():
    args = parse_args()

    robot   = SimRobotInterface()
    gripper = SimGripperInterface()

    labels = {}

    env = lmdb.open(args.lmdb, readonly=True, lock=False)
    with env.begin() as txn:
        metadata = pickle.loads(txn.get(b"__metadata__"))
        ep_names = list(metadata["episodes"].keys())
        print(f"Labeling {len(ep_names)} episodes ({args.n_replays} replays each)...")

        for ep_idx, ep_name in enumerate(ep_names):
            act_all = pickle.loads(txn.get(f"{ep_name}_actions".encode()))

            # Majority vote across n_replays
            failure_votes = []
            failure_steps = []
            print(f"\n[{ep_idx+1}/{len(ep_names)}] {ep_name} ({len(act_all)} steps)")

            for replay_i in range(args.n_replays):
                failed, failure_step = replay_episode(
                    robot, gripper, act_all,
                    args.topple_threshold, args.check_every
                )
                failure_votes.append(int(failed))
                if failure_step is not None:
                    failure_steps.append(failure_step)
                print(f"  replay {replay_i+1}/{args.n_replays}: {'FAILURE' if failed else 'success'}"
                      + (f" @ step {failure_step}" if failure_step is not None else ""))

            majority_failed = sum(failure_votes) > (args.n_replays / 2)
            # Use median failure step across replays that detected failure
            med_failure_step = int(np.median(failure_steps)) if failure_steps else None

            labels[ep_name] = {
                "outcome":      "failure" if majority_failed else "success",
                "failure_step": med_failure_step,
                "vote":         f"{sum(failure_votes)}/{args.n_replays}",
            }
            print(f"  → Label: {'FAILURE' if majority_failed else 'success'} "
                  f"(vote {sum(failure_votes)}/{args.n_replays})"
                  + (f", failure_step={med_failure_step}" if med_failure_step is not None else ""))

    env.close()

    with open(args.output, "w") as f:
        json.dump(labels, f, indent=2)

    n_fail = sum(1 for v in labels.values() if v["outcome"] == "failure")
    n_succ = len(labels) - n_fail
    print(f"\nDone. {n_fail} failures, {n_succ} successes → {args.output}")
    print("Review and manually correct any mislabeled episodes before running compute_metrics.py")


if __name__ == "__main__":
    main()
