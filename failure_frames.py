"""
Render close-up frames around the moment the automatic failure check fires.

Companion to visualize_failure_check.py. That script proves the checker runs; this one
lets you eyeball *what it saw* at the instant it fired, using a free camera zoomed on the
three blocks (the scene's cam_fixed sits too far back to judge tilt by eye).

For each episode it emits a montage: several steps before detection, the detection frame,
and several after -- each annotated with the live per-block tilt and the verdict.

Usage:
    python failure_frames.py --episodes 1 2 --speed 3
    python failure_frames.py --episodes 2 --threshold 45 --output-dir results/failure_frames
"""

import argparse
import glob
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import mujoco

from sim import SimRobotInterface, SimGripperInterface, SIM, TOPPLE_THRESHOLD_DEG

BLOCKS = ["block_left", "block_middle", "block_right"]
TRACKED = ["block_left", "block_right"]          # what check_failure actually watches


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir",  default="tasks/jenga_mujoco/episodes")
    p.add_argument("--episodes",    nargs="+", default=["2"])
    p.add_argument("--threshold",   type=float, default=TOPPLE_THRESHOLD_DEG)
    p.add_argument("--speed",       type=float, default=3.0)
    p.add_argument("--settle",      type=float, default=0.5)
    p.add_argument("--width",       type=int, default=520)
    p.add_argument("--height",      type=int, default=420)
    p.add_argument("--azimuth",     type=float, default=135.0)
    p.add_argument("--elevation",   type=float, default=-18.0)
    p.add_argument("--distance",    type=float, default=0.42)
    p.add_argument("--output-dir",  default="results/failure_frames")
    return p.parse_args()


def block_xpos(name):
    bid = mujoco.mj_name2id(SIM.model, mujoco.mjtObj.mjOBJ_BODY, name)
    return SIM.data.xpos[bid].copy() if bid != -1 else np.zeros(3)


def make_camera(args):
    """Free camera framed on the block cluster."""
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    centre = np.mean([block_xpos(b) for b in BLOCKS], axis=0)
    cam.lookat[:] = centre
    cam.distance = args.distance
    cam.azimuth = args.azimuth
    cam.elevation = args.elevation
    return cam


def annotate(frame, step, tilts, threshold, failed, worst_block, caption):
    img = frame.copy()
    h, w = img.shape[:2]

    strip = img[0:78, :].copy()
    img[0:78, :] = cv2.addWeighted(strip, 0.3, np.zeros_like(strip), 0.7, 0)

    cv2.putText(img, f"step {step}", (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    for i, b in enumerate(TRACKED):
        t = tilts[b]
        col = (0, 0, 255) if t > threshold else ((0, 165, 255) if t > 0.6 * threshold else (0, 255, 0))
        mark = "  <-- TRIGGER" if (failed and b == worst_block) else ""
        cv2.putText(img, f"{b.replace('block_',''):<5} {t:6.2f}d{mark}", (8, 42 + i * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2, cv2.LINE_AA)

    band = (0, 0, 255) if failed else (0, 150, 0)
    cv2.rectangle(img, (0, h - 26), (w, h), band, -1)
    cv2.putText(img, caption, (8, h - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    if failed:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 4)
    return img


def run(ep, traj_path, args, renderer, cam):
    with open(traj_path) as f:
        waypoints = json.load(f)["waypoints"]
    if not waypoints:
        return None

    robot = SimRobotInterface()
    SimGripperInterface()
    robot.reset()
    time.sleep(args.settle)

    # Re-aim after reset: block poses get randomised on every reset.
    with SIM.lock:
        cam.lookat[:] = np.mean([block_xpos(b) for b in BLOCKS], axis=0)

    frames = []          # (step, bgr, tilts, failed, worst)
    det_step = None
    t0_real, t0_sim = time.time(), waypoints[0]["timestamp"]

    for step, wp in enumerate(waypoints):
        target_t = (wp["timestamp"] - t0_sim) / args.speed
        el = time.time() - t0_real
        if target_t > el:
            time.sleep(target_t - el)

        wxyz = robot.target_quat
        robot.update_desired_ee_pose(
            torch.Tensor(wp["position"]),
            torch.tensor([wxyz[1], wxyz[2], wxyz[3], wxyz[0]]),
        )
        grip = wp["gripper"]
        is_closed = grip if isinstance(grip, bool) else grip > 0.5
        with SIM.lock:
            SIM.gripper_val = 0.0 if is_closed else 110

        with SIM.lock:
            tilts = {b: float(SIM.get_block_tilt(b)) for b in TRACKED}
            failed, worst, _ = SIM.check_failure(args.threshold)
            renderer.update_scene(SIM.data, camera=cam)
            rgb = renderer.render()

        if failed and det_step is None:
            det_step = step
        frames.append((step, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), tilts, failed, worst))

    if det_step is None:
        print(f"    no detection in episode {ep} this run -- nothing to show")
        return {"episode": ep, "detected": None}

    # sample around the detection moment
    offsets = [-40, -20, -8, 0, 8, 25]
    picks = []
    for off in offsets:
        idx = det_step + off
        if 0 <= idx < len(frames):
            picks.append((idx, off))

    tiles = []
    for idx, off in picks:
        step, bgr, tilts, failed, worst = frames[idx]
        if off < 0:
            cap = f"{abs(off)} steps BEFORE detection"
        elif off == 0:
            cap = f"DETECTION (step {step})"
        else:
            cap = f"{off} steps AFTER detection"
        tile = cv2.resize(bgr, (args.width, args.height))
        tiles.append(annotate(tile, step, tilts, args.threshold, failed, worst, cap))

    rows = [np.hstack(tiles[i:i + 3]) for i in range(0, len(tiles), 3)]
    wmax = max(r.shape[1] for r in rows)
    rows = [np.pad(r, ((0, 0), (0, wmax - r.shape[1]), (0, 0))) for r in rows]
    montage = np.vstack(rows)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"ep{ep}_detection.png"
    cv2.imwrite(str(path), montage)

    peak = max(max(f[2][b] for b in TRACKED) for f in frames)
    print(f"    detection at step {det_step}, peak tilt {peak:.1f} deg -> {path}")
    return {"episode": ep, "detected": det_step, "peak_tilt": round(peak, 2),
            "montage": str(path)}


def main():
    args = parse_args()
    renderer = mujoco.Renderer(SIM.model, height=args.height, width=args.width)
    cam = make_camera(args)

    out = []
    for ep in args.episodes:
        tj = sorted(glob.glob(os.path.join(args.source_dir, str(ep), "trajectory_*.json")))
        if not tj:
            print(f"episode {ep}: no trajectory json, skipping")
            continue
        print(f"episode {ep}:")
        r = run(str(ep), tj[0], args, renderer, cam)
        if r:
            out.append(r)

    if out:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(args.output_dir) / "frames_summary.json", "w") as f:
            json.dump(out, f, indent=2)
    print(f"\nDone -> {args.output_dir}/")


if __name__ == "__main__":
    main()
    import sys
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)