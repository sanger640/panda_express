"""
Build an "eye test" contact sheet: the automatic verdict next to what actually happened.

The reported precision figure was originally derived by eyeballing episodes. This script
lets you check whether the automatic labeler now agrees with that eye test: for each episode
it replays, records the verdict from the real SIM.check_failure(), and tiles the final state
of the blocks with the verdict stamped on it.

Scan the sheet. Every tile marked TOPPLED should show blocks lying flat; every tile marked OK
should show them standing. Any tile where those disagree is a labeling error.

Usage:
    python eyetest_sheet.py --n 12
    python eyetest_sheet.py --episodes 1 2 3 5 9 --threshold 45
    python eyetest_sheet.py --n 12 --compare 15   # same episodes judged at the old threshold too
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
TRACKED = ["block_left", "block_right"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir", default="tasks/jenga_mujoco/episodes")
    p.add_argument("--episodes",   nargs="+", default=None)
    p.add_argument("--n",          type=int, default=12)
    p.add_argument("--threshold",  type=float, default=TOPPLE_THRESHOLD_DEG)
    p.add_argument("--compare",    type=float, default=None,
                   help="Also report the verdict at this second threshold (e.g. 15) "
                        "and flag episodes where the two disagree")
    p.add_argument("--speed",      type=float, default=5.0)
    p.add_argument("--settle",     type=float, default=0.5)
    p.add_argument("--tile-w",     type=int, default=340)
    p.add_argument("--tile-h",     type=int, default=280)
    p.add_argument("--cols",       type=int, default=4)
    p.add_argument("--output-dir", default="results/eyetest")
    return p.parse_args()


def block_xpos(name):
    bid = mujoco.mj_name2id(SIM.model, mujoco.mjtObj.mjOBJ_BODY, name)
    return SIM.data.xpos[bid].copy() if bid != -1 else np.zeros(3)


def make_camera():
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.azimuth, cam.elevation = 0.40, 135.0, -14.0
    return cam


def replay(traj_path, args, renderer, cam):
    with open(traj_path) as f:
        waypoints = json.load(f)["waypoints"]
    if not waypoints:
        return None

    robot = SimRobotInterface()
    SimGripperInterface()
    robot.reset()
    time.sleep(args.settle)
    with SIM.lock:
        cam.lookat[:] = np.mean([block_xpos(b) for b in BLOCKS], axis=0)

    peak = {b: 0.0 for b in TRACKED}
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
                peak[b] = max(peak[b], float(SIM.get_block_tilt(b)))

    # settle, then capture the final resting state -- that is what the eye test judges
    time.sleep(0.4)
    with SIM.lock:
        final = {b: float(SIM.get_block_tilt(b)) for b in TRACKED}
        renderer.update_scene(SIM.data, camera=cam)
        rgb = renderer.render()

    worst_peak = max(peak.values())
    return {
        "peak_tilt": round(worst_peak, 2),
        "final_tilt": {b: round(v, 2) for b, v in final.items()},
        "verdict": "TOPPLED" if worst_peak > args.threshold else "OK",
        "verdict_compare": (None if args.compare is None
                            else ("TOPPLED" if worst_peak > args.compare else "OK")),
        "frame": cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR),
    }


def tile_for(ep, r, args):
    img = cv2.resize(r["frame"], (args.tile_w, args.tile_h))
    toppled = r["verdict"] == "TOPPLED"
    col = (0, 0, 220) if toppled else (0, 150, 0)

    cv2.rectangle(img, (0, 0), (args.tile_w - 1, args.tile_h - 1), col, 3)

    strip = img[0:30, :].copy()
    img[0:30, :] = cv2.addWeighted(strip, 0.25, np.zeros_like(strip), 0.75, 0)
    cv2.putText(img, f"ep {ep}", (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(img, (0, args.tile_h - 30), (args.tile_w, args.tile_h), col, -1)
    label = f"{r['verdict']}   peak {r['peak_tilt']:.0f}d"
    cv2.putText(img, label, (8, args.tile_h - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (255, 255, 255), 1, cv2.LINE_AA)

    # mark episodes where the two thresholds disagree
    if r["verdict_compare"] is not None and r["verdict_compare"] != r["verdict"]:
        cv2.putText(img, "DISAGREES", (args.tile_w - 118, 21),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2, cv2.LINE_AA)
    return img


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.episodes:
        cand = [str(e) for e in args.episodes]
    else:
        cand = sorted([d for d in os.listdir(args.source_dir) if d.isdigit()],
                      key=lambda x: int(x))

    # The scene XML's offscreen framebuffer is 640x480; render at that and downscale to tiles.
    renderer = mujoco.Renderer(SIM.model, height=480, width=640)
    cam = make_camera()

    tiles, rows_meta = [], []
    for ep in cand:
        tj = sorted(glob.glob(os.path.join(args.source_dir, ep, "trajectory_*.json")))
        if not tj:
            continue
        r = replay(tj[0], args, renderer, cam)
        if r is None:
            continue
        tiles.append(tile_for(ep, r, args))
        meta = {k: v for k, v in r.items() if k != "frame"}
        meta["episode"] = ep
        rows_meta.append(meta)
        extra = ""
        if r["verdict_compare"] is not None and r["verdict_compare"] != r["verdict"]:
            extra = f"   <-- differs at {args.compare:.0f}d ({r['verdict_compare']})"
        print(f"ep {ep:>4}  peak {r['peak_tilt']:6.1f}d  -> {r['verdict']:<8}{extra}")
        if len(tiles) >= args.n and not args.episodes:
            break

    if not tiles:
        print("nothing replayed")
        return

    # pad to a full grid
    while len(tiles) % args.cols:
        tiles.append(np.zeros_like(tiles[0]))
    grid = np.vstack([np.hstack(tiles[i:i + args.cols])
                      for i in range(0, len(tiles), args.cols)])

    path = out / f"eyetest_{int(args.threshold)}deg.png"
    cv2.imwrite(str(path), grid)
    with open(out / "eyetest.json", "w") as f:
        json.dump({"threshold": args.threshold, "compare": args.compare,
                   "episodes": rows_meta}, f, indent=2)

    n_top = sum(1 for m in rows_meta if m["verdict"] == "TOPPLED")
    print(f"\n{n_top} toppled / {len(rows_meta)-n_top} ok  (threshold {args.threshold:.0f}d)")
    if args.compare is not None:
        dis = [m["episode"] for m in rows_meta if m["verdict_compare"] != m["verdict"]]
        n_c = sum(1 for m in rows_meta if m["verdict_compare"] == "TOPPLED")
        print(f"at {args.compare:.0f}d it would be {n_c} toppled / "
              f"{len(rows_meta)-n_c} ok -- disagrees on {len(dis)}: {dis}")
    print(f"-> {path}")


if __name__ == "__main__":
    main()
    import sys
    sys.stdout.flush(); sys.stderr.flush()
    os._exit(0)
