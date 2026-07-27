"""
Visualize the automatic failure check (SIM.check_failure) on source episodes.

Replays raw teleop episodes from tasks/jenga_mujoco/episodes through the MuJoCo sim
and records, at every step, the tilt of both adjacent blocks and the verdict returned
by the *real* SIM.check_failure() used by generate_labels.py.

Unlike generate_labels.py this needs no LMDB — it reads the trajectory JSONs directly,
so it works before the noisy dataset has been regenerated.

Outputs per episode:
    <out>/<ep>/video.mp4    annotated replay (external cam) with live tilt + verdict
    <out>/<ep>/tilt.png     tilt-vs-step plot with threshold and detection point
    <out>/summary.json      per-episode verdict, peak tilt, detection step
    <out>/summary.png       tilt curves for all episodes on one axis

Usage:
    python visualize_failure_check.py --episodes 1 2 3 --speed 4
    python visualize_failure_check.py --n 5 --topple-threshold 15.0
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mujoco

# Importing sim launches the passive viewer and the physics thread.
from sim import SimRobotInterface, SimGripperInterface, SIM


BLOCKS = ["block_left", "block_right"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source-dir",       default="tasks/jenga_mujoco/episodes")
    p.add_argument("--episodes",         nargs="+", default=None,
                   help="Explicit episode ids, e.g. --episodes 1 7 42")
    p.add_argument("--n",                type=int, default=3,
                   help="If --episodes not given, replay the first N usable episodes")
    p.add_argument("--topple-threshold", type=float, default=15.0)
    p.add_argument("--speed",            type=float, default=4.0,
                   help="Playback speed multiplier (1.0 = original real-time)")
    p.add_argument("--settle",           type=float, default=0.5,
                   help="Seconds to let physics settle after reset")
    p.add_argument("--fps",              type=int, default=20, help="Output video fps")
    p.add_argument("--width",            type=int, default=640)
    p.add_argument("--height",           type=int, default=480)
    p.add_argument("--camera",           default="cam_fixed",
                   help="MuJoCo camera to render (cam_fixed | cam_wrist)")
    p.add_argument("--output-dir",       default="results/failure_check_viz")
    p.add_argument("--stop-on-failure",  action="store_true",
                   help="Stop replay at first detection (matches generate_labels.py behaviour)")
    p.add_argument("--repeat",           type=int, default=1,
                   help="Replay each episode N times with different random resets, to measure "
                        "how repeatable the verdict is (this is what generate_labels.py's "
                        "majority vote relies on)")
    p.add_argument("--no-video",         action="store_true",
                   help="Skip video writing (much faster for repeatability sweeps)")
    return p.parse_args()


def find_episodes(source_dir, episodes, n):
    """Return [(ep_name, trajectory_json_path), ...] for usable episodes."""
    if episodes:
        cand = [str(e) for e in episodes]
    else:
        cand = sorted(
            [d for d in os.listdir(source_dir) if d.isdigit()],
            key=lambda x: int(x),
        )

    out = []
    for ep in cand:
        tj = sorted(glob.glob(os.path.join(source_dir, ep, "trajectory_*.json")))
        if not tj:
            print(f"  skipping episode {ep}: no trajectory json")
            continue
        out.append((ep, tj[0]))
        if not episodes and len(out) >= n:
            break
    return out


def annotate(frame, step, n_steps, tilts, threshold, failed, det_step):
    """Draw tilt readouts and verdict banner onto a BGR frame."""
    img = frame.copy()
    h, w = img.shape[:2]

    # translucent header strip
    strip = img[0:96, :].copy()
    img[0:96, :] = cv2.addWeighted(strip, 0.35, np.zeros_like(strip), 0.65, 0)

    cv2.putText(img, f"step {step:04d}/{n_steps}", (10, 22),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

    for i, name in enumerate(BLOCKS):
        tilt = tilts[name]
        near = tilt > 0.6 * threshold
        color = (0, 0, 255) if tilt > threshold else ((0, 165, 255) if near else (0, 255, 0))
        label = f"{name.replace('block_',''):<5} {tilt:6.2f} deg"
        cv2.putText(img, label, (10, 46 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)
        # tilt bar, saturating at 2x threshold
        bar_x, bar_w = 210, 220
        frac = float(np.clip(tilt / (2 * threshold), 0, 1))
        y = 38 + i * 22
        cv2.rectangle(img, (bar_x, y), (bar_x + bar_w, y + 12), (90, 90, 90), 1)
        cv2.rectangle(img, (bar_x, y), (bar_x + int(bar_w * frac), y + 12), color, -1)
        # threshold tick at the halfway point of the bar
        tx = bar_x + bar_w // 2
        cv2.line(img, (tx, y - 2), (tx, y + 14), (255, 255, 255), 1)

    cv2.putText(img, f"threshold {threshold:.0f} deg", (bar_x + bar_w + 12, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    if failed:
        cv2.rectangle(img, (0, 0), (w - 1, h - 1), (0, 0, 255), 6)
        banner = f"TOPPLE DETECTED @ step {det_step}"
        cv2.putText(img, banner, (10, h - 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    return img


def plot_episode(ep, hist, threshold, det_step, path):
    steps = np.arange(len(hist[BLOCKS[0]]))
    plt.figure(figsize=(9, 4))
    for name in BLOCKS:
        plt.plot(steps, hist[name], label=name.replace("block_", ""), linewidth=1.6)
    plt.axhline(threshold, color="red", linestyle="--", linewidth=1.2,
                label=f"threshold ({threshold:.0f} deg)")
    if det_step is not None:
        plt.axvline(det_step, color="black", linestyle=":", linewidth=1.4,
                    label=f"detected @ {det_step}")
    plt.xlabel("replay step")
    plt.ylabel("tilt from reference upright (deg)")
    plt.title(f"episode {ep} — automatic failure check")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def replay(ep, traj_path, args, renderer, tag=""):
    with open(traj_path) as f:
        waypoints = json.load(f)["waypoints"]
    if not waypoints:
        return None

    ep_dir = Path(args.output_dir) / (ep + tag)
    ep_dir.mkdir(parents=True, exist_ok=True)

    robot = SimRobotInterface()
    SimGripperInterface()

    robot.reset()
    time.sleep(args.settle)

    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(
            str(ep_dir / "video.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            args.fps, (args.width, args.height),
        )

    hist = {b: [] for b in BLOCKS}
    det_step, peak_tilt, peak_block = None, 0.0, None

    t0_real = time.time()
    t0_sim = waypoints[0]["timestamp"]

    for step, wp in enumerate(waypoints):
        # pace playback against the original recording timestamps
        target_t = (wp["timestamp"] - t0_sim) / args.speed
        elapsed = time.time() - t0_real
        if target_t > elapsed:
            time.sleep(target_t - elapsed)

        # hold orientation fixed, exactly as generate_labels.py does
        wxyz = robot.target_quat
        quat = torch.tensor([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])
        robot.update_desired_ee_pose(torch.Tensor(wp["position"]), quat)

        grip = wp["gripper"]
        is_closed = grip if isinstance(grip, bool) else grip > 0.5
        with SIM.lock:
            SIM.gripper_val = 0.0 if is_closed else 110

        # read tilts + verdict from the real implementation under test
        with SIM.lock:
            tilts = {b: float(SIM.get_block_tilt(b)) for b in BLOCKS}
            failed, block, worst = SIM.check_failure(args.topple_threshold)
            rgb = None
            if writer is not None:
                renderer.update_scene(SIM.data, camera=args.camera)
                rgb = renderer.render()

        for b in BLOCKS:
            hist[b].append(tilts[b])
        if worst > peak_tilt:
            peak_tilt, peak_block = float(worst), block or peak_block

        if failed and det_step is None:
            det_step = step
            print(f"    TOPPLE at step {step}: {block} tilted {worst:.1f} deg")

        if writer is not None:
            frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            if frame.shape[:2] != (args.height, args.width):
                frame = cv2.resize(frame, (args.width, args.height))
            writer.write(annotate(frame, step, len(waypoints), tilts,
                                  args.topple_threshold, det_step is not None, det_step))

        if failed and args.stop_on_failure:
            break

    if writer is not None:
        writer.release()
        plot_episode(ep + tag, hist, args.topple_threshold, det_step, ep_dir / "tilt.png")

    return {
        "episode": ep,
        "n_steps": len(hist[BLOCKS[0]]),
        "outcome": "failure" if det_step is not None else "success",
        "detection_step": det_step,
        "peak_tilt_deg": round(peak_tilt, 2),
        "peak_block": peak_block,
        "final_tilt_deg": {b: round(hist[b][-1], 2) for b in BLOCKS},
        "video": str(ep_dir / "video.mp4"),
        "plot": str(ep_dir / "tilt.png"),
        "tilt_history": {b: [round(v, 3) for v in hist[b]] for b in BLOCKS},
    }


def summary_plot(results, threshold, path):
    plt.figure(figsize=(10, 5))
    for r in results:
        worst = max(BLOCKS, key=lambda b: max(r["tilt_history"][b]))
        series = r["tilt_history"][worst]
        style = "-" if r["outcome"] == "failure" else "--"
        plt.plot(series, style, linewidth=1.5,
                 label=f"ep {r['episode']} ({r['outcome']}, peak {r['peak_tilt_deg']:.1f})")
    plt.axhline(threshold, color="red", linestyle="--", linewidth=1.2,
                label=f"threshold ({threshold:.0f} deg)")
    plt.xlabel("replay step")
    plt.ylabel("worst block tilt (deg)")
    plt.title("Automatic failure check — worst-block tilt across episodes")
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    eps = find_episodes(args.source_dir, args.episodes, args.n)
    if not eps:
        print("No usable episodes found.")
        return
    print(f"Replaying {len(eps)} episode(s) at {args.speed}x, "
          f"threshold {args.topple_threshold} deg\n")

    renderer = mujoco.Renderer(SIM.model, height=args.height, width=args.width)

    results = []
    repeat_stats = {}
    for i, (ep, traj) in enumerate(eps):
        print(f"[{i+1}/{len(eps)}] episode {ep}")
        verdicts = []
        for rep in range(args.repeat):
            tag = "" if args.repeat == 1 else f"_r{rep+1}"
            t0 = time.time()
            r = replay(ep, traj, args, renderer, tag=tag)
            if r is None:
                print("    empty trajectory, skipped")
                break
            results.append(r)
            verdicts.append(r["outcome"])
            prefix = "   " if args.repeat == 1 else f"  r{rep+1}"
            print(f"{prefix} -> {r['outcome'].upper():8s} peak tilt {r['peak_tilt_deg']:6.2f} deg "
                  f"({r['peak_block']}) in {time.time()-t0:.1f}s")

        if args.repeat > 1 and verdicts:
            n_fail = verdicts.count("failure")
            agree = max(n_fail, len(verdicts) - n_fail) / len(verdicts)
            repeat_stats[ep] = {
                "verdicts": verdicts,
                "n_failure": n_fail,
                "n_success": len(verdicts) - n_fail,
                "agreement": round(agree, 3),
                "unanimous": n_fail in (0, len(verdicts)),
            }
            print(f"    verdicts: {n_fail} failure / {len(verdicts)-n_fail} success "
                  f"-> {'UNANIMOUS' if repeat_stats[ep]['unanimous'] else 'SPLIT (label unstable)'}")

    if repeat_stats:
        with open(out / "repeatability.json", "w") as f:
            json.dump(repeat_stats, f, indent=2)
        n_split = sum(1 for v in repeat_stats.values() if not v["unanimous"])
        print(f"\n--- repeatability over {args.repeat} replays ---")
        print(f"{'ep':>6} {'fail':>5} {'succ':>5} {'agreement':>10}  verdict")
        print("-" * 46)
        for ep, v in repeat_stats.items():
            print(f"{ep:>6} {v['n_failure']:>5} {v['n_success']:>5} {v['agreement']:>9.0%}  "
                  f"{'unanimous' if v['unanimous'] else 'SPLIT'}")
        print(f"\n{n_split}/{len(repeat_stats)} episode(s) gave unstable labels")

    if results:
        summary_plot(results, args.topple_threshold, out / "summary.png")
        with open(out / "summary.json", "w") as f:
            json.dump({"threshold": args.topple_threshold,
                       "episodes": results}, f, indent=2)

        print(f"\n{'ep':>6} {'outcome':>9} {'peak tilt':>10} {'detected':>9}")
        print("-" * 38)
        for r in results:
            print(f"{r['episode']:>6} {r['outcome']:>9} {r['peak_tilt_deg']:>9.2f}d "
                  f"{str(r['detection_step']):>9}")
        n_fail = sum(1 for r in results if r["outcome"] == "failure")
        print(f"\n{n_fail} failure(s), {len(results)-n_fail} success(es)")
        print(f"Artifacts in {out}/")


if __name__ == "__main__":
    main()
    # The passive viewer + renderer segfault on interpreter teardown. Exit hard to keep
    # the exit code clean -- but flush first, since os._exit discards buffered stdout.
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
