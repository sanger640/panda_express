"""
Safety monitor evaluation script. Replaces test_traj_noise_single_whole_max.py.

Sends LMDB trajectories through the safety monitor server, saves raw scores to JSON,
and optionally saves visualizations. Supports temporal aggregation for robustness.

Usage (basic):
    python test_monitor.py --lmdb tasks/jenga_mujoco_noise/jenga_single.lmdb

Usage (with all options):
    python test_monitor.py \
        --lmdb tasks/jenga_mujoco_noise/jenga_single.lmdb \
        --server-ip localhost --port 5556 \
        --n-perturb 50 --noise-std 0.05 \
        --num-hist 3 --num-pred 8 \
        --threshold 0.87 \
        --temporal-window 3 --temporal-agg max \
        --output-dir results/ftle \
        --visualize

Temporal aggregation (--temporal-agg):
    max  — rolling max over window: improves recall for gradual instability buildup
    mean — rolling mean over window: smooths spikes, reduces shadow false positives
    ema  — exponential moving average (alpha=0.6): balanced smoothing
"""

import argparse
import json
import os
import time
from collections import deque
from pathlib import Path

import cv2
import lmdb
import numpy as np
import pickle
import zmq


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lmdb",            required=True)
    p.add_argument("--server-ip",       default="localhost")
    p.add_argument("--port",            type=int, default=5556)
    p.add_argument("--n-perturb",       type=int, default=50)
    p.add_argument("--noise-std",       type=float, default=0.05)
    p.add_argument("--num-hist",        type=int, default=3)
    p.add_argument("--num-pred",        type=int, default=8)
    p.add_argument("--threshold",       type=float, default=0.87,
                   help="FTLE threshold for unsafe flag")
    p.add_argument("--temporal-window", type=int, default=1,
                   help="Rolling window size for temporal aggregation (1 = no aggregation)")
    p.add_argument("--temporal-agg",    choices=["max", "mean", "ema"], default="max",
                   help="Aggregation method over temporal window")
    p.add_argument("--ema-alpha",       type=float, default=0.6,
                   help="Alpha for EMA temporal aggregation")
    p.add_argument("--max-episodes",    type=int, default=None,
                   help="Only evaluate the first N episodes (quick validation runs)")
    p.add_argument("--output-dir",      default="results/monitor")
    p.add_argument("--visualize",       action="store_true",
                   help="Save visualization images for triggered steps")
    return p.parse_args()


def process_image(img_bytes):
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return np.transpose(img, (2, 0, 1))


def center_crop_resize(img, size=224):
    h, w = img.shape[:2]
    scale = size / min(h, w)
    img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    sh, sw = img.shape[:2]
    return img[(sh - size) // 2:(sh + size) // 2, (sw - size) // 2:(sw + size) // 2]


def add_label(img, text, color=(0, 255, 0)):
    res = img.copy()
    cv2.putText(res, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
    return res


class TemporalAggregator:
    def __init__(self, window, method, alpha=0.6):
        self.window = window
        self.method = method
        self.alpha  = alpha
        self.history = deque(maxlen=window)
        self._ema = None

    def update(self, score):
        """Returns effective score after temporal aggregation."""
        self.history.append(score)

        if self.method == "max":
            return max(self.history)
        elif self.method == "mean":
            return float(np.mean(self.history))
        elif self.method == "ema":
            if self._ema is None:
                self._ema = score
            else:
                self._ema = self.alpha * score + (1 - self.alpha) * self._ema
            return self._ema

    def reset(self):
        self.history.clear()
        self._ema = None


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.connect(f"tcp://{args.server_ip}:{args.port}")
    print(f"Connected to monitor server at {args.server_ip}:{args.port}")

    aggregator = TemporalAggregator(args.temporal_window, args.temporal_agg, args.ema_alpha)

    scores_out = {}  # {ep_name: {step: raw_score}}
    alt_out = {}     # {metric: {ep_name: {step: score}}}

    env = lmdb.open(args.lmdb, readonly=True, lock=False)
    with env.begin() as txn:
        metadata = pickle.loads(txn.get(b"__metadata__"))
        ep_names = list(metadata["episodes"].keys())
        if args.max_episodes:
            ep_names = ep_names[:args.max_episodes]
        print(f"Evaluating {len(ep_names)} episodes...")

        for ep_idx, ep_name in enumerate(ep_names):
            ep_meta  = metadata["episodes"][ep_name]
            c2_keys  = ep_meta["keys"]["cam2"]
            act_all  = pickle.loads(txn.get(f"{ep_name}_actions".encode()))
            prop_all = pickle.loads(txn.get(f"{ep_name}_proprio".encode()))

            actual_len = min(len(c2_keys), len(act_all), len(prop_all))
            max_start  = actual_len - (args.num_hist + args.num_pred) - 1
            if max_start <= 0:
                print(f"  Skipping {ep_name} (too short)")
                continue

            ep_dir = os.path.join(args.output_dir, ep_name)
            if args.visualize:
                os.makedirs(ep_dir, exist_ok=True)

            aggregator.reset()
            ep_scores = {}
            ep_alt = {}

            print(f"\n[{ep_idx+1}/{len(ep_names)}] {ep_name}")

            for start in range(0, max_start + 1, args.num_pred):
                vis_frames, gt_bgr, skip = [], [], False

                for t_off in range(args.num_hist + args.num_pred):
                    idx = start + t_off
                    if idx >= len(c2_keys):
                        skip = True
                        break
                    img_bytes = txn.get(c2_keys[idx].encode())
                    if img_bytes is None:
                        skip = True
                        break
                    frame = process_image(img_bytes)
                    vis_frames.append(frame)
                    gt_bgr.append(cv2.cvtColor(np.transpose(frame, (1, 2, 0)), cv2.COLOR_RGB2BGR))

                if skip:
                    break

                # Build perturbed batch
                clean_actions = act_all[start:start + args.num_hist + args.num_pred]
                T, D = clean_actions.shape
                batch_actions = np.tile(clean_actions[np.newaxis], (args.n_perturb, 1, 1)).astype(np.float32)
                noise = np.random.normal(0, args.noise_std, (args.n_perturb - 1, T, 3))
                batch_actions[1:, :, :3] += noise

                vis_hist  = np.tile(np.stack(vis_frames[:args.num_hist])[np.newaxis], (args.n_perturb, 1, 1, 1, 1))
                prop_hist = np.tile(prop_all[start:start + args.num_hist][np.newaxis], (args.n_perturb, 1, 1))

                t_req = time.time()
                socket.send_pyobj({
                    "visual":  vis_hist.astype(np.uint8),
                    "proprio": prop_hist.astype(np.float32),
                    "actions": batch_actions,
                    # decoding is only needed for the visualisation below
                    "return_states": bool(args.visualize),
                })
                resp = socket.recv_pyobj()
                latency = time.time() - t_req

                if "error" in resp:
                    print(f"  Server error at step {start}: {resp['error']}")
                    continue

                raw_score  = float(resp.get("max_lyapunov", 0.0))
                all_lyaps  = resp.get("all_lyapunovs", np.array([]))
                patch_idx  = resp.get("max_patch_idx", 0)
                pred_imgs  = resp.get("states")

                ep_scores[start] = raw_score
                alt = resp.get('alt_scores') or {}
                for k, v in alt.items():
                    ep_alt.setdefault(k, {})[start] = float(v)

                # Temporal aggregation
                eff_score = aggregator.update(raw_score)
                triggered = eff_score > args.threshold
                n_triggers = int(np.sum(all_lyaps > args.threshold)) if len(all_lyaps) > 0 else 0

                flag = "UNSAFE" if triggered else "safe"
                print(f"  step {start:04d} | raw={raw_score:.3f} | eff={eff_score:.3f} | {flag} | {latency*1000:.0f}ms")

                # Visualization (only for triggered steps)
                if args.visualize and triggered and pred_imgs is not None:
                    cols = []
                    for t in range(args.num_pred):
                        gt_idx = args.num_hist + t
                        gt_frame = center_crop_resize(gt_bgr[gt_idx])
                        col = [add_label(gt_frame, "GROUND TRUTH", (0, 255, 0))]

                        orig_pred = cv2.cvtColor(
                            np.ascontiguousarray(pred_imgs[0, gt_idx].transpose(1, 2, 0)),
                            cv2.COLOR_RGB2BGR
                        )
                        col.append(add_label(orig_pred, "ORIG PRED", (255, 255, 255)))

                        if pred_imgs.shape[0] > 1:
                            worst_pred = cv2.cvtColor(
                                np.ascontiguousarray(pred_imgs[1, gt_idx].transpose(1, 2, 0)),
                                cv2.COLOR_RGB2BGR
                            )
                            if t == args.num_pred - 1:
                                py, px = (patch_idx // 14) * 16, (patch_idx % 14) * 16
                                cv2.rectangle(orig_pred,  (px, py), (px + 16, py + 16), (0, 0, 255), 2)
                                cv2.rectangle(worst_pred, (px, py), (px + 16, py + 16), (0, 0, 255), 2)
                            label = f"WORST NOISY (LE:{raw_score:.2f}) [{n_triggers}]"
                            col.append(add_label(worst_pred, label, (0, 0, 255)))

                        cols.append(np.vstack(col))

                    viz = np.hstack(cols)
                    cv2.imwrite(os.path.join(ep_dir, f"step_{start:04d}.png"), viz)

            scores_out[ep_name] = ep_scores
            for k, v in ep_alt.items():
                alt_out.setdefault(k, {})[ep_name] = v

    env.close()

    # Save raw scores
    save_path = os.path.join(args.output_dir, "scores.json")
    meta = {
        "threshold": args.threshold,
        "n_perturb": args.n_perturb,
        "noise_std": args.noise_std,
        "num_hist": args.num_hist,
        "num_pred": args.num_pred,
        "temporal_window": args.temporal_window,
        "temporal_agg": args.temporal_agg,
    }
    with open(save_path, "w") as f:
        json.dump({"config": meta, "scores": scores_out, "alt_scores": alt_out}, f, indent=2)

    print(f"\nDone. Scores saved to {save_path}")


if __name__ == "__main__":
    main()
