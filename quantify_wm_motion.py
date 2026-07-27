"""
Quantify how much the world model's rollout actually moves, versus ground truth.

Reading motion off a montage thumbnail is unreliable, so this measures it. For a given chunk
it decodes the model's rollout, then reports for both ground truth and prediction:

  - frame-to-frame mean absolute difference (how much the image changes per step)
  - cumulative drift from the first predicted frame (how far the scene travels overall)

Both are reported for the full frame and for a block-region crop, so arm motion (which the
model demonstrably tracks) does not mask whether the blocks themselves move.

If PRED drift is near zero while GT drift is large, the model is producing a static scene.
If PRED drift tracks GT, it is modelling the dynamics and the flat FTLE has another cause.

Usage:
    python quantify_wm_motion.py --lmdb tasks/jenga_noise_50/jenga_single.lmdb \
        --labels labels_noise50.json --episodes 8 29 45 --offsets -16 -8 0
"""

import argparse
import json
import pickle
from pathlib import Path

import cv2
import lmdb
import numpy as np
import zmq


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lmdb", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--episodes", nargs="+", required=True)
    p.add_argument("--offsets", nargs="+", type=int, default=[-8])
    p.add_argument("--server-ip", default="localhost")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--n-perturb", type=int, default=50)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--num-hist", type=int, default=3)
    p.add_argument("--num-pred", type=int, default=8)
    # Block region in the 224x224 crop: lower-centre band where the three blocks sit.
    p.add_argument("--block-roi", nargs=4, type=int, default=[70, 0, 165, 80],
                   metavar=("X0", "Y0", "X1", "Y1"))
    p.add_argument("--output-dir", default="results/wm_motion")
    return p.parse_args()


def decode_img(b):
    a = np.frombuffer(b, dtype=np.uint8)
    return cv2.cvtColor(cv2.imdecode(a, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def center_crop_resize(img, size=224):
    h, w = img.shape[:2]
    s = size / min(h, w)
    img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    sh, sw = img.shape[:2]
    return img[(sh - size) // 2:(sh + size) // 2, (sw - size) // 2:(sw + size) // 2]


def drift_stats(seq, roi=None):
    """seq: (T,H,W,3) float. Returns (mean step-to-step diff, drift from frame 0)."""
    if roi is not None:
        x0, y0, x1, y1 = roi
        seq = seq[:, y0:y1, x0:x1]
    seq = seq.astype(np.float32)
    step = np.mean([np.abs(seq[i + 1] - seq[i]).mean() for i in range(len(seq) - 1)])
    drift = np.abs(seq[-1] - seq[0]).mean()
    return float(step), float(drift)


def main():
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    roi = tuple(args.block_roi)
    labels = json.load(open(args.labels))

    ctx = zmq.Context(); sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 180000)
    sock.connect(f"tcp://{args.server_ip}:{args.port}")

    env = lmdb.open(args.lmdb, readonly=True, lock=False)
    rows = []
    with env.begin() as txn:
        meta = pickle.loads(txn.get(b"__metadata__"))
        for ep in args.episodes:
            if ep not in meta["episodes"]:
                continue
            fs = labels[ep]["failure_step"]
            keys = meta["episodes"][ep]["keys"]["cam2"]
            acts = pickle.loads(txn.get(f"{ep}_actions".encode()))
            props = pickle.loads(txn.get(f"{ep}_proprio".encode()))
            span = args.num_hist + args.num_pred

            for off in args.offsets:
                start = fs + off
                if start < 0 or start + span >= len(keys):
                    continue
                raw = [decode_img(txn.get(keys[start + t].encode())) for t in range(span)]
                frames = [np.transpose(r, (2, 0, 1)) for r in raw]

                vis = np.tile(np.stack(frames[:args.num_hist])[None], (args.n_perturb, 1, 1, 1, 1))
                pro = np.tile(props[start:start + args.num_hist][None], (args.n_perturb, 1, 1))
                a = np.tile(acts[start:start + span][None], (args.n_perturb, 1, 1)).astype(np.float32)
                a[1:, :, :3] += np.random.normal(0, args.noise_std, (args.n_perturb - 1, span, 3))

                sock.send_pyobj({"visual": vis.astype(np.uint8),
                                 "proprio": pro.astype(np.float32), "actions": a,
                                 "return_states": True})
                r = sock.recv_pyobj()
                if "error" in r or r.get("states") is None:
                    continue
                st = r["states"]
                T = min(st.shape[1], span)

                gt = np.stack([center_crop_resize(raw[t]) for t in range(args.num_hist, T)])
                pr = np.stack([st[0, t].transpose(1, 2, 0) for t in range(args.num_hist, T)])

                g_s, g_d = drift_stats(gt)
                p_s, p_d = drift_stats(pr)
                gr_s, gr_d = drift_stats(gt, roi)
                pr_s, pr_d = drift_stats(pr, roi)
                rows.append(dict(ep=ep, off=off, ftle=float(r["max_lyapunov"]),
                                 gt_step=g_s, pred_step=p_s, gt_drift=g_d, pred_drift=p_d,
                                 gt_roi_step=gr_s, pred_roi_step=pr_s,
                                 gt_roi_drift=gr_d, pred_roi_drift=pr_d))

                # zoomed block-region strip so the motion is visible by eye
                x0, y0, x1, y1 = roi
                z = lambda im: cv2.resize(im[y0:y1, x0:x1], ((x1 - x0) * 4, (y1 - y0) * 4),
                                          interpolation=cv2.INTER_NEAREST)
                top = np.hstack([z(g) for g in gt])
                bot = np.hstack([z(p) for p in pr])
                sheet = np.vstack([top, bot])
                sheet = cv2.cvtColor(sheet, cv2.COLOR_RGB2BGR)
                cv2.putText(sheet, "GT", (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(sheet, "PRED", (4, top.shape[0] + 16), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 2)
                cv2.imwrite(str(out / f"zoom_ep{ep}_off{off:+d}.png"), sheet)
    env.close()

    print(f"{'ep':>4} {'off':>5} {'FTLE':>6} | {'FULL FRAME':>22} | {'BLOCK ROI':>22}")
    print(f"{'':>4} {'':>5} {'':>6} | {'gt_step':>7} {'pr_step':>7} {'ratio':>6} | "
          f"{'gt_step':>7} {'pr_step':>7} {'ratio':>6}")
    print("-" * 78)
    for r in rows:
        fr = r["pred_step"] / max(r["gt_step"], 1e-6)
        rr = r["pred_roi_step"] / max(r["gt_roi_step"], 1e-6)
        print(f"{r['ep']:>4} {r['off']:>+5} {r['ftle']:>6.3f} | {r['gt_step']:>7.2f} "
              f"{r['pred_step']:>7.2f} {fr:>6.2f} | {r['gt_roi_step']:>7.2f} "
              f"{r['pred_roi_step']:>7.2f} {rr:>6.2f}")
    if rows:
        rr = np.mean([r["pred_roi_step"] / max(r["gt_roi_step"], 1e-6) for r in rows])
        print(f"\nmean PRED/GT motion ratio in block region: {rr:.2f}")
        print("  ~1.0 = model reproduces block motion;  ~0.0 = model renders a static scene")
    json.dump(rows, open(out / "motion_stats.json", "w"), indent=2)
    print(f"-> {out}/")


if __name__ == "__main__":
    main()
