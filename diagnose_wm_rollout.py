"""
Does the world model actually predict the topple?

The FTLE profile turned out flat with respect to the failure (chunk-level AUC 0.588), which
has two very different explanations:

  (a) the world model predicts the fall, but FTLE fails to turn that into a usable score, or
  (b) the world model never predicts the fall at all, so there is no signal to extract.

This tells them apart by decoding the model's own rollout and putting it next to ground truth.
Each column is one timestep: top row is the real frame from the LMDB, bottom row is what the
model predicted for that timestep given only the first num_hist frames and the action sequence.

If the bottom row keeps the blocks upright while the top row shows them falling, it is (b),
and no amount of metric or threshold work will help.

Usage:
    python diagnose_wm_rollout.py --lmdb tasks/jenga_noise_50/jenga_single.lmdb \
        --labels labels_noise50.json --episodes 8 29 45
"""

import argparse
import json
import os
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
    p.add_argument("--server-ip", default="localhost")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--n-perturb", type=int, default=50)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--num-hist", type=int, default=3)
    p.add_argument("--num-pred", type=int, default=8)
    p.add_argument("--offsets", nargs="+", type=int, default=[-16, -8, 0],
                   help="Chunk starts to probe, as offsets from failure_step")
    p.add_argument("--output-dir", default="results/wm_diagnosis")
    return p.parse_args()


def decode_img(b):
    a = np.frombuffer(b, dtype=np.uint8)
    img = cv2.imdecode(a, cv2.IMREAD_COLOR)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def center_crop_resize(img, size=224):
    h, w = img.shape[:2]
    s = size / min(h, w)
    img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    sh, sw = img.shape[:2]
    return img[(sh - size) // 2:(sh + size) // 2, (sw - size) // 2:(sw + size) // 2]


def label(img, text, colour):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 18), (0, 0, 0), -1)
    cv2.putText(out, text, (4, 13), cv2.FONT_HERSHEY_SIMPLEX, 0.38, colour, 1, cv2.LINE_AA)
    return out


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    labels = json.load(open(args.labels))
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 180000)
    sock.connect(f"tcp://{args.server_ip}:{args.port}")

    env = lmdb.open(args.lmdb, readonly=True, lock=False)
    with env.begin() as txn:
        meta = pickle.loads(txn.get(b"__metadata__"))

        for ep in args.episodes:
            if ep not in meta["episodes"]:
                print(f"ep {ep}: not in LMDB")
                continue
            fs = labels[ep]["failure_step"]
            keys = meta["episodes"][ep]["keys"]["cam2"]
            acts = pickle.loads(txn.get(f"{ep}_actions".encode()))
            props = pickle.loads(txn.get(f"{ep}_proprio".encode()))

            for off in args.offsets:
                start = fs + off
                span = args.num_hist + args.num_pred
                if start < 0 or start + span >= len(keys):
                    continue

                frames = [np.transpose(decode_img(txn.get(keys[start + t].encode())), (2, 0, 1))
                          for t in range(span)]
                vis = np.tile(np.stack(frames[:args.num_hist])[None], (args.n_perturb, 1, 1, 1, 1))
                pro = np.tile(props[start:start + args.num_hist][None], (args.n_perturb, 1, 1))
                a = np.tile(acts[start:start + span][None], (args.n_perturb, 1, 1)).astype(np.float32)
                a[1:, :, :3] += np.random.normal(0, args.noise_std, (args.n_perturb - 1, span, 3))

                sock.send_pyobj({"visual": vis.astype(np.uint8),
                                 "proprio": pro.astype(np.float32),
                                 "actions": a, "return_states": True})
                r = sock.recv_pyobj()
                if "error" in r:
                    print(f"  ep {ep} start {start}: server error {r['error']}")
                    continue
                states = r.get("states")
                if states is None:
                    print(f"  ep {ep} start {start}: server returned no states")
                    continue

                cols = []
                T = min(states.shape[1], span)
                for t in range(args.num_hist, T):
                    gt = center_crop_resize(np.transpose(frames[t], (1, 2, 0)))
                    pr = np.ascontiguousarray(states[0, t].transpose(1, 2, 0))
                    is_hist = t < args.num_hist
                    cols.append(np.vstack([
                        label(cv2.cvtColor(gt, cv2.COLOR_RGB2BGR), f"GT t+{t-args.num_hist}", (0, 255, 0)),
                        label(cv2.cvtColor(pr, cv2.COLOR_RGB2BGR), "WM PRED", (255, 255, 255)),
                    ]))
                if not cols:
                    continue
                sheet = np.hstack(cols)
                banner = np.zeros((26, sheet.shape[1], 3), np.uint8)
                cv2.putText(banner,
                            f"ep {ep}  chunk start {start} = failure_step{off:+d}   "
                            f"FTLE {r['max_lyapunov']:.3f}",
                            (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
                sheet = np.vstack([banner, sheet])
                path = out / f"ep{ep}_start{start}_off{off:+d}.png"
                cv2.imwrite(str(path), sheet)
                print(f"  ep {ep} start {start:4d} (fs{off:+d})  FTLE {r['max_lyapunov']:.3f}  -> {path.name}")

    env.close()
    print(f"\n-> {out}/")


if __name__ == "__main__":
    main()
