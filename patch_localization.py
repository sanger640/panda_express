"""
Ground truth, world-model prediction, and the patch the monitor blames -- in one sheet.

test_monitor.py can draw the max-FTLE patch, but only on chunks that cross the threshold,
so when the monitor is silent (as in the sigma=0.05 baseline) no such image is ever produced.
This renders the overlay unconditionally, and adds the piece that was missing: where the
failure ACTUALLY is, derived from ground truth.

Per chunk it emits three rows:
    GT          real frames over the prediction horizon
    WM PRED     the model's rollout under the true actions
    WORST PERT  the model's rollout under the perturbation with the highest FTLE

Two boxes are drawn on the final column:
    red    the argmax-FTLE patch  -- where the monitor says the instability is
    green  the true motion region -- patches whose GT content changes most across the
           horizon, i.e. where the block actually moves

If the red box does not sit on or near the green region, the monitor is localising the
wrong thing, which is a different failure from simply scoring too low.

Usage:
    python patch_localization.py --lmdb tasks/jenga_noise_50/jenga_single.lmdb \
        --labels labels_noise50.json --episodes 8 29 45 --noise-std 0.002
"""

import argparse
import json
import pickle
from pathlib import Path

import cv2
import lmdb
import numpy as np
import zmq

GRID = 14          # 14x14 patch grid (see RESUME.md 5d)
PATCH_PX = 16      # 224 / 14


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lmdb", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--episodes", nargs="+", required=True)
    p.add_argument("--offsets", nargs="+", type=int, default=[-8])
    p.add_argument("--server-ip", default="localhost")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--n-perturb", type=int, default=50)
    p.add_argument("--noise-std", type=float, default=0.002)
    p.add_argument("--num-hist", type=int, default=3)
    p.add_argument("--num-pred", type=int, default=8)
    p.add_argument("--mask-top", type=int, default=28)
    p.add_argument("--output-dir", default="results/patch_localization")
    return p.parse_args()


def dec(b):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), 1), cv2.COLOR_BGR2RGB)


def crop224(img, size=224):
    h, w = img.shape[:2]
    s = size / min(h, w)
    img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    sh, sw = img.shape[:2]
    return img[(sh - size) // 2:(sh + size) // 2, (sw - size) // 2:(sw + size) // 2]


def true_motion_patches(gt, mask_top, topk=6):
    """Patches whose ground-truth content changes most from first to last frame."""
    d = np.abs(gt[-1].astype(float) - gt[0].astype(float)).mean(2)
    per = d.reshape(GRID, PATCH_PX, GRID, PATCH_PX).mean(axis=(1, 3)).reshape(-1)
    per[:mask_top] = -1
    return set(np.argsort(-per)[:topk].tolist()), per


def box(img, idx, colour, thick=2):
    r, c = idx // GRID, idx % GRID
    cv2.rectangle(img, (c * PATCH_PX, r * PATCH_PX),
                  ((c + 1) * PATCH_PX, (r + 1) * PATCH_PX), colour, thick)


def band(img, text, colour):
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 16), (0, 0, 0), -1)
    cv2.putText(out, text, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.35, colour, 1, cv2.LINE_AA)
    return out


def main():
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    labels = json.load(open(args.labels))
    ctx = zmq.Context(); sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 180000)
    sock.connect(f"tcp://{args.server_ip}:{args.port}")

    env = lmdb.open(args.lmdb, readonly=True, lock=False)
    summary = []
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
                raw = [dec(txn.get(keys[start + t].encode())) for t in range(span)]
                gt = np.stack([crop224(r) for r in raw])
                frames = np.stack([np.transpose(r, (2, 0, 1)) for r in raw])

                vis = np.tile(frames[:args.num_hist][None], (args.n_perturb, 1, 1, 1, 1))
                pro = np.tile(props[start:start + args.num_hist][None], (args.n_perturb, 1, 1))
                a = np.tile(acts[start:start + span][None], (args.n_perturb, 1, 1)).astype(np.float32)
                a[1:, :, :3] += np.random.normal(0, args.noise_std, (args.n_perturb - 1, span, 3))

                sock.send_pyobj({"visual": vis.astype(np.uint8), "proprio": pro.astype(np.float32),
                                 "actions": a, "return_states": True})
                r = sock.recv_pyobj()
                if "error" in r or r.get("states") is None:
                    print(f"  ep {ep} off {off}: no states")
                    continue
                st, pidx = r["states"], int(r["max_patch_idx"])

                true_set, per = true_motion_patches(gt[args.num_hist:], args.mask_top)
                hit = pidx in true_set
                # distance in patch grid from blamed patch to nearest true-motion patch
                pr_, pc_ = pidx // GRID, pidx % GRID
                dist = min(abs(pr_ - t // GRID) + abs(pc_ - t % GRID) for t in true_set)

                cols = []
                T = min(st.shape[1], span)
                for t in range(args.num_hist, T):
                    g = cv2.cvtColor(gt[t], cv2.COLOR_RGB2BGR).copy()
                    p0 = cv2.cvtColor(np.ascontiguousarray(st[0, t].transpose(1, 2, 0)), cv2.COLOR_RGB2BGR).copy()
                    p1 = cv2.cvtColor(np.ascontiguousarray(st[1, t].transpose(1, 2, 0)), cv2.COLOR_RGB2BGR).copy() \
                        if st.shape[0] > 1 else p0.copy()
                    if t == T - 1:
                        for im in (g, p0, p1):
                            for tp in true_set:
                                box(im, tp, (0, 200, 0), 1)
                            box(im, pidx, (0, 0, 255), 2)
                    cols.append(np.vstack([
                        band(g, f"GT t+{t-args.num_hist}", (0, 255, 0)),
                        band(p0, "WM PRED", (255, 255, 255)),
                        band(p1, "WORST PERT", (0, 165, 255)),
                    ]))
                sheet = np.hstack(cols)
                hdr = np.zeros((30, sheet.shape[1], 3), np.uint8)
                cv2.putText(hdr, f"ep {ep}  start {start} (fs{off:+d})  FTLE {r['max_lyapunov']:.3f}  "
                                 f"argmax patch {pidx} (r{pr_},c{pc_})  "
                                 f"true-motion hit={hit}  grid-dist={dist}   "
                                 f"[red=blamed  green=actual motion]",
                            (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
                path = out / f"ep{ep}_off{off:+d}.png"
                cv2.imwrite(str(path), np.vstack([hdr, sheet]))
                summary.append(dict(ep=ep, off=off, ftle=float(r["max_lyapunov"]),
                                    patch=pidx, hit=bool(hit), dist=int(dist)))
                print(f"  ep {ep:>3} off {off:+d}  FTLE {r['max_lyapunov']:.3f}  "
                      f"patch {pidx:>3} (r{pr_:>2},c{pc_:>2})  on-motion={hit}  dist={dist}")
    env.close()
    if summary:
        json.dump(summary, open(out / "summary.json", "w"), indent=2)
        h = sum(1 for s in summary if s["hit"]); d = np.mean([s["dist"] for s in summary])
        print(f"\nblamed patch lands on a true-motion patch: {h}/{len(summary)}")
        print(f"mean grid distance to nearest true-motion patch: {d:.1f} (0 = exact hit)")
    print(f"-> {out}/")


if __name__ == "__main__":
    main()
