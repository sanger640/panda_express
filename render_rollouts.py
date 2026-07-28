"""
Render the full rollout of every episode for visual inspection.

For each episode, walks the chunk grid exactly as test_monitor.py does and writes an MP4
with three panels side by side:

    GT           the real frame
    WM PRED      the model's rollout under the true action sequence
    WORST PERT   the model's rollout under the highest-FTLE perturbation

The argmax-FTLE patch is boxed in red on all three panels, so you can see what the monitor
is looking at while the scene evolves. A header carries the chunk's FTLE, the running
frame index, and -- for failure episodes -- how far the current frame is from failure_step,
turning green->amber->red as the topple approaches.

Also writes contact.png per episode: a wrapped contact sheet of the same frames, for a
quick overview without scrubbing.

Usage:
    python render_rollouts.py --lmdb tasks/jenga_noise_50/jenga_single.lmdb \
        --labels labels_noise50.json --noise-std 0.05 --output-dir results/rollouts
    python render_rollouts.py ... --episodes 8 29 45      # subset
"""

import argparse
import json
import pickle
from pathlib import Path

import cv2
import lmdb
import numpy as np
import zmq

GRID, PATCH_PX = 14, 16


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lmdb", required=True)
    p.add_argument("--labels", required=True)
    p.add_argument("--episodes", nargs="+", default=None, help="default: all in the LMDB")
    p.add_argument("--server-ip", default="localhost")
    p.add_argument("--port", type=int, default=5556)
    p.add_argument("--n-perturb", type=int, default=50)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--num-hist", type=int, default=3)
    p.add_argument("--num-pred", type=int, default=8)
    p.add_argument("--scale", type=float, default=1.4)
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--no-contact", action="store_true", help="skip the contact sheet")
    p.add_argument("--output-dir", default="results/rollouts")
    return p.parse_args()


def dec(b):
    return cv2.cvtColor(cv2.imdecode(np.frombuffer(b, np.uint8), 1), cv2.COLOR_BGR2RGB)


def crop224(img, size=224):
    h, w = img.shape[:2]; s = size / min(h, w)
    img = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
    sh, sw = img.shape[:2]
    return img[(sh - size) // 2:(sh + size) // 2, (sw - size) // 2:(sw + size) // 2]


def box(img, idx, colour, thick=2):
    r, c = idx // GRID, idx % GRID
    cv2.rectangle(img, (c * PATCH_PX, r * PATCH_PX),
                  ((c + 1) * PATCH_PX, (r + 1) * PATCH_PX), colour, thick)


def panel(img_rgb, title, colour, patch_idx):
    im = cv2.cvtColor(np.ascontiguousarray(img_rgb), cv2.COLOR_RGB2BGR).copy()
    if patch_idx is not None:
        box(im, patch_idx, (0, 0, 255), 2)
    cv2.rectangle(im, (0, 0), (im.shape[1], 16), (0, 0, 0), -1)
    cv2.putText(im, title, (3, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.36, colour, 1, cv2.LINE_AA)
    return im


def main():
    args = parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    labels = json.load(open(args.labels))
    ctx = zmq.Context(); sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 300000)
    sock.connect(f"tcp://{args.server_ip}:{args.port}")

    env = lmdb.open(args.lmdb, readonly=True, lock=False)
    NH, NP = args.num_hist, args.num_pred
    span = NH + NP
    index = []

    with env.begin() as txn:
        meta = pickle.loads(txn.get(b"__metadata__"))
        eps = args.episodes or sorted(meta["episodes"], key=lambda x: int(x) if x.isdigit() else 1 << 30)

        for ei, ep in enumerate(eps):
            if ep not in meta["episodes"]:
                continue
            info = labels.get(ep, {})
            outcome = info.get("outcome", "?")
            fs = info.get("failure_step")
            keys = meta["episodes"][ep]["keys"]["cam2"]
            acts = pickle.loads(txn.get(f"{ep}_actions".encode()))
            props = pickle.loads(txn.get(f"{ep}_proprio".encode()))
            max_start = min(len(keys), len(acts), len(props)) - span - 1
            if max_start <= 0:
                continue

            ep_dir = out / ep; ep_dir.mkdir(parents=True, exist_ok=True)
            W = int(224 * 3 * args.scale); H = int((224 + 22) * args.scale)
            vw = cv2.VideoWriter(str(ep_dir / "rollout.mp4"),
                                 cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (W, H))
            tiles, ftles = [], []

            for start in range(0, max_start + 1, NP):
                raw = [txn.get(keys[start + t].encode()) for t in range(span)]
                if any(r is None for r in raw):
                    break
                imgs = [dec(r) for r in raw]
                frames = np.stack([np.transpose(i, (2, 0, 1)) for i in imgs])

                vis = np.tile(frames[:NH][None], (args.n_perturb, 1, 1, 1, 1))
                pro = np.tile(props[start:start + NH][None], (args.n_perturb, 1, 1))
                a = np.tile(acts[start:start + span][None], (args.n_perturb, 1, 1)).astype(np.float32)
                a[1:, :, :3] += np.random.normal(0, args.noise_std, (args.n_perturb - 1, span, 3))

                sock.send_pyobj({"visual": vis.astype(np.uint8), "proprio": pro.astype(np.float32),
                                 "actions": a, "return_states": True})
                r = sock.recv_pyobj()
                if "error" in r or r.get("states") is None:
                    continue
                st, pidx = r["states"], int(r["max_patch_idx"])
                lam = float(r["max_lyapunov"])
                ftles.append((start, lam))

                T = min(st.shape[1], span)
                for t in range(NH, T):
                    gt = crop224(imgs[t])
                    p0 = st[0, t].transpose(1, 2, 0)
                    p1 = st[1, t].transpose(1, 2, 0) if st.shape[0] > 1 else p0
                    row = np.hstack([
                        panel(gt, f"GT  f{start+t}", (0, 255, 0), pidx),
                        panel(p0, "WM PRED", (255, 255, 255), pidx),
                        panel(p1, f"WORST PERT  FTLE {lam:.3f}", (0, 165, 255), pidx),
                    ])
                    hdr = np.zeros((22, row.shape[1], 3), np.uint8)
                    if fs is not None:
                        d = (start + t) - fs
                        col = (0, 0, 255) if d >= 0 else ((0, 165, 255) if d > -20 else (0, 200, 0))
                        note = f"{outcome.upper()}  failure_step {fs}  (frame {d:+d})"
                    else:
                        col = (0, 200, 0); note = f"{outcome.upper()}"
                    cv2.putText(hdr, f"ep {ep}  chunk {start:>4}  patch {pidx:>3}   {note}",
                                (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42, col, 1, cv2.LINE_AA)
                    fr = np.vstack([hdr, row])
                    fr = cv2.resize(fr, (W, H))
                    vw.write(fr)
                    if not args.no_contact and (t == T - 1):
                        tiles.append(row)
            vw.release()

            if tiles:
                per = 3
                while len(tiles) % per:
                    tiles.append(np.zeros_like(tiles[0]))
                sheet = np.vstack([np.hstack(tiles[i:i + per]) for i in range(0, len(tiles), per)])
                cv2.imwrite(str(ep_dir / "contact.png"), sheet)

            mx = max((l for _, l in ftles), default=float("nan"))
            index.append(dict(ep=ep, outcome=outcome, failure_step=fs,
                              chunks=len(ftles), max_ftle=mx))
            print(f"[{ei+1}/{len(eps)}] ep {ep:>3} {outcome:<8} chunks {len(ftles):>3} "
                  f"max FTLE {mx:.3f} -> {ep_dir}/rollout.mp4", flush=True)

    env.close()
    json.dump(index, open(out / "index.json", "w"), indent=2)
    print(f"\n{len(index)} episodes rendered -> {out}/")


if __name__ == "__main__":
    main()
