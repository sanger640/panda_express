"""
Evaluate the dual-camera world model, scoring divergence on the fixed camera only.

Index compatibility is the fiddly part. create_lmdb_full.py stores every frame without
aligning them to actions (502 frames vs 162 actions), whereas create_lmdb_single30.py
matches each action to its nearest frame and drops unmatched ones. labels_noise100.json
was realigned against the SINGLE LMDB, so to keep failure_step meaningful we reuse that
alignment: aligned cam2 keys and actions/proprio come from the single LMDB, and each
cam1 (wrist) frame is matched to its cam2 partner by timestamp from the unified LMDB.

The server receives visual as (B, T, 2, C, H, W) ordered [wrist, front], matching the
token layout the model's predict() documents: "# Tokens: [Wrist Patches, Front Patches]".

Usage:
    python test_monitor_dual.py \
        --single-lmdb tasks/jenga_noise_50/jenga_single_100.lmdb \
        --dual-lmdb   tasks/jenga_noise_50/jenga_unified.lmdb \
        --output-dir  results/eval100_dual
"""
import argparse, json, os, pickle, time
from pathlib import Path

import cv2, lmdb, numpy as np, zmq


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--single-lmdb", required=True, help="provides the aligned cam2 indexing")
    p.add_argument("--dual-lmdb", required=True, help="provides cam1 (wrist) frames")
    p.add_argument("--server-ip", default="localhost")
    p.add_argument("--port", type=int, default=5557)
    p.add_argument("--n-perturb", type=int, default=50)
    p.add_argument("--noise-std", type=float, default=0.05)
    p.add_argument("--num-hist", type=int, default=3)
    p.add_argument("--num-pred", type=int, default=8)
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--output-dir", default="results/eval100_dual")
    return p.parse_args()


def img_from(b):
    a = np.frombuffer(b, np.uint8)
    return np.transpose(cv2.cvtColor(cv2.imdecode(a, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB), (2, 0, 1))


def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    ctx = zmq.Context(); sock = ctx.socket(zmq.REQ)
    sock.setsockopt(zmq.RCVTIMEO, 600000)
    sock.connect(f"tcp://{args.server_ip}:{args.port}")

    env_s = lmdb.open(args.single_lmdb, readonly=True, lock=False)
    env_d = lmdb.open(args.dual_lmdb, readonly=True, lock=False)
    NH, NP = args.num_hist, args.num_pred
    span = NH + NP
    scores_out, alt_out = {}, {}

    with env_s.begin() as ts, env_d.begin() as td:
        meta_s = pickle.loads(ts.get(b"__metadata__"))
        meta_d = pickle.loads(td.get(b"__metadata__"))
        eps = list(meta_s["episodes"])
        if args.max_episodes:
            eps = eps[:args.max_episodes]
        print(f"Evaluating {len(eps)} episodes (dual view, scoring FRONT camera only)")

        for ei, ep in enumerate(eps):
            if ep not in meta_d["episodes"]:
                continue
            c2_aligned = meta_s["episodes"][ep]["keys"]["cam2"]      # aligned to actions
            c1_all = meta_d["episodes"][ep]["keys"]["cam1"]          # every wrist frame
            c1_ts = np.array([int(k.split("_")[-1]) for k in c1_all])
            acts = pickle.loads(ts.get(f"{ep}_actions".encode()))
            props = pickle.loads(ts.get(f"{ep}_proprio".encode()))

            # pair each aligned cam2 frame with its nearest-in-time wrist frame
            pairs = []
            for k2 in c2_aligned:
                t2 = int(k2.split("_")[-1])
                pairs.append((c1_all[int(np.argmin(np.abs(c1_ts - t2)))], k2))

            n = min(len(pairs), len(acts), len(props))
            max_start = n - span - 1
            if max_start <= 0:
                continue
            ep_scores, ep_alt = {}, {}
            print(f"\n[{ei+1}/{len(eps)}] {ep}")

            for start in range(0, max_start + 1, NP):
                frames = []
                bad = False
                for t in range(span):
                    k1, k2 = pairs[start + t]
                    b1, b2 = td.get(k1.encode()), ts.get(k2.encode())
                    if b1 is None or b2 is None:
                        bad = True; break
                    frames.append(np.stack([img_from(b1), img_from(b2)]))   # [wrist, front]
                if bad:
                    break
                vis = np.tile(np.stack(frames[:NH])[None], (args.n_perturb, 1, 1, 1, 1, 1))
                pro = np.tile(props[start:start + NH][None], (args.n_perturb, 1, 1))
                a = np.tile(acts[start:start + span][None], (args.n_perturb, 1, 1)).astype(np.float32)
                a[1:, :, :3] += np.random.normal(0, args.noise_std, (args.n_perturb - 1, span, 3))

                t0 = time.time()
                sock.send_pyobj({"visual": vis.astype(np.uint8),
                                 "proprio": pro.astype(np.float32),
                                 "actions": a, "return_states": False})
                r = sock.recv_pyobj()
                if "error" in r:
                    print(f"  server error @ {start}: {r['error']}")
                    continue
                ep_scores[start] = float(r["max_lyapunov"])
                for k, v in (r.get("alt_scores") or {}).items():
                    ep_alt.setdefault(k, {})[start] = float(v)
                print(f"  step {start:04d} | {r['max_lyapunov']:.4f} | {(time.time()-t0)*1000:.0f}ms")

            scores_out[ep] = ep_scores
            for k, v in ep_alt.items():
                alt_out.setdefault(k, {})[ep] = v

    env_s.close(); env_d.close()
    out = os.path.join(args.output_dir, "scores.json")
    with open(out, "w") as f:
        json.dump({"config": {"n_perturb": args.n_perturb, "noise_std": args.noise_std,
                              "num_hist": NH, "num_pred": NP, "view": "front",
                              "model": "dual"},
                   "scores": scores_out, "alt_scores": alt_out}, f, indent=2)
    print(f"\nDone -> {out}")


if __name__ == "__main__":
    main()
