"""
Computes precision/recall/F1/accuracy from saved monitor scores and episode labels.

Creates a comparison table across multiple ablation modes and sweeps thresholds
to find the best operating point per mode.

Usage:
    # Single mode
    python compute_metrics.py \
        --scores results/ftle/scores.json \
        --labels labels.json

    # Compare all ablation modes
    python compute_metrics.py \
        --scores-dir results/ \
        --labels labels.json \
        --modes ftle final_cosine mean_traj max_step ftle_mean_patch ftle_gap ftle_l2 ftle_topk

Labels JSON format — two formats accepted:

    Simple (binary):
        {"episode_001": 0, "episode_002": 1, ...}
        0 = success, 1 = failure

    Rich (from generate_labels.py):
        {"episode_001": {"outcome": "success", "failure_step": null},
         "episode_002": {"outcome": "failure", "failure_step": 24}, ...}

Note that scores.json is keyed by *chunk start index*, not by timestep: test_monitor.py
strides by num_pred, so there is one score per 8 timesteps, not one per timestep. What is
called "step-level" below is really chunk-level.

Metrics are computed at two levels, and both are reported by default:
    episode-level: TP = failed episode where the monitor triggered at least once within
                   the window where it could still have acted
    step-level:    chunks that genuinely forecast the topple are labeled unsafe; all
                   chunks in success episodes are safe (requires rich labels)

Two corrections are applied by default, both aimed at measuring *prediction* rather than
*observation*. generate_labels.py stops replaying at the first topple, but test_monitor.py
keeps scoring to the end of the episode, so later chunks observe a scene where the blocks
are already down:

  - Chunks past the cutoff are dropped (--no-truncate keeps them). Otherwise the monitor
    is charged a false positive for correctly reporting that a collapsed tower is
    unstable, and a monitor that never predicted the fall but spiked afterwards is
    credited a true positive.

  - The unsafe window counts only chunks whose observations end before the topple and
    whose prediction horizon still reaches it (--label-window legacy restores the old
    window). Otherwise the chunk that is *watching* the topple happen counts as a
    detection, which rewards observation rather than forecasting.

See chunk_bounds() for the exact index arithmetic.
"""

import argparse
import json
import os
import numpy as np
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scores",      help="Path to a single scores.json")
    p.add_argument("--scores-dir",  help="Directory containing mode subdirs with scores.json")
    p.add_argument("--modes",       nargs="+",
                   default=["ftle", "final_cosine", "mean_traj", "max_step",
                            "ftle_mean_patch", "ftle_gap", "ftle_l2", "ftle_topk",
                            "ftle_variance"],
                   help="Mode names to compare (subdirs of --scores-dir)")
    p.add_argument("--labels",      required=True, help="Path to labels.json")
    p.add_argument("--n-thresholds", type=int, default=100)
    p.add_argument("--level",       choices=["episode", "step", "both"], default="both")
    p.add_argument("--label-window", choices=["predictive", "legacy"], default="predictive",
                   help="Which scored chunks count as unsafe. 'predictive' (default) counts "
                        "only chunks whose observations end before the topple and whose "
                        "horizon still reaches it. 'legacy' also credits the chunk that is "
                        "watching the topple happen. See chunk_bounds().")
    p.add_argument("--no-truncate", action="store_true",
                   help="Keep steps recorded after failure_step. The labeler stops at the "
                        "first topple while the monitor keeps scoring, so these steps observe "
                        "an already-collapsed scene and are charged as false positives. Off by "
                        "default; use this only to reproduce the old numbers.")
    return p.parse_args()


def load_scores(path):
    with open(path) as f:
        data = json.load(f)
    return sanitize(data.get("scores", data)), data.get("config", {})


def sanitize(scores_by_ep):
    """Map non-finite chunk scores onto the finite minimum.

    le_cos() masks patches whose end-distance falls under a 1e-3 noise floor, and returns
    -inf for a perturbation where *every* patch is masked. With small deviator noise this
    happens for real: at sigma=0.002 about 1.4% of chunks come back -inf. Semantically that
    is "no patch showed significant drift", i.e. maximally safe, so the finite minimum is
    the right stand-in. Left raw, a single -inf makes np.linspace produce all-NaN thresholds
    and silently zeroes every metric.
    """
    vals = [v for s in scores_by_ep.values() for v in s.values()]
    finite = [v for v in vals if np.isfinite(v)]
    if not finite or len(finite) == len(vals):
        return scores_by_ep
    lo = min(finite)
    n = 0
    out = {}
    for ep, steps in scores_by_ep.items():
        out[ep] = {}
        for k, v in steps.items():
            if not np.isfinite(v):
                v = lo
                n += 1
            out[ep][k] = v
    print(f"  note: {n} non-finite chunk score(s) mapped to the finite minimum ({lo:.3f})")
    return out


def chunk_bounds(fs, num_hist, num_pred, window="predictive"):
    """Which scored chunks count as unsafe, and where to stop scoring, for a failure at fs.

    test_monitor.py strides by num_pred, so scores.json is keyed by chunk start index, not
    by timestep. A chunk at `start` observes [start, start+num_hist-1] and predicts
    [start+num_hist, start+num_hist+num_pred-1].

    "predictive" (default) counts only chunks that genuinely forecast the topple: their
    observations end strictly before fs, and their prediction horizon still reaches it.

        observations end before failure:  start + num_hist - 1 <  fs  -> start <= fs - num_hist
        horizon reaches failure:          start + num_hist + num_pred - 1 >= fs
                                                                       -> start >= fs - num_hist - num_pred + 1

    Chunks past the cutoff already have the topple inside their observation window, so a
    high score there is observation rather than prediction; they are dropped rather than
    counted as false positives.

    "legacy" reproduces the original window, (fs - num_pred) <= start <= fs, which also
    credits the chunk that is watching the topple happen.

    Returns (lo, hi, cutoff): chunks in [lo, hi] are unsafe, chunks with start > cutoff dropped.
    """
    if window == "legacy":
        return fs - num_pred, fs, fs
    return fs - num_hist - num_pred + 1, fs - num_hist, fs - num_hist


def episode_max_scores(scores_by_ep, labels, cutoffs=None, truncate=True):
    """Reduce each episode to one score: the max over chunks the monitor could still act on.

    generate_labels.py stops replaying at the first topple, but test_monitor.py keeps
    scoring to the end of the episode. Those later chunks show a world where the blocks are
    already down, so the world model legitimately reports high divergence. Counting them
    turns the metric into "did the monitor notice the wreckage" rather than "did it predict
    the fall". With truncate=True, failed episodes are cut at their cutoff.

    Success episodes have no failure step and always use all their chunks.
    """
    out = {}
    for ep, step_scores in scores_by_ep.items():
        if ep not in labels:
            continue
        cut = (cutoffs or {}).get(ep)
        if truncate and cut is not None:
            vals = [s for st, s in step_scores.items() if int(st) <= cut]
        else:
            vals = list(step_scores.values())
        out[ep] = max(vals) if vals else 0.0
    return out


def compute_pr_curve(scores_by_ep, labels, n_thresh=100, cutoffs=None, truncate=True):
    """
    scores_by_ep: {ep_name: {step: score}}
    labels: {ep_name: 0|1}
    Returns thresholds, precision, recall arrays.
    """
    ep_max = episode_max_scores(scores_by_ep, labels, cutoffs, truncate)

    all_scores = list(ep_max.values())
    if not all_scores:
        return np.array([]), np.array([]), np.array([])

    thresholds = np.linspace(min(all_scores), max(all_scores), n_thresh)
    precisions, recalls = [], []

    for thresh in thresholds:
        tp = fp = tn = fn = 0
        for ep, max_score in ep_max.items():
            label = labels[ep]
            triggered = max_score > thresh
            if label == 1 and triggered:     tp += 1
            elif label == 0 and triggered:   fp += 1
            elif label == 0 and not triggered: tn += 1
            elif label == 1 and not triggered: fn += 1

        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        precisions.append(prec)
        recalls.append(rec)

    return thresholds, np.array(precisions), np.array(recalls)


def best_f1_metrics(scores_by_ep, labels, n_thresh=100, cutoffs=None, truncate=True):
    """Returns dict of metrics at the threshold with best F1."""
    thresholds, precisions, recalls = compute_pr_curve(
        scores_by_ep, labels, n_thresh, cutoffs, truncate)

    if len(thresholds) == 0:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "threshold": 0}

    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = int(np.argmax(f1s))
    best_thresh = thresholds[best_idx]

    # Recompute full metrics at best threshold
    ep_max = episode_max_scores(scores_by_ep, labels, cutoffs, truncate)

    tp = fp = tn = fn = 0
    for ep, max_score in ep_max.items():
        label = labels[ep]
        triggered = max_score > best_thresh
        if label == 1 and triggered:       tp += 1
        elif label == 0 and triggered:     fp += 1
        elif label == 0 and not triggered: tn += 1
        elif label == 1 and not triggered: fn += 1

    total = tp + fp + tn + fn
    return {
        "accuracy":  (tp + tn) / (total + 1e-8) * 100,
        "precision": tp / (tp + fp + 1e-8) * 100,
        "recall":    tp / (tp + fn + 1e-8) * 100,
        "f1":        f1s[best_idx] * 100,
        "threshold": float(best_thresh),
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
    }


def step_level_metrics(scores_by_ep, labels, threshold, n_thresh=100, windows=None,
                       truncate=True):
    """
    Assigns per-chunk labels then finds the best-F1 threshold.

    `windows` maps episode -> (lo, hi, cutoff) from chunk_bounds(), computed only for failed
    episodes with a known failure step. Chunks in [lo, hi] are unsafe; chunks past `cutoff`
    are dropped (they observe an already-collapsed scene, so scoring them high is correct
    behaviour and must not be charged as a false positive). Episodes without an entry fall
    back to their episode label for every chunk.
    """
    step_preds = []  # (label, score) pairs
    n_dropped = 0
    for ep, step_scores in scores_by_ep.items():
        if ep not in labels:
            continue
        ep_label = labels[ep]
        bounds = (windows or {}).get(ep)

        for step_str, score in step_scores.items():
            step = int(step_str)
            if bounds is not None:
                lo, hi, cut = bounds
                if truncate and step > cut:
                    n_dropped += 1
                    continue
                step_label = 1 if lo <= step <= hi else 0
            else:
                step_label = ep_label
            step_preds.append((step_label, score))

    if n_dropped:
        print(f"  (dropped {n_dropped} post-failure steps; pass --no-truncate to keep them)")

    if not step_preds:
        return {}

    true_labels = np.array([x[0] for x in step_preds])
    pred_scores = np.array([x[1] for x in step_preds])

    thresholds = np.linspace(pred_scores.min(), pred_scores.max(), n_thresh)
    best_f1, best_met = 0, {}

    for thresh in thresholds:
        preds = (pred_scores > thresh).astype(int)
        tp = int(np.sum((preds == 1) & (true_labels == 1)))
        fp = int(np.sum((preds == 1) & (true_labels == 0)))
        tn = int(np.sum((preds == 0) & (true_labels == 0)))
        fn = int(np.sum((preds == 0) & (true_labels == 1)))
        total = tp + fp + tn + fn
        prec = tp / (tp + fp + 1e-8)
        rec  = tp / (tp + fn + 1e-8)
        f1   = 2 * prec * rec / (prec + rec + 1e-8)
        acc  = (tp + tn) / (total + 1e-8)
        if f1 > best_f1:
            best_f1 = f1
            best_met = {
                "accuracy":  acc * 100,
                "precision": prec * 100,
                "recall":    rec * 100,
                "f1":        f1 * 100,
                "threshold": float(thresh),
                "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            }

    return best_met


def print_table(results, level):
    header = f"{'Mode':<22} | {'Acc':>6} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | {'Thresh':>7}"
    print(f"\n{'='*len(header)}")
    print(f"  {level.upper()}-LEVEL METRICS (best F1 threshold per mode)")
    print(f"{'='*len(header)}")
    print(header)
    print("-" * len(header))
    for mode, m in sorted(results.items(), key=lambda x: -x[1].get("f1", 0)):
        print(f"{mode:<22} | {m.get('accuracy',0):>5.1f}% | {m.get('precision',0):>5.1f}% | "
              f"{m.get('recall',0):>5.1f}% | {m.get('f1',0):>5.1f}% | {m.get('threshold',0):>7.3f}")
    print(f"{'='*len(header)}\n")


def main():
    args = parse_args()

    with open(args.labels) as f:
        raw_labels = json.load(f)

    # Normalize to simple {ep: 0|1} regardless of input format
    labels = {}
    for ep, v in raw_labels.items():
        if isinstance(v, dict):
            labels[ep] = 1 if v.get("outcome") == "failure" else 0
        else:
            labels[ep] = int(v)

    # Rich labels for step-level metrics (failure_step per episode)
    failure_steps = {}
    for ep, v in raw_labels.items():
        if isinstance(v, dict) and v.get("failure_step") is not None:
            failure_steps[ep] = int(v["failure_step"])

    print(f"Labels: {sum(v==1 for v in labels.values())} failures, {sum(v==0 for v in labels.values())} successes"
          + (f" ({len(failure_steps)} with failure_step)" if failure_steps else ""))

    # Collect (mode_name, scores_by_ep, config) tuples
    entries = []
    if args.scores:
        scores_by_ep, config = load_scores(args.scores)
        mode = config.get("mode", Path(args.scores).parent.name)
        entries.append((mode, scores_by_ep, config))
    elif args.scores_dir:
        for mode in args.modes:
            path = os.path.join(args.scores_dir, mode, "scores.json")
            if not os.path.exists(path):
                print(f"  Skipping {mode}: {path} not found")
                continue
            scores_by_ep, config = load_scores(path)
            entries.append((mode, scores_by_ep, config))

    if not entries:
        print("No score files found.")
        return

    # Per-mode chunk windows: num_hist/num_pred come from the scores.json config, since
    # they determine which chunk could actually have predicted the failure.
    windows_for = {}
    for mode, _, config in entries:
        nh = config.get("num_hist", 3)
        npd = config.get("num_pred", 8)
        windows_for[mode] = {
            ep: chunk_bounds(fs, nh, npd, args.label_window)
            for ep, fs in failure_steps.items()
        }
    if failure_steps:
        nh = entries[0][2].get("num_hist", 3)
        npd = entries[0][2].get("num_pred", 8)
        ex_ep, ex_fs = next(iter(failure_steps.items()))
        lo, hi, cut = chunk_bounds(ex_fs, nh, npd, args.label_window)
        print(f"Label window '{args.label_window}' (num_hist={nh}, num_pred={npd}): "
              f"e.g. {ex_ep} fails at {ex_fs} -> unsafe chunks [{lo}, {hi}], drop start > {cut}")

    if args.level in ("episode", "both"):
        ep_results = {}
        for mode, scores_by_ep, config in entries:
            cutoffs = {ep: b[2] for ep, b in windows_for[mode].items()}
            ep_results[mode] = best_f1_metrics(
                scores_by_ep, labels, args.n_thresholds,
                cutoffs=cutoffs, truncate=not args.no_truncate)
        print_table(ep_results, "episode")

    if args.level in ("step", "both"):
        step_results = {}
        for mode, scores_by_ep, config in entries:
            thresh = config.get("threshold", 0.87)
            step_results[mode] = step_level_metrics(
                scores_by_ep, labels, thresh, args.n_thresholds,
                windows=windows_for[mode],
                truncate=not args.no_truncate,
            )
        print_table(step_results, "step")

    # Save results JSON
    all_results = {
        mode: {"episode": ep_results.get(mode, {}), "step": step_results.get(mode, {})}
        for mode, _, _ in entries
    }
    out_path = os.path.join(args.scores_dir or str(Path(args.scores).parent), "metrics_summary.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"Summary saved to {out_path}")


if __name__ == "__main__":
    main()
