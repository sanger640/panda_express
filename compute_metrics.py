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

Metrics are computed at two levels:
    episode-level: TP = failed episode where monitor triggered at least once
    step-level:    steps within NUM_PRED of failure_step are labeled unsafe;
                   all steps in success episodes are safe (requires rich labels)
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
    return p.parse_args()


def load_scores(path):
    with open(path) as f:
        data = json.load(f)
    return data.get("scores", data), data.get("config", {})


def compute_pr_curve(scores_by_ep, labels, n_thresh=100):
    """
    scores_by_ep: {ep_name: {step: score}}
    labels: {ep_name: 0|1}
    Returns thresholds, precision, recall arrays.
    """
    # Collect all scores with labels
    episode_max_scores = {}
    for ep, step_scores in scores_by_ep.items():
        if ep not in labels:
            continue
        episode_max_scores[ep] = max(step_scores.values()) if step_scores else 0.0

    all_scores = list(episode_max_scores.values())
    if not all_scores:
        return np.array([]), np.array([]), np.array([])

    thresholds = np.linspace(min(all_scores), max(all_scores), n_thresh)
    precisions, recalls = [], []

    for thresh in thresholds:
        tp = fp = tn = fn = 0
        for ep, max_score in episode_max_scores.items():
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


def best_f1_metrics(scores_by_ep, labels, n_thresh=100):
    """Returns dict of metrics at the threshold with best F1."""
    thresholds, precisions, recalls = compute_pr_curve(scores_by_ep, labels, n_thresh)

    if len(thresholds) == 0:
        return {"accuracy": 0, "precision": 0, "recall": 0, "f1": 0, "threshold": 0}

    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-8)
    best_idx = int(np.argmax(f1s))
    best_thresh = thresholds[best_idx]

    # Recompute full metrics at best threshold
    episode_max_scores = {
        ep: max(step_scores.values()) if step_scores else 0.0
        for ep, step_scores in scores_by_ep.items() if ep in labels
    }

    tp = fp = tn = fn = 0
    for ep, max_score in episode_max_scores.items():
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


def step_level_metrics(scores_by_ep, labels, threshold, n_thresh=100, failure_steps=None, num_pred=8):
    """
    Assigns per-step labels then finds the best-F1 threshold.

    If failure_steps is provided (from generate_labels.py), only the window of steps
    immediately before the failure are labeled unsafe (more precise than labeling the
    entire failed episode). Otherwise every step in a failed episode is labeled unsafe.
    """
    step_preds = []  # (label, score) pairs
    for ep, step_scores in scores_by_ep.items():
        if ep not in labels:
            continue
        ep_label = labels[ep]
        fs = (failure_steps or {}).get(ep)

        for step_str, score in step_scores.items():
            step = int(step_str)
            if ep_label == 1 and fs is not None:
                # Only the window of chunks up to num_pred steps before failure are unsafe
                step_label = 1 if (fs - num_pred) <= step <= fs else 0
            else:
                step_label = ep_label
            step_preds.append((step_label, score))

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

    if args.level in ("episode", "both"):
        ep_results = {}
        for mode, scores_by_ep, config in entries:
            ep_results[mode] = best_f1_metrics(scores_by_ep, labels, args.n_thresholds)
        print_table(ep_results, "episode")

    if args.level in ("step", "both"):
        step_results = {}
        for mode, scores_by_ep, config in entries:
            thresh = config.get("threshold", 0.87)
            step_results[mode] = step_level_metrics(
                scores_by_ep, labels, thresh, args.n_thresholds,
                failure_steps=failure_steps,
                num_pred=config.get("num_pred", 8),
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
