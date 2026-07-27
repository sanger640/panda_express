#!/bin/bash
# Run all ablation modes sequentially and compute metrics.
#
# Prerequisites:
#   - labels.json created (see format in compute_metrics.py docstring)
#   - LMDB dataset available
#   - World model checkpoint available
#
# Usage:
#   ./run_ablations.sh
#
# Or override defaults:
#   LMDB=/path/to/jenga_single.lmdb LABELS=labels.json ./run_ablations.sh

set -e

LMDB="${LMDB:-tasks/jenga_mujoco_noise/jenga_single.lmdb}"
LABELS="${LABELS:-labels.json}"
RESULTS_DIR="${RESULTS_DIR:-results}"
SERVER_IP="${SERVER_IP:-localhost}"
PORT="${PORT:-5556}"
N_PERTURB="${N_PERTURB:-50}"
NOISE_STD="${NOISE_STD:-0.05}"
THRESHOLD="${THRESHOLD:-0.87}"
NUM_HIST="${NUM_HIST:-3}"
NUM_PRED="${NUM_PRED:-8}"
DINO_WM_DIR="${DINO_WM_DIR:-../dino_wm}"

MODES=(
    "ftle"               # proposed method
    "final_cosine"       # baseline: final state cosine only
    "mean_traj"          # baseline: mean over trajectory
    "max_step"           # baseline: max at any step
    "ftle_mean_patch"    # ablation: mean over all patches
    "ftle_gap"           # ablation: global average pooled
    "ftle_l2"            # ablation: L2 instead of cosine
    "ftle_topk"          # ablation: top-5 patch mean
    "ftle_variance"      # fix 1: cross-perturbation spread (no original trajectory reference)
)

echo "=========================================="
echo "Ablation experiment"
echo "  LMDB:       $LMDB"
echo "  Labels:     $LABELS"
echo "  Results:    $RESULTS_DIR"
echo "  Modes:      ${MODES[*]}"
echo "=========================================="

if [ ! -f "$LABELS" ]; then
    echo "ERROR: labels.json not found at $LABELS"
    echo "Create it with format: {\"episode_001\": 0, \"episode_002\": 1, ...}"
    exit 1
fi

for MODE in "${MODES[@]}"; do
    echo ""
    echo "----------------------------------------"
    echo "Running mode: $MODE"
    echo "----------------------------------------"

    OUT_DIR="$RESULTS_DIR/$MODE"
    mkdir -p "$OUT_DIR"

    # Start the ablation server in the background
    echo "Starting server [mode=$MODE]..."
    pushd "$DINO_WM_DIR" > /dev/null
    python server_ablation.py mode="$MODE" &
    SERVER_PID=$!
    popd > /dev/null

    # Wait for server to be ready
    echo "Waiting for server to start..."
    sleep 8

    # Run evaluation
    python test_monitor.py \
        --lmdb "$LMDB" \
        --server-ip "$SERVER_IP" \
        --port "$PORT" \
        --n-perturb "$N_PERTURB" \
        --noise-std "$NOISE_STD" \
        --num-hist "$NUM_HIST" \
        --num-pred "$NUM_PRED" \
        --threshold "$THRESHOLD" \
        --output-dir "$OUT_DIR"

    # Save mode name into the scores.json config
    python -c "
import json
path = '$OUT_DIR/scores.json'
with open(path) as f: d = json.load(f)
d.setdefault('config', {})['mode'] = '$MODE'
with open(path, 'w') as f: json.dump(d, f, indent=2)
"

    # Stop server
    kill $SERVER_PID 2>/dev/null || true
    wait $SERVER_PID 2>/dev/null || true
    echo "Server stopped."
done

echo ""
echo "=========================================="
echo "Computing metrics across all modes..."
echo "=========================================="

python compute_metrics.py \
    --scores-dir "$RESULTS_DIR" \
    --modes "${MODES[@]}" \
    --labels "$LABELS" \
    --level both

echo "Done. Results in $RESULTS_DIR/metrics_summary.json"
