#!/bin/bash
# Full 100-episode evaluation, single-view checkpoint.
#
# Assumes replay_noisy.py has already produced 100 episodes in tasks/jenga_noise_50/.
# Builds the single-camera LMDB, extracts labels from the ground truth recorded during
# replay, starts the monitor server, and scores every chunk under all 14 candidate
# metrics in one pass.
set -e
PY=/home/sanger/miniforge3/envs/dino_wm/bin/python
PE=/home/sanger/wksp/panda_express
DW=/home/sanger/wksp/dino_wm
TASK=$PE/tasks/jenga_noise_50
SCRATCH=/tmp/claude-1000/-home-sanger-wksp/8db8a6b9-15bc-43dd-9e43-3a6f6584cad0/scratchpad

echo "=== 1/4 building single-camera LMDB ($(ls $TASK/episodes | wc -l) episodes) ==="
cd $DW
$PY create_lmdb_single30.py --data-path $TASK --lmdb-path $TASK/jenga_single_100.lmdb 2>&1 | grep -viE "^err :|it/s" | tail -2

echo "=== 2/4 extracting labels from recorded ground truth ==="
cd $PE
$PY extract_labels.py --episodes-dir $TASK/episodes --lmdb $TASK/jenga_single_100.lmdb \
    --output labels_noise100.json | tail -6

echo "=== 3/4 starting monitor server (single-view) ==="
cd $DW
DISPLAY=:1 nohup $PY -u server_single_max.py > $SCRATCH/server_100.log 2>&1 &
until grep -q "listening on" $SCRATCH/server_100.log 2>/dev/null; do sleep 3; done
echo "server up"

echo "=== 4/4 scoring all chunks (all 14 metrics) ==="
cd $PE
$PY test_monitor.py --lmdb $TASK/jenga_single_100.lmdb --n-perturb 50 --noise-std 0.05 \
    --threshold 0.0275 --output-dir results/eval100_single 2>&1 | tail -3

echo "=== done: results/eval100_single/scores.json ==="
