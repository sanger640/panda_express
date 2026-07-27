"""
Worked example: how the monitor metrics treat prediction vs observation.

============================================================================
THE NUMBERS THIS PRINTS ARE NOT MEASUREMENTS. The scores below are invented,
hand-picked to isolate the effect of each correction. They say nothing about
the Jenga task, the world model, or the monitor's real accuracy. This is a
regression check on the arithmetic in compute_metrics.py, nothing more.
Real numbers require a world model checkpoint and a run of test_monitor.py.
============================================================================

Builds a tiny synthetic labels.json + scores.json in a temp dir and runs compute_metrics.py
three times -- current defaults, legacy label window, and fully-old behaviour -- so each
correction is visible in isolation, without needing a checkpoint or an LMDB.

    python demo_metric_truncation.py

The scenario has six episodes:

    ep_01  failure @ step 40   monitor spikes AT step 40          -> genuine prediction
    ep_02  failure @ step 24   monitor spikes AT step 24          -> genuine prediction
    ep_06  failure @ step 32   monitor quiet until step 40        -> MISSED the fall,
                                                                     only saw the wreckage
    ep_03  success            monitor quiet throughout
    ep_04  success            one moderate blip
    ep_05  success            monitor quiet throughout

In every failed episode the scores after failure_step are large, because the world model is
looking at a scene where the blocks have already come down. That is correct behaviour, not
a false alarm -- but only if the metric knows to exclude those steps.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

LABELS = {
    "ep_01": {"outcome": "failure", "failure_step": 40, "vote": "3/3"},
    "ep_02": {"outcome": "failure", "failure_step": 24, "vote": "3/3"},
    "ep_06": {"outcome": "failure", "failure_step": 32, "vote": "3/3"},
    "ep_03": {"outcome": "success", "failure_step": None, "vote": "0/3"},
    "ep_04": {"outcome": "success", "failure_step": None, "vote": "0/3"},
    "ep_05": {"outcome": "success", "failure_step": None, "vote": "0/3"},
}

# Chunks are num_pred=8 apart, matching test_monitor.py's stride. With num_hist=3 and
# num_pred=8 the predictive chunk for a failure at fs is the one in [fs-10, fs-3]:
#     ep_01 fs=40 -> chunk 32     ep_02 fs=24 -> chunk 16     ep_06 fs=32 -> chunk 24
# ep_01/ep_02 spike on exactly that chunk (real forecast). ep_06 stays flat there and only
# spikes at fs itself, where the topple is already inside its observation window.
SCORES = {
    "ep_01": {"0": .30, "8": .35, "16": .40, "24": .45, "32": 1.20,
              "40": 2.10, "48": 2.40, "56": 2.30},
    "ep_02": {"0": .30, "8": .40, "16": 1.10,
              "24": 2.00, "32": 2.20, "40": 2.50, "48": 2.30},
    "ep_06": {"0": .30, "8": .35, "16": .40, "24": .45,
              "32": 2.20, "40": 2.40, "48": 2.35},
    "ep_03": {"0": .30, "8": .35, "16": .45, "24": .70, "32": .55, "40": .40, "48": .35},
    "ep_04": {"0": .32, "8": .38, "16": .90, "24": .60, "32": .42, "40": .38, "48": .36},
    "ep_05": {"0": .28, "8": .31, "16": .35, "24": .50, "32": .44, "40": .33, "48": .30},
}


def main():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "labels.json").write_text(json.dumps(LABELS, indent=2))
        (tmp / "scores.json").write_text(json.dumps(
            {"config": {"mode": "demo", "num_pred": 8, "threshold": 0.87},
             "scores": SCORES}, indent=2))

        for title, extra in [
            ("CURRENT — predictive window, post-failure chunks dropped (defaults)", []),
            ("LEGACY WINDOW — also credits the chunk watching the topple",
             ["--label-window", "legacy"]),
            ("FULLY OLD — legacy window and post-failure chunks kept",
             ["--label-window", "legacy", "--no-truncate"]),
        ]:
            print("\n" + "#" * 72)
            print("#  " + title)
            print("#" * 72)
            sys.stdout.flush()   # keep headers ordered against the subprocess output
            subprocess.run(
                [sys.executable, str(HERE / "compute_metrics.py"),
                 "--scores", str(tmp / "scores.json"),
                 "--labels", str(tmp / "labels.json")] + extra,
                check=True,
            )

    print("""
What to look at  (again: invented scores, not measurements)
-----------------------------------------------------------
ep_06 is the episode that matters. Its monitor stayed flat at 0.45 on chunk 24 -- the only
chunk that could have forecast the topple at step 32 -- and then jumped to 2.20 on chunk 32,
which already has the topple inside its 3-frame observation window. It predicted nothing; it
merely reported what it could see.

  CURRENT (predictive window, chunks dropped past the cutoff)
      ep_06 is judged on chunk 24 only -> 0.45, below any useful threshold -> counted as a
      miss. Recall 66.7%: two of three failures genuinely forecast. This is the honest number.

  LEGACY WINDOW
      The unsafe window widens to [fs-8, fs], so chunk 32 now counts as a detection
      opportunity for ep_06 -- and it scores 2.20. Recall jumps to 100%. The monitor is
      credited for watching a block fall over.

  FULLY OLD
      Post-failure chunks return as well. Episode metrics read a flawless 100% across the
      board, while step-level precision drops to 24% because every post-collapse chunk is
      labeled safe and scored high, each one booked as a false alarm.

The two errors push opposite ways -- recall flattered, precision punished -- so they do not
cancel out. They distort the precision/recall trade-off itself, which is the axis the
ablation table compares modes along.
""")


if __name__ == "__main__":
    main()
