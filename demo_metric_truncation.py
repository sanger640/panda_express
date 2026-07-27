"""
Worked example: why post-failure steps must be dropped before scoring the monitor.

Builds a tiny synthetic labels.json + scores.json in a temp dir and runs compute_metrics.py
twice -- once with the current (correct) behaviour, once with --no-truncate -- so the effect
is visible in isolation, without needing a world model checkpoint or an LMDB.

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

# Chunks are num_pred=8 apart, matching test_monitor.py's stride.
SCORES = {
    "ep_01": {"0": .30, "8": .35, "16": .40, "24": .45, "32": .60, "40": 1.20,
              "48": 2.10, "56": 2.40, "64": 2.30},
    "ep_02": {"0": .30, "8": .40, "16": .50, "24": 1.10,
              "32": 2.00, "40": 2.20, "48": 2.50, "56": 2.30},
    "ep_06": {"0": .30, "8": .35, "16": .40, "24": .42, "32": .45,
              "40": 2.20, "48": 2.40, "56": 2.35},
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
            ("CORRECT — post-failure steps dropped (default)", []),
            ("OLD — post-failure steps kept (--no-truncate)", ["--no-truncate"]),
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
What to look at
---------------
EPISODE level, recall: 66.7% correct vs 100.0% old.
    ep_06 never flagged the fall -- it only lit up at step 40, after the topple at 32.
    Keeping those steps lets its post-collapse score satisfy the trigger, so the old
    path books a true positive for an episode the monitor completely missed. A monitor
    that reports failures after they happen has no safety value; recall must not reward it.

STEP level, precision: 40.0% correct vs 24.0% old.
    The post-failure steps in ep_01/02/06 all score high. Labeled "safe" and kept, each
    becomes a false positive -- the monitor is charged for correctly noticing that a
    collapsed tower is unstable. Dropping them nearly doubles precision here.

Both distortions push in the flattering direction for recall and the punishing direction
for precision, so they do not cancel: they distort the precision/recall trade-off itself.
""")


if __name__ == "__main__":
    main()
