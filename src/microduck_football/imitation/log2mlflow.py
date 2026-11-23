#!/usr/bin/env python3
"""Stream a running `lerobot-train` log into MLflow (live curves in the MLflow UI).

  uv run python -m microduck_football.imitation.log2mlflow runs/act_e2e_v0.log --run-name act_e2e_v0 --log-freq 250

Parses LeRobot's periodic "step:... loss:... l1_loss:... kld_loss:..." lines; stops when the
training log says it is done (or after --idle-exit seconds without new lines).
"""

from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import mlflow  # noqa: E402

from microduck_football import paths  # noqa: E402

RUNS = paths.RUNS_DIR
KEYS = ("loss", "grdn", "lr", "data_s", "updt_s", "step_s", "l1_loss", "kld_loss")
PAT = re.compile(r"(\w+):([-+0-9.eE]+)(K|M)?")


def parse(line: str) -> dict:
    vals = {}
    for k, v, suf in PAT.findall(line):
        if k in KEYS:
            try:
                vals[k] = float(v) * {"K": 1e3, "M": 1e6}.get(suf, 1.0)
            except ValueError:
                pass
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", type=Path)
    ap.add_argument("--run-name", required=True)
    ap.add_argument("--log-freq", type=int, default=250, help="the run's --log_freq (x axis)")
    ap.add_argument("--experiment", default="microduck_football_act")
    ap.add_argument("--idle-exit", type=float, default=600)
    args = ap.parse_args()

    mlflow.set_tracking_uri(f"sqlite:///{RUNS / 'mlflow.db'}")
    mlflow.set_experiment(args.experiment)
    with mlflow.start_run(run_name=args.run_name):
        mlflow.set_tag("source_log", str(args.log))
        n, pos, last_new = 0, 0, time.time()
        while True:
            with open(args.log, "r", errors="ignore") as f:
                f.seek(pos)
                chunk = f.read()
                pos = f.tell()
            for line in chunk.replace("\r", "\n").split("\n"):
                if "ot_train.py" in line and " step:" in line and "loss:" in line:
                    vals = parse(line)
                    if vals:
                        n += 1
                        step = n * args.log_freq
                        if "step_s" in vals and vals["step_s"] > 0:
                            vals["steps_per_s"] = 1.0 / vals["step_s"]
                        mlflow.log_metrics(vals, step=step)
                        last_new = time.time()
                if "End of training" in line:
                    print("training finished; stopping")
                    return
            if time.time() - last_new > args.idle_exit:
                print("no new log lines; stopping")
                return
            time.sleep(5)


if __name__ == "__main__":
    main()
