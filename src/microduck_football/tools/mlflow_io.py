#!/usr/bin/env python3
"""Export the MLflow tracking database to a small portable folder, or rebuild a database from it.

  # rebuild the training dashboard from the committed export, then browse it
  uv run python -m microduck_football.tools.mlflow_io import
  uv run mlflow ui --backend-store-uri sqlite:///runs/mlflow.db

  # (maintainers) refresh the export from a local runs/mlflow.db
  uv run python -m microduck_football.tools.mlflow_io export

Export format (experiments/mlflow/): runs.json (experiment, run name, status, params, tags) and one
gzipped CSV per run with every logged metric point (key, step, timestamp, value).
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

from microduck_football import paths  # noqa: E402

EXPORT_DIR = paths.REPO_ROOT / "experiments" / "mlflow"
SKIP_RUNS = {"smoke", "smoke_2v2", "smoke_v3", "fpscheck"}       # plumbing tests, not experiments


def export(args):
    from mlflow import MlflowClient
    client = MlflowClient(args.uri)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    index = []
    for exp in client.search_experiments():
        for run in client.search_runs([exp.experiment_id], max_results=1000):
            name = run.info.run_name
            if name in SKIP_RUNS or run.info.status == "KILLED":
                continue
            fname = f"{exp.name}__{name}__{run.info.run_id[:8]}.csv.gz"
            n = 0
            with gzip.open(EXPORT_DIR / fname, "wt", newline="") as f:
                w = csv.writer(f)
                w.writerow(["key", "step", "timestamp", "value"])
                for key in sorted(run.data.metrics):
                    for m in client.get_metric_history(run.info.run_id, key):
                        w.writerow([key, m.step, m.timestamp, m.value])
                        n += 1
            index.append({"experiment": exp.name, "run_name": name, "status": run.info.status,
                          "start_time": run.info.start_time, "end_time": run.info.end_time,
                          "params": dict(run.data.params),
                          "tags": {k: v for k, v in run.data.tags.items() if not k.startswith("mlflow.")},
                          "metrics_file": fname, "n_points": n})
            print(f"{exp.name:28s} {name:32s} {n:7d} points")
    (EXPORT_DIR / "runs.json").write_text(json.dumps(index, indent=1))
    print(f"exported {len(index)} runs to {EXPORT_DIR}")


def import_(args):
    import mlflow
    from mlflow.entities import Metric, Param, RunTag
    from mlflow import MlflowClient
    index = json.loads((EXPORT_DIR / "runs.json").read_text())
    paths.RUNS_DIR.mkdir(exist_ok=True)
    client = MlflowClient(args.uri)
    for r in index:
        exp = client.get_experiment_by_name(r["experiment"])
        exp_id = exp.experiment_id if exp else client.create_experiment(r["experiment"])
        run = client.create_run(exp_id, run_name=r["run_name"], start_time=r["start_time"],
                                tags={**r["tags"], "imported_from": "experiments/mlflow"})
        rid = run.info.run_id
        params = [Param(k, str(v)[:6000]) for k, v in r["params"].items()]
        for i in range(0, len(params), 100):
            client.log_batch(rid, params=params[i:i + 100])
        metrics = []
        with gzip.open(EXPORT_DIR / r["metrics_file"], "rt") as f:
            for row in csv.DictReader(f):
                metrics.append(Metric(row["key"], float(row["value"]), int(row["timestamp"]), int(row["step"])))
        for i in range(0, len(metrics), 1000):
            client.log_batch(rid, metrics=metrics[i:i + 1000])
        client.set_terminated(rid, status=r["status"] if r["status"] in ("FINISHED", "FAILED") else "FINISHED",
                              end_time=r["end_time"])
        print(f"imported {r['experiment']} / {r['run_name']} ({len(metrics)} points)")
    print(f"done - run:  uv run mlflow ui --backend-store-uri {args.uri}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(required=True)
    e = sub.add_parser("export")
    e.add_argument("--uri", default=paths.TRACKING_URI)
    e.set_defaults(func=export)
    i = sub.add_parser("import")
    i.add_argument("--uri", default=paths.TRACKING_URI, help="target database (default runs/mlflow.db)")
    i.set_defaults(func=import_)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
