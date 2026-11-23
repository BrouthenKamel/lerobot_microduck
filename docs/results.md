# Results provenance

Every number in the README comes from [`results.json`](results.json), which records where each value came from.
All sources below are committed in this repository.

| Result | Source | Script that produces it |
|---|---|---|
| RL player, per-stage goal rates (20 new games each) | [`experiments/logs/eval/showreel_eval.log`](../experiments/logs/eval/showreel_eval.log) | `rl/make_showreel.py` |
| Camera player, attempt by attempt (20 new open-play games) | [`experiments/logs/eval/eval_*.log`](../experiments/logs/eval/) | `imitation/eval_student.py` |
| Camera player re-measured on the final simulator | [`experiments/logs/eval/act_verify_hub.log`](../experiments/logs/eval/act_verify_hub.log) | `imitation/eval_student.py` |
| "Why copying fails" and DAgger experiments | console output recorded verbatim in `results.json` | `imitation/bc_probe.py`, `imitation/dagger_probe.py` |
| Kick distance by floor | [`kick_rolling.json`](kick_rolling.json) | `rl/kick_lab.py rolling` |
| 2v2 checks during training | [`experiments/logs/eval/mappo_2v2_v3_fair/`](../experiments/logs/eval/mappo_2v2_v3_fair/) | `multiagent/train_2v2.py eval` |
| Training curves (curriculum, losses) | [`experiments/mlflow/`](../experiments/mlflow/) | rebuild with `tools/mlflow_io.py import` |

Figures: `uv run python -m microduck_football.tools.make_figures` (run `tools/mlflow_io.py import` first for the
training-curve figure).

**Evaluation noise.** Most evaluations use 20 games: a 16/20 vs 12/20 difference is about ±2 goals of noise each.
The simulator is deterministic for a given model, but adding bodies to the scene (the parked opponents) changes the
order of contact computations, after which identical starting positions play out differently.
