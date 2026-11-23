# Experiment records

Everything here is small, text-based and safe to commit. It lets you inspect every training run of the hackathon
without re-running anything.

| Folder | What | How to use it |
|---|---|---|
| `mlflow/` | every training run's settings and full metric history (curriculum stage, goal rates, losses, 2v2 scores): 14 runs, 111 k points | `uv run python -m microduck_football.tools.mlflow_io import` rebuilds `runs/mlflow.db`, then `uv run mlflow ui --backend-store-uri sqlite:///runs/mlflow.db` |
| `logs/eval/` | evaluation outputs, game by game (plain text) | open in any editor — these are the sources of the README numbers (`docs/results.json`) |
| `logs/train/` | training console logs (gzipped): PPO teacher, opponent curriculum, ACT + DAgger rounds, dataset generation, 2v2 MAPPO | `zcat logs/train/ppo_main_5m_t60.log.gz \| less` |

Run names, in the order they happened:

| Run | Chapter | What it is |
|---|---|---|
| `sanity_v1`, `ppo_fix_30min` | 1 | first PPO runs (plumbing, throughput fix) |
| `ppo_main_5m_t60` | 1 | the open-play RL player (5 M decisions, 60 % advancement threshold) |
| `ppo_obst_v1` | 1 | first opponent run — **invalid**: the defenders were see-through (kept as a record of the bug) |
| `ppo_opponents_v1` → `ppo_opponents_v2_keeper_ladder` | 1 | opponent curriculum with physical defenders, then the 3-step goalkeeper ladder |
| `act_e2e_v0`, `act_e2e_v1_twist` | 3 | first ACT attempts (button presses → smoothed steering) |
| `act_v2_mem_round0`, `_dagger1`, `_dagger2` | 3 | ACT with goal memory, then the two DAgger rounds (local folders `act_v2`, `act_v2_d1`, `act_v2_d2`) |
| `mappo_2v2_v1`, `v2`, `v3` | 4 | the three 2v2 self-play attempts |

Paths inside the logs are shortened to `<repo>`. Large checkpoints are not included (the final models are in
`models/` and on the Hugging Face Hub).
