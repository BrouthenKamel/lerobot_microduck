# Trained models

| File | What | Obs → action | Result (see `docs/results.json`) |
|---|---|---|---|
| `ppo_teacher_open_play.zip` | PPO teacher, open play (curriculum stages 0–3) | 29-D privileged → 9 primitives | 20/20 goals on stage 3 (full pitch), ~18 s |
| `ppo_teacher_obstacles.zip` | PPO teacher with opponents (warm-started, stages 4–10) | 49-D privileged → 9 primitives | 8/20 – 20/20 depending on the stage (keeper + defenders = 8/20) |
| `mappo_2v2_best.pt` | 2v2 MAPPO shared actor + centralised critic (`{"actor", "critic"}` state dicts) | 50-D team-frame obs → 10 actions | 0.40 vs a team of the 1v1 teacher (9W/30D/21L, 60 games; its best 20-game check during training was 0.65) |
| `act_camera_student/` *(not in git)* | LeRobot ACT camera student, DAgger round 2 | head camera + 32-D state → 4-D twist + kick | download: `uv run python -m microduck_football.tools.download_models` ([HF model card](https://huggingface.co/brouthen/microduck-football-act)) |

All three small models are loaded by default by the scripts (`src/microduck_football/paths.py`).
