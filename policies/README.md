---
license: apache-2.0
library_name: microduck
pipeline_tag: robotics
---

# Microduck motor skills (Pollen Robotics)

The complete policy set the microduck ships with: frozen ONNX policies from Pollen Robotics (Apache-2.0), trained
with [mjlab](https://github.com/mujocolab/mjlab) in [`microduck_rl`](https://github.com/pollen-robotics/microduck_rl).
All share one interface (61-D observation, 14 joint targets, 50 Hz), so they can be swapped at run time.
[`manifest.json`](manifest.json) is Pollen's description of the set (commands, durations, training provenance).
This project never modifies them; it only chooses which one runs.

| File | Kind | Skill | Used by this project |
|---|---|---|---|
| `alpha_walking.onnx` | continuous | omnidirectional walking (twist command) | **yes** — walk, turn, side-step, back |
| `alpha_stand.onnx` | continuous | standing, recovery after a fall | **yes** — stand |
| `ball_kick_left.onnx` / `ball_kick_right.onnx` | one-shot, 0.5 s | kick (trained for a ~1 m/s pass) | **yes** — kick left / right, aimed kick |
| `velstand.onnx` | continuous | walking that also stands still at zero command (Pollen's newer default) | not yet |
| `alpha_sitstand.onnx` | scripted | gentle sit down / stand up | not yet |
| `alpha_ground_pick.onnx` | one-shot, 2.8 s | pick something up from the ground | not yet |
| `roller.onnx`, `roller_crouch.onnx` | continuous / one-shot | skills for the roller-skate variant of the robot | not yet |
| `roulade.onnx` | one-shot, 1.0 s | roll | not yet |

To give the football player more skills (e.g. `velstand` for smoother stops): load the file where
`src/microduck_football/sim/duck_sim.py` builds its policy runner (`_Policies(...)`, which wraps the submodule's
`infer_policy.PolicyInference`, able to run every kind above), then expose it as a new entry in `PRIMITIVES`.
