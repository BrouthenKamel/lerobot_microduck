"""Every filesystem location the project uses, in one place.

Layout (repository root = REPO_ROOT):
  policies/                      frozen Pollen ONNX primitives (walk / stand / kicks)
  third_party/microduck_rl/      git submodule: infer_policy.py, robot MJCF + meshes
  src/microduck_football/sim/scenes/   pitch scenes (+ symlinks to the robot MJCF and meshes)
  models/                        small trained models shipped with the repo
  runs/   (gitignored)           training outputs, MLflow database
  data/   (gitignored)           local LeRobot datasets
  assets/media/                  videos
"""

from __future__ import annotations

import sys
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parent.parent

POLICY_DIR = REPO_ROOT / "policies"
THIRD_PARTY_DIR = REPO_ROOT / "third_party"
INFER_POLICY_DIR = THIRD_PARTY_DIR / "microduck_rl" / "scripts"

SCENE_DIR = PACKAGE_DIR / "sim" / "scenes"
SCENE_FOOTBALL = SCENE_DIR / "scene_football.xml"
SCENE_2V2 = SCENE_DIR / "scene_2v2.xml"
ROBOT_XML = SCENE_DIR / "robot_groundcontact.xml"

MODELS_DIR = REPO_ROOT / "models"
TEACHER_OPEN_PLAY = MODELS_DIR / "ppo_teacher_open_play.zip"      # PPO, open play (stages 0-3)
TEACHER_OBSTACLES = MODELS_DIR / "ppo_teacher_obstacles.zip"      # PPO, + opponents (stages 4-10)
MAPPO_2V2_BEST = MODELS_DIR / "mappo_2v2_best.pt"                 # 2v2 shared actor (+ critic)
ACT_STUDENT_DIR = MODELS_DIR / "act_camera_student"               # downloaded from the HF Hub

RUNS_DIR = REPO_ROOT / "runs"
DATA_DIR = REPO_ROOT / "data"
MEDIA_DIR = REPO_ROOT / "assets" / "media"
TRACKING_URI = f"sqlite:///{RUNS_DIR / 'mlflow.db'}"


def add_infer_policy_to_path() -> None:
    """infer_policy.py (the microduck RL team's CPU deployment runner) is a script, not a package:
    make it importable. Fails loudly if the submodule was not checked out."""
    if not (INFER_POLICY_DIR / "infer_policy.py").exists():
        raise FileNotFoundError(
            f"{INFER_POLICY_DIR / 'infer_policy.py'} not found - run `git submodule update --init`")
    p = str(INFER_POLICY_DIR)
    if p not in sys.path:
        sys.path.insert(0, p)
