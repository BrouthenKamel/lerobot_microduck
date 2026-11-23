#!/usr/bin/env python3
"""Download the ACT camera student (too large for git) from the Hugging Face Hub into models/.

  uv run python -m microduck_football.tools.download_models
  uv run python -m microduck_football.imitation.eval_student models/act_camera_student --stage 3 --episodes 20

The PPO teachers and the 2v2 MAPPO policy are small and ship with the repository (models/).
"""

from __future__ import annotations

import argparse

from microduck_football import paths

ACT_REPO = "brouthen/microduck-football-act"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", default=ACT_REPO)
    ap.add_argument("--out", default=str(paths.ACT_STUDENT_DIR))
    args = ap.parse_args()

    from huggingface_hub import snapshot_download
    path = snapshot_download(args.repo, local_dir=args.out)
    print(f"{args.repo} -> {path}")


if __name__ == "__main__":
    main()
