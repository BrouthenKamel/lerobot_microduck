#!/usr/bin/env python3
"""Publish a local LeRobot dataset to the Hugging Face Hub (needs `hf auth login` with a write token).

  uv run python -m microduck_football.imitation.push_dataset data/rl_football_v2 --name microduck-football-rl-camera --public
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="local dataset root (contains meta/, data/, videos/)")
    ap.add_argument("--name", required=True, help="dataset repo name (pushed under your HF user)")
    ap.add_argument("--public", action="store_true")
    args = ap.parse_args()

    from huggingface_hub import whoami
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    user = whoami()["name"]
    repo_id = f"{user}/{args.name}"
    root = args.root.resolve()
    ds = LeRobotDataset(f"local/{root.name}", root=root)
    ds.repo_id = repo_id
    ds.meta.repo_id = repo_id
    print(f"pushing {root} ({ds.meta.total_episodes} episodes, {ds.meta.total_frames} frames) -> {repo_id} "
          f"({'public' if args.public else 'private'})", flush=True)
    ds.push_to_hub(private=not args.public,
                   tags=["LeRobot", "microduck", "football", "simulation", "mujoco", "reinforcement-learning",
                         "imitation-learning"])
    print(f"https://huggingface.co/datasets/{repo_id}")
    print(f"viewer: https://huggingface.co/spaces/lerobot/visualize_dataset?path=%2F{repo_id}%2Fepisode_0")


if __name__ == "__main__":
    main()
