#!/usr/bin/env python3
"""Train the RL teacher (PPO) on FootballEnv with a reverse curriculum; log to MLflow.

  # train (7 parallel envs, starts at curriculum stage 0)
  uv run python -m microduck_football.rl.train_teacher train --run-name ppo_v1 --total-steps 5_000_000 --workers 7

  # smoke test (2 minutes)
  uv run python -m microduck_football.rl.train_teacher train --run-name smoke --total-steps 20_000 --workers 2

  # evaluate a checkpoint on a stage (3 = full random pitch)
  uv run python -m microduck_football.rl.train_teacher eval runs/ppo_v1/final.zip --stage 3 --episodes 50

  # look at the runs
  uv run mlflow ui --backend-store-uri sqlite:///runs/mlflow.db

Curriculum: the stage goes up when the goal rate over the last --window episodes OF THE CURRENT
STAGE reaches --advance-at. Episodes from earlier stages are replayed 20% of the time (football_env.py).
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import numpy as np  # noqa: E402

from microduck_football import paths  # noqa: E402

RUNS = paths.RUNS_DIR
TRACKING_URI = paths.TRACKING_URI
EXPERIMENT = "microduck_football_teacher"


def make_env(rank: int, seed: int, stage: int | None, obstacles: bool = False):
    def _init():
        from microduck_football.envs.football_env import FootballEnv
        return FootballEnv(seed=seed + 1000 * rank, stage=stage, obstacles=obstacles)
    return _init


def widen_from(model, old_path: str):
    """Warm start across an observation change: copy every weight from the old policy; first
    layers that got wider (new obs features appended at the end) keep the old columns and get
    ZERO weights for the new inputs, so the new policy initially behaves exactly like the old."""
    from stable_baselines3 import PPO
    old = PPO.load(old_path, device="cpu").policy.state_dict()
    new = model.policy.state_dict()
    copied, widened = 0, 0
    for k, v in old.items():
        if k not in new:
            continue
        if new[k].shape == v.shape:
            new[k].copy_(v)
            copied += 1
        elif v.ndim == 2 and new[k].shape[0] == v.shape[0] and new[k].shape[1] > v.shape[1]:
            new[k].zero_()
            new[k][:, :v.shape[1]] = v
            widened += 1
    model.policy.load_state_dict(new)
    print(f"warm start from {old_path}: {copied} tensors copied, {widened} first layers widened")


# ── MLflow output for SB3's logger ───────────────────────────────────────────
def mlflow_writer():
    import mlflow
    from stable_baselines3.common.logger import KVWriter

    class MLflowOutputFormat(KVWriter):
        def write(self, key_values, key_excluded, step=0):
            metrics = {}
            for k, v in key_values.items():
                if isinstance(v, (int, float, np.floating, np.integer)) and not isinstance(v, bool):
                    metrics[k.replace("/", "_")] = float(v)
            if metrics:
                mlflow.log_metrics(metrics, step=int(step))

        def close(self):
            pass

    return MLflowOutputFormat()


def curriculum_callback(start_stage: int, advance_at: float, window: int, max_stage: int):
    from stable_baselines3.common.callbacks import BaseCallback

    class Curriculum(BaseCallback):
        """Tracks finished episodes (VecMonitor puts 'episode' + our info keys in infos) and
        moves the whole VecEnv to the next stage when the current one is mastered."""

        def __init__(self):
            super().__init__()
            self.stage = start_stage
            self.recent = collections.deque(maxlen=window)     # successes at the current stage
            self.all_eps = collections.deque(maxlen=200)       # (success, kicks, falls, len_s, stage)

        def _on_training_start(self):
            self.training_env.env_method("set_stage", self.stage)

        def _on_step(self) -> bool:
            for info in self.locals["infos"]:
                if "episode" not in info:
                    continue
                st = int(info.get("stage", -1))
                ok = bool(info.get("is_success", False))
                self.all_eps.append((ok, info.get("n_kicks", 0), info.get("n_falls", 0),
                                     info.get("sim_time", 0.0), st))
                if st == self.stage:
                    self.recent.append(ok)
            if (len(self.recent) == window and np.mean(self.recent) >= advance_at
                    and self.stage < max_stage):
                self.stage += 1
                self.recent.clear()
                self.training_env.env_method("set_stage", self.stage)
                print(f"\n=== curriculum: stage -> {self.stage} at {self.num_timesteps} steps ===\n",
                      flush=True)
            return True

        def _on_rollout_end(self):
            self.logger.record("curriculum/stage", self.stage)
            if self.recent:
                self.logger.record("curriculum/goal_rate_current_stage", float(np.mean(self.recent)))
            if self.all_eps:
                arr = np.array(self.all_eps, dtype=float)
                self.logger.record("episodes/goal_rate_all", float(arr[:, 0].mean()))
                self.logger.record("episodes/kicks", float(arr[:, 1].mean()))
                self.logger.record("episodes/falls", float(arr[:, 2].mean()))
                self.logger.record("episodes/sim_seconds", float(arr[:, 3].mean()))

    return Curriculum()


def train(args):
    import mlflow
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.logger import HumanOutputFormat, Logger
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor

    from microduck_football.envs import football_env as E

    out = RUNS / args.run_name
    out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)

    fns = [make_env(i, args.seed, args.start_stage, args.obstacles) for i in range(args.workers)]
    venv = SubprocVecEnv(fns, start_method="fork") if args.workers > 1 else DummyVecEnv(fns)
    venv = VecMonitor(venv, info_keywords=("is_success", "n_kicks", "n_falls", "sim_time", "stage"))

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=args.run_name):
        params = {k: v for k, v in vars(args).items() if k != "func"}
        params.update(obs_dim=E.OBS_DIM, n_actions=len(E.ds.PRIMITIVES), decision_hz=E.DECISION_HZ,
                      w_ball=E.W_BALL_PROGRESS, w_duck=E.W_DUCK_PROGRESS, r_goal=E.R_GOAL,
                      r_fall=E.R_FALL, r_time=E.R_TIME, replay_earlier=E.REPLAY_EARLIER)
        mlflow.log_params(params)

        if args.resume:
            model = PPO.load(args.resume, env=venv, device="cpu")
            if args.lr_override is not None:
                model.learning_rate = args.lr_override
                model.lr_schedule = lambda _: args.lr_override
        else:
            model = PPO(
                "MlpPolicy", venv, device="cpu", seed=args.seed,
                n_steps=args.n_steps, batch_size=args.batch_size, n_epochs=args.n_epochs,
                learning_rate=args.lr, gamma=args.gamma, gae_lambda=0.95, clip_range=0.2,
                ent_coef=args.ent_coef, vf_coef=0.5, max_grad_norm=0.5,
                policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
            )
            if args.warm_start:
                widen_from(model, args.warm_start)
        model.set_logger(Logger(str(out), [HumanOutputFormat(sys.stdout), mlflow_writer()]))

        max_stage = (E.N_STAGES_OBSTACLES if args.obstacles else E.N_STAGES) - 1
        cur = curriculum_callback(args.start_stage, args.advance_at, args.window, max_stage)
        ckpt = CheckpointCallback(save_freq=max(args.save_every // args.workers, 1),
                                  save_path=str(out / "checkpoints"), name_prefix="ppo")
        t0 = time.time()
        try:
            model.learn(total_timesteps=args.total_steps, callback=CallbackList([cur, ckpt]),
                        reset_num_timesteps=not args.resume, progress_bar=False)
        finally:
            model.save(out / "final")
            mlflow.log_metric("wall_minutes", (time.time() - t0) / 60)
            mlflow.log_artifact(str(out / "final.zip"))
            venv.close()
        print(f"saved {out / 'final.zip'}  (final stage {cur.stage})")


def evaluate(args):
    from stable_baselines3 import PPO

    from microduck_football.sim import duck_sim as ds
    from microduck_football.envs import football_env as E
    from microduck_football.envs.football_env import FootballEnv

    model = PPO.load(args.model, device="cpu")
    obstacles = model.observation_space.shape[0] > E.OBS_DIM
    env = FootballEnv(seed=args.seed, stage=args.stage, obstacles=obstacles)
    E.REPLAY_EARLIER = 0.0
    res = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=args.seed + ep)
        done = False
        acts = collections.Counter()
        while not done:
            a, _ = model.predict(obs, deterministic=True)
            acts[ds.PRIMITIVE_NAMES[int(a)]] += 1
            obs, r, te, tr, info = env.step(a)
            done = te or tr
        res.append(info)
        print(f"ep {ep:3d}: {'GOAL   ' if info['is_success'] else 'timeout'} t={info['sim_time']:5.1f}s "
              f"kicks={info['n_kicks']} falls={info['n_falls']}  top actions {acts.most_common(3)}")
    g = [r for r in res if r["is_success"]]
    print(f"\nstage {args.stage}: goals {len(g)}/{len(res)}"
          + (f", mean time-to-goal {np.mean([r['sim_time'] for r in g]):.1f}s" if g else "")
          + f", falls/episode {np.mean([r['n_falls'] for r in res]):.2f}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(required=True)

    t = sub.add_parser("train")
    t.add_argument("--run-name", required=True)
    t.add_argument("--total-steps", type=int, default=5_000_000, help="env decisions (all workers)")
    t.add_argument("--workers", type=int, default=7, help="parallel envs (~260 MB RAM each)")
    t.add_argument("--start-stage", type=int, default=0)
    t.add_argument("--advance-at", type=float, default=0.7, help="goal rate to advance a stage")
    t.add_argument("--window", type=int, default=100, help="episodes in the goal-rate window")
    t.add_argument("--n-steps", type=int, default=512, help="rollout length per env")
    t.add_argument("--batch-size", type=int, default=512)
    t.add_argument("--n-epochs", type=int, default=10)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--gamma", type=float, default=0.995, help="5 Hz decisions: ~40 s horizon")
    t.add_argument("--ent-coef", type=float, default=0.01)
    t.add_argument("--save-every", type=int, default=250_000, help="checkpoint period (decisions)")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--resume", type=str, default=None, help="continue from a saved .zip")
    t.add_argument("--lr-override", type=float, default=None, help="learning rate when resuming")
    t.add_argument("--obstacles", action="store_true", help="obstacle features in the obs + stages 4-6")
    t.add_argument("--warm-start", type=str, default=None,
                   help="initialise from a policy with a smaller obs (e.g. the open-play teacher)")
    t.set_defaults(func=train)

    e = sub.add_parser("eval")
    e.add_argument("model")
    e.add_argument("--stage", type=int, default=3)
    e.add_argument("--episodes", type=int, default=20)
    e.add_argument("--seed", type=int, default=12345, help="use seeds not seen in training")
    e.set_defaults(func=evaluate)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
