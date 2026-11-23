#!/usr/bin/env python3
"""Information probe: is the camera student's failure a missing-information problem?

Behaviour cloning of the smoothed teacher (student_io.SmoothedTeacher, twist actions) from STATE
ONLY, with three feature sets, each evaluated closed-loop:
  a  the student's 25-D state (head angles point at the ball thanks to the tracking reflex)
  b  a + goal bearing (unit vector to the goal centre in the duck frame) + goal distance
  c  b + ball velocity in the duck frame
If a fails like the camera student while b/c succeed, the camera pipeline needs goal memory /
motion history; if all fail, plain behaviour cloning (distribution shift) is the bottleneck.

  uv run python -m microduck_football.imitation.bc_probe --episodes 200 --workers 6
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np  # noqa: E402

from microduck_football import paths  # noqa: E402

TEACHER = paths.TEACHER_OPEN_PLAY
VARIANTS = ("a", "b", "c", "d", "e")


def mem_reset(env, seed: int = 0):
    """Per-episode memory for variant e (call right after env.reset)."""
    from microduck_football.imitation import student_io as sio
    env._goalmem = sio.GoalMemory(seed)


def feats(env, prev_cmd, variant: str) -> np.ndarray:
    from microduck_football.sim import duck_sim as ds
    from microduck_football.imitation import student_io as sio
    from microduck_football.envs.football_env import OBS_SLICES
    sim = env.sim
    if variant == "d":       # the teacher's own (privileged) observation, minus its prev-primitive slot
        o = env._obs()
        keep = np.ones(len(o), bool)
        keep[OBS_SLICES["prev_action"]] = False
        return np.concatenate([o[keep], prev_cmd]).astype(np.float32)
    f = [sio.student_state(sim, prev_cmd)]
    if variant == "e":       # deployable: goal memory (glances + drifting odometry) + previous head angles
        if not hasattr(env, "_goalmem"):
            mem_reset(env)
        f.append(env._goalmem.features(sim))
        return np.concatenate(f).astype(np.float32)
    if variant in ("b", "c"):
        g = sim.to_duck_frame(ds.GOAL_CENTER)
        d = float(np.linalg.norm(g))
        f.append(np.array([g[0] / d, g[1] / d, d], np.float32))
    if variant == "c":
        f.append(sim.to_duck_frame(sim.ball_vel()[:2], is_vector=True).astype(np.float32))
    return np.concatenate(f).astype(np.float32)


def make_env():
    from microduck_football.envs import football_env as env_module
    from microduck_football.envs.football_env import FootballEnv
    env_module.REPLAY_EARLIER = 0.0
    return FootballEnv(seed=0, stage=3, head_tracking="privileged", glance=True, action_mode="twist")


def collect(args):
    wid, n_eps = args
    import torch
    from stable_baselines3 import PPO
    from microduck_football.sim import duck_sim as ds
    from microduck_football.imitation import student_io as sio
    torch.set_num_threads(1)
    rng = np.random.default_rng(2000 + wid)
    teacher = sio.SmoothedTeacher(PPO.load(TEACHER, device="cpu"))
    env = make_env()
    walkish = [i for i, (n, t) in enumerate(ds.PRIMITIVES) if t is not None]
    X = {v: [] for v in VARIANTS}
    Y = []
    for ep in range(n_eps):
        env.set_stage(3 if rng.random() < 0.7 else int(rng.integers(1, 3)))
        noisy = rng.random() < 0.25
        obs, _ = env.reset(seed=300000 + wid * 10000 + ep)
        mem_reset(env, seed=300000 + wid * 10000 + ep)
        teacher.reset()
        prev = np.zeros(3, np.float32)
        done = False
        while not done:
            label = teacher.act(obs)
            for v in VARIANTS:
                X[v].append(feats(env, prev, v))
            Y.append(label)
            act = label
            if noisy and rng.random() < 0.15:
                act = np.array([*sio.TWIST[ds.PRIMITIVE_NAMES[int(rng.choice(walkish))]], 0.0], np.float32)
            obs, r, te, tr, info = env.step(act)
            obs = teacher.fix_obs(obs)
            prev = act[:3].copy()
            done = te or tr
    return {v: np.array(X[v]) for v in VARIANTS}, np.array(Y)


def train(X, Y, epochs=60, seed=0):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    ymu, ysd = Y.mean(0), Y.std(0) + 1e-6
    net = nn.Sequential(nn.Linear(X.shape[1], 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                        nn.Linear(256, Y.shape[1]))
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32)
    Yt = torch.tensor((Y - ymu) / ysd, dtype=torch.float32)
    n = len(Xt)
    idx = np.random.default_rng(seed).permutation(n)
    split = int(0.9 * n)
    tr_i, va_i = idx[:split], idx[split:]
    for _ in range(epochs):
        perm = np.random.permutation(tr_i)
        for i in range(0, len(perm), 512):
            b = perm[i:i + 512]
            loss = nn.functional.l1_loss(net(Xt[b]), Yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        pred = net(Xt[va_i]).numpy() * ysd + ymu
    val_yaw_l1 = float(np.abs(pred[:, 2] - Y[va_i, 2]).mean())
    return {"net": net.state_dict(), "mu": mu, "sd": sd, "ymu": ymu, "ysd": ysd,
            "in_dim": X.shape[1], "out_dim": Y.shape[1]}, val_yaw_l1


def evaluate(args):
    variant, model, stage, seeds = args
    import torch
    import torch.nn as nn
    torch.set_num_threads(1)
    net = nn.Sequential(nn.Linear(model["in_dim"], 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                        nn.Linear(256, model["out_dim"]))
    net.load_state_dict(model["net"])
    env = make_env()
    env.set_stage(stage)
    lo, hi = env.action_space.low, env.action_space.high
    out = []
    for sd in seeds:
        env.reset(seed=sd)
        mem_reset(env, seed=sd)
        prev = np.zeros(3, np.float32)
        done = False
        while not done:
            x = (feats(env, prev, variant) - model["mu"]) / model["sd"]
            with torch.no_grad():
                a = net(torch.tensor(x, dtype=torch.float32)).numpy() * model["ysd"] + model["ymu"]
            a = np.clip(a, lo, hi).astype(np.float32)
            _, r, te, tr, info = env.step(a)
            prev = a[:3].copy()
            done = te or tr
        out.append((bool(info["is_success"]), float(info["sim_time"]), int(info["n_falls"])))
    return variant, stage, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", type=int, default=200)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval3", type=int, default=20, help="full-pitch eval episodes per variant")
    ap.add_argument("--eval1", type=int, default=10, help="stage-1 eval episodes per variant")
    args = ap.parse_args()
    ctx = mp.get_context("fork")

    t0 = time.time()
    per = [args.episodes // args.workers + (i < args.episodes % args.workers) for i in range(args.workers)]
    with ctx.Pool(args.workers) as pool:
        parts = pool.map(collect, list(enumerate(per)))
    X = {v: np.concatenate([p[0][v] for p in parts]) for v in VARIANTS}
    Y = np.concatenate([p[1] for p in parts])
    print(f"collected {len(Y)} decisions from {args.episodes} episodes in {time.time() - t0:.0f}s", flush=True)

    models = {}
    for v in VARIANTS:
        t1 = time.time()
        models[v], val = train(X[v], Y)
        print(f"variant {v}: {X[v].shape[1]} features, validation |yaw-rate error| {val:.3f} rad/s "
              f"(trained in {time.time() - t1:.0f}s)", flush=True)

    jobs = []
    for v in VARIANTS:
        s3 = list(range(777000, 777000 + args.eval3))
        for chunk in np.array_split(s3, 2):
            jobs.append((v, models[v], 3, [int(s) for s in chunk]))
        jobs.append((v, models[v], 1, list(range(777000, 777000 + args.eval1))))
    t2 = time.time()
    with ctx.Pool(args.workers) as pool:
        results = pool.map(evaluate, jobs)
    agg = {}
    for v, st, out in results:
        agg.setdefault((v, st), []).extend(out)
    print(f"\nclosed-loop eval ({time.time() - t2:.0f}s):")
    names = {"a": "student state (25)", "b": "+ goal bearing", "c": "+ goal bearing + ball velocity",
             "d": "teacher's own privileged obs", "e": "+ goal memory (drift) + prev head"}
    for v in VARIANTS:
        line = f"  {v} {names[v]:32s}"
        for st in (3, 1):
            o = agg[(v, st)]
            g = [t for ok, t, _ in o if ok]
            line += (f" | stage {st}: goals {len(g)}/{len(o)}"
                     + (f" ({np.mean(g):.1f}s)" if g else "")
                     + f", falls/ep {np.mean([f for *_, f in o]):.2f}")
        print(line)


if __name__ == "__main__":
    main()
