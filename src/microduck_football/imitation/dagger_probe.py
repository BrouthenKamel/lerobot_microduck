#!/usr/bin/env python3
"""DAgger probe (state-only students): does on-policy relabelling fix the distribution shift that
plain behaviour cloning suffers from?

Round 0 = behaviour cloning on teacher rollouts (bc_probe.collect). Each DAgger round: roll out the
current student (round 1 mixes in teacher actions with prob BETA), label EVERY visited state with
the teacher, aggregate, retrain from scratch, evaluate closed-loop.

Label on a student-visited state: the teacher's primitive -> twist target, smoothed from the
command the robot actually executed last (prev + ALPHA * (target - prev)), i.e. consistent with
the student's own state (which contains that previous command).

  uv run python -m microduck_football.imitation.dagger_probe --rounds 4 --variants a c
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

from microduck_football.imitation import bc_probe as bp  # noqa: E402


def _net(model):
    import torch.nn as nn
    net = nn.Sequential(nn.Linear(model["in_dim"], 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(),
                        nn.Linear(256, model["out_dim"]))
    net.load_state_dict(model["net"])
    return net


def dagger_collect(args):
    wid, n_eps, variant, model, beta, rnd = args
    import torch
    from stable_baselines3 import PPO
    from microduck_football.sim import duck_sim as ds
    from microduck_football.imitation import student_io as sio
    torch.set_num_threads(1)
    rng = np.random.default_rng(10_000 * rnd + wid)
    teacher = sio.SmoothedTeacher(PPO.load(bp.TEACHER, device="cpu"))
    net = _net(model)
    env = bp.make_env()
    lo, hi = env.action_space.low, env.action_space.high
    X, Y = [], []
    for ep in range(n_eps):
        env.set_stage(3 if rng.random() < 0.7 else int(rng.integers(1, 3)))
        obs, _ = env.reset(seed=500000 + rnd * 20000 + wid * 1000 + ep)
        bp.mem_reset(env, seed=500000 + rnd * 20000 + wid * 1000 + ep)
        teacher.reset()
        prev = np.zeros(3, np.float32)
        done = False
        while not done:
            prim = int(teacher.model.predict(obs, deterministic=True)[0])
            teacher.last_primitive = prim
            name = ds.PRIMITIVE_NAMES[prim]
            target = np.array(sio.TWIST.get(name, (0.0, 0.0, 0.0)), np.float32)
            label = np.array([*(prev + sio.ALPHA * (target - prev)), float(name.startswith("kick"))], np.float32)
            x = bp.feats(env, prev, variant)
            X.append(x)
            Y.append(label)
            if rng.random() < beta:
                act = label
            else:
                with torch.no_grad():
                    a = net(torch.tensor((x - model["mu"]) / model["sd"], dtype=torch.float32)).numpy()
                act = np.clip(a * model["ysd"] + model["ymu"], lo, hi).astype(np.float32)
            obs, r, te, tr, info = env.step(act)
            obs = teacher.fix_obs(obs)
            prev = act[:3].copy()
            done = te or tr
    return np.array(X), np.array(Y)


def eval_model(pool, variant, model, n3, n1):
    jobs = [(variant, model, 3, [int(s) for s in c]) for c in np.array_split(range(777000, 777000 + n3), 4)]
    jobs += [(variant, model, 1, [int(s) for s in c]) for c in np.array_split(range(777000, 777000 + n1), 2)]
    agg = {3: [], 1: []}
    for v, st, out in pool.map(bp.evaluate, jobs):
        agg[st].extend(out)
    def fmt(o):
        g = [t for ok, t, _ in o if ok]
        return f"{len(g)}/{len(o)}" + (f" ({np.mean(g):4.1f}s)" if g else "       ") + f" falls {np.mean([f for *_, f in o]):.2f}"
    return fmt(agg[3]), fmt(agg[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=["a", "c"])
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--bc-episodes", type=int, default=200)
    ap.add_argument("--dagger-episodes", type=int, default=100)
    ap.add_argument("--beta1", type=float, default=0.5, help="teacher-action share in round 1 (then 0)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--eval3", type=int, default=20)
    ap.add_argument("--eval1", type=int, default=10)
    args = ap.parse_args()
    ctx = mp.get_context("fork")
    t0 = time.time()

    with ctx.Pool(args.workers) as pool:
        per = [args.bc_episodes // args.workers + (i < args.bc_episodes % args.workers) for i in range(args.workers)]
        parts = pool.map(bp.collect, list(enumerate(per)))
        data = {v: (np.concatenate([p[0][v] for p in parts]), np.concatenate([p[1] for p in parts]))
                for v in args.variants}
        print(f"round 0 data: {len(data[args.variants[0]][1])} teacher decisions ({time.time() - t0:.0f}s)", flush=True)
        print(f"{'variant':8s} {'round':5s} {'decisions':>9s} | {'full pitch':26s} | {'stage 1':26s}", flush=True)
        for rnd in range(args.rounds + 1):
            for v in args.variants:
                X, Y = data[v]
                model, _ = bp.train(X, Y, seed=rnd)
                s3, s1 = eval_model(pool, v, model, args.eval3, args.eval1)
                print(f"{v:8s} {rnd:5d} {len(Y):9d} | {s3:26s} | {s1:26s}  ({time.time() - t0:.0f}s)", flush=True)
                if rnd == args.rounds:
                    continue
                beta = args.beta1 if rnd == 0 else 0.0
                per = [args.dagger_episodes // args.workers + (i < args.dagger_episodes % args.workers)
                       for i in range(args.workers)]
                new = pool.map(dagger_collect, [(i, n, v, model, beta, rnd + 1) for i, n in enumerate(per)])
                data[v] = (np.concatenate([X] + [p[0] for p in new]), np.concatenate([Y] + [p[1] for p in new]))


if __name__ == "__main__":
    main()
