#!/usr/bin/env python3
"""2v2 self-play with MAPPO (centralised critic, decentralised shared actor), plain PyTorch.

  # train (10 env workers), warm start from the open-play teacher, log to MLflow
  uv run python -m microduck_football.multiagent.train_2v2 train --run-name mappo_v1 --hours 2 --workers 10

  # evaluate a checkpoint against the teacher (or another checkpoint)
  uv run python -m microduck_football.multiagent.train_2v2 eval runs/mappo_v1/final.pt --games 40

Design
  * Actor: MLP 50 -> 256 -> 256 -> 10 on each duck's own team-frame obs (env_2v2.py), shared by
    all four ducks. Warm start: the open-play teacher's 29-input net, extra inputs zero-weighted,
    the new aimed_kick logit at ~3% probability (teacher argmax kept on 100% of states).
  * Critic: MLP 86 -> 256 -> 256 -> 1 on obs + global state (only used in training).
  * Decisions made while a duck is busy (kick running / aimed kick lining up) are ignored by the
    env, so they are masked out of the policy loss (still used for the value targets).
  * Self-play: each episode the learner plays a random side; the other side is (p = SELF_P) the
    current policy itself (both sides' data used) or an opponent from the pool (learner side
    only). Pool starts with the teacher. Promotion (like the single-duck curriculum): when the
    win rate against the pool over the last --window pool games reaches --promote-at, a frozen
    snapshot joins the pool (stage += 1). Draw = half a win.
"""

from __future__ import annotations

import argparse
import collections
import copy
import json
import multiprocessing as mp
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
import numpy as np  # noqa: E402

from microduck_football import paths  # noqa: E402

RUNS = paths.RUNS_DIR
TEACHER = paths.TEACHER_OPEN_PLAY
TRACKING_URI = paths.TRACKING_URI
EXPERIMENT = "microduck_football_2v2"


# ── networks ────────────────────────────────────────────────────────────────────
def make_nets():
    import torch.nn as nn
    from microduck_football.envs import env_2v2 as E

    actor = nn.Sequential(nn.Linear(E.OBS_DIM, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(),
                          nn.Linear(256, E.N_ACTIONS))
    critic = nn.Sequential(nn.Linear(E.STATE_DIM, 256), nn.Tanh(), nn.Linear(256, 256), nn.Tanh(),
                           nn.Linear(256, 1))
    return actor, critic


def warm_start_actor(actor, teacher_path=TEACHER, aimed_bias_gap=-2.0):
    """Copy the SB3 teacher (MlpPolicy pi=[256,256], tanh) into the actor: 29 obs columns copied,
    the 21 new ones zero; 9 action rows copied, aimed_kick row zero weights and a bias
    aimed_bias_gap below the mean teacher bias."""
    import torch
    from stable_baselines3 import PPO
    sd = PPO.load(teacher_path, device="cpu").policy.state_dict()
    w0, b0 = sd["mlp_extractor.policy_net.0.weight"], sd["mlp_extractor.policy_net.0.bias"]
    w1, b1 = sd["mlp_extractor.policy_net.2.weight"], sd["mlp_extractor.policy_net.2.bias"]
    wa, ba = sd["action_net.weight"], sd["action_net.bias"]
    with torch.no_grad():
        actor[0].weight.zero_()
        actor[0].weight[:, :w0.shape[1]] = w0
        actor[0].bias.copy_(b0)
        actor[2].weight.copy_(w1)
        actor[2].bias.copy_(b1)
        actor[4].weight.zero_()
        actor[4].weight[:wa.shape[0]] = wa
        actor[4].bias[:ba.shape[0]] = ba
        actor[4].bias[ba.shape[0]:] = ba.mean() - aimed_bias_gap
    assert isinstance(actor[1], torch.nn.Tanh)


def kickable(obs):
    """aimed_kick is only offered when the ball is within its reach (multi_sim.AimedKick gives up
    otherwise): ball 0-0.40 m ahead and |lateral| < 0.15 m, read from the duck's own obs (ball rel
    = obs[0:2]) -> the same rule in training, opponents, evaluation and deployment. (v2: an
    always-available aimed_kick became a free 'stand still', 48 attempts per game.)"""
    import torch
    bx, by = obs[..., 0], obs[..., 1]
    return (bx > 0.0) & (bx < 0.40) & (by.abs() < 0.15)


def masked_logits(actor, obs):
    import torch
    from microduck_football.sim import multi_sim as ms
    logits = actor(obs)
    mask = kickable(obs)
    logits[..., ms.AIMED_KICK] = torch.where(mask, logits[..., ms.AIMED_KICK],
                                             torch.full_like(logits[..., ms.AIMED_KICK], -1e9))
    return logits


class TeacherOpponent:
    """The single-duck teacher as a 2v2 opponent (first 29 obs dims, deterministic)."""

    def __init__(self, path=TEACHER):
        from stable_baselines3 import PPO
        self.model = PPO.load(path, device="cpu")

    def act(self, obs):
        from microduck_football.envs import env_2v2 as E
        return self.model.predict(obs[:, :E.TEACHER_DIM], deterministic=True)[0].astype(int)


class NetOpponent:
    def __init__(self, actor, deterministic=False):
        self.actor = actor
        self.deterministic = deterministic

    def act(self, obs):
        import torch
        with torch.no_grad():
            logits = masked_logits(self.actor, torch.as_tensor(obs, dtype=torch.float32))
            if self.deterministic:
                return logits.argmax(-1).numpy()
            return torch.distributions.Categorical(logits=logits).sample().numpy()


# ── env workers ──────────────────────────────────────────────────────────────────
def worker(remote, seed):
    import torch
    torch.set_num_threads(1)
    from microduck_football.envs import env_2v2 as E
    env = E.Env2v2(seed=seed)
    ep = 0
    while True:
        cmd, data = remote.recv()
        if cmd == "reset":
            obs, st = env.reset(seed=seed * 100_000 + ep)
            ep += 1
            remote.send((obs, st, np.array([d.busy() for d in env.sim.ducks])))
        elif cmd == "step":
            obs, st, r, te, tr, info = env.step(data)
            done = te or tr
            if done:
                info["final"] = True
                obs, st = env.reset(seed=seed * 100_000 + ep)
                ep += 1
            remote.send((obs, st, r, te, tr, info, np.array([d.busy() for d in env.sim.ducks])))
        elif cmd == "close":
            break


class Workers:
    def __init__(self, n, seed):
        ctx = mp.get_context("fork")
        self.remotes, self.procs = [], []
        for i in range(n):
            a, b = ctx.Pipe()
            p = ctx.Process(target=worker, args=(b, seed + 1000 * i), daemon=True)
            p.start()
            self.remotes.append(a)
            self.procs.append(p)

    def reset(self):
        for r in self.remotes:
            r.send(("reset", None))
        return [r.recv() for r in self.remotes]

    def step(self, actions):
        for r, a in zip(self.remotes, actions):
            r.send(("step", a))
        return [r.recv() for r in self.remotes]

    def close(self):
        for r in self.remotes:
            r.send(("close", None))


# ── training ─────────────────────────────────────────────────────────────────────
def gae(rew, val, done, last_val, gamma, lam):
    T = len(rew)
    adv = np.zeros(T, np.float32)
    last = 0.0
    for t in reversed(range(T)):
        nv = last_val if t == T - 1 else val[t + 1]
        nonterm = 1.0 - done[t]
        delta = rew[t] + gamma * nv * nonterm - val[t]
        last = delta + gamma * lam * nonterm * last
        adv[t] = last
    return adv, adv + val


def train(args):
    import mlflow
    import torch
    from microduck_football.envs import env_2v2 as E

    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    out = RUNS / args.run_name
    (out / "pool").mkdir(parents=True, exist_ok=True)

    actor, critic = make_nets()
    if args.resume:
        ck = torch.load(args.resume, weights_only=False)
        actor.load_state_dict(ck["actor"])
        critic.load_state_dict(ck["critic"])
    elif not args.no_warm_start:
        warm_start_actor(actor)
    opt_a = torch.optim.Adam(actor.parameters(), lr=args.lr_actor)
    opt_c = torch.optim.Adam(critic.parameters(), lr=args.lr_critic)

    pool = [("teacher", TeacherOpponent())]
    stage = 0
    W = Workers(args.workers, args.seed)
    starts = W.reset()
    obs = [s[0] for s in starts]
    sts = [s[1] for s in starts]
    busy = [s[2] for s in starts]

    def new_episode_cfg():
        learner_team = int(rng.integers(2))
        if rng.random() < args.self_p:
            return {"team": learner_team, "opp": "self"}
        # recent snapshots more often; the teacher stays in the pool
        k = len(pool)
        p = np.arange(1, k + 1, dtype=float)
        return {"team": learner_team, "opp": int(rng.choice(k, p=p / p.sum()))}

    cfgs = [new_episode_cfg() for _ in range(args.workers)]
    pool_results = collections.deque(maxlen=args.window)
    recent = collections.deque(maxlen=200)       # (kind, learner result, len, falls, kicks, aimed)
    total_samples, it, t0 = 0, 0, time.time()
    (out / "fair").mkdir(exist_ok=True)
    fair_proc, fair_seen, best_fair = None, set(), -1.0

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({k: v for k, v in vars(args).items() if k != "func"}
                          | {"obs_dim": E.OBS_DIM, "state_dim": E.STATE_DIM, "n_actions": E.N_ACTIONS,
                             "rolling": 0.0015, "r_goal": E.R_GOAL, "w_ball": E.W_BALL,
                             "w_approach": E.W_APPROACH})
        try:
            while time.time() - t0 < args.hours * 3600:
                it += 1
                # per worker, per duck trajectories
                buf = [[{"obs": [], "st": [], "act": [], "logp": [], "val": [], "rew": [], "done": [],
                         "valid": [], "learn": []} for _ in range(4)] for _ in range(args.workers)]
                for step in range(args.rollout):
                    O = np.stack(obs)                     # (W, 4, OBS)
                    S = np.stack(sts)
                    with torch.no_grad():
                        logits = masked_logits(actor, torch.as_tensor(O.reshape(-1, E.OBS_DIM), dtype=torch.float32))
                        dist = torch.distributions.Categorical(logits=logits)
                        a = dist.sample()
                        logp = dist.log_prob(a).numpy().reshape(args.workers, 4)
                        a = a.numpy().reshape(args.workers, 4)
                        v = critic(torch.as_tensor(S.reshape(-1, E.STATE_DIM), dtype=torch.float32)) \
                            .numpy().reshape(args.workers, 4)
                    actions = a.copy()
                    learn_mask = np.zeros((args.workers, 4), bool)
                    for w, c in enumerate(cfgs):
                        for i in range(4):
                            team = 0 if i < 2 else 1
                            learn_mask[w, i] = c["opp"] == "self" or team == c["team"]
                        if c["opp"] != "self":
                            opp_idx = [i for i in range(4) if (0 if i < 2 else 1) != c["team"]]
                            actions[w, opp_idx] = pool[c["opp"]][1].act(O[w, opp_idx])
                    res = W.step([list(map(int, actions[w])) for w in range(args.workers)])
                    for w, (o2, s2, r, te, tr, info, b2) in enumerate(res):
                        done = te or tr
                        for i in range(4):
                            if not learn_mask[w, i]:
                                continue
                            B = buf[w][i]
                            B["obs"].append(O[w, i]); B["st"].append(S[w, i]); B["act"].append(a[w, i])
                            B["logp"].append(logp[w, i]); B["val"].append(v[w, i]); B["rew"].append(r[i])
                            B["done"].append(float(done)); B["valid"].append(not busy[w][i])
                            B["learn"].append(True)        # time limit = end of game (draw penalty)
                        if done:
                            c = cfgs[w]
                            sc = info["scored"]
                            res_l = 0.5 if sc is None else float(sc == c["team"])
                            kind = "self" if c["opp"] == "self" else pool[c["opp"]][0]
                            lt = [i for i in range(4) if (0 if i < 2 else 1) == c["team"]]
                            recent.append((kind, res_l, info["sim_time"], sum(info["n_falls"]) / 4,
                                           sum(info["n_kicks"][i] for i in lt) / 2,
                                           sum(info["n_aimed"][i] for i in lt) / 2))
                            if c["opp"] != "self":
                                pool_results.append(res_l)
                            cfgs[w] = new_episode_cfg()
                        obs[w], sts[w], busy[w] = o2, s2, b2
                # bootstrap values for unfinished trajectories
                S = np.stack(sts)
                with torch.no_grad():
                    last_v = critic(torch.as_tensor(S.reshape(-1, E.STATE_DIM), dtype=torch.float32)) \
                        .numpy().reshape(args.workers, 4)
                cat = {k: [] for k in ("obs", "st", "act", "logp", "adv", "ret", "valid")}
                for w in range(args.workers):
                    for i in range(4):
                        B = buf[w][i]
                        if not B["rew"]:
                            continue
                        # trajectories of a duck can mix learner / non-learner episodes; we only stored
                        # learner steps, so episode boundaries are marked by done=1.
                        adv, ret = gae(np.array(B["rew"], np.float32), np.array(B["val"], np.float32),
                                       np.array(B["done"], np.float32), last_v[w, i], args.gamma, args.lam)
                        cat["obs"] += B["obs"]; cat["st"] += B["st"]; cat["act"] += B["act"]
                        cat["logp"] += B["logp"]; cat["adv"] += list(adv); cat["ret"] += list(ret)
                        cat["valid"] += B["valid"]
                T = {k: torch.as_tensor(np.array(v), dtype=torch.float32) for k, v in cat.items()}
                T["act"] = T["act"].long()
                valid = T["valid"] > 0.5
                n = len(T["act"])
                total_samples += int(valid.sum())
                advn = T["adv"].clone()
                advn[valid] = (advn[valid] - advn[valid].mean()) / (advn[valid].std() + 1e-8)
                stats = collections.defaultdict(list)
                for _ in range(args.epochs):
                    perm = torch.randperm(n)
                    for s0 in range(0, n, args.minibatch):
                        idx = perm[s0:s0 + args.minibatch]
                        m = valid[idx]
                        vpred = critic(T["st"][idx]).squeeze(-1)
                        vloss = ((vpred - T["ret"][idx]) ** 2).mean()
                        opt_c.zero_grad(); vloss.backward()
                        torch.nn.utils.clip_grad_norm_(critic.parameters(), 0.5); opt_c.step()
                        stats["vloss"].append(float(vloss))
                        if m.sum() < 8 or it <= args.critic_warmup:
                            continue                     # critic warm-up: actor frozen
                        ii = idx[m]
                        dist = torch.distributions.Categorical(logits=masked_logits(actor, T["obs"][ii]))
                        lp = dist.log_prob(T["act"][ii])
                        ratio = torch.exp(lp - T["logp"][ii])
                        A = advn[ii]
                        ploss = -torch.min(ratio * A, ratio.clamp(1 - args.clip, 1 + args.clip) * A).mean()
                        ent = dist.entropy().mean()
                        opt_a.zero_grad(); (ploss - args.ent_coef * ent).backward()
                        torch.nn.utils.clip_grad_norm_(actor.parameters(), 0.5); opt_a.step()
                        stats["ploss"].append(float(ploss))
                        stats["entropy"].append(float(ent))
                        stats["clipfrac"].append(float(((ratio - 1).abs() > args.clip).float().mean()))
                        stats["kl"].append(float((T["logp"][ii] - lp).mean()))
                # promotion
                promoted = False
                if len(pool_results) == args.window and np.mean(pool_results) >= args.promote_at:
                    stage += 1
                    snap = copy.deepcopy(actor).eval()
                    torch.save(snap.state_dict(), out / "pool" / f"stage{stage}.pt")
                    pool.append((f"stage{stage}", NetOpponent(snap)))
                    if len(pool) > args.pool_max:
                        pool.pop(1)                      # keep the teacher, drop the oldest snapshot
                    pool_results.clear()
                    promoted = True
                # logging
                el = time.time() - t0
                rec = np.array([r[1:] for r in recent], float) if recent else np.zeros((1, 5))
                vs_pool = [r[1] for r in recent if r[0] != "self"]
                vs_teacher = [r[1] for r in recent if r[0] == "teacher"]
                metrics = {"samples": total_samples, "samples_per_s": total_samples / el, "stage": stage,
                           "pool_size": len(pool),
                           "win_vs_pool_window": float(np.mean(pool_results)) if pool_results else 0.0,
                           "win_vs_pool_recent": float(np.mean(vs_pool)) if vs_pool else 0.0,
                           "win_vs_teacher_recent": float(np.mean(vs_teacher)) if vs_teacher else 0.0,
                           "episode_seconds": float(rec[:, 1].mean()), "falls_per_duck": float(rec[:, 2].mean()),
                           "kicks_per_duck": float(rec[:, 3].mean()), "aimed_per_duck": float(rec[:, 4].mean()),
                           "draw_rate": float(np.mean([r[1] == 0.5 for r in recent])) if recent else 0.0,
                           **{k: float(np.mean(v)) for k, v in stats.items()}}
                mlflow.log_metrics(metrics, step=total_samples)
                print(f"it {it:4d} | {el / 60:5.1f} min | samples {total_samples:8d} ({metrics['samples_per_s']:.0f}/s) | "
                      f"stage {stage} | win vs pool {metrics['win_vs_pool_window']:.2f} (n={len(pool_results)}) "
                      f"teacher {metrics['win_vs_teacher_recent']:.2f} | ep {metrics['episode_seconds']:.0f}s "
                      f"draws {metrics['draw_rate']:.2f} falls {metrics['falls_per_duck']:.2f} "
                      f"kicks {metrics['kicks_per_duck']:.1f} aimed {metrics['aimed_per_duck']:.1f} | "
                      f"ent {metrics.get('entropy', 0):.2f} kl {metrics.get('kl', 0):.3f}"
                      + ("  [critic warm-up]" if it <= args.critic_warmup else "")
                      + ("  *** PROMOTED ***" if promoted else ""), flush=True)
                if it % args.save_every == 0:
                    ck = out / f"ckpt_{total_samples}.pt"
                    torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "stage": stage,
                                "samples": total_samples}, ck)
                    if args.fair_eval_games and (fair_proc is None or fair_proc.poll() is not None):
                        fair_proc = subprocess.Popen(
                            ["nice", "-n", "15", sys.executable, "-m", "microduck_football.multiagent.train_2v2", "eval", str(ck),
                             "--games", str(args.fair_eval_games), "--workers", "2",
                             "--json", str(out / "fair" / f"{ck.stem}.json")],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # collect finished fair evals: log + keep the best checkpoint
                for jf in sorted((out / "fair").glob("*.json")):
                    if jf.name in fair_seen:
                        continue
                    fair_seen.add(jf.name)
                    r = json.loads(jf.read_text())
                    mlflow.log_metric("fair_score_vs_teacher", r["score"], step=int(r["samples"]))
                    print(f"  fair eval {jf.stem}: W {r['w']} / D {r['d']} / L {r['l']} -> score {r['score']:.2f}"
                          + ("  (new best)" if r["score"] > best_fair else ""), flush=True)
                    if r["score"] > best_fair:
                        best_fair = r["score"]
                        shutil.copy(out / f"{jf.stem}.pt", out / "best.pt")
                        mlflow.log_metric("best_fair_score", best_fair, step=int(r["samples"]))
        finally:
            torch.save({"actor": actor.state_dict(), "critic": critic.state_dict(), "stage": stage,
                        "samples": total_samples}, out / "final.pt")
            mlflow.log_metric("wall_minutes", (time.time() - t0) / 60)
            W.close()
        print(f"saved {out / 'final.pt'} (stage {stage}, {total_samples} samples)")


# ── evaluation ───────────────────────────────────────────────────────────────────
def load_actor(path):
    import torch
    actor, _ = make_nets()
    ck = torch.load(path, weights_only=False)
    actor.load_state_dict(ck["actor"] if "actor" in ck else ck)
    return actor.eval()


def eval_job(job):
    a_path, b_path, seeds = job
    import torch
    torch.set_num_threads(1)
    from microduck_football.envs import env_2v2 as E
    A = TeacherOpponent() if a_path == "teacher" else NetOpponent(load_actor(a_path), deterministic=True)
    B = TeacherOpponent() if b_path == "teacher" else NetOpponent(load_actor(b_path), deterministic=True)
    env = E.Env2v2(seed=seeds[0])
    out = []
    for k, sd in enumerate(seeds):
        obs, _ = env.reset(seed=sd)
        a_team = k % 2                           # alternate sides
        done = False
        while not done:
            acts = np.zeros(4, int)
            ia = [0, 1] if a_team == 0 else [2, 3]
            ib = [2, 3] if a_team == 0 else [0, 1]
            acts[ia] = A.act(obs[ia])
            acts[ib] = B.act(obs[ib])
            obs, st, r, te, tr, info = env.step(list(acts))
            done = te or tr
        sc = info["scored"]
        out.append((None if sc is None else int(sc == a_team), info["sim_time"], sum(info["n_falls"]),
                    sum(info["n_aimed"]), sum(info["n_kicks"])))
    return out


def evaluate(args):
    seeds = list(range(args.seed, args.seed + args.games))
    chunks = [list(map(int, c)) for c in np.array_split(seeds, args.workers)]
    with mp.get_context("fork").Pool(args.workers) as pool:
        res = sum(pool.map(eval_job, [(str(args.model), args.opponent, c) for c in chunks]), [])
    w = sum(r[0] == 1 for r in res)
    l = sum(r[0] == 0 for r in res)
    d = sum(r[0] is None for r in res)
    print(f"{Path(args.model).name} vs {args.opponent}: W {w} / D {d} / L {l} over {len(res)} games "
          f"(score {(w + 0.5 * d) / len(res):.2f}), mean {np.mean([r[1] for r in res]):.1f}s, "
          f"falls/game {np.mean([r[2] for r in res]):.1f}, aimed kicks/game {np.mean([r[3] for r in res]):.1f}, "
          f"kicks/game {np.mean([r[4] for r in res]):.1f}")
    if args.json:
        import torch
        ck = torch.load(args.model, weights_only=False)
        Path(args.json).write_text(json.dumps({"w": w, "d": d, "l": l, "score": (w + 0.5 * d) / len(res),
                                               "samples": int(ck.get("samples", 0)) if isinstance(ck, dict) else 0}))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(required=True)
    t = sub.add_parser("train")
    t.add_argument("--run-name", required=True)
    t.add_argument("--hours", type=float, default=2.0)
    t.add_argument("--workers", type=int, default=10)
    t.add_argument("--rollout", type=int, default=128, help="decisions per worker per update")
    t.add_argument("--epochs", type=int, default=5)
    t.add_argument("--minibatch", type=int, default=1024)
    t.add_argument("--lr-actor", type=float, default=1e-4)
    t.add_argument("--lr-critic", type=float, default=5e-4)
    t.add_argument("--gamma", type=float, default=0.995)
    t.add_argument("--lam", type=float, default=0.95)
    t.add_argument("--clip", type=float, default=0.2)
    t.add_argument("--ent-coef", type=float, default=0.001,
                   help="0.01 (v1) raised the policy entropy 0.57 -> 0.9 and eroded the teacher")
    t.add_argument("--critic-warmup", type=int, default=40,
                   help="iterations training only the critic (actor = frozen warm start): a fresh "
                        "critic's noisy advantages otherwise erode the teacher (v1: 0.50 -> 0.29 vs teacher)")
    t.add_argument("--self-p", type=float, default=0.5, help="share of episodes against itself")
    t.add_argument("--window", type=int, default=100, help="pool games in the promotion window")
    t.add_argument("--promote-at", type=float, default=0.55)
    t.add_argument("--pool-max", type=int, default=8)
    t.add_argument("--save-every", type=int, default=50, help="iterations between checkpoints")
    t.add_argument("--fair-eval-games", type=int, default=20,
                   help="deterministic games vs the teacher per checkpoint (background, nice); 0 = off")
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--resume", type=str, default=None)
    t.add_argument("--no-warm-start", action="store_true")
    t.set_defaults(func=train)
    e = sub.add_parser("eval")
    e.add_argument("model")
    e.add_argument("--opponent", default="teacher", help="'teacher' or a .pt checkpoint")
    e.add_argument("--games", type=int, default=40)
    e.add_argument("--workers", type=int, default=10)
    e.add_argument("--seed", type=int, default=31337)
    e.add_argument("--json", type=str, default=None, help="also write the result here")
    e.set_defaults(func=evaluate)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
