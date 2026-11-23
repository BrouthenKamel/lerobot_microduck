#!/usr/bin/env python3
"""Regenerate the README figures (docs/figures/*.png).

  uv run python -m microduck_football.tools.make_figures

Inputs: docs/results.json (every reported number + its provenance), docs/kick_rolling.json
(rl/kick_lab.py rolling), and the training curves in runs/mlflow.db (local; the figure that needs it is
skipped when it is absent).

Each figure answers one plain question (its title) and uses plain units: "goals out of 20", "hours of
simulated football" (one decision = 0.2 s of play).
"""

from __future__ import annotations

import json
import os

os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from microduck_football import paths  # noqa: E402

DOCS = paths.REPO_ROOT / "docs"
OUT = DOCS / "figures"
SECONDS_PER_DECISION = 0.2            # 5 Hz decisions

# validated categorical slots 1-3 (dataviz reference palette, all-pairs CVD-safe) + text/surface tokens
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED, LIGHT, GRID, SURFACE = "#0b0b0b", "#52514e", "#9a9993", "#d9d8d2", "#e6e5e0", "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10.5,
    "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False, "lines.linewidth": 2.0, "figure.dpi": 150,
})

STAGE_NAMES = {
    0: "Ball just in front of the goal", 1: "Ball 0.5-1.2 m out, duck behind it", 2: "Duck must get behind the ball",
    3: "Anywhere on the pitch", 4: "One defender in the way", 5: "Wall of 3 defenders, one gap",
    6: "2 defenders patrolling", 7: "Slow goalkeeper", 8: "Medium goalkeeper", 9: "Full-speed goalkeeper",
    10: "Goalkeeper + 2 defenders",
}


def _save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight")
    plt.close(fig)
    print("wrote", OUT / name)


def _subtitle(ax, text):
    ax.text(0, 1.015, text, transform=ax.transAxes, fontsize=9, color=INK2, va="bottom")


def _history(client, run_name, exp, key):
    runs = client.search_runs([exp], filter_string=f"tags.mlflow.runName = '{run_name}'",
                              order_by=["attributes.start_time DESC"])
    if not runs:
        return np.array([]), np.array([])
    h = sorted(client.get_metric_history(runs[0].info.run_id, key), key=lambda m: m.step)
    return np.array([m.step for m in h]), np.array([m.value for m in h])


def fig_teacher_learning(client):
    """Which stage the RL player has unlocked, against hours of simulated football."""
    if client is None:
        print("skip teacher_learning (no runs/mlflow.db)")
        return
    exp = client.get_experiment_by_name("microduck_football_teacher").experiment_id
    hours = lambda s: s * SECONDS_PER_DECISION / 3600  # noqa: E731
    s0, v0 = _history(client, "ppo_main_5m_t60", exp, "curriculum_stage")
    s1, v1 = _history(client, "ppo_opponents_v1", exp, "curriculum_stage")
    s2, v2 = _history(client, "ppo_opponents_v2_keeper_ladder", exp, "curriculum_stage")
    keep = s1 < (s2.min() if len(s2) else np.inf)
    s12, v12 = np.concatenate([s1[keep], s2]), np.concatenate([v1[keep], v2])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.25]}, sharey=True)
    for ax, s, v, c, title, sub in (
            (axes[0], s0, v0, BLUE, "Open play", "picked up from an earlier run at stage 2"),
            (axes[1], s12, v12, ORANGE, "With opponents", "starts from the open-play player")):
        ax.step(hours(s), v, where="post", color=c)
        ax.set_title(title, pad=16)
        _subtitle(ax, sub)
        ax.set_xlabel("hours of simulated football")
        ax.set_xlim(left=0)
    axes[0].set_yticks(range(11), [f"{k}  {STAGE_NAMES[k]}" for k in range(11)], fontsize=8.5)
    axes[0].set_ylim(-0.5, 10.5)
    fig.suptitle("The RL player unlocks the next stage once it scores often enough on the current one",
                 x=0.01, ha="left", fontsize=12.5, fontweight="bold", y=1.04)
    _save(fig, "teacher_learning.png")


def fig_teacher_stages(results):
    st = results["teacher_obstacles_per_stage"]["stages"]
    k = sorted(int(i) for i in st)
    goals = [st[str(i)]["goals"] for i in k]
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    y = np.arange(len(k))[::-1]
    cols = [BLUE if i < 4 else ORANGE for i in k]
    ax.barh(y, goals, color=cols, height=0.62, edgecolor=SURFACE, linewidth=2)
    for yy, g in zip(y, goals):
        ax.text(g + 0.3, yy, f"{g}/20", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, [STAGE_NAMES[i] for i in k], fontsize=9)
    ax.set_xlim(0, 22)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("goals out of 20 new random games")
    ax.grid(axis="y", visible=False)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=BLUE), plt.Rectangle((0, 0), 1, 1, color=ORANGE)],
              labels=["open play", "with opponents"], loc="lower right", bbox_to_anchor=(1.0, 0.0))
    ax.set_title("How often the final RL player scores, stage by stage", pad=16)
    _subtitle(ax, "same trained model on every stage; opponents get harder from top to bottom")
    _save(fig, "teacher_stages.png")


def fig_copying_vs_dagger(results):
    dp = results["dagger_probe"]["full_pitch_goals_per_round"]
    students = [("a: student state (25)", "sees only its own body\n(and where its head points)"),
                ("e: + goal memory (drift) + prev head", "+ remembers where it last\nsaw the goal (drifts)"),
                ("c: + goal bearing + ball velocity", "+ always knows exactly\nwhere the goal is")]
    fig, ax = plt.subplots(figsize=(8.6, 4.0))
    x = np.arange(len(students))
    before = [dp[s][0] for s, _ in students]
    after = [dp[s][-1] for s, _ in students]
    ax.bar(x - 0.19, before, width=0.36, color=LIGHT, edgecolor=SURFACE, linewidth=2)
    ax.bar(x + 0.19, after, width=0.36, color=BLUE, edgecolor=SURFACE, linewidth=2)
    for xx, b, a in zip(x, before, after):
        ax.text(xx - 0.19, b + 0.3, f"{b}/20", ha="center", va="bottom", fontsize=9, color=INK2)
        ax.text(xx + 0.19, a + 0.3, f"{a}/20", ha="center", va="bottom", fontsize=9, color=INK2)
    ax.set_xticks(x, [d for _, d in students], fontsize=9)
    ax.set_ylim(0, 20)
    ax.set_yticks([0, 5, 10, 15, 20])
    ax.set_ylabel("goals out of 20 games")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=LIGHT), plt.Rectangle((0, 0), 1, 1, color=BLUE)],
              labels=["copying the teacher only", "+ DAgger (4 rounds of corrections)"], loc="upper left")
    ax.set_title("Copying alone fails; DAgger works once the student can know where the goal is", pad=16)
    _subtitle(ax, "small state-only students (no camera), full pitch - a cheap test before training the camera model")
    _save(fig, "copying_vs_dagger.png")


def fig_act_progress(results):
    steps = [
        (results["act_first_attempt"], "1  Copy the teacher's button presses", LIGHT),
        (results["act_progression"][0], "2  Copy its smoothed steering instead", LIGHT),
        (results["act_progression"][1], "3  + remember where the goal was", LIGHT),
        (results["act_progression"][2], "4  + DAgger round 1", BLUE),
        (results["act_progression"][3], "5  + DAgger round 2", BLUE),
        (results["act_reverify_current_sim"], "    same model, re-measured on the final simulator", None),
    ]
    fig, ax = plt.subplots(figsize=(8.6, 3.9))
    y = np.arange(len(steps))[::-1]
    for yy, (p, label, c) in zip(y, steps):
        v = 20 * p["goals"] / p["n"]
        if c is None:
            ax.barh(yy, v, height=0.6, color=SURFACE, edgecolor=BLUE, hatch="////", linewidth=1.2)
        else:
            ax.barh(yy, v, height=0.6, color=c, edgecolor=SURFACE, linewidth=2)
        ax.text(v + 0.3, yy, f"{p['goals']}/{p['n']}", va="center", fontsize=9, color=INK2)
    ax.set_yticks(y, [s[1] for s in steps], fontsize=9)
    ax.set_xlim(0, 22)
    ax.set_xticks([0, 5, 10, 15, 20])
    ax.set_xlabel("goals per 20 new games (full pitch, open play)")
    ax.grid(axis="y", visible=False)
    ax.set_title("The camera player, attempt by attempt", pad=16)
    _subtitle(ax, "LeRobot ACT: sees only its head camera + its own body sensors; trained on open play (no opponents)")
    _save(fig, "act_progress.png")


def fig_kick():
    f = DOCS / "kick_rolling.json"
    if not f.exists():
        print("skip kick figure (run rl/kick_lab.py rolling --json docs/kick_rolling.json)")
        return
    d = json.loads(f.read_text())
    keys = sorted(d, key=float, reverse=True)
    names = {0.003: "grass-like\n(1v1 pitch)", 0.0015: "indoor floor\n(2v2 pitch)", 0.001: "smoother", 0.0005: "smoothest"}
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    for i, k in enumerate(keys):
        y = np.array(d[k])
        jitter = (np.random.default_rng(i).random(len(y)) - 0.5) * 0.28
        c = ORANGE if float(k) == 0.003 else (BLUE if float(k) == 0.0015 else MUTED)
        ax.scatter(i + jitter, y, s=22, color=c, edgecolor=SURFACE, linewidth=1, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [np.median(y)] * 2, color=INK, linewidth=2, zorder=4)
        ax.text(i + 0.3, np.median(y), f"{np.median(y):.1f} m", va="center", fontsize=9, color=INK2)
    ax.axhline(1.8, color=MUTED, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(len(keys) - 0.5, 1.83, "the wall (end of the test area)", ha="right", fontsize=8, color=INK2)
    ax.set_xticks(range(len(keys)), [names.get(float(k), k) for k in keys], fontsize=9)
    ax.set_ylim(0, 2.0)
    ax.set_xlabel("floor")
    ax.set_ylabel("distance the ball travels (m)")
    ax.set_title("How far one kick sends the ball", pad=16)
    _subtitle(ax, "24 well-placed kicks per floor; black line = typical kick")
    _save(fig, "kick_vs_floor.png")


def fig_2v2(results):
    hours = lambda s: s * SECONDS_PER_DECISION / 3600  # noqa: E731
    fig, ax = plt.subplots(figsize=(8.2, 3.9))
    ax.axhspan(0.5, 0.8, color=BLUE, alpha=0.06, linewidth=0)
    ax.axhline(0.5, color=MUTED, linewidth=1.2, linestyle=(0, (4, 3)))
    ax.text(0.01, 0.765, "better than the original player", transform=ax.get_yaxis_transform(), fontsize=8.5,
            color=INK2)
    ax.text(0.01, 0.235, "worse", transform=ax.get_yaxis_transform(), fontsize=8.5, color=INK2)
    v3 = results["mappo_v3_fair"]["points"]
    xs, ys = [hours(p["samples"]) for p in v3], [p["score"] for p in v3]
    ax.plot(xs, ys, color=BLUE, marker="o", markersize=5, label="final attempt (v3)")
    v2 = results["mappo_v2_fair"]["points"]
    ax.plot([hours(p["samples"]) for p in v2], [p["score"] for p in v2], color=ORANGE, marker="o", markersize=6,
            linestyle="none", label="earlier attempt (v2) - drifted to passive draws")
    b = int(np.argmax(ys))
    ax.annotate(f"best 20-game check: {v3[b]['w']}W {v3[b]['d']}D {v3[b]['l']}L", (xs[b], ys[b]),
                xytext=(-8, 10), textcoords="offset points", ha="right", fontsize=8.5, color=INK2)
    rt = results.get("mappo_best_retest")
    if rt:
        ax.plot([xs[b]], [rt["score"]], marker="D", markersize=8, color=INK, linestyle="none",
                label=f"same checkpoint re-tested on {rt['n']} new games")
        ax.annotate(f"{rt['score']:.2f} ({rt['w']}W {rt['d']}D {rt['l']}L)", (xs[b], rt["score"]),
                    xytext=(10, -4), textcoords="offset points", fontsize=8.5, color=INK)
    ax.set_ylim(0.2, 0.8)
    ax.set_xlim(left=0)
    ax.set_xlabel("hours of football played while training (all learning ducks)")
    ax.set_ylabel("match score (win 1, draw 0.5)")
    ax.legend(loc="lower right", fontsize=8.5)
    ax.set_title("2v2: does the new team beat a team of the original 1v1 players? Not yet", pad=16)
    _subtitle(ax, "each point = 20 matches (v2: 30), no randomness; picking the best point favours luck - see the re-test")
    _save(fig, "mappo_2v2.png")


def main():
    results = json.loads((DOCS / "results.json").read_text())
    client = None
    if (paths.RUNS_DIR / "mlflow.db").exists():
        from mlflow import MlflowClient
        client = MlflowClient(paths.TRACKING_URI)
    fig_teacher_learning(client)
    fig_teacher_stages(results)
    fig_copying_vs_dagger(results)
    fig_act_progress(results)
    fig_kick()
    fig_2v2(results)


if __name__ == "__main__":
    main()
