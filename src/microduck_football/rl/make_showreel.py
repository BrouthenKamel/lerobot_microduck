#!/usr/bin/env python3
"""Showreel: one PPO checkpoint plays every curriculum stage (0-10), with a title card per stage.

For each stage: evaluate --episodes unseen seeds (no rendering), then render ONE scoring episode
(the one with the median time-to-goal, i.e. representative rather than the best). The title card
shows the measured goal rate, so the reel stays honest.

  MUJOCO_GL=egl uv run python -m microduck_football.rl.make_showreel \
      --model models/ppo_teacher_obstacles.zip --out assets/media/showreel.mp4
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
from microduck_football import paths  # noqa: E402
import numpy as np  # noqa: E402

W, H = 960, 480            # frame: overview 640x480 | head camera 320x240 + info panel 320x240
FPS = 25
STAGES = {
    0: ("Finish", "ball 25-50 cm in front of the goal"),
    1: ("Approach", "ball 0.5-1.2 m out, duck behind it"),
    2: ("Get behind the ball", "duck off the ball-goal line"),
    3: ("Open play", "duck and ball anywhere on the pitch"),
    4: ("One defender", "static defender on the ball-goal line"),
    5: ("Defensive wall", "3 defenders across the line, one gap"),
    6: ("Moving defenders", "2 defenders patrolling across the line"),
    7: ("Goalkeeper (weak)", "slow keeper, long reaction delay"),
    8: ("Goalkeeper (medium)", "faster keeper, shorter delay"),
    9: ("Goalkeeper (full)", "full-speed keeper"),
    10: ("Keeper + defenders", "full keeper + 2 defenders"),
}


def make_env(model_path):
    from stable_baselines3 import PPO
    from microduck_football.envs import football_env as E
    E.REPLAY_EARLIER = 0.0
    model = PPO.load(model_path, device="cpu")
    env = E.FootballEnv(seed=0, stage=3, head_tracking="privileged", glance=True,
                        obstacles=model.observation_space.shape[0] > E.OBS_DIM)
    return model, env


def play(env, model, stage, seed, on_step=None):
    env.set_stage(stage)
    obs, info = env.reset(seed=seed)
    done = False
    while not done:
        a = int(model.predict(obs, deterministic=True)[0])
        if on_step:
            on_step(a)
        obs, r, te, tr, info = env.step(a)
        done = te or tr
    return bool(info["is_success"]), float(info["sim_time"]), int(info["n_falls"])


def evaluate(job):
    import torch
    torch.set_num_threads(1)
    model_path, stage, seeds = job
    model, env = make_env(model_path)
    return stage, [(s, *play(env, model, stage, s)) for s in seeds]


def font(size):
    from PIL import ImageFont
    for p in ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
              "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def render(job):
    import torch
    from PIL import Image, ImageDraw
    from microduck_football.sim import duck_sim as ds
    torch.set_num_threads(1)
    model_path, stage, seed, rate, out = job
    model, env = make_env(model_path)
    f_big, f_small = font(20), font(15)
    import imageio.v2 as iio
    writer = iio.get_writer(out, fps=FPS, codec="libx264", quality=8, macro_block_size=16)
    state = {"a": "stand", "n": 0}

    def grab(sim):
        state["n"] += 1
        if state["n"] % 2:            # 50 Hz control -> 25 fps
            return
        over = sim.render_overview(480, 640)
        if not state.get("zoomed"):          # closer camera for the big frame
            sim._overview_cam.distance, sim._overview_cam.elevation = 3.15, -60
            sim._overview_cam.lookat[:] = (0.5, 0.0, 0.0)
            state["zoomed"] = True
            over = sim.render_overview(480, 640)
        head = sim.render_head().copy()
        cy, cx = head.shape[0] // 2, head.shape[1] // 2
        head[cy - 1:cy + 2, cx - 8:cx + 9] = 255
        head[cy - 8:cy + 9, cx - 1:cx + 2] = 255
        img = Image.new("RGB", (W, H), (18, 18, 24))
        img.paste(Image.fromarray(over), (0, 0))
        img.paste(Image.fromarray(head), (640, 0))
        d = ImageDraw.Draw(img)
        d.rectangle([0, 0, 640, 30], fill=(0, 0, 0))
        d.text((10, 5), f"Stage {stage}: {STAGES[stage][0]}", font=f_big, fill=(255, 255, 255))
        d.text((648, 246), "duck head camera (ball tracking)", font=f_small, fill=(160, 160, 170))
        y = 280
        for k, v in (("sim time", f"{sim.t:5.1f} s"), ("primitive", state["a"]),
                     ("stage goal rate", rate)):
            d.text((648, y), k, font=f_small, fill=(160, 160, 170))
            d.text((790, y), v, font=f_small, fill=(255, 255, 255))
            y += 26
        d.text((648, H - 30), "PPO over frozen ONNX primitives", font=f_small, fill=(120, 120, 130))
        writer.append_data(np.asarray(img))

    env.on_control_step = grab
    ok, t, falls = play(env, model, stage, seed,
                        on_step=lambda a: state.__setitem__("a", ds.PRIMITIVE_NAMES[a]))
    writer.close()
    return stage, ok, t, falls


def title_card(stage, rate, mean_t, speed, n_frames):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (W, H), (18, 18, 24))
    d = ImageDraw.Draw(img)
    name, desc = STAGES[stage]
    d.text((W // 2, 150), f"Stage {stage}", font=font(28), fill=(150, 150, 165), anchor="mm")
    d.text((W // 2, 205), name, font=font(48), fill=(255, 255, 255), anchor="mm")
    d.text((W // 2, 265), desc, font=font(24), fill=(200, 200, 210), anchor="mm")
    d.text((W // 2, 340), f"goals {rate}" + (f"   mean time-to-goal {mean_t:.0f} s" if mean_t else ""),
           font=font(22), fill=(120, 220, 140), anchor="mm")
    if speed > 1:
        d.text((W // 2, 385), f"played at {speed}x speed", font=font(18), fill=(150, 150, 165), anchor="mm")
    return [np.asarray(img)] * n_frames


def text_card(lines, n_frames):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (W, H), (18, 18, 24))
    d = ImageDraw.Draw(img)
    y = H // 2 - 30 * len(lines) // 2
    for i, (txt, size, col) in enumerate(lines):
        d.text((W // 2, y), txt, font=font(size), fill=col, anchor="mm")
        y += size + 22
    return [np.asarray(img)] * n_frames


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path, default=paths.TEACHER_OBSTACLES)
    ap.add_argument("--out", type=Path, default=paths.MEDIA_DIR / "showreel_all_stages.mp4")
    ap.add_argument("--stages", type=int, nargs="+", default=list(range(11)))
    ap.add_argument("--episodes", type=int, default=20, help="eval episodes per stage")
    ap.add_argument("--seed", type=int, default=424242)
    ap.add_argument("--workers", type=int, default=11)
    ap.add_argument("--fast-above", type=float, default=20.0, help="play episodes longer than this at 2x")
    ap.add_argument("--from-log", type=Path, default=None,
                    help="reuse the eval lines ('stage k: goals a/b, mean t -> seed s') of an earlier run")
    args = ap.parse_args()
    tmp = paths.RUNS_DIR / f"{args.out.stem}_parts"
    tmp.mkdir(parents=True, exist_ok=True)
    ctx = mp.get_context("spawn")
    t0 = time.time()
    seeds = list(range(args.seed, args.seed + args.episodes))
    with ctx.Pool(args.workers) as pool:
        jobs, summary = [], {}
        if args.from_log:
            import re
            for line in args.from_log.read_text().splitlines():
                m = re.match(r"stage\s+(\d+): goals (\d+/\d+)(?:, mean ([\d.]+)s)? -> seed (\d+)", line)
                if m and int(m[1]) in args.stages:
                    st = int(m[1])
                    summary[st] = (m[2], float(m[3]) if m[3] else None, int(m[4]))
                    jobs.append((str(args.model), st, int(m[4]), m[2], str(tmp / f"s{st}.mp4")))
            res = {}
        else:
            res = dict(pool.map(evaluate, [(str(args.model), st, seeds) for st in args.stages]))
            print(f"eval done ({time.time() - t0:.0f}s)", flush=True)
        for st in res:
            goals = sorted([r for r in res[st] if r[1]], key=lambda r: r[2])
            rate = f"{len(goals)}/{len(res[st])}"
            mean_t = float(np.mean([r[2] for r in goals])) if goals else None
            pick = goals[len(goals) // 2][0] if goals else res[st][0][0]
            summary[st] = (rate, mean_t, pick)
            print(f"stage {st:2d}: goals {rate}" + (f", mean {mean_t:.1f}s" if goals else "") + f" -> seed {pick}")
            jobs.append((str(args.model), st, pick, rate, str(tmp / f"s{st}.mp4")))
        jobs = [j for j in jobs if not (Path(j[4]).exists() and Path(j[4]).stat().st_size > 10_000)]
        for st, ok, t, falls in pool.map(render, jobs):
            print(f"rendered stage {st}: {'GOAL' if ok else 'no goal'} {t:.1f}s falls {falls}", flush=True)

    import imageio.v2 as iio
    w = iio.get_writer(args.out, fps=FPS, codec="libx264", quality=8, macro_block_size=16)
    for f in text_card([("Microduck Football", 46, (255, 255, 255)),
                        ("RL teacher (PPO) over frozen walking / kicking policies", 22, (200, 200, 210)),
                        ("reverse curriculum: advance when the goal rate passes a threshold", 20, (150, 150, 165)),
                        (f"{len(args.stages)} stages, one final checkpoint", 20, (120, 220, 140))], FPS * 4):
        w.append_data(f)
    for st in args.stages:
        rate, mean_t, _ = summary[st]
        rd = iio.get_reader(tmp / f"s{st}.mp4")
        speed = 2 if rd.count_frames() / FPS > args.fast_above else 1
        for f in title_card(st, rate, mean_t, speed, int(FPS * 2.5)):
            w.append_data(f)
        last = None
        for i, f in enumerate(rd):
            if i % speed == 0:
                w.append_data(f)
            last = f
        rd.close()
        for _ in range(FPS):              # hold the last frame 1 s
            w.append_data(last)
    w.close()
    for p in tmp.glob("*.mp4"):
        p.unlink()
    tmp.rmdir()
    print(f"\n{args.out} written ({time.time() - t0:.0f}s total)")
    for st in args.stages:
        print(f"  stage {st:2d} {STAGES[st][0]:22s} goals {summary[st][0]}")


if __name__ == "__main__":
    main()
