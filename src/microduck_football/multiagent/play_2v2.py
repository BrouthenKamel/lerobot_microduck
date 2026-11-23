#!/usr/bin/env python3
"""Render 2v2 matches to an mp4: blue = --blue, red = --red ('teacher' or a train_2v2 .pt).

  MUJOCO_GL=egl uv run python -m microduck_football.multiagent.play_2v2 --blue runs/mappo_2v2_v1/final.pt --red teacher \
      --games 3 --video assets/media/2v2_vs_teacher.mp4
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np  # noqa: E402

W, H, FPS = 960, 600, 25


def font(size):
    from PIL import ImageFont
    p = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    return ImageFont.truetype(p, size) if os.path.exists(p) else ImageFont.load_default()


def player(spec):
    from microduck_football.multiagent import train_2v2 as T
    return T.TeacherOpponent() if spec == "teacher" else T.NetOpponent(T.load_actor(spec), deterministic=True)


def label(spec):
    return "teacher (1v1 dribbler)" if spec == "teacher" else Path(spec).parent.name + "/" + Path(spec).stem


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--blue", default="teacher")
    ap.add_argument("--red", default="teacher")
    ap.add_argument("--games", type=int, default=3)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--video", type=Path, required=True)
    ap.add_argument("--speed", type=int, default=1, help="frame skip (2 = 2x)")
    args = ap.parse_args()

    import imageio.v2 as iio
    from PIL import Image, ImageDraw
    from microduck_football.envs import env_2v2 as E
    from microduck_football.sim import multi_sim as ms

    blue, red = player(args.blue), player(args.red)
    env = E.Env2v2(seed=args.seed)
    f_big, f_mid, f_small = font(24), font(17), font(14)
    writer = iio.get_writer(args.video, fps=FPS, codec="libx264", quality=8, macro_block_size=16)
    score = [0, 0]
    state = {"n": 0, "game": 0, "acts": [0] * 4}

    def frame(e):
        state["n"] += 1
        if state["n"] % (2 * args.speed):
            return
        img = Image.new("RGB", (W, H), (18, 18, 24))
        img.paste(Image.fromarray(e.sim.render_overview(520, 960, distance=3.05, elevation=-64)), (0, 40))
        d = ImageDraw.Draw(img)
        d.text((12, 8), f"BLUE  {score[0]} - {score[1]}  RED", font=f_big, fill=(255, 255, 255))
        d.text((W - 12, 12), f"game {state['game'] + 1}/{args.games}   t = {e.sim.t:4.1f} s",
               font=f_mid, fill=(200, 200, 210), anchor="ra")
        for i, duck in enumerate(e.sim.ducks):
            a = state["acts"][i]
            if duck.macro is not None:
                txt = "aimed kick: lining up"
            elif duck.policy.behavior_mode is not None:
                txt = duck.policy.behavior_mode
            else:
                txt = ms.ACTION_NAMES[a]
            col = ms.TEAM_RGBA[duck.team]
            x = 12 + i * (W // 4)
            d.text((x, H - 30), f"{duck.prefix[:2].upper()}: {txt}", font=f_small,
                   fill=tuple(int(255 * c) for c in col[:3]))
        d.text((W // 2, 12), f"{label(args.blue)}  vs  {label(args.red)}", font=f_small,
               fill=(150, 150, 165), anchor="ma")
        writer.append_data(np.asarray(img))

    env.on_control_step = frame
    for g in range(args.games):
        state["game"] = g
        obs, _ = env.reset(seed=args.seed + g)
        done = False
        while not done:
            acts = np.zeros(4, int)
            acts[[0, 1]] = blue.act(obs[[0, 1]])
            acts[[2, 3]] = red.act(obs[[2, 3]])
            state["acts"] = list(acts)
            obs, st, r, te, tr, info = env.step(list(acts))
            done = te or tr
        sc = info["scored"]
        if sc is not None:
            score[sc] += 1
        print(f"game {g + 1}: {'BLUE scores' if sc == 0 else 'RED scores' if sc == 1 else 'no goal'} "
              f"at {info['sim_time']:.1f}s, falls {info['n_falls']}, aimed kicks {info['n_aimed']}", flush=True)
        for _ in range(FPS):           # hold 1 s with the updated score
            state["n"] = 0
            frame(env)
    writer.close()
    print(f"video: {args.video}  final score blue {score[0]} - {score[1]} red")


if __name__ == "__main__":
    main()
