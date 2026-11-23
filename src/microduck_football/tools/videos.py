#!/usr/bin/env python3
"""Walkthrough videos for the README: one clip per chapter, then one video that strings them together.

  MUJOCO_GL=egl uv run python -m microduck_football.tools.videos buttons      # Ch.0  the 9 skills
  MUJOCO_GL=egl uv run python -m microduck_football.tools.videos curriculum   # Ch.1  stages 0 -> 10 (cut from the showreel)
  MUJOCO_GL=egl uv run python -m microduck_football.tools.videos eyes         # Ch.2  head reflex: overview + head camera
  MUJOCO_GL=egl uv run python -m microduck_football.tools.videos kicks        # Ch.4  same kick, grass vs indoor floor
  MUJOCO_GL=egl uv run python -m microduck_football.tools.videos aimed        # Ch.4  aimed-kick skill close-up
  uv run python -m microduck_football.tools.videos dagger       # Ch.3  copying vs DAgger (needs 2 eval_student videos)
  uv run python -m microduck_football.tools.videos match        # Ch.4  2v2 clip (needs a play_2v2 video)
  uv run python -m microduck_football.tools.videos assemble     # walkthrough.mp4 + per-chapter GIFs

Clips go to assets/media/walkthrough/; GIFs to docs/media/. Ch.3 (copying vs DAgger) and the 2v2 match come
from imitation/eval_student.py and multiagent/play_2v2.py (see `assemble --help` for the expected file names).
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np  # noqa: E402

from microduck_football import paths  # noqa: E402

W, H, FPS = 960, 540, 25
OUT = paths.MEDIA_DIR / "walkthrough"
GIF_DIR = paths.REPO_ROOT / "docs" / "media"
FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BG = (18, 18, 24)
ACCENT = (255, 170, 40)


# ── drawing helpers ────────────────────────────────────────────────────────────────
def font(size, bold=True):
    from PIL import ImageFont
    p = FONT_B if bold else FONT_R
    return ImageFont.truetype(p, size) if os.path.exists(p) else ImageFont.load_default()


def overlay(frame, title=None, sub=None, corner=None):
    """Bottom caption band (title + subtitle) and an optional top-left tag."""
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame).convert("RGBA")
    layer = Image.new("RGBA", img.size, (0, 0, 0, 0))
    d = ImageDraw.Draw(layer)
    w, h = img.size
    if title:
        d.rectangle([0, h - (78 if sub else 52), w, h], fill=(10, 10, 14, 205))
        d.text((w / 2, h - (52 if sub else 26)), title, font=font(26), fill=(255, 255, 255, 255), anchor="mm")
        if sub:
            d.text((w / 2, h - 20), sub, font=font(16, False), fill=(205, 205, 212, 255), anchor="mm")
    if corner:
        tw = d.textlength(corner, font=font(17))
        d.rounded_rectangle([12, 12, 12 + tw + 20, 44], radius=8, fill=ACCENT + (255,))
        d.text((22, 17), corner, font=font(17), fill=(20, 20, 24, 255))
    return np.asarray(Image.alpha_composite(img, layer).convert("RGB"))


def card(lines, n_frames):
    from PIL import Image, ImageDraw
    img = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(img)
    y = H / 2 - sum(s + 18 for _, s, _ in lines) / 2
    for text, size, col in lines:
        d.text((W / 2, y + size / 2), text, font=font(size, size >= 26), fill=col, anchor="mm")
        y += size + 18
    return [np.asarray(img)] * n_frames


class Writer:
    def __init__(self, path):
        import imageio.v2 as iio
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.w = iio.get_writer(path, fps=FPS, codec="libx264", quality=8, macro_block_size=8,
                                ffmpeg_params=["-pix_fmt", "yuv420p"])
        self.n = 0

    def add(self, frame):
        self.w.append_data(frame)
        self.n += 1

    def close(self):
        self.w.close()
        print(f"wrote {self.path} ({self.n / FPS:.1f} s)")


def follow_renderer(model, h=H, w=W):
    import mujoco
    model.vis.global_.offwidth = max(model.vis.global_.offwidth, w)
    model.vis.global_.offheight = max(model.vis.global_.offheight, h)
    return mujoco.Renderer(model, height=h, width=w), mujoco.MjvCamera()


def shoot(r, cam, data, lookat, distance, azimuth, elevation):
    cam.lookat[:] = lookat
    cam.distance, cam.azimuth, cam.elevation = distance, azimuth, elevation
    r.update_scene(data, camera=cam)
    return r.render().copy()


def park_ball(sim, xy=(-0.8, 0.9)):
    import mujoco
    a = sim.ball_adr
    sim.data.qpos[a:a + 7] = [xy[0], xy[1], sim.model.geom("ball_geom").size[0], 1, 0, 0, 0]
    sim.data.qvel[sim.ball_vadr:sim.ball_vadr + 6] = 0.0
    mujoco.mj_forward(sim.model, sim.data)


# ── Ch.0: the 9 skills ─────────────────────────────────────────────────────────────
def buttons(args):
    from microduck_football.sim import duck_sim as ds
    from microduck_football.rl import kick_lab as kl
    sim = ds.DuckSim(seed=0)
    r, cam = follow_renderer(sim.model)
    wr = Writer(OUT / "ch0_buttons.mp4")
    desc = {"stand": "stand (also how it gets up)", "walk": "walk forward", "walk_left": "walk + turn left",
            "walk_right": "walk + turn right", "strafe_left": "side-step left", "strafe_right": "side-step right",
            "back": "walk back", "kick_left": "kick with the left foot", "kick_right": "kick with the right foot"}
    for i, (name, _) in enumerate(ds.PRIMITIVES):
        kl.open_reset(sim, x=0.2, y=-0.3, yaw=math.pi / 2)
        if name.startswith("kick"):
            sx, sy = ds.KICK_SPOT[name]
            kl.place_ball_rel(sim, sx - 0.02, sy)
        else:
            park_ball(sim)
        sim.apply_primitive(ds.P[name])
        steps = int((2.6 if name.startswith("kick") else 2.2) * ds.CONTROL_HZ)
        for k in range(steps):
            if not name.startswith("kick"):
                sim.apply_primitive(ds.P[name])
            sim.step()
            if k % 2:
                continue
            x, y, _ = sim.duck_pose()
            look = (x, y + (0.25 if name.startswith("kick") else 0.0), 0.1)
            f = shoot(r, cam, sim.data, look, 1.0 if name.startswith("kick") else 0.85, 135, -18)
            wr.add(overlay(f, f"{i + 1}/9   {desc[name]}", "a frozen Pollen skill network - we only choose which one runs",
                           corner="Chapter 0 - the duck's buttons"))
    wr.close()


# ── Ch.1: curriculum montage cut from the showreel ────────────────────────────────
STAGE_NAMES = {
    0: "Ball just in front of the goal", 1: "Ball further out", 2: "Get behind the ball first", 3: "Anywhere on the pitch",
    4: "One defender in the way", 5: "Wall of 3 defenders, one gap", 6: "2 defenders patrolling",
    7: "Slow goalkeeper", 8: "Medium goalkeeper", 9: "Full-speed goalkeeper", 10: "Goalkeeper + 2 defenders",
}


def _showreel_segments(path):
    """Gameplay intervals of the showreel = the stretches between its dark title cards. A frame is a card when
    it is mostly the card background (dark blue-grey); gameplay frames are mostly green pitch."""
    import imageio.v2 as iio
    rd = iio.get_reader(path)
    fps = rd.get_meta_data()["fps"]
    is_card = []
    for f in rd:
        small = f[::16, ::16].astype(int)
        dark = (np.abs(small - np.array(BG)).sum(-1) < 40).mean()
        is_card.append(dark > 0.6)
    rd.close()
    segs, start = [], None
    for i, c in enumerate(is_card + [True]):
        if not c and start is None:
            start = i
        elif c and start is not None:
            segs.append((start / fps, i / fps))
            start = None
    return [s for s in segs if s[1] - s[0] > 2.0]


def curriculum(args):
    import json
    import imageio.v2 as iio
    reel = paths.MEDIA_DIR / "showreel_all_stages.mp4"
    segs = _showreel_segments(reel)
    assert len(segs) == 11, f"expected 11 stage clips in the showreel, found {len(segs)}: {segs}"
    res = json.loads((paths.REPO_ROOT / "docs" / "results.json").read_text())["teacher_obstacles_per_stage"]["stages"]
    wr = Writer(OUT / "ch1_curriculum.mp4")
    rd = iio.get_reader(reel)
    fps = rd.get_meta_data()["fps"]
    for k, (a, b) in enumerate(segs):
        end = b - 1.0                                       # each clip ends with a 1 s hold on the goal
        start = max(a, end - (1.9 if k >= 4 else 1.4))
        for i in range(int(start * fps), int(end * fps)):
            f = rd.get_data(i)[30:480, 0:640]                # overview only (no stats panel)
            from PIL import Image
            f = np.asarray(Image.fromarray(f).resize((768, 540)))
            frame = np.full((H, W, 3), BG, np.uint8)
            frame[:, 96:864] = f
            g = res[str(k)]
            wr.add(overlay(frame, f"Stage {k}: {STAGE_NAMES[k]}", f"final RL player: {g['goals']}/20 goals on this stage",
                           corner="Chapter 1 - the RL player's curriculum"))
    rd.close()
    wr.close()


# ── Ch.2: head reflex ──────────────────────────────────────────────────────────────
def eyes(args):
    from PIL import Image
    from stable_baselines3 import PPO
    from microduck_football.envs import football_env as E
    E.REPLAY_EARLIER = 0.0
    model = PPO.load(paths.TEACHER_OPEN_PLAY, device="cpu")
    env = E.FootballEnv(seed=0, stage=3, head_tracking="privileged", glance=True)
    sim = env.sim
    r, cam = follow_renderer(sim.model, h=H, w=480)
    wr = Writer(OUT / "ch2_eyes.mp4")
    state = {"n": 0}

    def grab(s):
        state["n"] += 1
        if state["n"] % 2:
            return
        x, y, _ = s.duck_pose()
        over = shoot(r, cam, s.data, (x + 0.15, y, 0.05), 1.3, 135, -35)
        head = np.asarray(Image.fromarray(s.render_head()).resize((480, 360)))
        cy, cx = 180, 240
        head = head.copy()
        head[cy - 1:cy + 2, cx - 12:cx + 13] = 255
        head[cy - 12:cy + 13, cx - 1:cx + 2] = 255
        t = env.tracker
        mode = getattr(t, "mode", "ball")
        look = {"to_goal": "glancing at the GOAL", "hold_goal": "glancing at the GOAL", "to_ball": "back to the ball"}.get(
            mode, "tracking the BALL" if t.seen else "searching for the ball")
        frame = np.full((H, W, 3), BG, np.uint8)
        frame[:, :480] = over
        frame[60:420, 480:960] = head
        from PIL import ImageDraw
        img = Image.fromarray(frame)
        d = ImageDraw.Draw(img)
        d.text((720, 40), "what the head camera sees", font=font(17, False), fill=(220, 220, 228), anchor="mm")
        d.text((720, 440), look, font=font(20), fill=ACCENT, anchor="mm")
        wr.add(overlay(np.asarray(img), "The head follows the ball and glances at the goal",
                       "each decision is saved as camera image + body sensors + command -> LeRobot dataset",
                       corner="Chapter 2 - giving it eyes"))

    env.on_control_step = grab
    obs, _ = env.reset(seed=args.seed)
    done = False
    while not done and sim.t < args.seconds:
        obs, rew, te, tr, info = env.step(int(model.predict(obs, deterministic=True)[0]))
        done = te or tr
    wr.close()


# ── Ch.4: kicks ────────────────────────────────────────────────────────────────────
def _kick_frames(rolling, r, cam, seconds=3.2):
    from microduck_football.sim import duck_sim as ds
    from microduck_football.rl import kick_lab as kl
    sim = ds.DuckSim(seed=0)
    kl.open_reset(sim)
    sim.model.geom_friction[sim.model.geom("ball_geom").id, 2] = rolling
    sx, sy = ds.KICK_SPOT["kick_left"]
    kl.place_ball_rel(sim, sx - 0.02, sy)
    p0 = sim.ball_pos()[:2].copy()
    sim.apply_primitive(ds.P["kick_left"])
    frames, dist = [], []
    for k in range(int(seconds * ds.CONTROL_HZ)):
        sim.step()
        if k % 2 == 0:
            x, y, _ = sim.duck_pose()
            frames.append(shoot(r, cam, sim.data, (x, y + 0.85, 0.0), 1.75, 0, -22))
            dist.append(float(np.linalg.norm(sim.ball_pos()[:2] - p0)))
    return frames, dist


def kicks(args):
    from microduck_football.sim import duck_sim as ds
    model_owner = ds.DuckSim(seed=0)
    r, cam = follow_renderer(model_owner.model, h=H // 2 + 0, w=W)
    a, da = _kick_frames(0.003, r, cam)
    b, db = _kick_frames(0.0015, r, cam)
    wr = Writer(OUT / "ch4_kicks.mp4")
    from PIL import Image, ImageDraw
    for fa, fb, xa, xb in zip(a, b, da, db):
        top = np.asarray(Image.fromarray(fa).resize((W, 270)))
        bot = np.asarray(Image.fromarray(fb).resize((W, 270)))
        img = Image.fromarray(np.concatenate([top, bot]))
        d = ImageDraw.Draw(img)
        for y0, label, dist in ((0, "grass-like floor (1v1 pitch)", xa), (270, "indoor floor (2v2 pitch)", xb)):
            d.rectangle([0, y0, 330, y0 + 34], fill=(10, 10, 14))
            d.text((10, y0 + 7), f"{label}: {dist:.2f} m", font=font(17), fill=(255, 255, 255))
        wr.add(overlay(np.asarray(img), "Same kick, different floor",
                       "the kick is a gentle pass; grass stops the ball, an indoor floor lets it roll",
                       corner=None))
    for _ in range(FPS):
        wr.add(overlay(np.asarray(img), "Same kick, different floor",
                       "the kick is a gentle pass; grass stops the ball, an indoor floor lets it roll"))
    wr.close()


def aimed(args):
    from microduck_football.sim import duck_sim as ds
    from microduck_football.rl import kick_lab as kl
    sim = ds.DuckSim(seed=0)
    r, cam = follow_renderer(sim.model)
    kl.open_reset(sim)
    sim.model.geom_friction[sim.model.geom("ball_geom").id, 2] = 0.0015
    kl.place_ball_rel(sim, 0.26, 0.05)
    wr = Writer(OUT / "ch4_aimed_kick.mp4")
    step0, n = sim.step, {"k": 0, "phase": "walk up to the ball"}

    def step():
        step0()
        n["k"] += 1
        if n["k"] % 3:
            return
        x, y, _ = sim.duck_pose()
        f = shoot(r, cam, sim.data, (x, y + 0.2, 0.05), 0.95, 140, -20)
        busy = sim.policy.behavior_mode is not None
        phase = "kick!" if busy else ("settle, then creep in" if sim.primitive == ds.P["stand"] else "walk up to the ball")
        wr.add(overlay(f, f"Aimed kick: {phase}", "a 50 Hz routine on top of the skills: approach, wait until steady, fire",
                       corner="Chapter 4 - the aimed kick (1.5x)"))

    sim.step = step
    fired, foot, used = kl.aimed_kick(sim)
    for _ in range(int(1.6 * ds.CONTROL_HZ)):
        sim.step()
    wr.close()
    print(f"fired={fired} ({foot}) after {used:.1f} s")


# ── Ch.3: copying vs DAgger, same start ───────────────────────────────────────────
def dagger(args):
    """Stack two eval_student videos of the same seed (raw_act_copy_only.mp4 above raw_act_dagger.mp4), cut at
    --cut seconds (when the DAgger model scores), played at --speed."""
    import imageio.v2 as iio
    from PIL import Image, ImageDraw
    top = iio.get_reader(OUT / "raw_act_copy_only.mp4")
    bot = iio.get_reader(OUT / "raw_act_dagger.mp4")
    fps = top.get_meta_data()["fps"]
    n = int(args.cut * fps)
    wr = Writer(OUT / "ch3_dagger.mp4")
    last = None
    for i, (a, b) in enumerate(zip(top, bot)):
        if i >= n:
            break
        if i % args.speed:
            continue
        canvas = Image.new("RGB", (W, H), BG)
        for k, (f, label) in enumerate(((a, "copying only (behaviour cloning)"), (b, "+ 2 rounds of DAgger"))):
            img = Image.fromarray(f[16:240]).resize((654, 229))      # two views + the caption band fit in 540 px
            x0, y0 = (W - 654) // 2, 4 + k * 231
            canvas.paste(img, (x0, y0))
            d = ImageDraw.Draw(canvas)
            d.rectangle([x0, y0, x0 + d.textlength(label, font=font(16)) + 16, y0 + 28], fill=(10, 10, 14))
            d.text((x0 + 8, y0 + 5), label, font=font(16), fill=ACCENT if k else (230, 230, 235))
        last = overlay(np.asarray(canvas), None, None, corner=None)
        wr.add(overlay(last, "Same start, camera only: copying wanders, DAgger scores",
                       f"LeRobot ACT, open play; left half of each = overview for us, right half = what it sees ({args.speed}x)"))
    for _ in range(FPS):
        wr.add(overlay(last, "Same start, camera only: copying wanders, DAgger scores",
                       f"LeRobot ACT, open play; left half of each = overview for us, right half = what it sees ({args.speed}x)"))
    wr.close()


def match(args):
    """Chapter-4 2v2 clip from a play_2v2 video (raw_2v2_goal.mp4), trimmed to --start/--end seconds."""
    import imageio.v2 as iio
    rd = iio.get_reader(OUT / "raw_2v2_goal.mp4")
    fps = rd.get_meta_data()["fps"]
    wr = Writer(OUT / "ch4_2v2.mp4")
    for i, f in enumerate(rd):
        t = i / fps
        if t < args.start:
            continue
        if args.end and t > args.end:
            break
        wr.add(overlay(_to_canvas(f), "2v2 self-play: four ducks, one shared network",
                       "blue = trained team, red = original RL players (2x)"))
    wr.close()


# ── assembly ───────────────────────────────────────────────────────────────────────
CHAPTERS = [
    ("ch0_buttons.mp4", "Chapter 0", "The duck's 9 buttons", "Pollen's walking, standing and kicking skills - untouched"),
    ("ch1_curriculum.mp4", "Chapter 1", "An RL player that sees everything", "unlock the next stage once it scores often enough"),
    ("ch2_eyes.mp4", "Chapter 2", "Giving it eyes", "film its games from the head camera, save them as LeRobot datasets"),
    ("ch3_dagger.mp4", "Chapter 3", "A player that only uses its camera", "LeRobot ACT: copying alone fails, DAgger fixes it (open play)"),
    ("ch4_kicks.mp4", "Chapter 4", "Two against two", "fix the kick, add an aimed kick, then 2v2 self-play"),
    ("ch4_aimed_kick.mp4", None, None, None),
    ("ch4_2v2.mp4", None, None, None),
]


def _to_canvas(f):
    if f.shape[:2] == (H, W):
        return f
    from PIL import Image
    img = Image.fromarray(f)
    s = min(W / img.width, H / img.height)
    img = img.resize((int(img.width * s) // 2 * 2, int(img.height * s) // 2 * 2))
    canvas = np.full((H, W, 3), BG, np.uint8)
    x0, y0 = (W - img.width) // 2, (H - img.height) // 2
    canvas[y0:y0 + img.height, x0:x0 + img.width] = np.asarray(img)
    return canvas


def gif(src, dst, width=640, fps=10, speed=1.0):
    vf = f"setpts=PTS/{speed},fps={fps},scale={width}:-1:flags=lanczos"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-filter_complex",
                    f"[0:v]{vf},split[a][b];[a]palettegen=max_colors=96:stats_mode=diff[p];"
                    f"[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle", "-loop", "0", str(dst)],
                   check=True)
    print(f"wrote {dst} ({dst.stat().st_size / 1e6:.1f} MB)")


def assemble(args):
    import imageio.v2 as iio
    wr = Writer(paths.MEDIA_DIR / "walkthrough.mp4")
    for fr in card([("Microduck Football", 48, (255, 255, 255)),
                    ("teaching a 25 cm robot duck to play football in simulation", 22, (205, 205, 212)),
                    ("RL  ->  LeRobot dataset  ->  ACT + DAgger  ->  2v2", 22, ACCENT)], int(3 * FPS)):
        wr.add(fr)
    for clip, chap, title, sub in CHAPTERS:
        p = OUT / clip
        if not p.exists():
            print(f"(missing {p.name} - skipped)")
            continue
        if chap:
            for fr in card([(chap, 22, (160, 160, 172)), (title, 38, (255, 255, 255)), (sub, 19, (205, 205, 212))],
                           int(2.2 * FPS)):
                wr.add(fr)
        for f in iio.get_reader(p):
            wr.add(_to_canvas(f))
    for fr in card([("Datasets and model on the Hugging Face Hub", 26, (255, 255, 255)),
                    ("brouthen/microduck-football-rl-camera  -  brouthen/microduck-football-obstacles-camera", 16,
                     (205, 205, 212)),
                    ("brouthen/microduck-football-act", 16, (205, 205, 212)),
                    ("credits: Pollen Robotics (microduck), Hugging Face LeRobot, Rhoban BAM, MuJoCo", 15,
                     (160, 160, 172))], int(3.5 * FPS)):
        wr.add(fr)
    wr.close()
    if not args.no_gifs:
        for clip, name, speed in (("ch1_curriculum.mp4", "ch1_curriculum.gif", 1.0), ("ch3_dagger.mp4", "ch3_dagger.gif", 1.0),
                                  ("ch4_kicks.mp4", "ch4_kicks.gif", 1.0), ("ch4_aimed_kick.mp4", "ch4_aimed_kick.gif", 1.0),
                                  ("ch4_2v2.mp4", "ch4_2v2.gif", 1.0)):
            if (OUT / clip).exists():
                gif(OUT / clip, GIF_DIR / name, speed=speed)
        for clip, name, speed in (("ch0_buttons.mp4", "ch0_buttons.gif", 2.0), ("ch2_eyes.mp4", "ch2_eyes.gif", 1.6)):
            if (OUT / clip).exists():                 # long clips: smaller and faster to stay ~2.5 MB
                gif(OUT / clip, GIF_DIR / name, width=480, fps=8, speed=speed)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(required=True)
    sub.add_parser("buttons").set_defaults(func=buttons)
    sub.add_parser("curriculum").set_defaults(func=curriculum)
    e = sub.add_parser("eyes")
    e.add_argument("--seed", type=int, default=12346)
    e.add_argument("--seconds", type=float, default=14.0)
    e.set_defaults(func=eyes)
    sub.add_parser("kicks").set_defaults(func=kicks)
    sub.add_parser("aimed").set_defaults(func=aimed)
    d = sub.add_parser("dagger")
    d.add_argument("--cut", type=float, default=22.0)
    d.add_argument("--speed", type=int, default=2)
    d.set_defaults(func=dagger)
    m = sub.add_parser("match")
    m.add_argument("--start", type=float, default=0.0)
    m.add_argument("--end", type=float, default=None)
    m.set_defaults(func=match)
    a = sub.add_parser("assemble")
    a.add_argument("--no-gifs", action="store_true")
    a.set_defaults(func=assemble)
    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
