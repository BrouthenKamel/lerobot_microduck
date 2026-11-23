#!/usr/bin/env python3
"""Teleoperate the microduck on a football pitch and record demonstrations.

The frozen RL skills (walk / stand / kick ONNX) do all the balancing. YOU supply the
high-level commands; those commands are what gets recorded as the imitation-learning action:

    action = [vx, vy, yaw_rate, kick]                         (4-D, kick is 0/1)
    observation.images.head = head-camera frame               (what the duck sees)
    observation.state       = [gyro(3), gravity(3), joint_pos(14)]   (proprioception, 20-D)

Run from the project root:   uv run python -m microduck_football.teleop.football_teleop --help
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from microduck_football import paths

paths.add_infer_policy_to_path()
import infer_policy as ip  # noqa: E402  (the RL team's CPU deployment rehearsal)

ROOT = paths.REPO_ROOT
SCENE = paths.SCENE_FOOTBALL
POLICY_DIR = paths.POLICY_DIR

# ── Pitch (must match scene_football.xml) ────────────────────────────────────
GOAL_X = 1.7            # goal line
PITCH_X = (-0.9, 1.7)   # playing area (walls are ball-only: the duck can walk through them)
PITCH_Y = (-1.0, 1.0)
OUT_MARGIN = 0.15       # duck centre this far past a wall -> episode discarded (left the pitch)
FALL_GIVEUP = 10.0      # s the duck gets to stand back up before the episode is discarded
GOAL_HALF_WIDTH = 0.25
GOAL_HEIGHT = 0.2
BALL_R = ip.BALL_RADIUS

# ── Recording ────────────────────────────────────────────────────────────────
CONTROL_HZ = 50
RECORD_HZ = 10
DECIMATION = 4          # physics substeps per control step (dt = 5 ms)
IMG_W, IMG_H = 320, 240   # final (upright, landscape) size
CAM_FOVY = 90.0            # degrees; the head camera is mounted a quarter-turn off, so this ends up horizontal
HEAD_PITCH = 0.6           # rad head-pitch command while playing. With FOV 90 the camera then sees
                           # both the ball at the feet (when to shoot) and the goal ~1 m ahead.
TASK = "walk to the ball and kick it into the goal"
ACTION_NAMES = ["vx", "vy", "yaw_rate", "kick"]
STATE_NAMES = (
    ["gyro_x", "gyro_y", "gyro_z", "grav_x", "grav_y", "grav_z"]
    + [f"joint_{i}" for i in range(14)]
)

# ── Teleop commands ──────────────────────────────────────────────────────────
# Measured in sim: the walk policy only starts walking for vx >= ~0.25 (below that it stands),
# and strafing / turning only do anything while it is walking forward. So forward is on/off.
VX_WALK = 0.4          # top of the training range: 0.15 m/s (0.3 -> 0.10 m/s), no falls in tests
VX_BACK = -0.4        # backward only starts at >= ~0.35 in magnitude (then holds at less)
VY_STRAFE = 0.2
STEER_RATE = 1.0        # rad/s while a steering key is held (joystick-style, momentary)
STEER_HOLD = 0.35       # s a key press keeps steering (terminals send no key-release; holding
                        # the key refreshes it through auto-repeat)
KICK_SETTLE = 1.2       # s the duck must stand still before a kick starts. Measured braking from
                        # a walk (14 assisted kicks): 0.5 s -> 3 hit, 1.0 s -> 9, 1.5 s -> 10.
KICK_SETTLE_MAX = 2.0   # ... and its speed must be below KICK_SETTLE_SPEED, but never wait longer
KICK_SETTLE_SPEED = 0.02
# "In reach" window for the HUD and foot choice (without magnet): far edge +1.2 cm, sides 3.5 cm.
REACH_DX_MAX = 0.012
REACH_DY = 0.035
# Grass: linear drag on the ball (N per m/s). Measured: 0 -> kicks travel 0.2-1.8 m and the duck
# can never catch up; 0.01 -> kicks a consistent ~1.1 m and walking into the ball dribbles it.
BALL_DRAG = 0.01
GAME_SPEED = 1.5        # sim runs this many times faster than real time (physics unchanged)
# Kick magnet (sim assist, like a football game's "first touch"): the kick window sits right where
# a walking duck's toes first touch the ball, so it bumps the ball away before it can brake. When a
# kick fires, a ball within this distance of the foot's sweet spot is nudged onto it. 0 disables.
KICK_MAGNET = 0.06
YAW_MAX = 1.5

# The kick policies were trained with the ball here, in the duck's yaw frame (x fwd, y left).
KICK_SPOT = {"kick_right": (ip.BALL_OFFSET_X, -ip.BALL_OFFSET_ABS_Y),
             "kick_left": (ip.BALL_OFFSET_X, +ip.BALL_OFFSET_ABS_Y)}
# Measured in sim (standing duck, ball placed around the sweet spot): a ball up to ~4 cm CLOSER
# than the spot, or ~3 cm to either side, still gets kicked; a ball more than ~1 cm FURTHER than
# the spot is missed completely. So the window is asymmetric.
KICK_DX = (-0.04, +0.01)
KICK_DY = 0.03


class FootballPolicy(ip.PolicyInference):
    """Same skills as infer_policy.py, with two changes for real football."""

    def _place_ball(self, behavior):
        # infer_policy teleports the ball in front of the foot on every kick, to match training.
        # That is cheating for football: the ball must stay where it physically is.
        pass

    def set_vel_cmd(self, lin_vel_x=0.0, lin_vel_y=0.0, ang_vel_z=0.0):
        # Same as the parent without the console print (it would trample the HUD line).
        self.vel_cmd = np.array([lin_vel_x, lin_vel_y, ang_vel_z], dtype=np.float32)
        self._update_policy_session()
        self._update_command()


class Sim:
    def __init__(self, args):
        bam_model = ip.load_bam_model(args.kp_fw, args.vin, None)
        self.model, self.data, self.bam, _ = ip.load_mujoco_with_bam(
            str(SCENE), bam_model, 0.005, args.vin_drop_gain, ip.BAM_VIN_MIN)
        p = POLICY_DIR
        self.policy = FootballPolicy(
            self.model, self.data, bam_ctrl=self.bam,
            walking_onnx_path=str(p / "alpha_walking.onnx"),
            standing_onnx_path=str(p / "alpha_stand.onnx"),
            kick_left_onnx_path=str(p / "ball_kick_left.onnx"),
            kick_right_onnx_path=str(p / "ball_kick_right.onnx"),
            new_cmd_obs=True, use_projected_gravity=True, kick_duration=args.kick_duration)
        self.policy.vel_max_x, self.policy.vel_min_x = VX_WALK, VX_BACK
        self.policy.vel_max_y, self.policy.vel_min_y = VY_STRAFE, -VY_STRAFE
        self.policy.vel_max_ang = YAW_MAX

        self.dt = DECIMATION * self.model.opt.timestep
        self.rng = np.random.default_rng(args.seed)
        self.trunk_adr = self.policy._trunk_qpos_adr
        self.trunk_vadr = int(self.model.jnt_dofadr[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")])
        # Walls/posts/nets use collision bit 4 (see scene_football.xml): give the ball bits 1+4 so only
        # it collides with them and the duck can't trip over an edge.
        bg = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
        self.model.geom_contype[bg] = 5
        self.model.geom_conaffinity[bg] = 5
        # Grass: linear drag on the ball's translational dofs only (damping the spin as well
        # kills the roll and kicks go 7 cm).
        bv = int(self.model.jnt_dofadr[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")])
        self.model.dof_damping[bv:bv + 3] = args.ball_drag
        self.ball_adr = self.policy.ball_qpos_adr
        self.ball_vadr = self.policy.ball_qvel_adr

        # Head camera. The model's camera faces backwards; same 180° fix as sim/camera.py.
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
        fixed = np.zeros(4)
        mujoco.mju_mulQuat(fixed, self.model.cam_quat[self.cam_id], np.array([0.0, 1.0, 0.0, 0.0]))
        self.model.cam_quat[self.cam_id] = fixed
        self.model.cam_fovy[self.cam_id] = CAM_FOVY
        self.renderer = None
        self.args = args
        self.magnet = args.magnet_cm / 100.0
        self.head_pitch = args.head_pitch
        self.pending_kick = None   # kick committed, waiting for the duck to stop and settle
        self.stood_since = 0.0     # sim time since the duck was last commanded to move
        self.t = 0.0

    # -- camera -------------------------------------------------------------
    def render_head(self) -> np.ndarray:
        if self.renderer is None:
            # The head camera is mounted a quarter-turn off (see sim/camera.py), so render
            # portrait and rotate to get an upright landscape image.
            self.renderer = mujoco.Renderer(self.model, height=IMG_W, width=IMG_H)
        self.renderer.update_scene(self.data, camera=self.cam_id)
        return np.ascontiguousarray(np.rot90(self.renderer.render(), k=-1))

    # -- geometry helpers ---------------------------------------------------
    def duck_pose(self):
        a = self.trunk_adr
        x, y = float(self.data.qpos[a]), float(self.data.qpos[a + 1])
        qw, qx, qy, qz = self.data.qpos[a + 3:a + 7]
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return x, y, yaw

    def ball_pos(self):
        return self.data.qpos[self.ball_adr:self.ball_adr + 3].copy()

    def ball_in_duck_frame(self):
        x, y, yaw = self.duck_pose()
        b = self.ball_pos()
        dx, dy = b[0] - x, b[1] - y
        return (math.cos(yaw) * dx + math.sin(yaw) * dy, -math.sin(yaw) * dx + math.cos(yaw) * dy)

    def kick_ready(self):
        f = self.foot_in_reach()
        return [f] if f else []

    def scored(self):
        b = self.ball_pos()
        return b[0] > GOAL_X + BALL_R and abs(b[1]) < GOAL_HALF_WIDTH and b[2] < GOAL_HEIGHT

    def out_of_bounds(self):
        x, y, _ = self.duck_pose()
        return (x < PITCH_X[0] - OUT_MARGIN or x > PITCH_X[1] + OUT_MARGIN
                or y < PITCH_Y[0] - OUT_MARGIN or y > PITCH_Y[1] + OUT_MARGIN)

    def get_up(self):
        """Hand control to the stand policy with a zero command: it recovers from face-down,
        on-back and on-side falls in under a second (measured)."""
        pol = self.policy
        self.cancel_kick()
        if pol.behavior_mode is not None:
            pol._end_behavior()
        pol.set_vel_cmd(0.0, 0.0, 0.0)

    def push(self, speed: float = 1.0):
        """Shove the duck (same as infer_policy's P key) - handy for practising get-ups."""
        ang = self.rng.uniform(0, 2 * math.pi)
        self.data.qvel[self.trunk_vadr:self.trunk_vadr + 2] = [speed * math.cos(ang), speed * math.sin(ang)]

    def fallen(self):
        return float(self.policy.get_projected_gravity()[2]) > -0.5

    # -- episode reset ------------------------------------------------------
    def reset(self):
        """Randomise duck + ball, let the duck settle standing for 1 s."""
        m, d, pol, rng = self.model, self.data, self.policy, self.rng
        mujoco.mj_resetData(m, d)
        for _ in range(100):
            dx, dy = rng.uniform(-0.65, 0.1), rng.uniform(-0.6, 0.6)
            bx, by = rng.uniform(0.15, 1.0), rng.uniform(-0.7, 0.7)
            if math.hypot(bx - dx, by - dy) > 0.35:
                break
        yaw = rng.uniform(-math.pi / 2, math.pi / 2)
        a = self.trunk_adr
        d.qpos[a:a + 3] = [dx, dy, 0.125]
        d.qpos[a + 3:a + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        for i, qi in enumerate(pol.joint_qpos_indices):
            d.qpos[qi] = pol.default_pose[i]
        d.qpos[self.ball_adr:self.ball_adr + 7] = [bx, by, BALL_R, 1, 0, 0, 0]
        d.qvel[:] = 0.0
        self.bam.reset(d.qpos)
        pol.last_action[:] = 0.0
        pol.behavior_mode = None
        pol.set_vel_cmd(0.0, 0.0, 0.0)
        pol.set_position_targets(pol.default_pose)
        mujoco.mj_forward(m, d)
        for _ in range(CONTROL_HZ):  # 1 s settle
            self.step()

    # -- stepping -----------------------------------------------------------
    def shoot(self):
        """The one kick button: stop, let the duck settle, then kick with the foot on the ball's
        side (the kick policies only work from a standing start - mid-stride they swing at air).
        A ball within the magnet radius of that foot's sweet spot is nudged onto it."""
        if self.pending_kick is None and self.policy.behavior_mode is None:
            self.policy.set_vel_cmd(0.0, 0.0, 0.0)
            self.pending_kick = "shoot"     # foot is chosen at kick time, after the duck settles

    def choose_foot(self) -> str:
        foot = self.foot_in_reach()
        if foot:
            return foot
        return "kick_left" if self.ball_in_duck_frame()[1] > 0 else "kick_right"

    def cancel_kick(self):
        self.pending_kick = None

    def spot_offset(self, name):
        bx, by = self.ball_in_duck_frame()
        sx, sy = KICK_SPOT[name]
        return bx - sx, by - sy

    def reach_window(self):
        m = self.magnet
        return min(KICK_DX[0], -m), max(REACH_DX_MAX, m), max(REACH_DY, m)

    def foot_in_reach(self):
        """The foot whose window the ball is in (or about to be in), else None."""
        lo, hi, dy_max = self.reach_window()
        best = None
        for name in KICK_SPOT:
            dxo, dyo = self.spot_offset(name)
            if lo <= dxo <= hi and abs(dyo) <= dy_max:
                if best is None or abs(dyo) < best[0]:
                    best = (abs(dyo), name)
        return best[1] if best else None

    def snap_ball(self, name):
        """Kick magnet: nudge a nearby ball exactly onto the foot's sweet spot."""
        if self.magnet <= 0:
            return
        dxo, dyo = self.spot_offset(name)
        if math.hypot(dxo, dyo) > 1.2 * self.magnet:
            return
        x, y, yaw = self.duck_pose()
        sx, sy = KICK_SPOT[name]
        self.data.qpos[self.ball_adr:self.ball_adr + 3] = [
            x + math.cos(yaw) * sx - math.sin(yaw) * sy,
            y + math.sin(yaw) * sx + math.cos(yaw) * sy, BALL_R]
        self.data.qvel[self.ball_vadr:self.ball_vadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _speed(self) -> float:
        return float(np.linalg.norm(self.data.qvel[self.trunk_vadr:self.trunk_vadr + 2]))

    def step(self):
        pol = self.policy
        self.t += self.dt
        if np.any(pol.vel_cmd != 0.0):
            self.stood_since = self.t
        # Head tilted down while playing (so the camera sees the ball at the feet), level while a
        # kick is pending/running: the kick policies were trained with a level head and miss
        # almost every kick with it tilted (1/14 vs 13/14 measured).
        want = 0.0 if (self.pending_kick is not None or pol.behavior_mode is not None) else self.head_pitch
        if pol.head_offset[1] != want:
            pol.head_offset[1] = want
            pol._update_command()
        if self.pending_kick is not None:
            if np.any(pol.vel_cmd != 0.0):
                self.pending_kick = None      # operator moved again: cancel
            else:
                waited = self.t - self.stood_since
                if waited >= KICK_SETTLE_MAX or (waited >= KICK_SETTLE and self._speed() < KICK_SETTLE_SPEED):
                    foot = self.choose_foot()
                    self.snap_ball(foot)
                    pol.trigger_behavior(foot)
                    self.pending_kick = None
        pol.update_behavior(self.dt)
        pol.apply_action(pol.infer())
        for _ in range(DECIMATION):
            self.bam.update()
            mujoco.mj_step(self.model, self.data)

    def action_vector(self) -> np.ndarray:
        v = self.policy.vel_cmd
        kick = self.policy.behavior_mode or self.pending_kick   # intent counts from the key press
        return np.array([v[0], v[1], v[2], float(bool(kick))],
                        dtype=np.float32)

    def state_vector(self) -> np.ndarray:
        return self.policy.get_observations()[:20].astype(np.float32)  # gyro, gravity, joint pos


class CameraWindow:
    """The duck's-eye view in a small pygame window (OpenCV here is a headless build)."""

    def __init__(self, scale: int = 2):
        # SDL otherwise swallows Ctrl+C / SIGTERM and turns them into window events.
        os.environ.setdefault("SDL_NO_SIGNAL_HANDLERS", "1")
        import pygame
        self.pg = pygame
        pygame.display.init()
        self.size = (IMG_W * scale, IMG_H * scale)
        self.screen = pygame.display.set_mode(self.size)
        pygame.display.set_caption("duck camera (what the policy sees)")

    def show(self, rgb: np.ndarray):
        surf = self.pg.surfarray.make_surface(rgb.swapaxes(0, 1))
        self.screen.blit(self.pg.transform.scale(surf, self.size), (0, 0))
        self.pg.display.flip()
        self.pg.event.pump()


class Recorder:
    def __init__(self, repo_id: str, root: Path):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        if (root / "meta" / "info.json").exists():
            self.ds = LeRobotDataset.resume(repo_id, root=root)
            print(f"Resuming dataset at {root} ({self.ds.num_episodes} episodes so far)")
        else:
            features = {
                "observation.images.head": {"dtype": "video", "shape": (IMG_H, IMG_W, 3),
                                            "names": ["height", "width", "channels"]},
                "observation.state": {"dtype": "float32", "shape": (len(STATE_NAMES),), "names": STATE_NAMES},
                "action": {"dtype": "float32", "shape": (len(ACTION_NAMES),), "names": ACTION_NAMES},
            }
            self.ds = LeRobotDataset.create(repo_id, fps=RECORD_HZ, features=features, root=root,
                                            robot_type="microduck", use_videos=True)
            print(f"Created dataset at {root}")
        self.n_frames = 0

    def add(self, image, state, action):
        self.ds.add_frame({"observation.images.head": image, "observation.state": state,
                           "action": action, "task": TASK})
        self.n_frames += 1

    def save(self):
        self.ds.save_episode()
        self.n_frames = 0

    def discard(self):
        self.ds.clear_episode_buffer()
        self.n_frames = 0

    def close(self):
        self.ds.finalize()


HELP = """
 Movement
   UP              walk forward (latched: keeps walking)
   DOWN            walk backward (latched)
   S               stop
   LEFT / RIGHT    steer, joystick-style: HOLD the key to turn, release to go straight.
                   (works while walking; turning in place is very slow)
   A / E           strafe left / right, same hold-to-apply behaviour (only while walking)
   U               get up (the duck also gets up by itself when it falls; the episode goes on)
   P               shove the duck (to practise falling / getting up)
 Shooting
   SPACE           SHOOT.  The duck stops, settles (~1.2 s of sim time: kicks only work from a
                   standing start), then kicks with the foot on the ball's side.  The ball goes
                   ~1 m straight ahead of where the duck faces.  A ball within ~6 cm of the foot is
                   nudged onto it (--magnet-cm 0 = off).  UP/DOWN/S during the pause cancel it.
 Dribbling: just walk into the ball - the grass keeps it close, so you can push it along.
 Episode control
   G or ENTER      start recording (duck + ball were randomised and the duck is standing)
   X               discard this attempt and reset
   Q               quit (dataset is finalised)
 Ending an episode: a goal auto-saves it.  A timeout (90 s), leaving the pitch, or failing to
get up within 10 s discards it.  An ordinary fall does not end the episode.
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo-id", default="local/microduck_football")
    ap.add_argument("--root", type=Path, default=paths.DATA_DIR / "microduck_football")
    ap.add_argument("--no-record", action="store_true", help="just play: no dataset written")
    ap.add_argument("--max-seconds", type=float, default=90.0, help="episode timeout (discarded)")
    ap.add_argument("--steer-hold", type=float, default=STEER_HOLD,
                    help="seconds one steering key press lasts (raise if turning stutters while holding)")
    ap.add_argument("--steer-rate", type=float, default=STEER_RATE, help="turn rate while steering (rad/s)")
    ap.add_argument("--magnet-cm", type=float, default=KICK_MAGNET * 100,
                    help="kick magnet radius in cm: a ball this close to the foot's sweet spot is "
                         "nudged onto it when the kick fires (0 = off, pure physics)")
    ap.add_argument("--speed", type=float, default=GAME_SPEED,
                    help="game speed: sim time per real second (physics and the dataset are unchanged)")
    ap.add_argument("--head-pitch", type=float, default=HEAD_PITCH,
                    help="head tilt (rad) while playing, so the camera sees the ball at the feet")
    ap.add_argument("--ball-drag", type=float, default=BALL_DRAG,
                    help="grass drag on the ball (0 = original ice-like rolling)")
    ap.add_argument("--kick-duration", type=float, default=0.5,
                    help="seconds the kick policy runs (manifest: 0.5)")
    ap.add_argument("--show-camera", action="store_true",
                    help="open a window with the duck's own camera view (play with only this for honest demos)")
    ap.add_argument("--no-viewer", action="store_true", help="skip the 3D viewer")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--vin", type=float, default=7.4)
    ap.add_argument("--vin-drop-gain", type=float, default=0.1)
    ap.add_argument("--kp-fw", type=float, default=ip.BAM_KP_FW)
    ap.add_argument("--selftest", type=Path, default=None, metavar="OUTDIR",
                    help="non-interactive smoke test: scripted walk+kick, saves a camera frame, "
                         "writes a 1-episode dataset in OUTDIR")
    args = ap.parse_args()

    sim = Sim(args)
    if args.selftest:
        return selftest(sim, args)

    recorder = None if args.no_record else Recorder(args.repo_id, args.root)
    cam_view = CameraWindow(scale=2) if args.show_camera else None

    pol = sim.policy
    print(HELP)
    sim.reset()
    state = "WAIT"  # WAIT -> REC -> (saved | discarded) -> reset -> WAIT
    t_rec = 0.0
    step_i = 0
    saved = 0
    rec_every = CONTROL_HZ // RECORD_HZ

    def cmd(vx=None, vy=None, dyaw=0.0):
        v = pol.vel_cmd
        pol.set_vel_cmd(v[0] if vx is None else vx, v[1] if vy is None else vy,
                        float(np.clip(v[2] + dyaw, -YAW_MAX, YAW_MAX)))

    def end_episode(outcome: str):
        nonlocal state, saved
        if recorder is not None and state == "REC":
            if outcome == "GOAL" and recorder.n_frames > 0:
                recorder.save()
                saved += 1
            else:
                recorder.discard()
        print(f"\n== episode {outcome}"
              + (f"  (saved: {saved})" if recorder else "") + "  -> resetting, then press G to start")
        sim.reset()
        state = "WAIT"

    viewer_cm = (mujoco.viewer.launch_passive(sim.model, sim.data, show_left_ui=False, show_right_ui=False)
                 if not args.no_viewer else None)
    try:
        with ip.TerminalInput() as term:
            viewer = viewer_cm.__enter__() if viewer_cm else None
            if viewer:
                viewer.cam.lookat[:] = (0.4, 0.0, 0.0)
                viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = 3.6, 90, -55
            quit_ = False
            loop_i = 0
            fall_since = None
            steer_yaw = strafe = 0.0
            steer_until = strafe_until = 0.0
            deadline = time.perf_counter()
            while not quit_ and (viewer is None or viewer.is_running()):
                loop_i += 1
                now = time.perf_counter()
                for key in term.get_keys():
                    if key == " ":
                        if fall_since is None:            # no kicking from the floor
                            steer_until = strafe_until = 0.0
                            sim.shoot()
                    elif key == "u":
                        sim.get_up()
                        steer_until = strafe_until = 0.0
                    elif key == "p":
                        sim.push()
                    elif (pol.behavior_mode is not None or fall_since is not None) \
                            and key in ("up", "down", "left", "right", "a", "e", "s"):
                        pass                              # kicking / getting up: ignore movement
                    elif key == "up": cmd(vx=VX_WALK)
                    elif key == "down": cmd(vx=VX_BACK)
                    elif key == "left": steer_yaw, steer_until = args.steer_rate, now + args.steer_hold
                    elif key == "right": steer_yaw, steer_until = -args.steer_rate, now + args.steer_hold
                    elif key == "a": strafe, strafe_until = VY_STRAFE, now + args.steer_hold
                    elif key == "e": strafe, strafe_until = -VY_STRAFE, now + args.steer_hold
                    elif key == "s":
                        steer_until = strafe_until = 0.0
                        sim.cancel_kick()
                        pol.set_vel_cmd(0.0, 0.0, 0.0)
                    elif key in ("g", "\n", "\r") and state == "WAIT":
                        state, t_rec, step_i = "REC", 0.0, 0
                        print("\n== RECORDING ...")
                    elif key == "x": end_episode("DISCARDED")
                    elif key == "q": quit_ = True

                if pol.behavior_mode is None and sim.pending_kick is None and fall_since is None:
                    yaw_t = steer_yaw if now < steer_until else 0.0
                    vy_t = strafe if now < strafe_until else 0.0
                    if yaw_t != pol.vel_cmd[2] or vy_t != pol.vel_cmd[1]:
                        pol.set_vel_cmd(float(pol.vel_cmd[0]), vy_t, yaw_t)

                # record BEFORE stepping: (observation_t, command_t)
                if state == "REC" and recorder is not None and step_i % rec_every == 0:
                    recorder.add(sim.render_head(), sim.state_vector(), sim.action_vector())
                sim.step()
                if state == "REC":
                    step_i += 1
                    t_rec += sim.dt
                    if sim.scored():
                        end_episode("GOAL")
                    elif sim.out_of_bounds():
                        end_episode("DISCARDED (duck left the pitch)")
                    elif fall_since is not None and sim.t - fall_since > FALL_GIVEUP:
                        end_episode("DISCARDED (could not get up)")
                    elif t_rec > args.max_seconds:
                        end_episode("DISCARDED (timeout)")

                # Falls don't end the episode: hand over to the stand policy to get up.
                if sim.fallen():
                    if fall_since is None:
                        fall_since = sim.t
                        steer_until = strafe_until = 0.0
                        sim.get_up()
                        print("\n== duck fell - getting up (press U to retry)")
                elif fall_since is not None:
                    fall_since = None

                if loop_i % rec_every == 0:
                    bx, by = sim.ball_in_duck_frame()
                    v = pol.vel_cmd
                    ready = "/".join(r.split("_")[1][0].upper() for r in sim.kick_ready()) or "-"
                    mode = ("GETTING UP" if fall_since is not None else
                            pol.behavior_mode or ("SHOOT..." if sim.pending_kick else "walk/stand"))
                    print(f"\r[{state:4s} {t_rec:5.1f}s] cmd vx={v[0]:+.1f} vy={v[1]:+.1f} yaw={v[2]:+.1f}"
                          f" | {mode:12s} | ball ahead={bx:+.2f} left={by:+.2f}"
                          f" | kick-ready: {ready:3s} | saved={saved}   ", end="", flush=True)
                if cam_view is not None and loop_i % rec_every == 0:
                    cam_view.show(sim.render_head())
                if viewer:
                    viewer.sync()
                # Absolute-deadline pacing: a slow frame (camera render) is made up by the next
                # steps instead of permanently slowing the sim down.
                deadline += sim.dt / args.speed
                slack = deadline - time.perf_counter()
                if slack > 0:
                    time.sleep(slack)
                elif slack < -0.5:
                    deadline = time.perf_counter()  # far behind (e.g. paused): don't sprint
    finally:
        if viewer_cm:
            viewer_cm.__exit__(None, None, None)
        if recorder is not None:
            recorder.close()
        print(f"\nDone. Episodes saved this session: {saved}")


def selftest(sim: Sim, args):
    """Scripted smoke test (no viewer, no keyboard): walk to the ball, kick, record 1 episode."""
    import imageio.v3 as iio
    out = args.selftest
    out.mkdir(parents=True, exist_ok=True)
    pol = sim.policy
    # Deterministic layout: duck at origin facing +x, ball 0.4 m ahead and slightly right.
    sim.reset()
    a = sim.trunk_adr
    sim.data.qpos[a:a + 7] = [0, 0, 0.125, 1, 0, 0, 0]
    sim.data.qpos[sim.ball_adr:sim.ball_adr + 7] = [0.40, -ip.BALL_OFFSET_ABS_Y, BALL_R, 1, 0, 0, 0]
    sim.data.qvel[:] = 0
    mujoco.mj_forward(sim.model, sim.data)

    rec = Recorder("local/selftest", out / "dataset") if not (out / "dataset" / "meta").exists() else None
    iio.imwrite(out / "head_cam_t0.png", sim.render_head())

    # Play it like a person: walk at the ball, press SHOOT when it is at the feet (~13 cm ahead).
    pol.set_vel_cmd(VX_WALK, 0.0, 0.0)
    for i in range(CONTROL_HZ * 8):
        if i % (CONTROL_HZ // RECORD_HZ) == 0 and rec:
            rec.add(sim.render_head(), sim.state_vector(), sim.action_vector())
        sim.step()
        if sim.ball_in_duck_frame()[0] < 0.13:
            break
    print(f"walked {i / CONTROL_HZ:.1f}s; ball in duck frame: {np.round(sim.ball_in_duck_frame(), 3)}; "
          f"kick-ready: {sim.kick_ready()}; fallen: {sim.fallen()}")
    iio.imwrite(out / "head_cam_before_kick.png", sim.render_head())
    b0 = sim.ball_pos()
    sim.shoot()
    for i in range(CONTROL_HZ * 5):
        if i % (CONTROL_HZ // RECORD_HZ) == 0 and rec:
            rec.add(sim.render_head(), sim.state_vector(), sim.action_vector())
        sim.step()
    b1 = sim.ball_pos()
    print(f"ball moved {np.linalg.norm(b1[:2] - b0[:2]):.2f} m: {b0[:2]} -> {b1[:2]}; "
          f"scored: {sim.scored()}; fallen: {sim.fallen()}")
    if rec:
        rec.save()
        rec.close()
        print(f"dataset written to {out / 'dataset'}")


if __name__ == "__main__":
    main()
