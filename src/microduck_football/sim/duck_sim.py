"""Microduck football world: physics + the frozen primitive policies, with no gameplay helpers.

The duck is driven only through PRIMITIVES (stand / walk variants / kicks), each of which feeds a
command to one of the pretrained ONNX policies in ``policies/`` exactly as the robot's runtime
would. Nothing here nudges the ball, picks a foot, waits for the duck to settle or stands it back
up: whoever drives the duck (heuristic, RL agent, imitation policy) has to do that itself.
"""

from __future__ import annotations

import contextlib
import io
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

from microduck_football import paths

paths.add_infer_policy_to_path()
import infer_policy as ip  # noqa: E402  (the RL team's CPU deployment rehearsal)

ROOT = paths.REPO_ROOT
SCENE = paths.SCENE_FOOTBALL
POLICY_DIR = paths.POLICY_DIR

# ── Pitch (must match scene_football.xml) ────────────────────────────────────
GOAL_X = 1.7
GOAL_HALF_WIDTH = 0.25
GOAL_HEIGHT = 0.3
GOAL_CENTER = np.array([GOAL_X, 0.0])
PITCH_X = (-0.9, 1.7)
PITCH_Y = (-1.0, 1.0)
BALL_R = ip.BALL_RADIUS

# ── Timing ───────────────────────────────────────────────────────────────────
CONTROL_HZ = 50          # policy rate (the ONNX policies were trained at 50 Hz)
DECIMATION = 4           # physics substeps per control step (dt = 5 ms)
KICK_DURATION = 0.5      # s a kick policy runs before handing back (policies/manifest.json)

# ── Camera ───────────────────────────────────────────────────────────────────
IMG_W, IMG_H = 320, 240  # upright landscape head-camera image
CAM_FOVY = 90.0          # degrees (ends up horizontal: the camera is mounted a quarter-turn off)

# Where the kick policies were trained to find the ball, in the duck's yaw frame (x fwd, y left).
KICK_SPOT = {"kick_right": (ip.BALL_OFFSET_X, -ip.BALL_OFFSET_ABS_Y),
             "kick_left": (ip.BALL_OFFSET_X, +ip.BALL_OFFSET_ABS_Y)}

# ── Primitives ───────────────────────────────────────────────────────────────
# (name, twist command) - twist None = episodic kick behaviour. Command values respect what the
# walk policy actually does (measured): it only starts walking for vx >= ~0.25 (backward <= -0.35),
# and turning / strafing only work while walking.
PRIMITIVES = [
    ("stand", (0.0, 0.0, 0.0)),
    ("walk", (0.4, 0.0, 0.0)),
    ("walk_left", (0.4, 0.0, 1.0)),
    ("walk_right", (0.4, 0.0, -1.0)),
    ("strafe_left", (0.3, 0.3, 0.0)),
    ("strafe_right", (0.3, -0.3, 0.0)),
    ("back", (-0.4, 0.0, 0.0)),
    ("kick_left", None),
    ("kick_right", None),
]
PRIMITIVE_NAMES = [p[0] for p in PRIMITIVES]
P = {name: i for i, name in enumerate(PRIMITIVE_NAMES)}

# Opponents (scene_football.xml, mocap bodies): N_DEF defenders + 1 goalkeeper.
N_DEF = 3
DEF_R = 0.09                       # defender radius (18 cm wide)
KEEPER_HALF_W = 0.08               # goalkeeper half width (16 cm wide box)
KEEPER_X = GOAL_X - 0.06           # keeper slides along this line, just in front of the goal
KEEPER_Y = GOAL_HALF_WIDTH - 0.06  # |y| limit of the keeper centre
# Goalkeeper strength levels: (max speed m/s, reaction delay s). The full keeper is slower than
# the duck (beatable by dribbling) but the jump from no keeper to it was too big to learn by
# exploration (stuck at ~5% goals for 900k decisions), hence the weak -> medium -> full ladder.
KEEPER_LEVELS = {"keeper_weak": (0.02, 1.2), "keeper_medium": (0.04, 0.9), "keeper": (0.06, 0.6),
                 "keeper_defenders": (0.06, 0.6)}
OBSTACLE_STAGES = {                # curriculum stage -> opponent layout (stages 0-3: none)
    4: "one_static",               # 1 defender ON the ball->goal line
    5: "wall",                     # 3 defenders across the line with one gap
    6: "patrol",                   # 2 defenders patrolling across the line
    7: "keeper_weak",              # slow goalkeeper with a long reaction delay
    8: "keeper_medium",
    9: "keeper",                   # full goalkeeper
    10: "keeper_defenders",        # full goalkeeper + 2 static defenders
}
_PARK = [(-3.0, -2.0 - 0.5 * i) for i in range(N_DEF + 1)]

STATE_NAMES = (["gyro_x", "gyro_y", "gyro_z", "grav_x", "grav_y", "grav_z"]
               + [f"joint_{i}" for i in range(14)])


class _Policies(ip.PolicyInference):
    """infer_policy's policy runner minus its two conveniences."""

    def _place_ball(self, behavior):
        pass  # infer_policy teleports the ball to the foot on every kick; we never do.

    def set_vel_cmd(self, lin_vel_x=0.0, lin_vel_y=0.0, ang_vel_z=0.0):
        self.vel_cmd = np.array([lin_vel_x, lin_vel_y, ang_vel_z], dtype=np.float32)
        self._update_policy_session()
        self._update_command()


@contextlib.contextmanager
def _single_thread_onnx():
    """infer_policy creates its sessions with onnxruntime's default thread pools (one thread per
    core). With several sims in parallel (RL workers) those pools fight over the CPU: forcing one
    thread per session measured +30% total throughput at 7 workers. Same outputs."""
    ort = ip.ort
    original = ort.InferenceSession

    def session(path, sess_options=None, *args, **kwargs):
        so = sess_options or ort.SessionOptions()
        so.intra_op_num_threads = 1
        so.inter_op_num_threads = 1
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        return original(path, so, *args, **kwargs)

    ort.InferenceSession = session
    try:
        yield
    finally:
        ort.InferenceSession = original


def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class DuckSim:
    """One duck, one ball, one goal. Step at 50 Hz with ``step()``, choose behaviour with
    ``apply_primitive()``."""

    def __init__(self, seed: int | None = None, kick_duration: float = KICK_DURATION,
                 vin: float = 7.4, vin_drop_gain: float = 0.1, kp_fw: float = ip.BAM_KP_FW,
                 verbose: bool = False):
        out = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
        with out, _single_thread_onnx():
            bam_model = ip.load_bam_model(kp_fw, vin, None)
            self.model, self.data, self.bam, _ = ip.load_mujoco_with_bam(
                str(SCENE), bam_model, 0.005, vin_drop_gain, ip.BAM_VIN_MIN)
            p = POLICY_DIR
            self.policy = _Policies(
                self.model, self.data, bam_ctrl=self.bam,
                walking_onnx_path=str(p / "alpha_walking.onnx"),
                standing_onnx_path=str(p / "alpha_stand.onnx"),
                kick_left_onnx_path=str(p / "ball_kick_left.onnx"),
                kick_right_onnx_path=str(p / "ball_kick_right.onnx"),
                new_cmd_obs=True, use_projected_gravity=True, kick_duration=kick_duration)
        self.verbose = verbose
        # BAM rewrites these MODEL fields on every update (friction budget, damping); mj_resetData
        # does not restore model fields, so keep the compiled values and put them back on reset.
        self._model_defaults = {f: getattr(self.model, f).copy()
                                for f in ("dof_frictionloss", "dof_damping", "dof_armature")}
        self.dt = DECIMATION * self.model.opt.timestep
        self.rng = np.random.default_rng(seed)
        self.trunk_adr = self.policy._trunk_qpos_adr
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "trunk_base_freejoint")
        self.trunk_vadr = int(self.model.jnt_dofadr[jid])
        self.ball_adr = self.policy.ball_qpos_adr
        self.ball_vadr = self.policy.ball_qvel_adr

        # Head camera: the model's camera faces backwards; same 180° fix as sim/camera.py.
        self.cam_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
        fixed = np.zeros(4)
        mujoco.mju_mulQuat(fixed, self.model.cam_quat[self.cam_id], np.array([0.0, 1.0, 0.0, 0.0]))
        self.model.cam_quat[self.cam_id] = fixed
        self.model.cam_fovy[self.cam_id] = CAM_FOVY
        self._head_renderer = None
        self._overview_renderer = None
        def mocap(name):
            return int(self.model.body_mocapid[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)])
        self.def_mocap = [mocap(f"def_{i}") for i in range(N_DEF)]
        self.keeper_mocap = mocap("keeper")
        self.def_gids = [mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"def_{i}") for i in range(N_DEF)]
        self.keeper_gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "keeper")
        self._clear_opponents()

        self.t = 0.0
        self.stand_since = 0.0     # sim time the current stand command started (None-like: t)
        self.primitive = P["stand"]

    # -- quiet wrapper (infer_policy prints on behaviour start/end) ---------
    def _q(self):
        return contextlib.nullcontext() if self.verbose else contextlib.redirect_stdout(io.StringIO())

    # -- geometry -------------------------------------------------------------
    def duck_pose(self):
        a = self.trunk_adr
        x, y = float(self.data.qpos[a]), float(self.data.qpos[a + 1])
        qw, qx, qy, qz = self.data.qpos[a + 3:a + 7]
        yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
        return x, y, yaw

    def ball_pos(self) -> np.ndarray:
        return self.data.qpos[self.ball_adr:self.ball_adr + 3].copy()

    def ball_vel(self) -> np.ndarray:
        return self.data.qvel[self.ball_vadr:self.ball_vadr + 3].copy()

    def to_duck_frame(self, xy, is_vector: bool = False):
        """World xy point (or vector) -> duck yaw frame (x forward, y left)."""
        x, y, yaw = self.duck_pose()
        dx, dy = (xy[0], xy[1]) if is_vector else (xy[0] - x, xy[1] - y)
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([c * dx + s * dy, -s * dx + c * dy])

    def ball_in_duck_frame(self):
        return self.to_duck_frame(self.ball_pos()[:2])

    def trunk_vel_duck_frame(self):
        v = self.data.qvel[self.trunk_vadr:self.trunk_vadr + 2]
        return self.to_duck_frame(v, is_vector=True)

    def yaw_rate(self) -> float:
        return float(self.data.qvel[self.trunk_vadr + 5])

    def speed(self) -> float:
        return float(np.linalg.norm(self.data.qvel[self.trunk_vadr:self.trunk_vadr + 2]))

    def gravity(self) -> np.ndarray:
        return self.policy.get_projected_gravity().astype(np.float32)

    def fallen(self) -> bool:
        return float(self.gravity()[2]) > -0.5

    def scored(self) -> bool:
        b = self.ball_pos()
        return b[0] > GOAL_X + BALL_R and abs(b[1]) < GOAL_HALF_WIDTH and b[2] < GOAL_HEIGHT

    def stand_time(self) -> float:
        """Seconds the duck has been commanded to stand (0 while walking or kicking)."""
        if self.primitive != P["stand"] or self.policy.behavior_mode is not None:
            return 0.0
        return self.t - self.stand_since

    def state_vector(self) -> np.ndarray:
        return self.policy.get_observations()[:20].astype(np.float32)  # gyro, gravity, joint pos

    # -- reset ------------------------------------------------------------------
    # -- opponents (defenders + goalkeeper) -----------------------------------------
    def _clear_opponents(self):
        self.defenders = []        # dicts: base (xy), axis (xy), amp, speed, phase, pos, vel
        self.keeper = None         # dict: y, vel_y, history (ball y), active
        for i, mid in enumerate(self.def_mocap):
            self.data.mocap_pos[mid] = [*_PARK[i], 0.1]
        self.data.mocap_pos[self.keeper_mocap] = [*_PARK[N_DEF], 0.1]

    def _valid_def(self, p, duck, ball, others, ball_clear=0.25):
        aim = np.array([GOAL_X, 0.0])
        return (PITCH_X[0] + 0.15 < p[0] < GOAL_X - 0.25 and PITCH_Y[0] + 0.15 < p[1] < PITCH_Y[1] - 0.15
                and np.linalg.norm(p - ball) > ball_clear and np.linalg.norm(p - duck) > 0.35
                and np.linalg.norm(p - aim) > 0.3 and all(np.linalg.norm(p - q) > 2 * DEF_R + 0.02 for q in others))

    def setup_opponents(self, layout: str | None, duck_xy, ball_xy):
        """Place the opponents for a curriculum layout (see OBSTACLE_STAGES); None = no opponents."""
        self._clear_opponents()
        if layout is None:
            return
        rng = self.rng
        duck, ball = np.asarray(duck_xy, float), np.asarray(ball_xy, float)
        aim = np.array([GOAL_X, 0.0])
        d = aim - ball
        dist = float(np.linalg.norm(d)) + 1e-9
        u = d / dist                                   # ball -> goal direction
        nrm = np.array([-u[1], u[0]])                  # its left normal
        defs = []

        def add_static(t_lo, t_hi, lateral):
            for _ in range(100):
                p = ball + u * dist * rng.uniform(t_lo, t_hi) + nrm * rng.uniform(-lateral, lateral)
                if self._valid_def(p, duck, ball, [q["base"] for q in defs]):
                    defs.append(dict(base=p, axis=nrm, amp=0.0, speed=0.0, phase=0.0))
                    return

        if layout == "one_static":
            add_static(0.4, 0.7, 0.08)
        elif layout == "wall":
            for _ in range(100):
                t = rng.uniform(0.4, 0.6)
                centre = ball + u * dist * t
                gap, c = rng.uniform(0.35, 0.45), rng.uniform(-0.25, 0.25)
                lat = [c - gap / 2 - DEF_R, c - gap / 2 - 3 * DEF_R - 0.02, c + gap / 2 + DEF_R]
                pts = [centre + nrm * l for l in lat]
                if all(self._valid_def(p, duck, ball, [], ball_clear=0.2) for p in pts):
                    defs = [dict(base=p, axis=nrm, amp=0.0, speed=0.0, phase=0.0) for p in pts]
                    break
        elif layout == "patrol":
            for t in (0.35, 0.7):
                for _ in range(100):
                    p = ball + u * dist * t + nrm * rng.uniform(-0.1, 0.1)
                    if self._valid_def(p, duck, ball, [q["base"] for q in defs], ball_clear=0.2):
                        defs.append(dict(base=p, axis=nrm, amp=rng.uniform(0.25, 0.4),
                                         speed=rng.uniform(0.05, 0.1), phase=rng.uniform(0, 4)))
                        break
        elif layout == "keeper_defenders":
            add_static(0.3, 0.45, 0.1)
            add_static(0.6, 0.75, 0.1)
        if layout in KEEPER_LEVELS:
            vmax, delay = KEEPER_LEVELS[layout]
            y0 = float(np.clip(ball[1], -KEEPER_Y, KEEPER_Y))
            self.keeper = dict(y=y0, vel_y=0.0, vmax=vmax, history=[y0] * int(round(delay / self.dt)))
        for q in defs:
            q["pos"], q["vel"] = self._patrol_pos(q, 0.0), np.zeros(2)
        self.defenders = defs
        self._apply_opponents()

    @staticmethod
    def _patrol_pos(q, t):
        if q["amp"] <= 0:
            return q["base"].copy()
        period = 4 * q["amp"] / q["speed"]             # triangle wave, constant speed
        x = ((t + q["phase"] * period / 4) % period) / period
        tri = 4 * x - 1 if x < 0.5 else 3 - 4 * x      # -1 .. 1
        return q["base"] + q["axis"] * q["amp"] * tri

    def _apply_opponents(self):
        for i, mid in enumerate(self.def_mocap):
            xy = self.defenders[i]["pos"] if i < len(self.defenders) else _PARK[i]
            self.data.mocap_pos[mid] = [xy[0], xy[1], 0.1]
        if self.keeper is not None:
            self.data.mocap_pos[self.keeper_mocap] = [KEEPER_X, self.keeper["y"], 0.1]

    def update_opponents(self):
        """Advance moving defenders and the goalkeeper by one control step."""
        if not self.defenders and self.keeper is None:
            return
        for q in self.defenders:
            if q["amp"] > 0:
                new = self._patrol_pos(q, self.t)
                q["vel"] = (new - q["pos"]) / self.dt
                q["pos"] = new
        if self.keeper is not None:
            k = self.keeper
            k["history"].append(float(self.ball_pos()[1]))
            target = float(np.clip(k["history"].pop(0), -KEEPER_Y, KEEPER_Y))
            step = float(np.clip(target - k["y"], -k["vmax"] * self.dt, k["vmax"] * self.dt))
            k["y"] += step
            k["vel_y"] = step / self.dt
        self._apply_opponents()

    def opponent_states(self):
        """[(xy, vel_xy)] for active defenders, and (xy, vel_xy) or None for the keeper (world)."""
        defs = [(q["pos"], q["vel"]) for q in self.defenders]
        keeper = None if self.keeper is None else (np.array([KEEPER_X, self.keeper["y"]]),
                                                    np.array([0.0, self.keeper["vel_y"]]))
        return defs, keeper

    def sample_layout(self, difficulty="full"):
        """Duck (x, y, yaw) and ball (x, y) for an episode start.

        Curriculum stages (reverse curriculum: start next to success, then move away from it):
          0  ball 0.25-0.5 m in front of the goal mouth, duck right behind it on the line to the
             goal, facing it                                   -> learn stand -> kick / dribble in
          1  ball 0.5-1.2 m from the goal, duck behind it on the line, facing it
                                                               -> repeat approach + kick
          2  ball 0.5-1.6 m from the goal, duck 0.4-0.8 m from the ball within +-60° of the line
             behind it, heading within +-45° of the ball      -> get behind the ball
          3  "full": duck and ball anywhere in the own half / middle, heading in [-90°, 90°]
        "easy" (fixed, used for heuristic baselines): duck facing the goal, ball 20-35 cm ahead.
        """
        rng = self.rng
        if difficulty == "easy":
            dx, dy = rng.uniform(0.6, 1.0), rng.uniform(-0.3, 0.3)
            yaw = math.atan2(-dy, GOAL_X - dx) + rng.uniform(-0.15, 0.15)
            dist = rng.uniform(0.20, 0.35)
            return (dx, dy, yaw), (dx + dist * math.cos(yaw), dy + dist * math.sin(yaw))
        stage = 3 if difficulty == "full" else int(difficulty)
        stage = min(stage, 3)           # stages 4-6 = full layout + obstacles (see reset)
        margin = 0.15
        for _ in range(200):
            if stage == 3:
                dx, dy = rng.uniform(-0.65, 0.1), rng.uniform(-0.6, 0.6)
                bx, by = rng.uniform(0.15, 1.0), rng.uniform(-0.7, 0.7)
                yaw = rng.uniform(-math.pi / 2, math.pi / 2)
                if math.hypot(bx - dx, by - dy) > 0.35:
                    return (dx, dy, yaw), (bx, by)
                continue
            # ball at distance r from a point inside the goal, within an angle of the goal axis
            r_lo, r_hi, ang = {0: (0.25, 0.5, 0.35), 1: (0.5, 1.2, 0.6), 2: (0.5, 1.6, 0.8)}[stage]
            r, th = rng.uniform(r_lo, r_hi), rng.uniform(-ang, ang)
            aim = np.array([GOAL_X + 0.1, 0.0])
            ball = aim + r * np.array([-math.cos(th), math.sin(th)])
            to_goal = (aim - ball) / np.linalg.norm(aim - ball)
            if stage in (0, 1):
                back = rng.uniform(0.15, 0.25) if stage == 0 else rng.uniform(0.2, 0.35)
                duck = ball - to_goal * back
                yaw = math.atan2(to_goal[1], to_goal[0]) + rng.uniform(-0.17, 0.17)
            else:
                dist, off = rng.uniform(0.4, 0.8), rng.uniform(-math.pi / 3, math.pi / 3)
                duck = ball - dist * np.array([math.cos(math.atan2(to_goal[1], to_goal[0]) + off),
                                               math.sin(math.atan2(to_goal[1], to_goal[0]) + off)])
                face_ball = math.atan2(ball[1] - duck[1], ball[0] - duck[0])
                yaw = face_ball + rng.uniform(-math.pi / 4, math.pi / 4)
            inside = lambda p: (PITCH_X[0] + margin < p[0] < PITCH_X[1] - margin
                                and PITCH_Y[0] + margin < p[1] < PITCH_Y[1] - margin)
            if inside(duck) and inside(ball):
                return (float(duck[0]), float(duck[1]), wrap(yaw)), (float(ball[0]), float(ball[1]))
        raise RuntimeError(f"could not sample a layout for stage {stage}")

    def reset(self, difficulty="full"):
        """Place duck and ball (see sample_layout: "easy", "full", or curriculum stage 0-3), then
        let the duck settle standing for 1 s."""
        m, d, pol = self.model, self.data, self.policy
        mujoco.mj_resetData(m, d)
        for f, v in self._model_defaults.items():
            getattr(m, f)[:] = v
        (dx, dy, yaw), (bx, by) = self.sample_layout(difficulty)
        layout = None
        if difficulty not in ("easy", "full"):
            layout = OBSTACLE_STAGES.get(int(difficulty))
        self.setup_opponents(layout, (dx, dy), (bx, by))
        a = self.trunk_adr
        d.qpos[a:a + 3] = [dx, dy, 0.125]
        d.qpos[a + 3:a + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        for i, qi in enumerate(pol.joint_qpos_indices):
            d.qpos[qi] = pol.default_pose[i]
        d.qpos[self.ball_adr:self.ball_adr + 7] = [bx, by, BALL_R, 1, 0, 0, 0]
        d.qvel[:] = 0.0
        self.bam.reset(d.qpos)
        # bam's reset() clears targets and torques but not its clock: after mj_resetData the sim
        # time is 0 again, so the first update would see a large negative dt. (Episodes were not
        # reproducible from the seed because of this.)
        self.bam.last_ts = d.time
        pol.last_action[:] = 0.0
        pol.behavior_mode = None
        pol.head_offset[:] = 0.0
        pol.set_position_targets(pol.default_pose)
        mujoco.mj_forward(m, d)
        self.t = 0.0
        self.stand_since = 0.0
        self.primitive = P["stand"]
        self.apply_primitive(P["stand"])
        for _ in range(CONTROL_HZ):  # 1 s settle
            self.step()

    # -- control ----------------------------------------------------------------
    def apply_primitive(self, i: int):
        """Switch the active primitive. Kicks start the kick policy (which then runs for
        kick_duration and hands back to stand); a kick already running is not interrupted."""
        pol = self.policy
        if pol.behavior_mode is not None:
            return
        name, twist = PRIMITIVES[i]
        if twist is None:
            with self._q():
                pol.trigger_behavior(name)
            self.primitive = P["stand"]       # what the duck returns to after the kick
            self.stand_since = self.t + pol.behavior_time_left
            return
        if i == P["stand"] and self.primitive != P["stand"]:
            self.stand_since = self.t
        self.primitive = i
        with self._q():
            pol.set_vel_cmd(*twist)

    def apply_twist(self, vx: float, vy: float, yaw: float, kick: bool = False):
        """Continuous alternative to apply_primitive: send a twist command to the walk policy
        directly (|cmd| tiny -> stand policy, which also gets the duck up). kick=True starts the
        right-foot kick (like primitive kick_right)."""
        if self.policy.behavior_mode is not None:
            return
        if kick:
            self.apply_primitive(P["kick_right"])
            return
        moving = abs(vx) + abs(vy) + abs(yaw) > 0.05
        if not moving:
            self.apply_primitive(P["stand"])
            return
        self.primitive = P["walk"]               # "moving" for stand_time purposes
        with self._q():
            self.policy.set_vel_cmd(float(vx), float(vy), float(yaw))

    def step(self):
        """One 50 Hz control step (4 physics substeps)."""
        pol = self.policy
        self.t += self.dt
        with self._q():
            pol.update_behavior(self.dt)
        if pol.behavior_mode is None and self.primitive == P["stand"] and pol.current_policy != "standing":
            with self._q():
                pol.set_vel_cmd(0.0, 0.0, 0.0)   # kick just ended: _end_behavior hands to the walk net
        self.update_opponents()
        pol.apply_action(pol.infer())
        for _ in range(DECIMATION):
            self.bam.update()
            mujoco.mj_step(self.model, self.data)

    # -- rendering --------------------------------------------------------------
    def render_head(self) -> np.ndarray:
        if self._head_renderer is None:
            # Rendered portrait and rotated: the head camera is mounted a quarter-turn off.
            self._head_renderer = mujoco.Renderer(self.model, height=IMG_W, width=IMG_H)
        self._head_renderer.update_scene(self.data, camera=self.cam_id)
        return np.ascontiguousarray(np.rot90(self._head_renderer.render(), k=-1))

    def render_overview(self, height: int = 240, width: int = 320) -> np.ndarray:
        if self._overview_renderer is None:
            self._overview_renderer = mujoco.Renderer(self.model, height=height, width=width)
            self._overview_cam = mujoco.MjvCamera()
            self._overview_cam.lookat[:] = (0.4, 0.0, 0.0)
            self._overview_cam.distance, self._overview_cam.azimuth = 3.6, 90
            self._overview_cam.elevation = -55
        self._overview_renderer.update_scene(self.data, camera=self._overview_cam)
        return self._overview_renderer.render().copy()
