"""2v2 microduck football world: four ducks (two per team), one ball, two goals.

Same physics stack as duck_sim.DuckSim (MuJoCo + BAM actuators + the frozen ONNX primitive
policies), one copy per duck:
  * scene_2v2.xml (pitch only) + robot_groundcontact.xml attached 4x with name prefixes
    b0_/b1_ (blue, attacks +x) and r0_/r1_ (red, attacks -x);
  * per duck: its own BAM controller (own battery voltage-drop) and its own policy runner
    (infer_policy.PolicyInference re-pointed at that duck's joints/sensors).
Ducks collide with each other through a massless trunk capsule (contype/conaffinity 4).

Floor: ball rolling friction ROLLING_2V2 = 0.0015 (indoor / short carpet) instead of the
single-duck scene's turf (0.003): measured in rl/kick_lab.py, the kick policies (trained for a
1 m/s ball) then pass ~1.5 m instead of ~0.5 m, and the open-play teacher still scores 26/30.

Actions per duck: duck_sim.PRIMITIVES (9) + "aimed_kick" (index 9): a 50 Hz closed-loop skill
(walk in, settle, fire the kick with the foot on the ball's side; kick_lab.aimed_kick, here as a
state machine so the other ducks keep playing while one lines up a kick).

Nothing here is single-duck code changed: duck_sim.py / football_env.py are untouched.
"""

from __future__ import annotations

import contextlib
import io
import math

import mujoco
import numpy as np

from microduck_football.sim import duck_sim as ds
from microduck_football.sim.duck_sim import ROOT, POLICY_DIR, ip, _Policies, _single_thread_onnx, wrap

from microduck_football import paths

SCENE_2V2 = paths.SCENE_2V2
ROBOT_XML = paths.ROBOT_XML

GOAL_X = 1.7                  # goals at x = +-GOAL_X, same mouth as the single-duck scene
GOAL_HALF_WIDTH = ds.GOAL_HALF_WIDTH
GOAL_HEIGHT = ds.GOAL_HEIGHT
PITCH_X = (-1.7, 1.7)
PITCH_Y = (-1.0, 1.0)
BALL_R = ds.BALL_R
ROLLING_2V2 = 0.0015

PREFIXES = ["b0_", "b1_", "r0_", "r1_"]
TEAM_RGBA = {0: (0.15, 0.35, 1.0, 1.0), 1: (1.0, 0.2, 0.15, 1.0)}

AIMED_KICK = len(ds.PRIMITIVES)          # action index 9
ACTION_NAMES = ds.PRIMITIVE_NAMES + ["aimed_kick"]
N_ACTIONS = len(ACTION_NAMES)

# aimed-kick window (kick_lab.grid: works 0-4 cm closer than the trained spot, +-3 cm lateral)
AIM_X_FIRE, AIM_X_MIN = 0.085, 0.05
AIM_Y_OUT, AIM_Y_IN = 0.03, 0.015
AIM_REACH = 0.40
AIM_MAX_S = 5.0


def _bamify(spec, force_limit):
    """infer_policy.load_mujoco_with_bam's spec edit, prefix-aware: every non-passive position
    actuator -> torque motor with the voltage-bounded force range; its joint's damping and
    frictionloss zeroed (BAM computes them) with the stiff friction constraint."""
    names = []
    joints = {j.name: j for j in spec.joints}
    for act in spec.actuators:
        tgt = act.target
        tgt_name = tgt.name if hasattr(tgt, "name") else str(tgt)
        if tgt_name.split("_", 1)[-1].startswith("passive_"):
            continue
        act.set_to_motor()
        act.forcelimited = True
        act.forcerange = (-force_limit, force_limit)
        act.ctrllimited = False
        act.gear = [1.0, 0, 0, 0, 0, 0]
        names.append(act.name)
        j = joints.get(tgt_name)
        if j is not None:
            j.damping = np.zeros((3, 1))
            j.frictionloss = 0.0
            j.solref_friction = ip.BAM_STIFF_SOLREF_FRICTION
            j.solimp_friction = ip.BAM_STIFF_SOLIMP_FRICTION
    return names


class AimedKick:
    """kick_lab.aimed_kick as a 50 Hz state machine (one tick per control step)."""

    def __init__(self, duck: "Duck"):
        self.d = duck
        _, by = duck.ball_rel()
        self.foot = "kick_left" if by > 0 else "kick_right"
        self.tx, self.ty = ds.KICK_SPOT[self.foot]
        self.out = 1.0 if self.ty > 0 else -1.0
        self.t0 = duck.sim.t
        self.lead = -0.005
        self.burst = 16
        self.state, self.n = "approach", 0
        self.still_since = None
        self.fired = False
        bx, by = duck.ball_rel()
        if bx < AIM_X_FIRE + 0.05 and self._lat_ok(by - self.ty):
            self._to_settle()

    def _lat_ok(self, ey):
        return -AIM_Y_IN < self.out * ey < AIM_Y_OUT

    def _in_window(self, bx, ey, slack=0.0):
        return AIM_X_MIN < bx <= AIM_X_FIRE + slack and self._lat_ok(ey)

    def _to_settle(self):
        self.d._prim(ds.P["stand"])
        self.state, self.n, self.still_since = "settle", 0, None

    def tick(self) -> bool:
        """Advance one control step. Returns False once finished (fired or gave up)."""
        d, t = self.d, self.d.sim.t
        bx, by = d.ball_rel()
        ey = by - self.ty
        if t - self.t0 > AIM_MAX_S or d.fallen():
            d._prim(ds.P["stand"])
            return False
        if self.state == "approach":
            if not (0.0 < bx < AIM_REACH and abs(by) < 0.15):
                d._prim(ds.P["stand"])
                return False
            if self._in_window(bx, ey) or bx <= AIM_X_FIRE + self.lead:
                self._to_settle()
            else:
                d._twist(0.3, float(np.clip(4.0 * (ey - self.out * 0.007), -0.3, 0.3)), 0.0)
            return True
        if self.state in ("burst", "back"):
            self.n -= 1
            if self.state == "burst":
                d._twist(0.3, float(np.clip(4.0 * (ey - self.out * 0.007), -0.3, 0.3)), 0.0)
            else:
                d._twist(-0.4, float(np.clip(4.0 * ey, -0.3, 0.3)), 0.0)
            if self.n <= 0:
                if self.state == "burst":
                    self._to_settle()
                else:
                    self.state = "approach"
            return True
        # settle: stand until the trunk is still (the kick only connects from a settled stance)
        self.n += 1
        if np.linalg.norm(d.gyro()) < 0.15:
            self.still_since = t if self.still_since is None else self.still_since
            still = self.n * d.sim.dt >= 0.3 and t - self.still_since >= 0.2
        else:
            self.still_since, still = None, False
        if not (still or self.n * d.sim.dt >= 1.2):
            return True
        if self._in_window(bx, ey, slack=0.005):
            d._prim(ds.P[self.foot])
            self.fired = True
            return False
        if AIM_X_FIRE < bx < AIM_X_FIRE + 0.05 and self._lat_ok(ey):
            self.burst += 5                      # stopped short: creep in (walk off needs ~0.4 s)
            self.state, self.n = "burst", self.burst
        elif bx <= AIM_X_MIN or not self._lat_ok(ey):
            self.state, self.n = "back", int(0.6 * ds.CONTROL_HZ)
        else:
            self.state = "approach"
        return True


class Duck:
    """One duck inside a MultiDuckSim: pose / frames / sensors and primitive control, mirroring
    the DuckSim API. ``sign`` = +1 for blue (attacks +x), -1 for red."""

    def __init__(self, sim: "MultiDuckSim", idx: int, prefix: str, policy, bam):
        self.sim, self.idx, self.prefix = sim, idx, prefix
        self.team = 0 if prefix.startswith("b") else 1
        self.sign = 1.0 if self.team == 0 else -1.0
        self.policy, self.bam = policy, bam
        m = sim.model
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, prefix + "trunk_base_freejoint")
        self.trunk_adr = int(m.jnt_qposadr[jid])
        self.trunk_vadr = int(m.jnt_dofadr[jid])
        self.cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, prefix + "head_camera")
        self.primitive = ds.P["stand"]
        self.stand_since = 0.0
        self.macro: AimedKick | None = None
        self.n_kicks = 0           # kicks started (raw or aimed)
        self.n_falls = 0

    # -- geometry / sensors ------------------------------------------------------
    def pose(self):
        q = self.sim.data.qpos
        a = self.trunk_adr
        qw, qx, qy, qz = q[a + 3:a + 7]
        return float(q[a]), float(q[a + 1]), math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))

    def xy(self):
        return self.sim.data.qpos[self.trunk_adr:self.trunk_adr + 2].copy()

    def vel_xy(self):
        return self.sim.data.qvel[self.trunk_vadr:self.trunk_vadr + 2].copy()

    def to_frame(self, xy, is_vector=False):
        x, y, yaw = self.pose()
        dx, dy = (xy[0], xy[1]) if is_vector else (xy[0] - x, xy[1] - y)
        c, s = math.cos(yaw), math.sin(yaw)
        return np.array([c * dx + s * dy, -s * dx + c * dy])

    def ball_rel(self):
        return self.to_frame(self.sim.ball_pos()[:2])

    def yaw_rate(self):
        return float(self.sim.data.qvel[self.trunk_vadr + 5])

    def gravity(self):
        return self.policy.get_projected_gravity().astype(np.float32)

    def gyro(self):
        return self.policy.get_base_ang_vel()

    def fallen(self):
        return float(self.gravity()[2]) > -0.5

    def stand_time(self):
        if self.primitive != ds.P["stand"] or self.policy.behavior_mode is not None or self.macro is not None:
            return 0.0
        return self.sim.t - self.stand_since

    def kick_time_left(self):
        return self.policy.behavior_time_left if self.policy.behavior_mode is not None else 0.0

    def busy(self):
        """True while a kick runs or an aimed kick is lining up: decisions are ignored."""
        return self.policy.behavior_mode is not None or self.macro is not None

    # -- control -------------------------------------------------------------------
    def _prim(self, i: int):
        pol = self.policy
        if pol.behavior_mode is not None:
            return
        name, twist = ds.PRIMITIVES[i]
        if twist is None:
            with self.sim._q():
                pol.trigger_behavior(name)
            self.primitive = ds.P["stand"]
            self.stand_since = self.sim.t + pol.behavior_time_left
            self.n_kicks += 1
            return
        if i == ds.P["stand"] and self.primitive != ds.P["stand"]:
            self.stand_since = self.sim.t
        self.primitive = i
        with self.sim._q():
            pol.set_vel_cmd(*twist)

    def _twist(self, vx, vy, yaw):
        if self.policy.behavior_mode is not None:
            return
        self.primitive = ds.P["walk"]
        with self.sim._q():
            self.policy.set_vel_cmd(float(vx), float(vy), float(yaw))

    def act(self, a: int):
        """Agent decision (5 Hz). Ignored while busy."""
        if self.busy():
            return
        if a == AIMED_KICK:
            self.macro = AimedKick(self)
            if not self.macro.tick():
                self.macro = None
        else:
            self._prim(int(a))

    def control_step(self):
        """Per 50 Hz control step, before physics."""
        pol = self.policy
        with self.sim._q():
            pol.update_behavior(self.sim.dt)
        if pol.behavior_mode is None and self.primitive == ds.P["stand"] and pol.current_policy != "standing" \
                and self.macro is None:
            with self.sim._q():
                pol.set_vel_cmd(0.0, 0.0, 0.0)
        if self.macro is not None and not self.macro.tick():
            self.macro = None
        pol.apply_action(pol.infer())


class MultiDuckSim:
    """Four ducks, one ball, two goals. ``step()`` = one 50 Hz control step for everyone."""

    def __init__(self, seed: int | None = None, rolling: float = ROLLING_2V2, kick_duration=ds.KICK_DURATION,
                 vin: float = 7.4, vin_drop_gain: float = 0.1, kp_fw: float = ip.BAM_KP_FW, verbose=False):
        from bam.mujoco import MujocoController
        self.verbose = verbose
        out = contextlib.nullcontext() if verbose else contextlib.redirect_stdout(io.StringIO())
        with out, _single_thread_onnx():
            bams = [ip.load_bam_model(kp_fw, vin, None) for _ in PREFIXES]
            kt, R = bams[0].kt.value, bams[0].R.value
            force_limit = bams[0].actuator.vin * kt / R
            spec = mujoco.MjSpec.from_file(str(SCENE_2V2))
            for i, p in enumerate(PREFIXES):
                robot = mujoco.MjSpec.from_file(str(ROBOT_XML))
                trunk = robot.body("trunk_base")
                trunk.add_geom(name="bib", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.03, 0, 0],
                               pos=[-0.02, 0, 0.19], mass=0.0,          # team marker above the head (visual) rgba=TEAM_RGBA[0 if p.startswith("b") else 1],
                               contype=0, conaffinity=0, group=1)
                # duck-duck collision proxy: the groundcontact model only has collision geoms on
                # the legs/feet (bodies passed through each other). contype/conaffinity bit 4
                # only: collides with the other ducks' proxies, not with its own parts, the
                # ball or the world. Massless (the trunk has an explicit <inertial>).
                trunk.add_geom(name="body_proxy", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.075, 0, 0],
                               fromto=[0.02, 0, -0.03, 0.02, 0, 0.12], contype=4, conaffinity=4,
                               group=3, rgba=[1, 1, 1, 0.2], mass=0.0)
                frame = spec.worldbody.add_frame(pos=[-0.5 - i, 0, 0] if i < 2 else [0.5 + i, 0, 0])
                frame.attach_body(trunk, p, "")
            names = _bamify(spec, force_limit)
            self.model = spec.compile()
            self.model.opt.timestep = 0.005
            self.data = mujoco.MjData(self.model)
            for p in PREFIXES:          # team markers: the robot's default class gives them a material
                g = self.model.geom(p + "bib").id
                self.model.geom_matid[g] = -1
                self.model.geom_rgba[g] = TEAM_RGBA[0 if p.startswith("b") else 1]
            gid = self.model.geom("ball_geom").id
            self.model.geom_friction[gid, 2] = rolling
            self.ducks = []
            for i, p in enumerate(PREFIXES):
                mine = [n for n in names if n.startswith(p)]
                bam = MujocoController(bams[i], mine, self.model, self.data,
                                       vin_drop_gain=vin_drop_gain, vin_min=ip.BAM_VIN_MIN)
                pol = _Policies(self.model, self.data, bam_ctrl=bam,
                                walking_onnx_path=str(POLICY_DIR / "alpha_walking.onnx"),
                                standing_onnx_path=str(POLICY_DIR / "alpha_stand.onnx"),
                                kick_left_onnx_path=str(POLICY_DIR / "ball_kick_left.onnx"),
                                kick_right_onnx_path=str(POLICY_DIR / "ball_kick_right.onnx"),
                                new_cmd_obs=True, use_projected_gravity=True, kick_duration=kick_duration)
                self._repoint(pol, p, mine)
                self.ducks.append(Duck(self, i, p, pol, bam))
        self._model_defaults = {f: getattr(self.model, f).copy()
                                for f in ("dof_frictionloss", "dof_damping", "dof_armature")}
        self.dt = ds.DECIMATION * self.model.opt.timestep
        self.rng = np.random.default_rng(seed)
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
        self.ball_adr = int(self.model.jnt_qposadr[jid])
        self.ball_vadr = int(self.model.jnt_dofadr[jid])
        for d in self.ducks:          # head cameras face backwards in the model (DuckSim fix)
            fixed = np.zeros(4)
            mujoco.mju_mulQuat(fixed, self.model.cam_quat[d.cam_id], np.array([0.0, 1.0, 0.0, 0.0]))
            self.model.cam_quat[d.cam_id] = fixed
            self.model.cam_fovy[d.cam_id] = ds.CAM_FOVY
        self._renderers = {}
        self._overview_cam = None
        self.t = 0.0

    def _repoint(self, pol, prefix, act_names):
        """PolicyInference looked everything up by the single-duck names at init: point it at
        this duck's prefixed sensors / bodies / 14 actuated joints instead."""
        m = self.model
        pol.imu_ang_vel_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SENSOR, prefix + "imu_ang_vel")
        pol.trunk_base_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, prefix + "trunk_base")
        jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, prefix + "trunk_base_freejoint")
        pol._trunk_qpos_adr = int(m.jnt_qposadr[jid])
        ids = [m.actuator(n).id for n in act_names]
        pol.n_joints = len(ids)
        pol.joint_qpos_indices = [int(m.jnt_qposadr[m.actuator_trnid[i, 0]]) for i in ids]
        pol.joint_qvel_indices = [int(m.jnt_dofadr[m.actuator_trnid[i, 0]]) for i in ids]
        pol.default_pose = ip.DEFAULT_POSE[:pol.n_joints].copy()
        pol.last_action = np.zeros(pol.n_joints, np.float32)
        if getattr(pol, "use_delay", False):
            pol.action_buffer = [np.zeros(pol.n_joints, np.float32) for _ in pol.action_buffer]
        assert pol.imu_ang_vel_id >= 0 and pol.trunk_base_id >= 0 and pol.n_joints == 14

    def _q(self):
        return contextlib.nullcontext() if self.verbose else contextlib.redirect_stdout(io.StringIO())

    # -- world ---------------------------------------------------------------------
    def ball_pos(self):
        return self.data.qpos[self.ball_adr:self.ball_adr + 3].copy()

    def ball_vel(self):
        return self.data.qvel[self.ball_vadr:self.ball_vadr + 3].copy()

    def goal_scored(self):
        """0 if blue scored (ball in the +x goal), 1 if red scored, else None."""
        b = self.ball_pos()
        if abs(b[1]) < GOAL_HALF_WIDTH and b[2] < GOAL_HEIGHT:
            if b[0] > GOAL_X + BALL_R:
                return 0
            if b[0] < -GOAL_X - BALL_R:
                return 1
        return None

    # -- reset -----------------------------------------------------------------------
    def sample_kickoff(self, rng=None, spread=1.0):
        """Ducks in their own half, ball near the centre. spread in [0, 1] scales the
        randomisation (0 = fixed formation). Returns [(x, y, yaw)] x4, (bx, by)."""
        rng = rng or self.rng
        poses = []
        for d in self.ducks:
            k = d.idx % 2
            base = (-0.45, 0.35) if k == 0 else (-1.1, -0.3)
            x = base[0] + spread * rng.uniform(-0.2, 0.2)
            y = base[1] * (1 if rng.random() < 0.5 else -1) + spread * rng.uniform(-0.2, 0.2) if spread else base[1]
            yaw = spread * rng.uniform(-0.6, 0.6)
            poses.append((d.sign * x, d.sign * y, yaw if d.sign > 0 else wrap(math.pi + yaw)))
        ball = (spread * rng.uniform(-0.15, 0.15), spread * rng.uniform(-0.3, 0.3))
        # keep ducks apart
        for i in range(4):
            for j in range(i):
                if math.hypot(poses[i][0] - poses[j][0], poses[i][1] - poses[j][1]) < 0.35:
                    x, y, yaw = poses[i]
                    poses[i] = (x, y + (0.4 if y <= poses[j][1] else -0.4), yaw)
        return poses, ball

    def reset(self, poses=None, ball=None, settle_s=1.0):
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        for f, v in self._model_defaults.items():
            getattr(m, f)[:] = v
        if poses is None:
            poses, ball = self.sample_kickoff()
        for duck, (x, y, yaw) in zip(self.ducks, poses):
            a = duck.trunk_adr
            d.qpos[a:a + 3] = [x, y, 0.125]
            d.qpos[a + 3:a + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
            pol = duck.policy
            for i, qi in enumerate(pol.joint_qpos_indices):
                d.qpos[qi] = pol.default_pose[i]
        d.qpos[self.ball_adr:self.ball_adr + 7] = [ball[0], ball[1], BALL_R, 1, 0, 0, 0]
        d.qvel[:] = 0.0
        for duck in self.ducks:
            duck.bam.reset(d.qpos)
            duck.bam.last_ts = d.time
            pol = duck.policy
            pol.last_action[:] = 0.0
            pol.behavior_mode = None
            pol.head_offset[:] = 0.0
            pol.set_position_targets(pol.default_pose)
            duck.macro, duck.n_kicks, duck.n_falls = None, 0, 0
        mujoco.mj_forward(m, d)
        self.t = 0.0
        for duck in self.ducks:
            duck.stand_since, duck.primitive = 0.0, ds.P["stand"]
            duck._prim(ds.P["stand"])
        for _ in range(int(settle_s * ds.CONTROL_HZ)):
            self.step()

    # -- step ------------------------------------------------------------------------
    def step(self):
        self.t += self.dt
        for duck in self.ducks:
            duck.control_step()
        for _ in range(ds.DECIMATION):
            for duck in self.ducks:
                duck.bam.update()
            mujoco.mj_step(self.model, self.data)

    # -- rendering -------------------------------------------------------------------
    def _renderer(self, h, w):
        if (h, w) not in self._renderers:
            self._renderers[(h, w)] = mujoco.Renderer(self.model, height=h, width=w)
        return self._renderers[(h, w)]

    def render_overview(self, height=480, width=800, distance=3.7, elevation=-62):
        if self._overview_cam is None:
            self._overview_cam = mujoco.MjvCamera()
        cam = self._overview_cam
        cam.lookat[:] = (0.0, 0.0, 0.0)
        cam.distance, cam.azimuth, cam.elevation = distance, 90, elevation
        r = self._renderer(height, width)
        r.update_scene(self.data, camera=cam)
        return r.render().copy()

    def render_head(self, idx: int):
        r = self._renderer(ds.IMG_W, ds.IMG_H)
        r.update_scene(self.data, camera=self.ducks[idx].cam_id)
        return np.ascontiguousarray(np.rot90(r.render(), k=-1))
