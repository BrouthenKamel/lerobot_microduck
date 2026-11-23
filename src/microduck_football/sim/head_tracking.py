"""Ball-tracking head reflex: keeps the ball in the head camera by commanding head yaw/pitch.

The walk / stand policies take a head-pose command (obs slots [neck_pitch, head_pitch, head_yaw,
head_roll]) and move the head themselves; this reflex only chooses that command. Two "eyes"
decide where the ball is:

  privileged  project the simulator's true ball position into the camera, if it is inside the
              image AND in line of sight (a ray from the camera must reach the ball first) - a
              "perfect detector" that works for any object and respects occlusion (default)
  detector    find the orange ball in the rendered camera image (what a real duck could do)

Both return the ball's angular offset from the image centre, so the controller is identical.
Measured: a +head_yaw command moves the ball RIGHT in the image (head turns left); +head_pitch
moves it UP (head looks down). The two eyes agree to ~1 px when the ball is visible.
"""

from __future__ import annotations

import math

import numpy as np

import mujoco

from microduck_football.sim import duck_sim as ds

F_PX = (ds.IMG_W / 2) / math.tan(math.radians(ds.CAM_FOVY) / 2)   # 160 px for FOV 90° / 320 px
CX, CY = ds.IMG_W / 2, ds.IMG_H / 2

YAW_LIM = 1.4        # rad, training range of the head_yaw command
PITCH_LIM = (-0.3, 1.0)   # rad, head_pitch command (positive = look down); training cap ±1.1
GAIN = 0.5           # fraction of the angular error corrected per update
MAX_STEP = 0.15      # rad per update (10 Hz -> 1.5 rad/s): the head is heavy, keep it gentle
SEARCH_RATE = 0.8    # rad/s head-yaw sweep while the ball is not seen
SEARCH_PITCH = 0.35  # rad, head pitch while searching (slightly down: the ball is on the floor)
UPDATE_EVERY = 5     # control steps (50 Hz / 5 = 10 Hz, like a camera pipeline)

# Glances (football-player gaze): while the ball is close, look at the goal now and then.
GLANCE_EVERY = 4.0        # s between glances (tuned: ball in view 79%, goal 40%)
GLANCE_HOLD = 0.4         # s looking at the goal
GLANCE_BALL_DIST = 0.8    # m: only glance when the ball is this close (it is "at the feet")
GLANCE_MAX_STEP = 0.35    # rad per update (3.5 rad/s): a glance is a quick head turn
GLANCE_PITCH = 0.1        # rad: near level, the goal is far
REACQUIRE_PITCH = 0.5     # rad: looking down toward the ball while turning back to it
GOAL_AIM = np.array([ds.GOAL_X, 0.0])


def detect_ball(img: np.ndarray):
    """Orange-blob detector. Returns (col, row, n_pixels) of the largest-area orange mask or None."""
    r = img[..., 0].astype(np.int16)
    g = img[..., 1].astype(np.int16)
    b = img[..., 2].astype(np.int16)
    mask = (r > 120) & (b < 90) & (g > 40) & (r - g > 40)
    n = int(mask.sum())
    if n < 6:
        return None
    rows, cols = np.nonzero(mask)
    return float(cols.mean()), float(rows.mean()), n


_VISIBLE_GROUPS = np.array([1, 1, 1, 0, 0, 0], dtype=np.uint8)   # rendered geom groups only


def line_of_sight(sim: ds.DuckSim, target_geom: str = "ball_geom") -> bool:
    """True if any part of ``target_geom`` (a sphere) is visible from the head camera: rays to its
    centre and to 4 points on its silhouette (up/down/left/right as seen from the camera); visible
    if at least one of them hits the ball before any other rendered geom. (A single ray to the
    centre calls a ball "hidden" when its top still peeks over a wall.)"""
    m, d = sim.model, sim.data
    gid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, target_geom)
    start = d.cam_xpos[sim.cam_id].copy()
    centre = d.geom_xpos[gid]
    radius = float(m.geom_size[gid][0]) * 0.85
    view = (centre - start) / max(np.linalg.norm(centre - start), 1e-9)
    side = np.cross(view, [0.0, 0.0, 1.0])
    side = side / max(np.linalg.norm(side), 1e-9)
    up = np.cross(side, view)
    hit = np.array([-1], dtype=np.int32)
    head_body = int(m.cam_bodyid[sim.cam_id])     # the camera sits inside the head: skip it
    for off in (np.zeros(3), up, -up, side, -side):
        mujoco.mj_ray(m, d, start, centre + radius * off - start, _VISIBLE_GROUPS, 1, head_body, hit)
        if int(hit[0]) == gid:
            return True
    return False


def project_ball(sim: ds.DuckSim, occlusion: bool = True):
    """True ball position -> (col, row) in the upright head image, or None if it is outside the
    view or (with occlusion=True) hidden behind something."""
    pos = sim.data.cam_xpos[sim.cam_id]
    rot = sim.data.cam_xmat[sim.cam_id].reshape(3, 3)
    pc = rot.T @ (sim.ball_pos() - pos)
    depth = -pc[2]                       # MuJoCo cameras look along -z
    if depth <= 0.02:
        return None
    # rendered portrait (x right, y up), then rotated 90° clockwise into the upright image
    col = (CX - 1) + F_PX * pc[1] / depth
    row = CY + F_PX * pc[0] / depth
    if not (0 <= col < ds.IMG_W and 0 <= row < ds.IMG_H):
        return None
    if occlusion and not line_of_sight(sim):
        return None
    return col, row


GLANCE_ONLY_IF_UNSEEN = True   # skip a glance when the goal is already in the frame


def goal_in_view(sim: ds.DuckSim) -> bool:
    """Goal mouth centre (on the line, mid-height) inside the head image."""
    pos = sim.data.cam_xpos[sim.cam_id]
    rot = sim.data.cam_xmat[sim.cam_id].reshape(3, 3)
    pc = rot.T @ (np.array([ds.GOAL_X, 0.0, 0.15]) - pos)
    if -pc[2] <= 0.02:
        return False
    col = (CX - 1) + F_PX * pc[1] / -pc[2]
    row = CY + F_PX * pc[0] / -pc[2]
    return 0 <= col < ds.IMG_W and 0 <= row < ds.IMG_H


def bearing(sim: ds.DuckSim, xy) -> float:
    """Angle of a world xy point from the duck's heading (left +), radians."""
    p = sim.to_duck_frame(xy)
    return math.atan2(p[1], p[0])


class BallTracker:
    """Head reflex: keeps the ball in view; with glance=True also looks at the goal now and then
    (only when the ball is close, the goal is within the head's yaw range, and no kick is on).
    When the ball is out of view after a glance it turns back toward where the ball is, like a
    player's short-term memory, instead of sweeping blindly."""

    def __init__(self, sim: ds.DuckSim, eye: str = "privileged", glance: bool = False):
        assert eye in ("privileged", "detector")
        self.sim, self.eye, self.glance = sim, eye, glance
        self.reset()

    def reset(self):
        self.yaw, self.pitch = 0.0, 0.0
        self.mode = "ball"               # ball | to_goal | hold_goal | to_ball
        self.mode_since = 0.0
        self.last_glance = 0.0
        self.search_dir = 1.0
        self._n = 0
        self.seen = False
        self.frames = 0
        self.frames_seen = 0
        self._apply()

    def _apply(self):
        pol = self.sim.policy
        pol.head_offset[:] = [0.0, self.pitch, self.yaw, 0.0]
        pol._update_command()            # kicks override with their own all-zero command

    def ball_offset(self):
        """Angular offset of the ball from the image centre: (left +, down +) radians, or None."""
        if self.eye == "privileged":
            p = project_ball(self.sim)
        else:
            d = detect_ball(self.sim.render_head())
            p = None if d is None else d[:2]
        if p is None:
            return None
        col, row = p
        return math.atan((CX - col) / F_PX), math.atan((row - CY) / F_PX)

    def on_control_step(self):
        self._n += 1
        if self._n % UPDATE_EVERY:
            return
        dt = UPDATE_EVERY * self.sim.dt
        sim = self.sim
        if sim.policy.behavior_mode is not None:
            # kicks only work with a level head: hold it level while a kick runs
            self.yaw, self.pitch = 0.0, 0.0
            self.mode = "ball"
            self._apply()
            return
        off = self.ball_offset()
        self.frames += 1
        self.seen = off is not None
        if self.glance and self._glance_step(off):
            if off is not None:
                self.frames_seen += 1
            self._apply()
            return
        if off is not None:
            self.frames_seen += 1
            dyaw = float(np.clip(GAIN * off[0], -MAX_STEP, MAX_STEP))
            dpitch = float(np.clip(GAIN * off[1], -MAX_STEP, MAX_STEP))
            self.yaw = float(np.clip(self.yaw + dyaw, -YAW_LIM, YAW_LIM))
            self.pitch = float(np.clip(self.pitch + dpitch, *PITCH_LIM))
        else:
            # search: sweep the head left-right, slightly looking down
            self.yaw += self.search_dir * SEARCH_RATE * dt
            if abs(self.yaw) >= YAW_LIM:
                self.yaw = math.copysign(YAW_LIM, self.yaw)
                self.search_dir *= -1
            self.pitch += float(np.clip(SEARCH_PITCH - self.pitch, -MAX_STEP, MAX_STEP))
        self._apply()

    # -- glances ------------------------------------------------------------------
    def _move_to(self, yaw, pitch, step):
        self.yaw += float(np.clip(np.clip(yaw, -YAW_LIM, YAW_LIM) - self.yaw, -step, step))
        self.pitch += float(np.clip(np.clip(pitch, *PITCH_LIM) - self.pitch, -step, step))
        return abs(self.yaw - np.clip(yaw, -YAW_LIM, YAW_LIM)) < 0.08

    def _glance_step(self, off) -> bool:
        """Runs the glance state machine. Returns True if it moved the head this update
        (otherwise the normal ball tracking runs)."""
        sim, t = self.sim, self.sim.t
        if self.mode == "ball":
            ball_dist = float(np.linalg.norm(sim.ball_in_duck_frame()))
            goal_b = bearing(sim, GOAL_AIM)
            if (off is not None and t - self.last_glance >= GLANCE_EVERY
                    and ball_dist < GLANCE_BALL_DIST and abs(goal_b) < YAW_LIM
                    and not (GLANCE_ONLY_IF_UNSEEN and goal_in_view(sim))):
                self.mode, self.mode_since = "to_goal", t
                self.pre_glance = (self.yaw, self.pitch, bearing(sim, sim.ball_pos()[:2]))
            else:
                return False
        if self.mode == "to_goal":
            if self._move_to(bearing(sim, GOAL_AIM), GLANCE_PITCH, GLANCE_MAX_STEP) or t - self.mode_since > 1.0:
                self.mode, self.mode_since = "hold_goal", t
            return True
        if self.mode == "hold_goal":
            self._move_to(bearing(sim, GOAL_AIM), GLANCE_PITCH, GLANCE_MAX_STEP)
            if t - self.mode_since >= GLANCE_HOLD:
                self.mode, self.mode_since, self.last_glance = "to_ball", t, t
            return True
        # to_ball: return to the head pose used just before the glance (corrected for how much
        # the ball's bearing changed meanwhile), then hand back to tracking once it is seen
        if off is not None or t - self.mode_since > 1.0:
            self.mode = "ball"
            return False
        yaw0, pitch0, b0 = self.pre_glance
        self._move_to(yaw0 + (bearing(sim, sim.ball_pos()[:2]) - b0), pitch0, GLANCE_MAX_STEP)
        return True
