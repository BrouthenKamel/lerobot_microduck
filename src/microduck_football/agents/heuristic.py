"""Hand-coded football player for FootballEnv: uses only the env's observation and primitives.

A baseline to check that the primitives + physics can play football at all (and later a
reference for the RL agent). It always kicks with the right foot: its kicks curl ~13° to the
right (measured), so it lines up ~13° left of the goal.

Close to the goal it dribbles instead (walks through the ball along the aim line): kicks only
carry ~0.5 m on this turf, and pushing the ball while walking is more reliable near the goal.

States
  APPROACH  get to a staging point behind the ball on the (aim) line, detouring around the ball
  FINAL     walk in along the line, strafing to stay on it; stand when close enough
  SETTLE    stand still >= 1 s (kicks only work from a settled stand), then check the ball:
            in the kick window -> kick; a bit too far -> one short walk; otherwise re-approach
  BACKOFF   step back a little, then APPROACH again
"""

from __future__ import annotations

import math

import numpy as np

from microduck_football.sim import duck_sim as ds
from microduck_football.envs.football_env import OBS_SLICES

AIM_OFFSET = math.radians(13)   # right-foot kicks curl ~13° right: aim that much left
SX, SY = ds.KICK_SPOT["kick_right"]
STAGING = 0.40                  # m behind the kick position (room to turn: radius ~15 cm)
ON_LINE = 0.10                  # m lateral tolerance to switch to the final approach
STOP_AT = 0.03                  # m before the kick position: stand (braking covers the rest)
SETTLE_S = 1.0
KICK_DX = (-0.04, 0.012)        # ball offset from the sweet spot that still gets kicked (measured)
KICK_DY = 0.03
TURN = math.radians(20)
AIM_DEPTH = 0.15                # aim at a point this far INSIDE the goal (not at the line centre)
AIM_SPREAD = 0.12               # ... shifted toward the ball's side by up to this much
DRIBBLE_DIST = 0.6              # ball this close to the aim point: walk through it, don't kick


def _unit(v):
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def _rot(v, a):
    c, s = math.cos(a), math.sin(a)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]])


class HeuristicPlayer:
    def __init__(self):
        self.reset()

    def reset(self):
        self.state = "APPROACH"
        self.backoff_left = 0
        self.nudges = 0
        self.dribble = False

    # -- geometry in the duck frame ------------------------------------------
    def _plan(self, obs):
        b = obs[OBS_SLICES["ball_pos"]]
        pl, pr = obs[OBS_SLICES["post_left"]], obs[OBS_SLICES["post_right"]]
        g = 0.5 * (pl + pr)
        t = _unit(pl - pr)                          # along the goal line
        inward = _rot(t, -math.pi / 2)              # into the goal
        aim = g + inward * AIM_DEPTH + t * float(np.clip((b - g) @ t, -AIM_SPREAD, AIM_SPREAD))
        self.dribble = float(np.linalg.norm(aim - b)) < DRIBBLE_DIST
        offset = 0.0 if self.dribble else AIM_OFFSET
        h = _rot(_unit(aim - b), offset)            # desired heading at the kick
        n = _rot(h, math.pi / 2)                    # its left normal
        k = b - h * SX - n * SY                     # where the duck's centre must be to kick
        a = float(k @ h)                            # how far the duck still is behind k (along h)
        e = float(-(k @ n))                         # duck's lateral offset from the line (left +)
        return b, h, n, k, a, e

    def _clamp_to_pitch(self, obs, p):
        """Keep targets ~12 cm inside the walls (duck frame in, duck frame out)."""
        sy, cy = obs[OBS_SLICES["duck_heading"]]
        xy = obs[OBS_SLICES["duck_xy"]]
        w = xy + np.array([cy * p[0] - sy * p[1], sy * p[0] + cy * p[1]])
        w[0] = np.clip(w[0], ds.PITCH_X[0] + 0.12, ds.PITCH_X[1] - 0.12)
        w[1] = np.clip(w[1], ds.PITCH_Y[0] + 0.12, ds.PITCH_Y[1] - 0.12)
        d = w - xy
        return np.array([cy * d[0] + sy * d[1], -sy * d[0] + cy * d[1]])

    @staticmethod
    def _steer(target, allow_strafe_err=None):
        ang = math.atan2(target[1], target[0])
        if ang > TURN:
            return ds.P["walk_left"]
        if ang < -TURN:
            return ds.P["walk_right"]
        if allow_strafe_err is not None and abs(allow_strafe_err) > 0.02:
            return ds.P["strafe_right"] if allow_strafe_err > 0 else ds.P["strafe_left"]
        return ds.P["walk"]

    # -- policy -----------------------------------------------------------------
    def act(self, obs) -> int:
        grav_z = float(obs[OBS_SLICES["gravity"]][2])
        if grav_z > -0.5:                       # fallen: the stand policy gets it up
            self.state = "APPROACH"
            return ds.P["stand"]

        b, h, n, k, a, e = self._plan(obs)

        if self.state == "BACKOFF":
            self.backoff_left -= 1
            if self.backoff_left <= 0:
                self.state = "APPROACH"
            return ds.P["back"]

        if self.state == "SETTLE":
            still = float(obs[OBS_SLICES["stand_time"]][0])
            speed = float(np.linalg.norm(obs[OBS_SLICES["duck_vel"]]))
            if still < SETTLE_S or speed > 0.03:
                return ds.P["stand"]
            dxo, dyo = b[0] - SX, b[1] - SY
            if KICK_DX[0] <= dxo <= KICK_DX[1] and abs(dyo) <= KICK_DY:
                self.state, self.nudges = "APPROACH", 0
                return ds.P["kick_right"]
            if KICK_DX[1] < dxo < 0.10 and abs(dyo) <= 0.05 and self.nudges < 3:
                self.nudges += 1                # a little too far: one short step, settle again
                self.state = "NUDGE"
                return ds.P["walk"]
            self.state, self.backoff_left, self.nudges = "BACKOFF", 3, 0
            return ds.P["back"]

        if self.state == "NUDGE":
            self.state = "SETTLE"
            return ds.P["stand"]

        if self.state == "FINAL":
            if self.dribble and abs(e) < 0.12 and a > -0.10:
                proj = k - h * a                    # keep walking along the line, through the ball
                return self._steer(proj + h * 0.15, allow_strafe_err=e)
            if a <= STOP_AT:
                self.state = "SETTLE"
                return ds.P["stand"]
            if abs(e) > 0.12 or a > STAGING + 0.15:
                self.state = "APPROACH"
            else:
                proj = k - h * a
                carrot = proj + h * min(0.15, a)
                return self._steer(carrot, allow_strafe_err=e)

        # APPROACH
        facing = abs(math.atan2(h[1], h[0])) < math.radians(60)
        if a > 0.0 and abs(e) < ON_LINE and facing:   # already behind the ball, on the line
            self.state = "FINAL"
            return self._steer(k - h * a + h * min(0.15, a), allow_strafe_err=e)
        staging = self._clamp_to_pitch(obs, k - h * STAGING)
        if np.linalg.norm(staging) < 0.15 and a > 0.0:
            # At the staging point but facing the wrong way: the duck can only turn while walking
            # (radius ~15 cm), so turn toward the aim direction instead of circling the point.
            return ds.P["walk_left"] if math.atan2(h[1], h[0]) > 0 else ds.P["walk_right"]
        target = staging
        # Detour if the straight path to the staging point passes close to the ball.
        seg = staging
        t = float(np.clip((b @ seg) / max(seg @ seg, 1e-9), 0.0, 1.0))
        if np.linalg.norm(b - seg * t) < 0.15 and np.linalg.norm(b) < np.linalg.norm(staging):
            side = n if e > 0 else -n
            target = self._clamp_to_pitch(obs, b + side * 0.25)
        return self._steer(target)
