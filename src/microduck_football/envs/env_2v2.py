"""2v2 football environment on MultiDuckSim, for centralised training / decentralised execution.

Every duck observes the game in its TEAM FRAME (the world rotated 180 deg for red), so one shared
policy plays both sides and always attacks +x.

Actor obs (OBS_DIM = 50), per duck:
  [0:29]  EXACTLY the single-duck teacher's layout (football_env.py OBS_FIELDS), computed in the team frame:
          ball rel (2), ball vel (2), attacking goal posts rel (2+2), x/y (2), sin/cos yaw (2),
          trunk vel (2), yaw rate, gravity (3), stand time, kick time left, prev primitive one-hot (9)
          -> the open-play teacher warm-starts every duck (train_teacher.widen_from).
  [29:50] aimed kick in progress, prev action was aimed kick, own goal posts rel (4),
          teammate rel pos/vel + fallen (5), two opponents nearest-first rel pos/vel + fallen (10).
Critic state (STATE_DIM = OBS_DIM + 36): the duck's own obs + global team-frame state
  (ball pos/vel, then ego, teammate, opponents: x, y, sin, cos, vx, vy, fallen, busy).

Actions: 10 (multi_sim.ACTION_NAMES: 9 primitives + aimed_kick), 5 Hz; ignored while busy.

Rewards (per duck; zero-sum between teams except the individual terms):
  goal +-R_GOAL (team), ball potential W_BALL * delta[(|b - own goal| - |b - opp goal|) / 2] (team),
  approach W_APPROACH * delta[-max(d_ball - 0.2, 0)] for the team's duck closest to the ball only
  (the other one is free to position), fall R_FALL, time R_TIME (individual).
  draw R_DRAW for everyone when MAX_SECONDS runs out without a goal.
Episode: ends on a goal or after MAX_SECONDS (a real end of game: no bootstrapping past it).
"""

from __future__ import annotations

import math

import numpy as np

from microduck_football.sim import duck_sim as ds
from microduck_football.sim import multi_sim as ms

OBS_DIM = 50
TEACHER_DIM = 29
STATE_DIM = OBS_DIM + 4 + 4 * 8
N_ACTIONS = ms.N_ACTIONS

DECISION_EVERY = 10          # control steps per decision (50 Hz / 10 = 5 Hz, as the teacher)
MAX_SECONDS = 60.0
R_GOAL = 10.0
W_BALL = 2.0
W_APPROACH = 0.5
R_FALL = -1.0
R_TIME = -0.005
R_DRAW = -3.0                # both teams, when the time runs out without a goal (v2 drifted to passive draws)


class Env2v2:
    def __init__(self, seed: int = 0, max_seconds: float = MAX_SECONDS, spread: float = 1.0, **sim_kw):
        self.sim = ms.MultiDuckSim(seed=seed, **sim_kw)
        self.rng = np.random.default_rng(seed)
        self.max_seconds = max_seconds
        self.spread = spread
        self.prev_action = [ds.P["stand"]] * 4
        self.on_control_step = None

    # -- team-frame helpers ------------------------------------------------------------
    @staticmethod
    def _tf(p, sign):
        return np.asarray(p, np.float64)[:2] * sign

    def _goal_posts(self, sign, attacking=True):
        gx = ms.GOAL_X * (1 if attacking else -1)
        # team frame (gx, +hw), (gx, -hw)  -> world = sign * team
        return [np.array([gx, ms.GOAL_HALF_WIDTH]) * sign, np.array([gx, -ms.GOAL_HALF_WIDTH]) * sign]

    def _duck_feats(self, ego, other):
        rel = ego.to_frame(other.xy())
        vel = ego.to_frame(other.vel_xy(), is_vector=True)
        return [rel[0], rel[1], vel[0], vel[1], float(other.fallen())]

    def obs(self, i: int) -> np.ndarray:
        s, d = self.sim, self.sim.ducks[i]
        x, y, yaw = d.pose()
        sign = d.sign
        tx, ty = sign * x, sign * y
        tyaw = yaw if sign > 0 else ms.wrap(yaw + math.pi)
        onehot = np.zeros(len(ds.PRIMITIVES), np.float32)
        pa = self.prev_action[i]
        if pa < len(ds.PRIMITIVES):
            onehot[pa] = 1.0
        att = self._goal_posts(sign, True)
        own = self._goal_posts(sign, False)
        mates = [o for o in s.ducks if o.team == d.team and o is not d]
        opps = sorted([o for o in s.ducks if o.team != d.team],
                      key=lambda o: float(np.linalg.norm(o.xy() - d.xy())))
        parts = [
            d.ball_rel(),
            d.to_frame(s.ball_vel()[:2], is_vector=True),
            d.to_frame(att[0]), d.to_frame(att[1]),
            [tx, ty], [math.sin(tyaw), math.cos(tyaw)],
            d.to_frame(d.vel_xy(), is_vector=True), [d.yaw_rate()],
            d.gravity(), [min(d.stand_time(), 3.0)], [d.kick_time_left()],
            onehot,
            # extras
            [float(d.macro is not None), float(pa == ms.AIMED_KICK)],
            d.to_frame(own[0]), d.to_frame(own[1]),
            self._duck_feats(d, mates[0]),
            self._duck_feats(d, opps[0]), self._duck_feats(d, opps[1]),
        ]
        o = np.concatenate([np.asarray(p, np.float32).ravel() for p in parts])
        assert o.shape == (OBS_DIM,)
        return o

    def state(self, i: int) -> np.ndarray:
        """Centralised critic input for duck i: its obs + the global state in its team frame."""
        s, d = self.sim, self.sim.ducks[i]
        sign = d.sign
        b, bv = s.ball_pos()[:2] * sign, s.ball_vel()[:2] * sign
        order = [d] + [o for o in s.ducks if o.team == d.team and o is not d] + \
            sorted([o for o in s.ducks if o.team != d.team], key=lambda o: o.idx)
        g = [b, bv]
        for o in order:
            x, y, yaw = o.pose()
            tyaw = yaw if sign > 0 else ms.wrap(yaw + math.pi)
            g.append([sign * x, sign * y, math.sin(tyaw), math.cos(tyaw), *(o.vel_xy() * sign),
                      float(o.fallen()), float(o.busy())])
        return np.concatenate([self.obs(i)] + [np.asarray(p, np.float32).ravel() for p in g])

    def all_obs(self):
        return np.stack([self.obs(i) for i in range(4)])

    def all_states(self):
        return np.stack([self.state(i) for i in range(4)])

    # -- potentials ------------------------------------------------------------------------
    def _phi_ball(self, team):
        b = self.sim.ball_pos()[:2]
        sign = 1.0 if team == 0 else -1.0
        opp_goal = np.array([ms.GOAL_X * sign, 0.0])
        own_goal = -opp_goal
        return 0.5 * (float(np.linalg.norm(b - own_goal)) - float(np.linalg.norm(b - opp_goal)))

    def _approach(self):
        """per duck: -max(d_ball - 0.2, 0), and whether it is its team's closest duck."""
        b = self.sim.ball_pos()[:2]
        dist = [float(np.linalg.norm(d.xy() - b)) for d in self.sim.ducks]
        closest = [dist[i] <= min(dist[j] for j in range(4) if self.sim.ducks[j].team == self.sim.ducks[i].team)
                   for i in range(4)]
        return [-max(x - 0.2, 0.0) for x in dist], closest

    # -- gym-ish API ---------------------------------------------------------------------------
    def reset(self, seed: int | None = None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
            self.sim.rng = np.random.default_rng(seed + 1)
        poses, ball = self.sim.sample_kickoff(self.rng, spread=self.spread)
        self.sim.reset(poses, ball)
        self.prev_action = [ds.P["stand"]] * 4
        self._phi = [self._phi_ball(0), self._phi_ball(1)]
        self._appr, _ = self._approach()
        self._fallen = [d.fallen() for d in self.sim.ducks]
        self.stats = {"n_falls": [0] * 4, "n_kicks": [0] * 4, "n_aimed": [0] * 4, "touches": [0] * 4}
        return self.all_obs(), self.all_states()

    def step(self, actions):
        s = self.sim
        for i, a in enumerate(actions):
            d = s.ducks[i]
            if not d.busy():
                if int(a) == ms.AIMED_KICK:
                    self.stats["n_aimed"][i] += 1
                d.act(int(a))
                self.prev_action[i] = int(a)
        scored = None
        for _ in range(DECISION_EVERY):
            s.step()
            if self.on_control_step:
                self.on_control_step(self)
            scored = s.goal_scored()
            if scored is not None:
                break
        rew = np.full(4, R_TIME, np.float32)
        phi = [self._phi_ball(0), self._phi_ball(1)]
        appr, closest = self._approach()
        for i, d in enumerate(s.ducks):
            rew[i] += W_BALL * (phi[d.team] - self._phi[d.team])
            if closest[i]:
                rew[i] += W_APPROACH * (appr[i] - self._appr[i])
            f = d.fallen()
            if f and not self._fallen[i]:
                rew[i] += R_FALL
                self.stats["n_falls"][i] += 1
            self._fallen[i] = f
            if scored is not None:
                rew[i] += R_GOAL if scored == d.team else -R_GOAL
            self.stats["n_kicks"][i] = d.n_kicks
        self._phi, self._appr = phi, appr
        terminated = scored is not None
        truncated = s.t >= self.max_seconds
        if truncated and not terminated:
            rew += R_DRAW
        info = {"scored": scored, "sim_time": s.t, **{k: list(v) for k, v in self.stats.items()}}
        return self.all_obs(), self.all_states(), rew, terminated, truncated, info
