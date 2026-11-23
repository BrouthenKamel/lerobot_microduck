"""Gymnasium env: microduck football driven by discrete primitives (RL-ready).

Action  : Discrete(9) - one of duck_sim.PRIMITIVES. EVERY decision lasts 0.2 s, kicks included:
          a kick starts the 0.5 s kick policy, and while it runs the agent's actions are ignored
          (obs "kick_time_left" says so). Uniform decisions keep the time cost and the discount
          per decision honest - with a 0.6 s macro-kick priced like a 0.2 s step, the first
          agent learned to stand and spam kicks (~100 per episode) as the cheapest way to wait.
Obs     : privileged state in the duck's yaw frame (see OBS_FIELDS), float32.
Reward  : potential-based progress (ball -> goal, duck -> ball), +10 goal, -1 per fall, small
          time cost.
Episode : ends on a goal; truncated after max_seconds of sim time.
"""

from __future__ import annotations

from typing import Callable

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from microduck_football.sim import duck_sim as ds

DECISION_HZ = 5
STEPS_PER_DECISION = ds.CONTROL_HZ // DECISION_HZ          # 10 control steps = 0.2 s

# Observation layout: name -> size (in order).
OBS_FIELDS = [
    ("ball_pos", 2),        # ball xy, duck frame (m)
    ("ball_vel", 2),        # ball velocity, duck frame (m/s)
    ("post_left", 2),       # left goal post (y=+0.25), duck frame
    ("post_right", 2),      # right goal post (y=-0.25), duck frame
    ("duck_xy", 2),         # duck position on the pitch (world, m)
    ("duck_heading", 2),    # sin(yaw), cos(yaw)
    ("duck_vel", 2),        # trunk velocity, duck frame
    ("yaw_rate", 1),
    ("gravity", 3),         # projected gravity in the trunk frame (z ~ -1 upright)
    ("stand_time", 1),      # s commanded to stand (clipped at 3)
    ("kick_time_left", 1),  # s left in a running kick (actions are ignored while > 0)
    ("prev_action", len(ds.PRIMITIVES)),
]
OBS_SLICES = {}
_i = 0
for _name, _n in OBS_FIELDS:
    OBS_SLICES[_name] = slice(_i, _i + _n)
    _i += _n
OBS_DIM = _i

# Reward weights
W_BALL_PROGRESS = 2.0       # per metre the ball gets closer to the goal centre
W_DUCK_PROGRESS = 0.5       # per metre the duck gets closer to the ball (while > 0.2 m away)
R_GOAL = 10.0
R_FALL = -1.0
R_TIME = -0.005             # per decision

# Curriculum (see duck_sim.DuckSim.sample_layout): episode time limit per stage, and the share of
# episodes replayed from earlier stages so the agent does not forget them.
STAGE_SECONDS = {0: 20.0, 1: 40.0, 2: 60.0, 3: 90.0, **{k: 90.0 for k in range(4, 11)}}
N_STAGES = 4                 # open-play curriculum (stages 0-3)
N_STAGES_OBSTACLES = 11      # + stages 4-10 (duck_sim.OBSTACLE_STAGES): on-line defender, wall,
                             #   patrolling defenders, weak / medium / full goalkeeper, keeper + defenders
# Opponent features (obstacles=True), duck frame: N_DEF defender slots (nearest first) + 1 keeper
# slot, each [x, y, vx, vy, present].
OBST_FEATS = 5 * (ds.N_DEF + 1)
REPLAY_EARLIER = 0.25


class FootballEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": DECISION_HZ}

    def __init__(self, difficulty: str = "full", max_seconds: float = 90.0, seed: int | None = None,
                 render_mode: str | None = None, stage: int | None = None,
                 head_tracking: str | None = None, glance: bool = False, action_mode: str = "primitive",
                 obstacles: bool = False):
        """difficulty: "easy" / "full" (fixed). stage: curriculum stage 0-3 instead (overrides
        difficulty and max_seconds; change it during training with set_stage).
        head_tracking: None (head level), "privileged" or "detector" (see head_tracking.py);
        glance: also look at the goal now and then while the ball is close.
        action_mode: "primitive" (Discrete(9)) or "twist" (Box [vx, vy, yaw_rate, kick>0.5]).
        obstacles: append obstacle features to the obs (OBS_DIM + OBST_FEATS); stages 4-6 place them."""
        super().__init__()
        self.sim = ds.DuckSim(seed=seed)
        self.difficulty = difficulty
        self.max_seconds = max_seconds
        self.stage = stage
        self._episode_stage = stage
        self._limit = max_seconds
        self.render_mode = render_mode
        self.action_mode = action_mode
        if action_mode == "twist":
            self.action_space = spaces.Box(np.array([-0.4, -0.3, -1.0, 0.0], np.float32),
                                           np.array([0.4, 0.3, 1.0, 1.0], np.float32))
        else:
            self.action_space = spaces.Discrete(len(ds.PRIMITIVES))
        self.obstacles = obstacles
        self.observation_space = spaces.Box(-np.inf, np.inf, (OBS_DIM + (OBST_FEATS if obstacles else 0),),
                                            np.float32)
        self.on_control_step: Callable[[ds.DuckSim], None] | None = None   # e.g. video capture
        self.tracker = None
        if head_tracking is not None:
            from microduck_football.sim.head_tracking import BallTracker
            self.tracker = BallTracker(self.sim, eye=head_tracking, glance=glance)
        self._prev_action = ds.P["stand"]

    # -- helpers ----------------------------------------------------------------
    def _obs(self) -> np.ndarray:
        s = self.sim
        x, y, yaw = s.duck_pose()
        onehot = np.zeros(len(ds.PRIMITIVES), np.float32)
        onehot[self._prev_action] = 1.0
        parts = [
            s.ball_in_duck_frame(),
            s.to_duck_frame(s.ball_vel()[:2], is_vector=True),
            s.to_duck_frame((ds.GOAL_X, ds.GOAL_HALF_WIDTH)),
            s.to_duck_frame((ds.GOAL_X, -ds.GOAL_HALF_WIDTH)),
            [x, y],
            [np.sin(yaw), np.cos(yaw)],
            s.trunk_vel_duck_frame(),
            [s.yaw_rate()],
            s.gravity(),
            [min(s.stand_time(), 3.0)],
            [s.policy.behavior_time_left if s.policy.behavior_mode is not None else 0.0],
            onehot,
        ]
        if self.obstacles:
            defs, keeper = s.opponent_states()
            feats = np.zeros((ds.N_DEF + 1, 5), np.float32)
            rows = [(s.to_duck_frame(p), s.to_duck_frame(v, is_vector=True)) for p, v in defs]
            rows.sort(key=lambda r: float(np.linalg.norm(r[0])))
            for k, (p, v) in enumerate(rows[:ds.N_DEF]):
                feats[k] = [p[0], p[1], v[0], v[1], 1.0]
            if keeper is not None:
                p, v = s.to_duck_frame(keeper[0]), s.to_duck_frame(keeper[1], is_vector=True)
                feats[ds.N_DEF] = [p[0], p[1], v[0], v[1], 1.0]
            parts.append(feats.ravel())
        return np.concatenate([np.asarray(p, np.float32).ravel() for p in parts])

    def _potentials(self):
        s = self.sim
        b = s.ball_pos()[:2]
        x, y, _ = s.duck_pose()
        phi_ball = -float(np.linalg.norm(b - ds.GOAL_CENTER))
        phi_duck = -max(float(np.hypot(b[0] - x, b[1] - y)) - 0.2, 0.0)
        return phi_ball, phi_duck

    def _info(self, scored: bool) -> dict:
        b = self.sim.ball_pos()[:2]
        return {"is_success": scored, "fallen": self.sim.fallen(), "n_falls": self._n_falls,
                "stage": -1 if self._episode_stage is None else self._episode_stage,
                "n_kicks": self._n_kicks, "sim_time": self.sim.t,
                "ball_dist_to_goal": float(np.linalg.norm(b - ds.GOAL_CENTER))}

    # -- gym API ----------------------------------------------------------------
    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self.sim.rng = np.random.default_rng(seed)
        options = options or {}
        if "difficulty" in options:
            difficulty, self._episode_stage, self._limit = options["difficulty"], None, self.max_seconds
        elif self.stage is not None:
            st = self.stage
            if st > 0 and self.sim.rng.random() < REPLAY_EARLIER:
                st = int(self.sim.rng.integers(0, st))
            difficulty, self._episode_stage, self._limit = st, st, STAGE_SECONDS[st]
        else:
            difficulty, self._episode_stage, self._limit = self.difficulty, None, self.max_seconds
        self.sim.reset(difficulty)
        if self.tracker is not None:
            self.tracker.reset()
        self._prev_action = ds.P["stand"]
        self._phi = self._potentials()
        self._was_fallen = self.sim.fallen()
        self._n_falls = 0
        self._n_kicks = 0
        return self._obs(), self._info(False)

    def step(self, action):
        s = self.sim
        if self.action_mode == "twist":
            a = np.asarray(action, np.float32)
            kick = bool(a[3] > 0.5)
            if s.policy.behavior_mode is None:
                self._n_kicks += int(kick)
                s.apply_twist(float(a[0]), float(a[1]), float(a[2]), kick=kick)
            action = ds.P["kick_right"] if (kick or s.policy.behavior_mode is not None) else ds.P["walk"]
        else:
            action = int(action)
            if s.policy.behavior_mode is not None:
                action = self._prev_action          # a kick is running: this decision is ignored
            else:
                self._n_kicks += int(ds.PRIMITIVES[action][1] is None)
                s.apply_primitive(action)

        reward, scored = 0.0, False
        for _ in range(STEPS_PER_DECISION):
            s.step()
            if self.tracker is not None:
                self.tracker.on_control_step()
            if self.on_control_step is not None:
                self.on_control_step(s)
            fallen = s.fallen()
            if fallen and not self._was_fallen:
                self._n_falls += 1
                reward += R_FALL
            self._was_fallen = fallen
            if s.scored():
                scored = True
                break

        phi = self._potentials()
        reward += W_BALL_PROGRESS * (phi[0] - self._phi[0]) + W_DUCK_PROGRESS * (phi[1] - self._phi[1])
        reward += R_TIME + (R_GOAL if scored else 0.0)
        self._phi = phi
        self._prev_action = action
        truncated = (not scored) and s.t >= self._limit
        return self._obs(), float(reward), scored, truncated, self._info(scored)

    def set_stage(self, stage: int):
        """Curriculum hook (called by the trainer through VecEnv.env_method)."""
        self.stage = int(stage)

    def render(self):
        return self.sim.render_overview()
