"""What the LeRobot student sees and outputs - shared by gen_dataset.py and eval_student.py so the
two can never drift apart.

Observation (all available on a real duck):
  observation.images.head  (240, 320, 3) head camera
  observation.state        (25,) gyro(3), gravity(3), 14 joint positions, previous executed command
                           (vx, vy, yaw_rate), time standing, kick time left
Action ("twist" mode): (4,) [vx, vy, yaw_rate, kick] - the smoothed teacher command, executed
directly by FootballEnv(action_mode="twist") (kick > 0.5 starts a right-foot kick).

The teacher is the PPO primitive policy, executed through an EMA low-pass (TAU s): measured 20/20
goals at TAU=0.4 vs 3.4 -> 2.0 turn-direction flips per second, i.e. an expert ACT can imitate.
"""

from __future__ import annotations

import numpy as np

from microduck_football.sim import duck_sim as ds

TAU = 0.4
ALPHA = min(1.0, 0.2 / TAU)          # EMA factor per 0.2 s decision
TWIST = {"stand": (0.0, 0.0, 0.0), "walk": (0.4, 0.0, 0.0), "walk_left": (0.4, 0.0, 1.0),
         "walk_right": (0.4, 0.0, -1.0), "strafe_left": (0.3, 0.3, 0.0),
         "strafe_right": (0.3, -0.3, 0.0), "back": (-0.4, 0.0, 0.0)}
ACTION_NAMES = ["vx", "vy", "yaw_rate", "kick"]
STATE_NAMES = list(ds.STATE_NAMES) + ["prev_vx", "prev_vy", "prev_yaw_rate", "stand_time", "kick_time_left"]
# v2 (+ GoalMemory): goal estimate from glances with odometry drift + previous head angles (32-D).
# The DAgger probe showed this information is what scoring needs (0/20 -> 9/20 with DAgger).
MEM_NAMES = ["goal_dir_x", "goal_dir_y", "goal_dist", "goal_time_since_seen", "goal_seen",
             "prev_head_pitch", "prev_head_yaw"]
STATE_NAMES_MEM = STATE_NAMES + MEM_NAMES
TASK = "dribble the ball into the goal"


def features(memory: bool = False):
    names = STATE_NAMES_MEM if memory else STATE_NAMES
    return {
        "observation.images.head": {"dtype": "video", "shape": (ds.IMG_H, ds.IMG_W, 3),
                                    "names": ["height", "width", "channels"]},
        "observation.state": {"dtype": "float32", "shape": (len(names),), "names": names},
        "action": {"dtype": "float32", "shape": (len(ACTION_NAMES),), "names": ACTION_NAMES},
    }


def student_state(sim: ds.DuckSim, prev_cmd, mem: "GoalMemory | None" = None) -> np.ndarray:
    """25-D state, or 32-D with a GoalMemory (call exactly once per decision: it updates)."""
    pol = sim.policy
    kick_left = pol.behavior_time_left if pol.behavior_mode is not None else 0.0
    extra = np.array([*prev_cmd, min(sim.stand_time(), 3.0), kick_left], np.float32)
    parts = [sim.state_vector(), extra]
    if mem is not None:
        parts.append(mem.features(sim))
    return np.concatenate(parts).astype(np.float32)


def teacher_label(teacher: "SmoothedTeacher", priv_obs, prev_cmd) -> np.ndarray:
    """DAgger label on ANY state: the teacher's primitive -> twist target, smoothed from the command
    the robot actually executed last (consistent with the student's own state)."""
    prim = int(teacher.model.predict(priv_obs, deterministic=True)[0])
    teacher.last_primitive = prim
    name = ds.PRIMITIVE_NAMES[prim]
    target = np.array(TWIST.get(name, (0.0, 0.0, 0.0)), np.float32)
    prev = np.asarray(prev_cmd, np.float32)
    return np.array([*(prev + ALPHA * (target - prev)), float(name.startswith("kick"))], np.float32)


class SmoothedTeacher:
    """PPO primitive teacher -> EMA-smoothed twist command (the label and, without noise, the
    executed action)."""

    def __init__(self, model):
        self.model = model
        self.reset()

    def reset(self):
        self.cmd = np.zeros(3, np.float32)
        self.last_primitive = ds.P["stand"]

    def act(self, priv_obs) -> np.ndarray:
        a = int(self.model.predict(priv_obs, deterministic=True)[0])
        self.last_primitive = a
        name = ds.PRIMITIVE_NAMES[a]
        target = np.array(TWIST.get(name, (0.0, 0.0, 0.0)), np.float32)
        self.cmd = self.cmd + ALPHA * (target - self.cmd)
        return np.array([*self.cmd, float(name.startswith("kick"))], np.float32)

    def fix_obs(self, priv_obs):
        """In twist mode the env records a generic prev_action; the teacher expects its own."""
        from microduck_football.envs.football_env import OBS_SLICES
        sl = OBS_SLICES["prev_action"]
        priv_obs[sl] = 0.0
        priv_obs[sl.start + self.last_primitive] = 1.0
        return priv_obs


class GoalMemory:
    """Deployable-style goal estimate: refreshed only when the goal is inside the camera frame,
    carried between sightings by the duck's own motion with simulated odometry drift (heading
    random walk while moving + ~5% position error per metre walked). Also remembers the previous
    head angles (the head tracks the ball, so their change = the ball's relative motion).

    features(): [goal_dir_x, goal_dir_y, goal_dist, time_since_seen, seen_flag,
                 prev_head_pitch, prev_head_yaw]
    """

    YAW_DRIFT = 0.04      # rad per sqrt(s) of walking
    POS_DRIFT = 0.05      # m of position error per m walked
    N = 7

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        self.seen = False
        self.goal_rel_world = None      # goal - duck at the last sighting, plus accumulated drift
        self.drift_yaw = 0.0
        self.drift_pos = np.zeros(2)
        self.t_seen = 0.0
        self.last_xy = None
        self.prev_head = np.zeros(2, np.float32)

    def features(self, sim: ds.DuckSim) -> np.ndarray:
        from microduck_football.sim.head_tracking import goal_in_view
        x, y, yaw = sim.duck_pose()
        dt = 0.2
        if self.last_xy is not None:
            step = float(np.hypot(x - self.last_xy[0], y - self.last_xy[1]))
            if step > 1e-3:
                self.drift_yaw += self.YAW_DRIFT * np.sqrt(dt) * self.rng.standard_normal()
                self.drift_pos += self.POS_DRIFT * step * self.rng.standard_normal(2)
        self.last_xy = (x, y)
        if goal_in_view(sim):
            self.seen, self.t_seen = True, sim.t
            self.drift_yaw, self.drift_pos = 0.0, np.zeros(2)
        if self.seen:
            rel = sim.to_duck_frame(ds.GOAL_CENTER)                     # true relative goal ...
            c, s = np.cos(self.drift_yaw), np.sin(self.drift_yaw)
            rel = np.array([c * rel[0] - s * rel[1], s * rel[0] + c * rel[1]]) + self.drift_pos
            d = float(np.linalg.norm(rel)) + 1e-6                      # ... seen through drift
            goal = [rel[0] / d, rel[1] / d, d, min(sim.t - self.t_seen, 10.0), 1.0]
        else:
            goal = [0.0, 0.0, 0.0, 10.0, 0.0]
        q = sim.data.qpos
        idx = sim.policy.joint_qpos_indices
        head_now = np.array([q[idx[6]], q[idx[7]]], np.float32)       # head_pitch, head_yaw joints
        out = np.array([*goal, *self.prev_head], np.float32)
        self.prev_head = head_now
        return out
