"""Smoke tests: the package imports, the scenes compile, every environment resets and steps with the
documented observation sizes, and the shipped models load. CPU only, ~1 min.

  uv run pytest
"""

import importlib
import os
import pkgutil

import numpy as np
import pytest

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")

import microduck_football  # noqa: E402
from microduck_football import paths  # noqa: E402


def test_every_module_imports():
    for m in pkgutil.walk_packages(microduck_football.__path__, "microduck_football."):
        importlib.import_module(m.name)


def test_paths_exist():
    for p in (paths.POLICY_DIR / "alpha_walking.onnx", paths.SCENE_FOOTBALL, paths.SCENE_2V2, paths.ROBOT_XML,
              paths.INFER_POLICY_DIR / "infer_policy.py", paths.TEACHER_OPEN_PLAY, paths.TEACHER_OBSTACLES,
              paths.MAPPO_2V2_BEST):
        assert p.exists(), f"missing {p} (submodule checked out? `git submodule update --init`)"


def test_duck_sim_walks():
    from microduck_football.sim import duck_sim as ds
    sim = ds.DuckSim(seed=0)
    sim.reset(3)
    x0 = np.array(sim.duck_pose()[:2])
    sim.apply_primitive(ds.P["walk"])
    for _ in range(100):
        sim.step()
    assert not sim.fallen()
    assert np.linalg.norm(np.array(sim.duck_pose()[:2]) - x0) > 0.1


@pytest.mark.parametrize("obstacles,stage,dim", [(False, 3, 29), (True, 6, 49)])
def test_football_env(obstacles, stage, dim):
    from microduck_football.envs import football_env as E
    env = E.FootballEnv(seed=0, stage=stage, obstacles=obstacles)
    obs, _ = env.reset(seed=1)
    assert obs.shape == (dim,)
    obs, r, te, tr, info = env.step(env.action_space.sample())
    assert obs.shape == (dim,) and np.isfinite(r)


def test_teachers_load_and_act():
    from stable_baselines3 import PPO
    from microduck_football.envs import football_env as E
    for path, obstacles in ((paths.TEACHER_OPEN_PLAY, False), (paths.TEACHER_OBSTACLES, True)):
        model = PPO.load(path, device="cpu")
        env = E.FootballEnv(seed=0, stage=4 if obstacles else 3, obstacles=obstacles)
        obs, _ = env.reset(seed=2)
        a = int(model.predict(obs, deterministic=True)[0])
        assert 0 <= a < env.action_space.n


def test_student_io_features():
    from microduck_football.imitation import student_io as sio
    assert sio.features(False)["observation.state"]["shape"] == (25,)
    assert sio.features(True)["observation.state"]["shape"] == (32,)


def test_env_2v2_and_mappo_model():
    import torch
    from microduck_football.envs import env_2v2 as E
    from microduck_football.multiagent import train_2v2 as T
    env = E.Env2v2(seed=0)
    obs, st = env.reset(seed=3)
    assert obs.shape == (4, E.OBS_DIM) == (4, 50) and st.shape == (4, E.STATE_DIM) == (4, 86)
    actor = T.load_actor(paths.MAPPO_2V2_BEST)
    with torch.no_grad():
        a = T.masked_logits(actor, torch.as_tensor(obs)).argmax(-1).numpy()
    obs, st, r, te, tr, info = env.step(list(a))
    assert obs.shape == (4, 50) and r.shape == (4,)
