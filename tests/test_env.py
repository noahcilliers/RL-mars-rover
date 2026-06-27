"""Sanity checks for RoverNavEnv: SB3 compatibility, API shapes, and that a
random episode runs cleanly and terminates/truncates."""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from stable_baselines3.common.env_checker import check_env

import marsnav.envs  # noqa: F401  (registers marsnav/RoverNav-v0)
from marsnav.envs import RoverNavEnv


def test_sb3_check_env():
    # Raises on any SB3/Gymnasium API violation.
    check_env(RoverNavEnv(), warn=True, skip_render_check=True)


def test_registered_make():
    env = gym.make("marsnav/RoverNav-v0")
    obs, info = env.reset(seed=0)
    assert env.observation_space.contains(obs)
    env.close()


def test_reset_and_step_shapes():
    env = RoverNavEnv()
    obs, info = env.reset(seed=1)
    assert obs.shape == (8,) and obs.dtype == np.float32
    assert env.observation_space.contains(obs)
    out = env.step(env.action_space.sample())
    assert len(out) == 5
    obs, reward, terminated, truncated, info = out
    assert env.observation_space.contains(obs)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool) and isinstance(truncated, bool)


def test_episode_terminates_or_truncates():
    env = RoverNavEnv()
    env.reset(seed=2)
    done = False
    steps = 0
    while not done and steps <= env.max_steps + 1:
        _, _, terminated, truncated, _ = env.step(env.action_space.sample())
        done = terminated or truncated
        steps += 1
    assert done
    assert steps <= env.max_steps + 1  # never runs past the time limit


def test_progress_reward_sign():
    """Driving straight at a goal dead ahead should yield positive reward."""
    env = RoverNavEnv(goal_distance=(10.0, 10.0))
    env.reset(seed=3)
    # place goal straight ahead of the rover's heading
    yaw = env._yaw()
    pos = env.data.xpos[env._chassis_id][:2].copy()
    env._goal = pos + 8.0 * np.array([np.cos(yaw), np.sin(yaw)])
    env._prev_dist = env._goal_dist()
    total = 0.0
    for _ in range(20):
        _, r, term, trunc, _ = env.step(np.array([1.0, 0.0], dtype=np.float32))
        total += r
        if term or trunc:
            break
    assert total > 0.0
