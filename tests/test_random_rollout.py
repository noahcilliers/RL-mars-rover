from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
import pytest

import marsnav.envs  # noqa: F401
from marsnav.eval.random_rollout import RolloutConfig, run_random_rollout, sample_random_actions
from marsnav.utils.timing import BatchTimer


def test_batch_timer_reports_batch_and_item_means():
    timer = BatchTimer()
    timer.add(2_000_000, items=4)
    timer.add(4_000_000, items=4)

    assert timer.batches == 2
    assert timer.items == 8
    assert timer.mean_batch_ms == pytest.approx(3.0)
    assert timer.mean_item_ms == pytest.approx(0.75)


def test_sample_random_actions_matches_vector_shape_and_bounds():
    env = gym.make("marsnav/RoverNav-v0")
    try:
        action_space = env.action_space
        rng = np.random.default_rng(123)
        actions = sample_random_actions(action_space, n_envs=5, rng=rng)
    finally:
        env.close()

    assert actions.shape == (5, 2)
    assert actions.dtype == np.float32
    assert np.all(actions >= action_space.low)
    assert np.all(actions <= action_space.high)


def test_random_rollout_writes_tensorboard_and_summary(tmp_path: Path):
    config = RolloutConfig(
        n_envs=2,
        total_steps=6,
        seed=7,
        vec_env="dummy",
        log_dir=tmp_path,
        run_name="smoke",
        log_interval=1,
        env_kwargs={"max_episode_seconds": 0.05},
    )

    summary = run_random_rollout(config)
    run_dir = tmp_path / "smoke"

    assert summary["env_steps"] >= config.total_steps
    assert summary["episodes"] > 0
    assert summary["decision_ms_per_env_mean"] >= 0.0
    assert summary["env_step_ms_per_env_mean"] > 0.0
    assert (run_dir / "summary.json").exists()
    assert list(run_dir.glob("events.out.tfevents.*"))
