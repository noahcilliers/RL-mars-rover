from __future__ import annotations

from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

import marsnav.envs  # noqa: F401
from marsnav.train.ppo_flat import PPOFlatConfig, train_ppo_flat


def test_tiny_ppo_flat_training_run_writes_model_and_summary(tmp_path: Path):
    config = PPOFlatConfig(
        total_timesteps=8,
        n_envs=2,
        seed=11,
        vec_env="dummy",
        run_dir=tmp_path,
        run_name="smoke",
        n_steps=4,
        batch_size=4,
        n_epochs=1,
        eval_freq=0,
        eval_episodes=2,
        checkpoint_freq=0,
        env_kwargs={"max_episode_seconds": 0.05},
    )

    summary = train_ppo_flat(config)
    run_dir = tmp_path / "smoke"

    assert Path(summary["final_model_path"]).exists()
    assert summary["best_model_path"] is None
    assert summary["final_eval"]["episodes"] == 2
    assert summary["final_eval"]["policy_ms_per_env_decision_mean"] >= 0.0
    assert (run_dir / "config.json").exists()
    assert (run_dir / "summary.json").exists()
    assert (run_dir / "final_eval" / "summary.json").exists()

    model = PPO.load(summary["final_model_path"], device="cpu")
    env = gym.make("marsnav/RoverNav-v0", max_episode_seconds=0.05)
    try:
        obs, _info = env.reset(seed=3)
        action, _state = model.predict(obs, deterministic=True)
    finally:
        env.close()

    assert action.shape == (2,)
    assert np.all(action >= -1.0)
    assert np.all(action <= 1.0)
