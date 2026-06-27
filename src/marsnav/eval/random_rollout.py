"""Vectorized random-policy rollouts with timing and TensorBoard logging.

This is Phase 0 instrumentation: the "policy" is deliberately random, but the
measurement path is the same one PPO inference and planner checks will use.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import gymnasium as gym
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from torch.utils.tensorboard import SummaryWriter

import marsnav.envs  # noqa: F401  (registers marsnav/RoverNav-v0)
from marsnav.utils.timing import BatchTimer, elapsed_ns, now_ns

VecEnvKind = Literal["dummy", "subproc"]


@dataclass
class RolloutConfig:
    env_id: str = "marsnav/RoverNav-v0"
    n_envs: int = 8
    total_steps: int = 4_000
    seed: int = 0
    vec_env: VecEnvKind = "subproc"
    start_method: str = "spawn"
    log_dir: Path = Path("runs/random-rollout")
    run_name: str | None = None
    log_interval: int = 20
    env_kwargs: dict[str, Any] = field(default_factory=dict)


def _validate_config(config: RolloutConfig) -> None:
    if config.n_envs <= 0:
        raise ValueError("n_envs must be positive")
    if config.total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if config.log_interval <= 0:
        raise ValueError("log_interval must be positive")
    if config.vec_env not in ("dummy", "subproc"):
        raise ValueError("vec_env must be 'dummy' or 'subproc'")


def _make_env(env_id: str, seed: int, rank: int, env_kwargs: dict[str, Any]):
    def _init():
        import marsnav.envs  # noqa: F401

        env = gym.make(env_id, **env_kwargs)
        env.reset(seed=seed + rank)
        return env

    return _init


def make_vector_env(config: RolloutConfig) -> VecEnv:
    """Create a vectorized environment from the rollout config."""

    _validate_config(config)
    env_fns = [
        _make_env(config.env_id, config.seed, rank, dict(config.env_kwargs))
        for rank in range(config.n_envs)
    ]
    if config.vec_env == "dummy":
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns, start_method=config.start_method)


def sample_random_actions(
    action_space: gym.spaces.Box,
    *,
    n_envs: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample one bounded random action per environment."""

    low = np.broadcast_to(action_space.low, (n_envs, *action_space.shape))
    high = np.broadcast_to(action_space.high, (n_envs, *action_space.shape))
    return rng.uniform(low, high).astype(action_space.dtype)


def _run_name(config: RolloutConfig) -> str:
    if config.run_name:
        return config.run_name
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return f"random_policy_{stamp}_seed{config.seed}_envs{config.n_envs}"


def _outcome(info: dict[str, Any]) -> str:
    if info.get("is_success"):
        return "success"
    if info.get("rollover"):
        return "rollover"
    if info.get("out_of_bounds"):
        return "out_of_bounds"
    if info.get("TimeLimit.truncated"):
        return "timeout"
    return "other_done"


def _rate(count: int, total: int) -> float | None:
    if total == 0:
        return None
    return count / total


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.fmean(values)


def _serializable_config(config: RolloutConfig) -> dict[str, Any]:
    data = asdict(config)
    data["log_dir"] = str(config.log_dir)
    return data


def _add_optional_scalar(
    writer: SummaryWriter,
    tag: str,
    value: float | int | None,
    step: int,
) -> None:
    if value is not None:
        writer.add_scalar(tag, value, step)


def run_random_rollout(config: RolloutConfig) -> dict[str, Any]:
    """Run a vectorized random-policy rollout and return aggregate metrics."""

    _validate_config(config)
    run_dir = config.log_dir / _run_name(config)
    run_dir.mkdir(parents=True, exist_ok=False)

    rng = np.random.default_rng(config.seed)
    vec_env = make_vector_env(config)
    writer = SummaryWriter(str(run_dir))
    writer.add_text("rollout/config", json.dumps(_serializable_config(config), indent=2))

    decision_timer = BatchTimer()
    env_step_timer = BatchTimer()
    interval_decision_timer = BatchTimer()
    interval_env_step_timer = BatchTimer()

    episode_returns = np.zeros(config.n_envs, dtype=np.float64)
    episode_lengths = np.zeros(config.n_envs, dtype=np.int64)
    completed_returns: list[float] = []
    completed_lengths: list[int] = []
    outcome_counts = {
        "success": 0,
        "rollover": 0,
        "out_of_bounds": 0,
        "timeout": 0,
        "other_done": 0,
    }

    env_steps = 0
    vector_steps = 0
    wall_start = perf_counter()
    interval_wall_start = wall_start

    try:
        vec_env.reset()
        while env_steps < config.total_steps:
            start = now_ns()
            actions = sample_random_actions(
                vec_env.action_space,
                n_envs=config.n_envs,
                rng=rng,
            )
            decision_ns = elapsed_ns(start)
            decision_timer.add(decision_ns, items=config.n_envs)
            interval_decision_timer.add(decision_ns, items=config.n_envs)

            start = now_ns()
            _obs, rewards, dones, infos = vec_env.step(actions)
            env_ns = elapsed_ns(start)
            env_step_timer.add(env_ns, items=config.n_envs)
            interval_env_step_timer.add(env_ns, items=config.n_envs)

            vector_steps += 1
            env_steps += config.n_envs
            episode_returns += rewards
            episode_lengths += 1

            for idx, done in enumerate(dones):
                if not done:
                    continue
                completed_returns.append(float(episode_returns[idx]))
                completed_lengths.append(int(episode_lengths[idx]))
                outcome_counts[_outcome(infos[idx])] += 1
                episode_returns[idx] = 0.0
                episode_lengths[idx] = 0

            if vector_steps % config.log_interval == 0 or env_steps >= config.total_steps:
                interval_wall_s = perf_counter() - interval_wall_start
                interval_env_steps = config.n_envs * interval_decision_timer.batches
                steps_per_s = interval_env_steps / interval_wall_s if interval_wall_s > 0 else 0.0

                writer.add_scalar("timing/decision_batch_ms", interval_decision_timer.mean_batch_ms, env_steps)
                writer.add_scalar("timing/decision_ms_per_env", interval_decision_timer.mean_item_ms, env_steps)
                writer.add_scalar("timing/env_step_batch_ms", interval_env_step_timer.mean_batch_ms, env_steps)
                writer.add_scalar("timing/env_step_ms_per_env", interval_env_step_timer.mean_item_ms, env_steps)
                writer.add_scalar("throughput/env_steps_per_second", steps_per_s, env_steps)
                writer.add_scalar("rollout/vector_steps", vector_steps, env_steps)
                writer.add_scalar("rollout/completed_episodes", len(completed_returns), env_steps)
                _add_optional_scalar(writer, "episode/reward_mean", _mean(completed_returns), env_steps)
                _add_optional_scalar(writer, "episode/length_mean", _mean([float(x) for x in completed_lengths]), env_steps)

                interval_decision_timer.reset()
                interval_env_step_timer.reset()
                interval_wall_start = perf_counter()
    finally:
        vec_env.close()
        writer.flush()
        writer.close()

    wall_time_s = perf_counter() - wall_start
    episodes = len(completed_returns)
    summary = {
        "run_dir": str(run_dir),
        "env_id": config.env_id,
        "vec_env": config.vec_env,
        "n_envs": config.n_envs,
        "seed": config.seed,
        "requested_env_steps": config.total_steps,
        "env_steps": env_steps,
        "vector_steps": vector_steps,
        "episodes": episodes,
        "wall_time_s": wall_time_s,
        "env_steps_per_second": env_steps / wall_time_s if wall_time_s > 0 else 0.0,
        "decision_batch_ms_mean": decision_timer.mean_batch_ms,
        "decision_ms_per_env_mean": decision_timer.mean_item_ms,
        "env_step_batch_ms_mean": env_step_timer.mean_batch_ms,
        "env_step_ms_per_env_mean": env_step_timer.mean_item_ms,
        "episode_reward_mean": _mean(completed_returns),
        "episode_length_mean": _mean([float(x) for x in completed_lengths]),
        "outcomes": outcome_counts,
        "success_rate": _rate(outcome_counts["success"], episodes),
        "rollover_rate": _rate(outcome_counts["rollover"], episodes),
        "out_of_bounds_rate": _rate(outcome_counts["out_of_bounds"], episodes),
        "timeout_rate": _rate(outcome_counts["timeout"], episodes),
    }

    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def format_summary(summary: dict[str, Any]) -> str:
    """Human-readable summary for the CLI."""

    def fmt(value: float | int | None, suffix: str = "") -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.4f}{suffix}"
        return f"{value}{suffix}"

    tensorboard_dir = Path(summary["run_dir"]).parent
    lines = [
        "Random rollout complete",
        f"  run_dir                 {summary['run_dir']}",
        f"  env_steps               {summary['env_steps']} ({summary['vector_steps']} vector steps)",
        f"  episodes                {summary['episodes']}",
        f"  throughput              {fmt(summary['env_steps_per_second'], ' env steps/s')}",
        f"  decision cost           {fmt(summary['decision_ms_per_env_mean'], ' ms/env decision')}",
        f"  env step cost           {fmt(summary['env_step_ms_per_env_mean'], ' ms/env step')}",
        f"  success_rate            {fmt(summary['success_rate'])}",
        f"  rollover_rate           {fmt(summary['rollover_rate'])}",
        f"  out_of_bounds_rate      {fmt(summary['out_of_bounds_rate'])}",
        f"  timeout_rate            {fmt(summary['timeout_rate'])}",
        f"  tensorboard             uv run tensorboard --logdir {tensorboard_dir}",
    ]
    return "\n".join(lines)
