"""Phase 1 PPO training on the flat RoverNav environment."""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from torch.utils.tensorboard import SummaryWriter

import marsnav.envs  # noqa: F401  (registers marsnav/RoverNav-v0)
from marsnav.utils.timing import BatchTimer, elapsed_ns, now_ns

VecEnvKind = Literal["dummy", "subproc"]


@dataclass
class PPOFlatConfig:
    """Configuration for a flat-ground PPO training run."""

    env_id: str = "marsnav/RoverNav-v0"
    total_timesteps: int = 200_000
    n_envs: int = 8
    seed: int = 0
    vec_env: VecEnvKind = "subproc"
    start_method: str = "spawn"
    run_dir: Path = Path("runs/ppo-flat")
    run_name: str | None = None
    device: str = "cpu"
    learning_rate: float = 3e-4
    n_steps: int = 1024
    batch_size: int = 256
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    eval_freq: int = 20_000
    eval_episodes: int = 10
    eval_n_envs: int = 1
    checkpoint_freq: int = 50_000
    log_interval: int = 1
    env_kwargs: dict[str, Any] = field(default_factory=dict)
    policy_kwargs: dict[str, Any] = field(default_factory=dict)


def _validate_config(config: PPOFlatConfig) -> None:
    if config.total_timesteps <= 0:
        raise ValueError("total_timesteps must be positive")
    if config.n_envs <= 0:
        raise ValueError("n_envs must be positive")
    if config.n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if config.batch_size <= 1:
        raise ValueError("batch_size must be greater than 1")
    if config.n_epochs <= 0:
        raise ValueError("n_epochs must be positive")
    if config.eval_episodes <= 0:
        raise ValueError("eval_episodes must be positive")
    if config.eval_n_envs <= 0:
        raise ValueError("eval_n_envs must be positive")
    if config.log_interval <= 0:
        raise ValueError("log_interval must be positive")
    if config.vec_env not in ("dummy", "subproc"):
        raise ValueError("vec_env must be 'dummy' or 'subproc'")

    rollout_batch = config.n_steps * config.n_envs
    if rollout_batch <= 1:
        raise ValueError("n_steps * n_envs must be greater than 1")
    if config.batch_size > rollout_batch:
        raise ValueError("batch_size must be <= n_steps * n_envs")


def _run_name(config: PPOFlatConfig) -> str:
    if config.run_name:
        return config.run_name
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    return f"ppo_flat_{stamp}_seed{config.seed}_envs{config.n_envs}"


def _serializable_config(config: PPOFlatConfig) -> dict[str, Any]:
    data = asdict(config)
    data["run_dir"] = str(config.run_dir)
    return data


def _make_ppo_env(
    config: PPOFlatConfig,
    *,
    n_envs: int,
    seed: int,
    monitor_dir: Path | None,
) -> VecEnv:
    vec_cls = DummyVecEnv if config.vec_env == "dummy" else SubprocVecEnv
    vec_kwargs: dict[str, Any] | None = None
    if config.vec_env == "subproc":
        vec_kwargs = {"start_method": config.start_method}
    if monitor_dir is not None:
        monitor_dir.mkdir(parents=True, exist_ok=True)

    return make_vec_env(
        config.env_id,
        n_envs=n_envs,
        seed=seed,
        monitor_dir=str(monitor_dir) if monitor_dir is not None else None,
        env_kwargs=dict(config.env_kwargs),
        vec_env_cls=vec_cls,
        vec_env_kwargs=vec_kwargs,
    )


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.fmean(values)


def _rate(count: int, total: int) -> float | None:
    if total == 0:
        return None
    return count / total


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


def _callback_freq(env_step_freq: int, n_envs: int) -> int:
    return max(env_step_freq // n_envs, 1)


def _build_callbacks(
    config: PPOFlatConfig,
    *,
    run_dir: Path,
    eval_env: VecEnv | None,
):
    callbacks = []
    checkpoints_dir = run_dir / "checkpoints"

    if config.checkpoint_freq > 0:
        checkpoints_dir.mkdir(parents=True, exist_ok=True)
        callbacks.append(
            CheckpointCallback(
                save_freq=_callback_freq(config.checkpoint_freq, config.n_envs),
                save_path=str(checkpoints_dir),
                name_prefix="ppo_flat",
            )
        )

    if config.eval_freq > 0 and eval_env is not None:
        callbacks.append(
            EvalCallback(
                eval_env,
                n_eval_episodes=config.eval_episodes,
                eval_freq=_callback_freq(config.eval_freq, config.n_envs),
                log_path=str(run_dir / "eval"),
                best_model_save_path=str(run_dir / "best_model"),
                deterministic=True,
                render=False,
                verbose=1,
            )
        )

    if not callbacks:
        return None
    return CallbackList(callbacks)


def evaluate_ppo_policy(
    model: PPO,
    config: PPOFlatConfig,
    *,
    run_dir: Path,
) -> dict[str, Any]:
    """Run deterministic eval and time policy inference separately."""

    eval_env = _make_ppo_env(
        config,
        n_envs=config.eval_n_envs,
        seed=config.seed + 20_000,
        monitor_dir=run_dir / "final_eval_monitor",
    )

    policy_timer = BatchTimer()
    env_step_timer = BatchTimer()
    episode_returns = np.zeros(config.eval_n_envs, dtype=np.float64)
    episode_lengths = np.zeros(config.eval_n_envs, dtype=np.int64)
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
    wall_start = perf_counter()

    try:
        obs = eval_env.reset()
        while len(completed_returns) < config.eval_episodes:
            start = now_ns()
            actions, _ = model.predict(obs, deterministic=True)
            policy_timer.add(elapsed_ns(start), items=config.eval_n_envs)

            start = now_ns()
            obs, rewards, dones, infos = eval_env.step(actions)
            env_step_timer.add(elapsed_ns(start), items=config.eval_n_envs)
            env_steps += config.eval_n_envs

            episode_returns += rewards
            episode_lengths += 1

            for idx, done in enumerate(dones):
                if not done:
                    continue
                if len(completed_returns) < config.eval_episodes:
                    completed_returns.append(float(episode_returns[idx]))
                    completed_lengths.append(int(episode_lengths[idx]))
                    outcome_counts[_outcome(infos[idx])] += 1
                episode_returns[idx] = 0.0
                episode_lengths[idx] = 0
    finally:
        eval_env.close()

    wall_time_s = perf_counter() - wall_start
    episodes = len(completed_returns)
    summary = {
        "episodes": episodes,
        "env_steps": env_steps,
        "wall_time_s": wall_time_s,
        "env_steps_per_second": env_steps / wall_time_s if wall_time_s > 0 else 0.0,
        "policy_batch_ms_mean": policy_timer.mean_batch_ms,
        "policy_ms_per_env_decision_mean": policy_timer.mean_item_ms,
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

    eval_dir = run_dir / "final_eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    writer = SummaryWriter(str(run_dir / "tb" / "final_eval"))
    try:
        for key, value in summary.items():
            if isinstance(value, (int, float)):
                writer.add_scalar(f"final_eval/{key}", value, config.total_timesteps)
    finally:
        writer.flush()
        writer.close()

    return summary


def train_ppo_flat(config: PPOFlatConfig) -> dict[str, Any]:
    """Train PPO on flat-ground RoverNav and return run metadata."""

    _validate_config(config)
    run_dir = config.run_dir / _run_name(config)
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "config.json").write_text(
        json.dumps(_serializable_config(config), indent=2),
        encoding="utf-8",
    )

    train_env = _make_ppo_env(
        config,
        n_envs=config.n_envs,
        seed=config.seed,
        monitor_dir=run_dir / "monitor",
    )
    eval_env = None

    try:
        if config.eval_freq > 0:
            eval_env = _make_ppo_env(
                config,
                n_envs=1,
                seed=config.seed + 10_000,
                monitor_dir=run_dir / "eval_monitor",
            )

        callbacks = _build_callbacks(config, run_dir=run_dir, eval_env=eval_env)
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=config.learning_rate,
            n_steps=config.n_steps,
            batch_size=config.batch_size,
            n_epochs=config.n_epochs,
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
            clip_range=config.clip_range,
            ent_coef=config.ent_coef,
            vf_coef=config.vf_coef,
            max_grad_norm=config.max_grad_norm,
            tensorboard_log=str(run_dir / "tb"),
            policy_kwargs=dict(config.policy_kwargs),
            verbose=1,
            seed=config.seed,
            device=config.device,
        )
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=callbacks,
            log_interval=config.log_interval,
            tb_log_name="train",
            progress_bar=False,
        )

        final_model_path = run_dir / "final_model.zip"
        model.save(final_model_path)
        final_eval = evaluate_ppo_policy(model, config, run_dir=run_dir)
    finally:
        train_env.close()
        if eval_env is not None:
            eval_env.close()

    best_model_path = run_dir / "best_model" / "best_model.zip"
    summary = {
        "run_dir": str(run_dir),
        "final_model_path": str(final_model_path),
        "best_model_path": str(best_model_path) if best_model_path.exists() else None,
        "tensorboard_logdir": str(run_dir / "tb"),
        "config": _serializable_config(config),
        "final_eval": final_eval,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def format_training_summary(summary: dict[str, Any]) -> str:
    """Human-readable summary for the training CLI."""

    final_eval = summary["final_eval"]

    def fmt(value: float | int | None, suffix: str = "") -> str:
        if value is None:
            return "n/a"
        if isinstance(value, float):
            return f"{value:.4f}{suffix}"
        return f"{value}{suffix}"

    lines = [
        "PPO flat-ground training complete",
        f"  run_dir                 {summary['run_dir']}",
        f"  final_model             {summary['final_model_path']}",
        f"  best_model              {summary['best_model_path'] or 'n/a'}",
        f"  tensorboard             uv run tensorboard --logdir {summary['tensorboard_logdir']}",
        f"  eval_episodes           {final_eval['episodes']}",
        f"  eval_success_rate       {fmt(final_eval['success_rate'])}",
        f"  eval_rollover_rate      {fmt(final_eval['rollover_rate'])}",
        f"  eval_timeout_rate       {fmt(final_eval['timeout_rate'])}",
        f"  policy decision cost    {fmt(final_eval['policy_ms_per_env_decision_mean'], ' ms/env decision')}",
    ]
    return "\n".join(lines)
