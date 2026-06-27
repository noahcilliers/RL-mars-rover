"""Train Phase 1 PPO on the flat RoverNav environment.

Example:
    uv run python scripts/train_ppo_flat.py --total-timesteps 200000 --n-envs 8
    uv run tensorboard --logdir runs/ppo-flat
"""

from __future__ import annotations

import argparse
from pathlib import Path

from marsnav.train.ppo_flat import PPOFlatConfig, format_training_summary, train_ppo_flat


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--env-id", default="marsnav/RoverNav-v0")
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vec-env", choices=("subproc", "dummy"), default="subproc")
    parser.add_argument("--start-method", default="spawn")
    parser.add_argument("--run-dir", type=Path, default=Path("runs/ppo-flat"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--eval-freq", type=int, default=20_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--eval-n-envs", type=int, default=1)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--max-episode-seconds", type=float, default=None)
    parser.add_argument("--goal-distance-min", type=float, default=None)
    parser.add_argument("--goal-distance-max", type=float, default=None)
    parser.add_argument("--success-radius", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_kwargs = {}
    if args.max_episode_seconds is not None:
        env_kwargs["max_episode_seconds"] = args.max_episode_seconds
    if args.goal_distance_min is not None or args.goal_distance_max is not None:
        if args.goal_distance_min is None or args.goal_distance_max is None:
            raise ValueError("set both --goal-distance-min and --goal-distance-max")
        env_kwargs["goal_distance"] = (args.goal_distance_min, args.goal_distance_max)
    if args.success_radius is not None:
        env_kwargs["success_radius"] = args.success_radius

    config = PPOFlatConfig(
        env_id=args.env_id,
        total_timesteps=args.total_timesteps,
        n_envs=args.n_envs,
        seed=args.seed,
        vec_env=args.vec_env,
        start_method=args.start_method,
        run_dir=args.run_dir,
        run_name=args.run_name,
        device=args.device,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        vf_coef=args.vf_coef,
        max_grad_norm=args.max_grad_norm,
        eval_freq=args.eval_freq,
        eval_episodes=args.eval_episodes,
        eval_n_envs=args.eval_n_envs,
        checkpoint_freq=args.checkpoint_freq,
        log_interval=args.log_interval,
        env_kwargs=env_kwargs,
    )
    summary = train_ppo_flat(config)
    print(format_training_summary(summary))


if __name__ == "__main__":
    main()
