"""Run a vectorized random policy and log timing/episode metrics.

Example:
    uv run python scripts/random_rollout.py --n-envs 8 --total-steps 4000
    uv run tensorboard --logdir runs/random-rollout
"""

from __future__ import annotations

import argparse
from pathlib import Path

from marsnav.eval.random_rollout import RolloutConfig, format_summary, run_random_rollout


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-id", default="marsnav/RoverNav-v0")
    parser.add_argument("--n-envs", type=int, default=8)
    parser.add_argument(
        "--total-steps",
        type=int,
        default=4_000,
        help="Total environment transitions to collect across all envs.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vec-env", choices=("subproc", "dummy"), default="subproc")
    parser.add_argument(
        "--start-method",
        default="spawn",
        help="Multiprocessing start method for --vec-env subproc.",
    )
    parser.add_argument("--log-dir", type=Path, default=Path("runs/random-rollout"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument(
        "--max-episode-seconds",
        type=float,
        default=None,
        help="Override the env episode limit, useful for quick smoke runs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env_kwargs = {}
    if args.max_episode_seconds is not None:
        env_kwargs["max_episode_seconds"] = args.max_episode_seconds

    config = RolloutConfig(
        env_id=args.env_id,
        n_envs=args.n_envs,
        total_steps=args.total_steps,
        seed=args.seed,
        vec_env=args.vec_env,
        start_method=args.start_method,
        log_dir=args.log_dir,
        run_name=args.run_name,
        log_interval=args.log_interval,
        env_kwargs=env_kwargs,
    )
    summary = run_random_rollout(config)
    print(format_summary(summary))


if __name__ == "__main__":
    main()
