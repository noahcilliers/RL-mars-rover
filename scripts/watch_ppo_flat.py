"""Watch a trained Phase 1 PPO policy drive the rover.

Examples:
    uv run mjpython scripts/watch_ppo_flat.py
    uv run mjpython scripts/watch_ppo_flat.py --model runs/ppo-flat/<run>/final_model.zip
    uv run python scripts/watch_ppo_flat.py --headless --episodes 3
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

import marsnav.envs  # noqa: F401  (registers marsnav/RoverNav-v0)


def latest_model_path(runs_dir: Path, *, prefer: str = "best") -> Path:
    """Find the newest saved PPO model under ``runs_dir``."""

    patterns = {
        "best": ["*/best_model/best_model.zip", "*/final_model.zip"],
        "final": ["*/final_model.zip", "*/best_model/best_model.zip"],
    }[prefer]
    for pattern in patterns:
        candidates = list(runs_dir.glob(pattern))
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime)

    raise FileNotFoundError(
        f"no PPO model zips found under {runs_dir}; pass --model explicitly"
    )


def make_env(args: argparse.Namespace):
    env_kwargs = {"render_mode": "rgb_array"} if args.headless_render else {}
    if args.max_episode_seconds is not None:
        env_kwargs["max_episode_seconds"] = args.max_episode_seconds
    if args.goal_distance_min is not None or args.goal_distance_max is not None:
        if args.goal_distance_min is None or args.goal_distance_max is None:
            raise ValueError("set both --goal-distance-min and --goal-distance-max")
        env_kwargs["goal_distance"] = (args.goal_distance_min, args.goal_distance_max)
    if args.success_radius is not None:
        env_kwargs["success_radius"] = args.success_radius
    return gym.make(args.env_id, **env_kwargs)


def print_episode(
    episode: int,
    *,
    steps: int,
    total_reward: float,
    terminated: bool,
    truncated: bool,
    info: dict,
) -> None:
    if info.get("is_success"):
        outcome = "success"
    elif info.get("rollover"):
        outcome = "rollover"
    elif info.get("out_of_bounds"):
        outcome = "out_of_bounds"
    elif truncated or info.get("TimeLimit.truncated"):
        outcome = "timeout"
    elif terminated:
        outcome = "terminated"
    else:
        outcome = "done"
    print(
        f"episode {episode}: outcome={outcome} "
        f"steps={steps} reward={total_reward:.3f}"
    )


def run_policy(args: argparse.Namespace) -> None:
    model_path = args.model or latest_model_path(args.runs_dir, prefer=args.prefer)
    print(f"loading model: {model_path}")
    model = PPO.load(model_path, device=args.device)

    env = make_env(args)
    rover_env = env.unwrapped
    viewer = None

    try:
        if not args.headless:
            from mujoco import viewer as mujoco_viewer

            try:
                viewer = mujoco_viewer.launch_passive(rover_env.model, rover_env.data)
            except RuntimeError as exc:
                if "mjpython" in str(exc):
                    raise SystemExit(
                        "Interactive MuJoCo viewer on macOS requires mjpython.\n"
                        "Re-run with: uv run mjpython scripts/watch_ppo_flat.py\n"
                        "For non-GUI playback, use: uv run python scripts/watch_ppo_flat.py --headless"
                    ) from exc
                raise

        for episode in range(1, args.episodes + 1):
            obs, _info = env.reset(seed=args.seed + episode - 1)
            total_reward = 0.0
            steps = 0
            done = False

            if viewer is not None:
                viewer.sync()

            terminated = False
            truncated = False
            while not done:
                action, _state = model.predict(obs, deterministic=not args.stochastic)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += float(reward)
                steps += 1
                done = bool(terminated or truncated)

                if args.print_actions and steps % args.print_every == 0:
                    action_arr = np.asarray(action, dtype=float).reshape(-1)
                    print(
                        f"  step={steps:04d} action=[{action_arr[0]:+.3f}, "
                        f"{action_arr[1]:+.3f}] reward={float(reward):+.3f}"
                    )

                if viewer is not None:
                    if not viewer.is_running():
                        return
                    viewer.sync()
                    if args.realtime:
                        time.sleep(max(0.0, 1.0 / args.control_hz))

            print_episode(
                episode,
                steps=steps,
                total_reward=total_reward,
                terminated=bool(terminated),
                truncated=bool(truncated),
                info=info,
            )
            if viewer is not None and args.pause_on_done:
                time.sleep(args.pause_on_done)
    finally:
        if viewer is not None:
            viewer.close()
        env.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model", type=Path, default=None, help="PPO .zip to load")
    parser.add_argument("--runs-dir", type=Path, default=Path("runs/ppo-flat"))
    parser.add_argument("--prefer", choices=("best", "final"), default="best")
    parser.add_argument("--env-id", default="marsnav/RoverNav-v0")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--headless", action="store_true", help="run without opening viewer")
    parser.add_argument(
        "--headless-render",
        action="store_true",
        help="enable rgb_array rendering while running headless",
    )
    parser.add_argument("--stochastic", action="store_true", help="sample from policy")
    parser.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--control-hz", type=float, default=20.0)
    parser.add_argument("--pause-on-done", type=float, default=1.0)
    parser.add_argument("--print-actions", action="store_true")
    parser.add_argument("--print-every", type=int, default=20)
    parser.add_argument("--max-episode-seconds", type=float, default=None)
    parser.add_argument("--goal-distance-min", type=float, default=None)
    parser.add_argument("--goal-distance-max", type=float, default=None)
    parser.add_argument("--success-radius", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_policy(args)


if __name__ == "__main__":
    main()
