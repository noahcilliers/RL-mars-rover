# marsnav — Cheap, Confident Navigation for Mars Rovers

Learned vs. Planned navigation for Mars surface traversal. A wheeled rover in
MuJoCo, controlled by a PPO policy that senses only its **local** surroundings,
benchmarked against a strong **traversability- and safety-aware classical
planner** that is handed the **full** terrain map. The headline metric is
**assurance cost per decision and exploration throughput** — not path length.

See the numbered design docs in `docs/` for the full framing, thesis, and
phased plan, and `research-docs/` for the literature review.

## Status

**Phase 1 — flat-ground PPO.** Toolchain, repo scaffold, rover MJCF, a custom
Gymnasium env, instrumented random-policy rollout, and an SB3 PPO training entry
point for flat-ground goal reaching.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and is built for Apple Silicon
(CPU/MPS, no CUDA).

```bash
uv sync --extra dev      # create .venv (Python 3.12) and install deps
```

## Layout

```
src/marsnav/
  assets/      MJCF rover model + heightfield
  envs/        Gymnasium environment(s)
  baselines/   straight-line floor (v1); safety-aware planner (Phase 2+)
  train/       SB3 PPO entry points
  eval/        metrics + the assurance-cost benchmark
  utils/       timing / instrumentation helpers
scripts/       runnable entry points (viewer, manual drive, rollout)
tests/         env sanity checks
```

## Commands

```bash
uv run python scripts/view_rover.py      # inspect the rover in the MuJoCo viewer
uv run python scripts/manual_drive.py    # sanity-check skid-steer control
uv run python scripts/random_rollout.py  # instrumented random-policy rollout
uv run tensorboard --logdir runs/random-rollout
uv run python scripts/train_ppo_flat.py  # Phase 1 PPO training on flat ground
uv run mjpython scripts/watch_ppo_flat.py  # watch the newest trained PPO policy
uv run tensorboard --logdir runs/ppo-flat
uv run pytest                            # env sanity checks
```
