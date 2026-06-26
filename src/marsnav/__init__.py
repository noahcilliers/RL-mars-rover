"""marsnav — Cheap, Confident Navigation for Mars Rovers (Learned vs. Planned).

v1 scope: a wheeled rover in MuJoCo, a PPO navigation policy using local sensing
only, benchmarked against a strong safety-aware classical planner. The headline
metric is assurance cost per decision and exploration throughput — not path length.
See the numbered design docs at the repo root.
"""

__version__ = "0.1.0"
