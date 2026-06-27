# Project Description

**Working title:** Cheap, Confident Navigation for Mars Rovers — Learned vs. Planned
**Type:** Portfolio + research project (public write-ups planned; no formal paper)
**Platform:** MuJoCo, trained on a Mac Mini (Apple Silicon, CPU/MPS — no CUDA)
**Companion doc:** see `00-problem-statement.md` for the underlying problem this addresses.

## Summary

A Mars rover's real limitation is not crashing — it is how little ground it covers in the time it has. Exploration is throttled by the **cost of being confident the next move is safe**: onboard compute is tiny, and the classical algorithms that assure a safe path are expensive to run, so the rover spends most of its time deciding rather than driving (see the problem statement).

This project tests whether a reinforcement-learning policy can navigate rough Martian terrain with **the same confidence as a strong classical planner, but at a fraction of the cost per decision** — and therefore let a rover explore more. We train a PPO policy that senses only its **local** surroundings and compare it against a **traversability- and safety-aware classical planner** (not a naive shortest-path baseline) that is handed the **full** terrain map. The question is whether the learned policy can match the planner's success and safety while deciding in constant time and depending far less on a prepared global map.

The work runs entirely in simulation on modest hardware. Terrain is generated procedurally to mimic Martian surface statistics (slopes, rocks, craters) under Mars gravity, with real NASA elevation patches (HiRISE) reserved for evaluation and demo footage.

## Motivation

Classical planners are excellent when they have an accurate global map and the compute budget to validate every candidate path. On a rover, neither holds comfortably: the processor is radiation-hardened and weak, the safety/feasibility checking that dominates each planning cycle is costly, and much of Mars lacks high-resolution maps at rover scale in the first place. Each of those is a tax on exploration — paid in slow, cautious, ground-vetted driving. A learned policy is attractive precisely on this axis: a single forward pass produces a decision in constant time, and it can act on local sensing instead of a map uploaded from Earth. If "good enough" safety can be had far more cheaply, the rover covers more ground per sol.

## What the project is

A single, focused navigation benchmark centered on **decision cost**, not path length:

- A wheeled rover modeled in MuJoCo, controlled by a PPO policy that observes a local egocentric terrain scan, its own state (orientation, velocity), and the direction/distance to a known goal — but **not** a global map.
- A **strong classical opponent**: Dijkstra/A\* run over a **traversability-weighted** cost map with a **safety/feasibility checker** that rejects geometrically short but unsafe routes — given the full map. Plus a trivial straight-line controller as a sanity floor.
- A comparison whose headline is **assurance cost and exploration throughput** — how much compute each safe decision takes and how much ground gets covered per unit of it — alongside success rate, safety (rollover and getting stuck), generalization to unseen terrain, and how the planner degrades as map quality drops.

The project is deliberately scoped so that v1 is *only* navigation. Training begins on flat, featureless ground (a sanity check), then moves to procedural Mars-like terrain, and finally is evaluated on real Mars elevation data.

## Goals and definition of success

- A working PPO navigation policy that reaches goals over rough terrain with comparable safety to the planner.
- An honest benchmark showing whether the learned policy delivers **comparable success and safety at materially lower assurance cost**, and whether it **degrades more gracefully** when the planner's map is poor or absent.
- Clear evidence for or against the thesis (see the thesis document), suitable for public write-ups.
- A clean, reproducible codebase and environment that can be extended later.

Success is **not** "the reward curve went up," and it is **not** "a shorter path." Success is a defensible answer to whether learning buys the same confidence more cheaply — and so more exploration.

## Scope

**In scope (v1):** rover model, procedural + real Mars terrain, PPO navigation to a known goal, a strong traversability/safety-aware classical baseline, the assurance-cost comparison, and a write-up.

**Planned next (v2):** replace the privileged local terrain scan with an onboard depth camera, learned via teacher–student distillation.

**Deferred (future vision):** the broader "prepare a habitable site" idea — site survey/selection, manipulation (excavation, deliver-and-place), multi-skill switching via a finite-state machine, and eventually a learned high-level controller. Explicitly out of scope for now, recorded only as the long-term direction.

## Constraints

- All training on a Mac Mini: no NVIDIA GPU, no CUDA, no Isaac Sim. This rules out GPU-accelerated planetary simulators and pushes us toward lightweight, CPU-friendly tooling (MuJoCo, vectorized CPU environments) — which also keeps the project honest about modest compute, the very constraint it studies.
- Solo / small-team effort, so scope is kept tight and reuse is favored over building from scratch.
