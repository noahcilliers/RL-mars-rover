# Terrain and Simulation Environment

Brief reference for the simulator choice, terrain generation, and training scale. Companion to `03-project-plan.md`.

## Platform: MuJoCo

MuJoCo is the v1 environment. It is sufficient — not just convenient — for this project because the expensive "assurance" computation our thesis targets (ACE/ENav-style feasibility checking) is **geometric**: body clearance, attitude, and suspension/tilt over a heightmap, not soil mechanics. MuJoCo models rigid geometry well, runs natively on Apple Silicon with no CUDA, has Python/MJCF, cameras, IMU, and rangefinder sensing, and gives fast CPU stepping for RL. It is also open and reproducible, unlike the closed JPL ENav Sim used by the prior art — an open benchmark is itself a contribution.

Known gap: MuJoCo is not a terramechanics engine (no wheel sinkage or regolith deformation). See "Hazard scope" below.

## Terrain generation

Terrain is **generated, not stored.** We sample from a procedural distribution (domain randomization), not a fixed dataset:

- A generator maps random parameters — noise-based roughness, rock density, crater count, slope, goal distance — to a heightfield array (sub-millisecond in numpy).
- At each episode reset, overwrite the heightfield values **in place** (`model.hfield_data`) on a fixed-resolution `hfield` slot, then reset the sim state. The heightfield dimensions stay constant, so **no model recompile is needed** — this is the key technique that makes unlimited terrain variety effectively free. (Rebuilding/recompiling the MJCF each episode is the trap to avoid.)
- Store **seeds, not terrains**, for reproducibility. Keep a fixed held-out set of seeds for consistent evaluation.

## Real Mars terrain

For held-out evaluation, import real elevation data as heightfields: **HiRISE** (rover-scale, stereo DTMs), with **MOLA/CTX** for coarser context. This is geometry only. For the v2 camera/perception phase, **MARTIAN** (NASA's open Blender renderer that ingests HiRISE DTMs/orthoimages) supplies realistic Mars imagery.

## Hazard scope (v1)

v1 hazards are **geometric**: tip-over, clearance, and traversability over slopes, rocks, and craters — exactly what MuJoCo models and what ACE/ENav actually check. Soil-driven **getting stuck** (sinkage/slip, the Spirit-style failure) is only crudely approximated via low-traction patches and is explicitly flagged as such. True terramechanics is deferred to a later **Project Chrono** track, not attempted in v1 (and Chrono's high-fidelity soil models are GPU-bound, a poor fit for the Mac anyway).

## Training scale

Terrain *quantity* is nearly free; the cost is **total environment steps on CPU**, not the number of landscapes.

- Whether the agent sees 1k or 1M distinct terrains costs the same wall-clock — each is a cheap regenerate-at-reset.
- Throughput comes from **vectorized CPU environments** (8–16 parallel copies across cores) plus a small state-based MLP policy (no pixels in v1).
- Distinct terrains experienced ≈ total_steps / episode_length — on the order of tens of thousands for a typical run, which is plenty for generalization. Chasing literal millions of distinct terrains is unnecessary.
- The real difficulty is that a wide terrain distribution converges slower, so use a **curriculum**: start near-flat, widen randomization (roughness, rocks, slope, goal distance) as success rate climbs.
- Rough budget: a few million to a few tens of millions of steps per run → hours to ~1–2 days on a multi-core M-series Mac. The main time sink is repeated runs for the benchmark, seeds, and ablations (plan for overnight runs).

## Platform roadmap

MuJoCo now (fast policy iteration + the benchmark) → **Chrono** later (terramechanics / getting-stuck fidelity check) → **Gazebo** if ROS 2 systems integration is needed → **MARTIAN / Ames Stereo Pipeline** for terrain and vision realism in the perception phase.
