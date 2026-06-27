# Thesis

## Central claim

**A PPO policy that navigates from local sensing alone can match a strong classical planner's navigation confidence — comparable success and safety over rough Martian terrain — while costing far less per decision and depending far less on a prepared global map. Because the real bottleneck in rover navigation is the compute cost of assured decisions, not path optimality, a learned policy that is "good enough" on safety but dramatically cheaper to run lets a rover explore more in the same time.**

## The argument

A rover explores slowly because every autonomous step carries an expensive assurance cost: on a weak, radiation-hardened processor, the dominant work is checking whether a candidate move is safe — clearance, attitude, slip, getting stuck — and rejecting the ones that aren't. Classical planners are strong when they have an accurate global map and the budget to run those checks, but that budget is exactly what a rover lacks. The expense of assured planning, not the quality of the path, is what limits how far the rover gets (see the problem statement).

That points to two distinct ways a learned policy can win:

1. **Cheaper assurance at comparable safety.** A trained policy produces a decision in constant time — one forward pass — instead of an expensive search-and-safety-check loop. If it reaches goals about as reliably and as safely as the planner while spending a small fraction of the compute per decision, it directly relieves the bottleneck and buys more exploration.
2. **Necessity when maps are poor or absent.** A planner can only plan over a map it trusts, and much of Mars is not mapped at rover resolution. A locally-sensing policy doesn't depend on that map at all. Where good maps don't exist, it may be the only viable navigator — so its competitiveness is not merely interesting but operationally important.

## What "as good as or better" means

The claim is **not** about producing a shorter path on a known map — that is the planner's game, and given the map it would win. "Better" is defined on what actually constrains a rover:

- **Assurance / decision cost** *(the headline win condition)* — compute per safe decision: policy inference vs. the planner's search-plus-feasibility-checking burden, and how that cost scales with terrain complexity and map size.
- **Exploration throughput** — ground covered per unit of compute and time, the mission-level consequence of cheaper decisions.
- **Success rate** — fraction of episodes that reach the goal (must be comparable or better).
- **Safety** — rollover rate and **getting-stuck / immobilization** rate (must be comparable or better).
- **Graceful degradation** — how each method holds up as the planner's map quality drops toward the real Mars condition.

The learned policy must be *comparable* on success and safety, and *much cheaper and less map-dependent* on the rest. Winning a shorter path is explicitly not the goal.

## Why it is falsifiable

The thesis can fail in two clear ways: if PPO cannot reach comparable success and safety against a strong, safety-aware planner; or if its cheaper per-decision cost does not hold up once it is forced to match the planner's safety. Either outcome refutes the claim, and a clean negative result — "the planner's assurance is worth its cost" — is itself worth reporting, because no existing study pits local-only learned control against a full-map safety-aware planner on this axis.

## Why it matters

If a locally-sensing learned policy can deliver the same confidence far more cheaply, it attacks the actual bottleneck on Mars exploration — assured decisions on scarce compute — rather than a path-length problem that was never the constraint. And it degrades gracefully exactly where Mars is hardest: the unmapped, unfamiliar ground a rover must cross with no Earth-vetted route to follow.
