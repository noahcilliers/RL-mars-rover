# Project Plan

Cheap, Confident Navigation for Mars Rovers — implementation plan.

This plan captures every design decision and breaks the work into phases. Each phase lists objectives, the tasks involved, deliverables, and exit criteria (what must be true to move on). It assumes the framing in `00-problem-statement.md`, the claim in `02-thesis.md`, and the environment notes in `04-terrain-and-simulation-environment.md`.

---

## 1. Objective

Build a PPO-trained rover navigation policy in MuJoCo and benchmark it against a **strong, traversability- and safety-aware classical planner** (Dijkstra/A\* over a terrain-weighted cost map plus a feasibility/safety checker) on Mars-like terrain. The learned policy uses **local sensing only**; the classical planner gets the **full map**. The benchmark is centered on **assurance cost and exploration throughput** — how much compute each safe decision takes and how much ground gets covered per unit of it — with success, safety (rollover and clearance/traversability), generalization, and degradation under poor maps as the supporting axes. Path length is explicitly not the figure of merit.

## 2. Hardware and constraints

- **Machine:** Mac Mini, Apple Silicon. MuJoCo runs natively.
- **No CUDA / no NVIDIA GPU.** This rules out Isaac Sim and GPU-batched planetary simulators. Training relies on **CPU-vectorized environments** (and MPS where it helps). The modest compute is also thematically honest: limited compute is the constraint the project studies.
- **Small team / solo.** Favor reuse over building from scratch; keep scope tight.

## 3. Software stack

- **Physics:** MuJoCo (native Apple Silicon).
- **Env API:** Gymnasium — a custom environment wrapping the MuJoCo model and exposing our observation/action spaces.
- **RL algorithm:** Stable-Baselines3 (SB3) PPO for v1 — a trustworthy, off-the-shelf PPO so our effort goes into the environment, the baseline, and the comparison, not into reimplementing the optimizer. CleanRL held in reserve for v2 (camera / distillation).
- **Parallelism:** vectorized CPU environments (multiple env copies across cores) — the main lever for PPO sample throughput without a GPU.
- **Logging:** Weights & Biases or TensorBoard, tracking the **evaluation metrics** (not just reward), including per-decision compute, so runs produce post-ready charts.
- **Classical baseline:** our own traversability/safety-aware Dijkstra/A\* planner + path-follower (see §7), plus a trivial straight-line-to-goal controller.

### Division of labor (what we define vs. what SB3 does)

- **We define:** the MuJoCo world and rover, the observation space, the action space, the reward function, episode termination/reset, terrain generation, the curriculum, and the classical baseline.
- **SB3 handles:** rollout collection, advantage/credit assignment, the PPO update, the policy/value networks, backpropagation and weight updates, plus vectorized-env handling and logging hooks.

## 4. The agent

- **Embodiment:** a wheeled rover (4-wheel skid-steer). Wheeled, not legged — legged locomotion is its own hard RL problem and is the realistic Mars form anyway. (A robot arm + gripper, reused from MuJoCo Menagerie, e.g. uFactory Lite 6 / SO-ARM100 + Robotiq 2F-85, belongs to the deferred manipulation phases — not v1.)
- **Action space (v1 navigation):** continuous, abstracted to `[forward velocity, yaw rate]` driving position/velocity actuators with PD control — far more stable to train than raw wheel torques.
- **Observation space (v1 = the "teacher" sense):**
  - **Local terrain:** an egocentric, forward-biased height-scan covering roughly **8–10 m** at modest resolution. Local and moving, never the global map; resolution kept coarse so we don't accidentally hand the policy a minimap.
  - **Proprioception:** body orientation/tilt (IMU), linear and angular velocity, wheel states. (Wheel-vs-body velocity mismatch also gives a slip signal, useful only if the optional stuck-approximation is enabled.)
  - **Goal vector:** direction and distance to the goal. Fair — the rover knows its destination coordinates; it just has no map of the terrain in between.
  - Optional: a few rangefinder beams for crisp obstacle detection. **No camera in v1.**

## 5. Environment

- **Terrain (hybrid):** procedurally generated Mars-like **rigid** heightfields (slopes, scattered rocks, small craters) with domain randomization for training. Real NASA **HiRISE** DEM patches reserved for evaluation and demo footage. (See `04-terrain-and-simulation-environment.md` for terrain generation and training-scale details.)
- **Hazard scope (geometric):** v1 hazards are **geometric** — slopes, rocks, and craters that create tip-over, clearance, and traversability challenges. This is exactly what MuJoCo models and what ACE/ENav actually check. Soil-driven **getting stuck** (the Spirit-style failure) is *not* a v1 headline hazard; it can only be crudely approximated with optional low-traction patches and is otherwise deferred to the Chrono terramechanics track.
- **Physics:** **Mars gravity (3.71 m/s²)**; tuned wheel friction/slip. **No deformable regolith** in v1 (MuJoCo is rigid-body; true granular soil is a research project on its own — any low-traction patches are a flagged approximation, not real terramechanics).
- **World scale:** a bounded arena (~50–100 m).

## 6. Reward design

- **Core:** dense **progress-to-goal** reward (reward for reducing distance to the goal each step).
- **Safety:** large terminal penalty for **rollover/tip-over**; mild penalty for excessive tilt. (An optional getting-stuck penalty — commanded motion but ~zero net displacement over a window — can be added against low-traction patches, but it is a flagged approximation, not a core v1 term.)
- **Staging:** start with *only* reach-the-goal + rollover safety. Add energy/time efficiency (and, optionally, the stuck-approximation term) **later as a curriculum**, once basic reaching works. Multi-term reward functions up front are where RL projects quietly die.
- **Episode termination:** goal reached (success), tip-over (fail), timeout, or out-of-bounds. (Optional: stuck/immobilized as a failure only when the low-traction approximation is enabled.)

## 7. The classical baseline (made strong on purpose)

The comparison is only meaningful if the planner is the kind real rover systems use, not a strawman. Per the literature (ACE, ENav, MLNav), the expensive, decisive part of rover planning is feasibility/safety checking — which is exactly the assurance cost this project is about. So the baseline has three parts:

- **Traversability-weighted cost map:** per-cell cost derived from the heightmap (slope, roughness), so the search prefers safe ground rather than the geometrically shortest line.
- **Search:** Dijkstra/A\* over that cost map. Given the **full** map.
- **Safety / feasibility checker:** before accepting a route, verify the rover body can cross it within tilt, clearance, and rollover limits; reject short-but-unsafe routes. This is the operation we **instrument and count**, because its cost is the headline variable.

A trivial straight-line-to-goal controller is also kept as a sanity floor.

## 8. Evaluation methodology

Instrument these from the very first run:

- **Assurance / decision cost (headline):** compute per safe decision — PPO inference time vs. the planner's search-plus-feasibility-check burden (wall-clock per decision on the same machine; number of expensive feasibility checks per traverse; replanning frequency; how cost scales with terrain complexity and map size; episodes where the planner fails to return a feasible route in budget).
- **Exploration throughput (headline):** ground covered per unit of compute / per fixed time budget — the mission-level consequence.
- **Success rate:** reached goal without tipping (and, when the optional stuck-approximation is enabled, without immobilization).
- **Safety:** rollover rate (headline). Stuck/immobilization rate reported only when the low-traction approximation is enabled, and labeled approximate.
- **Generalization gap:** training terrain vs. unseen procedural seeds and held-out HiRISE patches.
- **Map-quality sweep:** planner performance as its map is degraded (downsampled, noised, partially missing) toward the real Mars condition — the policy, using local sensing, is unaffected by this and serves as the reference.
- **Path efficiency:** reported for completeness only — *not* a win condition.

**Fairness protocol:** the planner gets the full (then progressively degraded) map; PPO gets local sensing only. "Better" = comparable success and safety at **lower assurance cost**, higher **throughput**, and better **degradation under poor maps** — not shortest path.

The headline deliverable is an **assurance-cost / throughput comparison**, not a reward curve and not a shortest-path table.

---

## 9. Phased implementation

### Phase 0 — Setup and scaffolding
**Objective:** a runnable skeleton.
- Repo structure, environment management, install MuJoCo + Gymnasium + SB3 + logging.
- Author the rover MJCF (chassis + 4 wheels + sensors); load and inspect it in the MuJoCo viewer.
- Custom Gymnasium env wrapping the model, exposing the v1 observation/action spaces.
- Wire up vectorized CPU envs, logging, and a per-decision compute timer.

**Deliverable:** rover loads, env steps, a random policy runs end-to-end with metrics + timing logged.
**Exit criteria:** env passes a Gymnasium sanity check; rollouts run across multiple parallel envs.

### Phase 1 — Navigation on clean terrain (sanity)
**Objective:** prove the learning loop works before adding difficulty.
- Flat, featureless ground, Mars gravity.
- Train PPO on reach-a-known-goal with the dense progress reward + rollover penalty.
- Implement the straight-line baseline.

**Deliverable:** PPO reliably reaches goals on flat ground and beats the straight-line baseline.
**Exit criteria:** high success rate on flat ground; metrics + timing behaving sensibly; reproducible run.

### Phase 2 — Mars-like terrain + the strong planner
**Objective:** the real training environment and a worthy opponent.
- Procedural Mars-like rigid heightfields with rocks, craters, and slopes (geometric hazards); optional low-traction patches as a flagged stuck approximation; domain randomization.
- Retrain/curriculum the PPO policy on rough terrain with the local height-scan.
- Build the **traversability/safety-aware** planner (cost map + Dijkstra/A\* + feasibility checker) and instrument its per-decision and per-traverse compute.

**Deliverable:** a PPO policy that traverses rough terrain; a strong, instrumented classical planner.
**Exit criteria:** PPO achieves solid success without frequent rollovers; planner produces valid, safety-checked traversals with logged assurance cost.

### Phase 3 — The benchmark (the core result)
**Objective:** answer the thesis question.
- Run PPO (local sensing) vs. the strong planner (full map) vs. straight-line on a fixed evaluation suite.
- Sweep map quality for the planner (full → degraded → sparse).
- Evaluate generalization on unseen procedural seeds and real HiRISE patches.
- Produce the assurance-cost / throughput comparison plus success/safety/generalization tables and figures.

**Deliverable:** the headline comparison + map-degradation curve; **first public write-up/post**.
**Exit criteria:** a clear, defensible verdict on whether PPO matches the planner's success/safety at lower assurance cost and degrades better under poor maps — including a negative result if that's what the data shows.

### Phase 4 — v2: from privileged sensing to a camera (teacher–student)
**Objective:** make perception realistic.
- Treat the Phase 1–3 height-scan policy as the **teacher** (privileged local terrain).
- Add an onboard depth camera; train a **student** that uses camera + proprioception only, distilled from the teacher.
- Switch to CleanRL here for the editable PPO needed for distillation.

**Deliverable:** a camera-only navigation policy + a write-up on the privileged→onboard transition.
**Exit criteria:** student approaches teacher performance using only onboard sensing.

### Phase 5 — Deferred future vision (recorded, not scheduled)
The original "prepare a habitable site" idea, plus the terramechanics track, kept as direction rather than committed work:
- **Terramechanics fidelity (Chrono):** real soft-soil sinkage/slip and the Spirit-style getting-stuck failure, which MuJoCo cannot model — the natural home for the "does rigid-body sim mask the real mobility problem?" study.
- **Survey/select** skill — confirm a site is suitable.
- **Build/place** skill — manipulation (excavation, deliver-and-place) using a reused arm + gripper.
- **FSM switching** between traverse → survey → build skills (scripted first).
- **Learned high-level manager** — replace the FSM with a hierarchical RL controller (the most novel, post-worthy extension).

---

## 10. Risks and mitigations

- **Unfair baseline (strawman planner).** A naive distance-only A\* would dodge the exact assurance cost we're studying. *Mitigation:* the §7 traversability/safety-aware planner with an instrumented feasibility checker.
- **Apples-to-oranges compute comparison.** PPO inference and A\* planning are different shapes of computation. *Mitigation:* measure on the same machine, report multiple cost proxies (wall-clock per decision, count of expensive feasibility checks, scaling with map size/complexity), and be explicit about what each captures.
- **Soil mechanics are out of scope in v1.** Rigid MuJoCo can't model soft-soil sinkage or the Spirit-style getting-stuck failure, so v1 hazards are geometric (tip-over / clearance). *Mitigation:* optional low-traction patches as a clearly-flagged approximation only; true terramechanics deferred to the Chrono track (see `04-terrain-and-simulation-environment.md`).
- **Slow training on a Mac.** No GPU caps throughput. *Mitigation:* state-based observations (no pixels in v1), vectorized CPU envs, coarse height-scan resolution, curriculum.
- **Reward hacking / collapse.** Over-shaped rewards produce degenerate behavior. *Mitigation:* minimal reward first, efficiency/optional terms added as curriculum.
- **Overfitting to terrain.** *Mitigation:* domain randomization in training; held-out seeds + HiRISE for evaluation.
- **Pixel/perception creep into v1.** *Mitigation:* camera is explicitly Phase 4, via teacher–student.

## 11. Open items

- **Team / roles:** "we" not yet defined — confirm whether this is solo or a team, and split ownership of env, baseline, and evaluation accordingly.
- **Logging choice:** W&B vs. TensorBoard (minor; either works).
- **Exact arm model** for the deferred manipulation phases (Lite 6 vs. SO-ARM100) — decide when Phase 5 becomes real.
