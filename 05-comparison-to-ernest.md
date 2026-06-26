# Comparison: This Project vs. NASA's ERNEST Rover

A side-by-side of this project (*Cheap, Confident Navigation for Mars Rovers — Learned vs. Planned*) and NASA/JPL's **ERNEST** rover, which surfaced in NASA news on June 18, 2026. Companion to `00`–`04`. The goal is honest positioning: where the two genuinely overlap, where they diverge, what ERNEST does to the thesis, and what is worth borrowing.

A one-line caution up front: most of what is public about ERNEST is press material (NASA/JPL news, June 2026), not a paper. Mechanical and field-test facts are well sourced; the exact RL algorithm, observation space, and onboard-compute story are **not** disclosed, so claims about ERNEST's *learning internals* below are deliberately hedged.

---

## 1. What ERNEST is (in one paragraph)

**ERNEST** (Exploration Rover for Navigating Extreme Sloped Terrain) is a JPL prototype, work begun in 2022, hardware completed September 2024. It is ~4 ft (1.2 m) long, four-wheeled with **mesh wheels** and an **active suspension**: two powered front joints articulate a gimbal so the rover can lift individual wheels and drive with non-standard gaits — *squirming, wheel-walking, obstacle-climbing* — and a clutch lets it drop into a passive, less-capable but more energy-efficient mode. Four steerable wheels let it drive in any direction, including sideways. The headline result: in a March 2026 field campaign in the Colorado Desert it covered **16 miles (26 km) in 37 hours of driving at up to 0.6 mph**, with minimal human intervention — about an **order of magnitude faster** than Curiosity/Perseverance can navigate. Autonomy was trained with **reinforcement learning** in JPL's high-fidelity DARTS simulator (calibrated with measured hardware/terrain-response data), running thousands of simulated driving hours on an HPC cluster, then validated on Mars Yard obstacle courses and in the desert. It is primarily a **testbed for a future long-range lunar rover** (Mars secondary).

---

## 2. The strong similarities

These are real, and worth naming plainly rather than downplaying:

- **Same root problem.** Both target the fact that Earth-in-the-loop driving is slow and that the win is *covering more ground*, not finding optimal paths. ERNEST's stated point is deciding "without waiting for commands from Earth"; this project's headline is exploration throughput. The framings rhyme almost exactly.
- **RL as the lever for onboard autonomy.** Both replace slow, vetted, human-supervised driving with a *learned* policy that decides locally. ERNEST literally replaced human joysticking with an RL policy; this project replaces (benchmarks against) an expensive classical planner.
- **Train in sim, then transfer.** Both are sim-first. ERNEST trains in DARTS and transfers to hardware; this project trains in MuJoCo and evaluates on held-out procedural seeds and real HiRISE patches.
- **Throughput/speed is the figure of merit — not path length.** ERNEST's brag is 10× speed and 16 miles; this project explicitly disowns shortest-path as a win condition in favor of assurance cost and ground-per-unit-time. Same instinct about what actually matters.
- **Wheeled, four-wheel, steering-abstracted embodiment.** ERNEST: four steerable wheels. This project: 4-wheel skid-steer with a `[forward velocity, yaw rate]` abstraction. Both deliberately avoid legged locomotion.
- **Geometric terrain hazards and a graded obstacle progression.** ERNEST's "extreme sloped terrain," sand ripples, rubble piles, steps, steep slopes ≈ this project's slopes, rocks, craters under a curriculum. Both lean on terrain variety and a difficulty ramp.

The takeaway: a current, well-funded JPL program is betting on essentially the same premise this project argues — that learned, locally-deciding autonomy is how a rover stops being slow. That is good news for the project's *relevance*, and slightly complicating for its *novelty* (see §4–5).

---

## 3. The important differences

This is where the two projects are actually doing different things.

**a. What the RL controls — locomotion vs. navigation.** This is the biggest distinction. ERNEST's learning is, as far as the public material describes, **low-level locomotion and suspension control**: how to place wheels, when to wheel-walk or squirm, how to climb a specific obstacle. Longer-range *navigation* (path planning, choosing what to go over vs. around) is described as a **new project they are just starting**. This project is the inverse: PPO learns **navigation** (`[v, yaw]` toward a goal over local terrain) on top of PD-controlled actuators, and explicitly does *not* learn gaits or torque-level locomotion. ERNEST = learned getting-over-obstacles; this project = learned getting-to-the-goal.

**b. Where the novelty lives — hardware vs. comparison.** ERNEST's central novelty is **mechanical**: an active suspension that advances the 30-year rocker-bogie design, validated by building two earlier prototypes and testing 11 suspension configurations in regolith simulant. This project introduces no new hardware; its novelty is the **controlled head-to-head** — learned local-sensing policy vs. a strong, *safety-aware* classical planner, with **assurance (decision) cost instrumented as the headline variable**. ERNEST has no published learned-vs-planner compute-cost benchmark; it is a capability/speed demonstration against today's rovers.

**c. Sim fidelity and the reality gap.** ERNEST uses JPL's **high-fidelity DARTS sim, calibrated against measured hardware response on real terrain types**, and closes the loop on real hardware in the desert. This project uses **rigid-body MuJoCo on a Mac Mini**, with no deformable regolith, no real hardware, and terramechanics/getting-stuck explicitly deferred to a later Chrono track. ERNEST's choices implicitly endorse "fidelity matters for transfer," which sharpens this project's own stated risk — though note the project's defense holds: its thesis targets *geometric* assurance (clearance, tilt, rollover), which MuJoCo does model faithfully.

**d. Compute philosophy — and a distinction worth protecting.** ERNEST trains on an **HPC cluster** (thousands of sim-hours per weekend). This project is deliberately constrained to a **Mac Mini, no CUDA**, both as a real limit and as a thematic mirror of the rover's scarce onboard compute. Important: these are *training*-compute facts, and they say nothing against the thesis, which is about **onboard inference** cost. In fact ERNEST quietly supports that thesis — its onboard decision is a cheap learned forward pass, which is exactly the "constant-time assured decision" this project argues for. The project should keep training-compute and onboard-compute rigorously separate in write-ups so no reader conflates them.

**e. Perception.** This project's v1 uses a privileged local height-scan + proprioception, **no camera** (camera is v2, via teacher–student). ERNEST operates with real onboard perception in the field (mast head, varied lighting). Different rung on the perception ladder.

**f. Scale and target body.** ERNEST: real-world, **kilometer-scale** traverse (26 km), **lunar-rover-first** (Moon polar lighting, Mars secondary). This project: bounded **~50–100 m sim arena**, **Mars-first** (Mars gravity 3.71 m/s², HiRISE). Cross-applicable, but different operating points.

**g. Open vs. closed.** ERNEST's sim (DARTS) and results are internal JPL. This project is built on **open, reproducible** tooling (MuJoCo, SB3, public HiRISE) — and an open benchmark is itself part of the contribution, since the prior art (ENav Sim, DARTS) is closed.

---

## 4. What ERNEST does to the thesis

**It validates the premise, not the specific claim.**

*Validation.* ERNEST is concrete, current, first-party evidence that (1) the bottleneck really is decision/assurance cycle time and Earth-in-the-loop latency, not path optimality; (2) RL-for-rover-autonomy works well enough that JPL is putting real money and a field campaign behind it; and (3) a learned onboard policy buys dramatically more ground per unit time (their 10×). Every one of those is load-bearing for this project's problem statement and thesis. ERNEST is the best available "this matters in the real world, right now" citation for the intro.

*Where it complicates things.* Four honest caveats the project should preempt rather than hide:

1. **Some of ERNEST's speed-up is hardware, not policy.** Active suspension lets it cross terrain that stops a rocker-bogie rover at all. A fixed-embodiment benchmark (this project) can't capture that lever, so it must be careful to claim only what it measures: cheaper *assured navigation decisions on a given rover*, not "learning makes rovers 10× faster."
2. **ERNEST does not answer this project's question.** There is no public learned-vs-safety-aware-planner assurance-cost comparison from ERNEST. That is good for novelty, but it also means the project cannot lean on ERNEST as proof — it still has to produce the controlled comparison itself.
3. **JPL is moving toward learned + planned, not learned *instead of* planned.** ERNEST's just-started follow-on explicitly aims to *integrate* the learned suspension/traversal skill with "longer-range intelligent navigation" that "plans an efficient path… to tackle surmountable obstacles and circumnavigate hazardous ones." That is a **hybrid** architecture. It gently undercuts a hard "learned vs. planned" dichotomy — the real frontier may be the combination. The project's framing survives (it's measuring the trade, which is exactly what you'd want before building a hybrid), but the write-up should acknowledge hybridization as the likely endpoint rather than implying planners are obsolete.
4. **Fidelity.** ERNEST's investment in a hardware-calibrated sim is a mild reminder that MuJoCo-only, rigid-body results carry transfer risk. Defensible here because the thesis is scoped to geometric assurance — but worth stating, not glossing.

Net: ERNEST **strengthens the motivation and the "throughput, not path length" stance**, leaves the **central comparison untouched and still open**, and adds one nuance — hybrid learned+planned — the project should name.

---

## 5. Positioning and novelty (for the public write-ups)

The project is **not** made redundant by ERNEST. Clean way to say what's still distinct:

> ERNEST shows, at the *system* level, that learned onboard autonomy lets a rover go an order of magnitude faster. It does so with new hardware, a closed high-fidelity simulator, an HPC training budget, and no published account of *what the autonomy costs per decision* or how it compares to the classical planning it replaces. This project isolates exactly that missing piece: an **open, reproducible, controlled benchmark** of a local-sensing learned policy against a **strong, safety-aware classical planner**, with **assurance cost and exploration throughput as the headline metrics**, on commodity hardware, including a **map-degradation sweep** that probes the unmapped-terrain regime where planners fail and local policies don't.

Distinct contributions that remain fully intact:

- **The decision-cost axis** as the figure of merit (ERNEST reports speed/distance; it does not report compute-per-assured-decision).
- **An open benchmark** versus closed ENav/DARTS sims — reproducibility as a contribution.
- **The fairness protocol**: local-sensing learned vs. full-map planner, with graceful-degradation-under-poor-maps as a first-class result — operationally central for Mars/Moon ground that isn't mapped at rover scale.

Recommended framing: cite ERNEST as **real-world validation of the problem**, and position this work as the **controlled study of the cost trade underneath it**. Avoid over-claiming "as good as a planner"; given JPL is heading toward learned+planned hybrids, frame results as *measuring the trade*, not declaring a winner.

---

## 6. Concrete ideas worth borrowing

- **Calibrate the sim, even cheaply.** ERNEST fed measured terrain-response data into its simulator. Full terramechanics is out of scope, but the project can tune MuJoCo friction/slip against published rover-terrain data and *document* the calibration and its limits — turning a stated risk into a methods strength.
- **Define named, fixed obstacle courses (a "Mars Yard" analogue).** ERNEST's Mars Yard course — sand ripples, rubble, steps, steep slopes — is a repeatable, interpretable, demo-friendly eval. The project should freeze a small set of named procedural eval courses (plus the HiRISE patches) for consistent reporting and good footage.
- **Validate the deferred high-level manager direction.** ERNEST's new "decide *when/how* to use active suspension, integrated with navigation" is essentially the **hierarchical / mode-switching controller** this project parks in Phase 5 (FSM → learned manager). JPL pursuing it is evidence that the deferred direction is the right long-term bet, not scope creep.
- **Model the capability-vs-efficiency trade explicitly.** ERNEST's clutch (active = capable but costly; passive = efficient) mirrors the project's planned **energy/time-efficiency reward term**. ERNEST confirms that trade is real and worth a curriculum stage rather than an afterthought.
- **Add throughput/intervention metrics alongside compute-per-decision.** ERNEST reports distance-per-time and "minimal intervention." The project can adopt **ground-covered-per-fixed-time** and an **autonomy/intervention rate** as reader-legible companions to the harder assurance-cost numbers.
- **Lighting/shadow stress tests for v2.** ERNEST deliberately drove at dawn/dusk/night to mimic lunar long shadows. When the project reaches the v2 camera phase (MARTIAN renderer is already in the plan), vary lighting and shadow length as a perception robustness axis.
- **The "build small, argue it scales" narrative.** ERNEST is explicitly a prototype meant to show a 2×-scale version is feasible. A similar "small, honest, commodity-hardware study that points at the real question" arc is a strong shape for the project's write-ups.

---

## 7. One-glance summary

| Axis | This project | ERNEST |
|---|---|---|
| Core novelty | Learned-vs-planned **assurance-cost benchmark** | **Active-suspension hardware** + learned locomotion |
| What RL controls | Navigation (`[v, yaw]` to goal) | Low-level locomotion / gaits / suspension |
| Baseline / comparison | Strong safety-aware A*/Dijkstra, instrumented | Capability/speed vs. today's rocker-bogie rovers |
| Headline metric | Compute per assured decision; throughput | Speed & distance (16 mi, 10× faster) |
| Simulator | MuJoCo, rigid-body, open | DARTS, high-fidelity, hardware-calibrated, closed |
| Compute | Mac Mini, no GPU (by design) | HPC cluster for training |
| Sensing (now) | Local height-scan, no camera (v1) | Onboard perception in the field |
| Terramechanics | Deferred (Chrono later) | Real terrain in sim + real desert |
| Scale / body | ~50–100 m arena, Mars-first | 26 km traverse, Moon-first |
| Hardware | None (sim only) | Real prototype, field-tested |
| Status of the key question | The whole project | Not publicly answered; new nav project just starting |

---

## Sources

- [NASA Testing Advanced Capabilities for Moon, Mars Rovers — NASA](https://www.nasa.gov/solar-system/moon/nasa-testing-advanced-capabilities-for-moon-mars-rovers/) (Jun 18, 2026)
- [NASA Testing Advanced Capabilities for Moon, Mars Rovers — JPL](https://www.jpl.nasa.gov/news/nasa-testing-advanced-capabilities-for-moon-mars-rovers/) (Jun 18, 2026)
- [Meet ERNEST, NASA's Next-Generation Rover — Gizmodo](https://gizmodo.com/meet-ernest-nasas-next-generation-rover-designed-to-be-faster-and-tougher-2000774216)
- [NASA's new prototype rover navigates 16 miles in extreme terrain — Interesting Engineering](https://interestingengineering.com/ai-robotics/nasas-new-prototype-rover-completes-16-mile-autonomous-desert-trek)
- ERNEST TechPort record (NASA): https://techport.nasa.gov/projects/182846
