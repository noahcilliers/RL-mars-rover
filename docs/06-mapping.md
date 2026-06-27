# Mapping

Brief reference for the kind of terrain representation this project uses, where it comes from in simulation, and how it would be produced on real hardware. Companion to `03-project-plan.md` and `04-terrain-and-simulation-environment.md`.

## What we mean by "mapping"

The terrain representation is a **2.5D heightmap** (an elevation grid), not a full 3D model. A regular 2D grid of cells, each cell storing a single height value `z = f(x, y)` — the elevation of the surface at that point. It is called **2.5D** because it captures surface relief but not true volumetric 3D: one height per `(x, y)`, so it cannot represent overhangs, caves, or anything stacked vertically. For Mars/Moon surface traversal that is a fine trade — the hazards that matter (slopes, rocks, craters, steps) are all surface relief, which a heightmap captures exactly, and a grid is cheap to store, sample, and reason over.

Two distinct heightmaps appear in this project, and they belong to different agents (see `05-comparison-to-ernest.md` and the map discussion in the plan):

- The **policy's input** is a *local, egocentric* heightmap — a forward-biased window roughly **8–10 m** around the rover, re-sampled every step as the rover moves. It is local sensing, never a global map.
- The **classical planner's input** is the *full, global* heightmap of the arena (then progressively degraded in the map-quality sweep). Only the planner depends on a prior map.

This doc is about the local 2.5D heightmap the policy uses.

## Where the heightmap comes from in v1 (the simulator)

In v1 there is **no perception pipeline.** The terrain is procedurally generated and stored as MuJoCo's heightfield (`model.hfield_data`; see `04`). The local height-scan handed to the policy is **sampled directly from that ground-truth heightfield** around the rover's current pose — a cheap array lookup each step, optionally supplemented by raycast rangefinder beams. There is no stereo, no reconstruction, no SLAM: we simply index the terrain we generated.

This makes the v1 observation **privileged** — the policy sees clean, exact local geometry that a real rover would have to work to obtain. That is deliberate. It lets v1 isolate the *navigation* question (can a local-sensing policy match the planner's success/safety at lower assurance cost) without entangling it with the *perception* question. The "iterative" creation of the map in v1 is therefore just: each timestep, re-sample the heightfield window at the rover's new position. Cheap and exact.

## How the heightmap would be produced physically

On real hardware the same 2.5D local heightmap is not free — it has to be **built from onboard sensing and updated iteratively as the rover drives.** The standard pipeline:

1. **Range sensing.** An onboard sensor measures depth — **stereo cameras** (the Mars-heritage choice: disparity between a stereo pair yields a 3D point cloud), or alternatively a depth camera or lidar.
2. **Pose estimation.** Each frame's points are placed into a consistent frame using the rover's pose from **visual(-inertial) odometry / SLAM**, which tracks where the rover is and corrects drift.
3. **Elevation fusion.** The points are merged into an **egocentric 2.5D grid** centered on the rover — each cell accumulating an estimated height (and typically a variance). As the rover moves, the grid **shifts with it** and cells update incrementally as new measurements arrive. This incremental fuse-and-shift is what makes the map "iterative": it is continuously rebuilt around the rover rather than computed once.

This is exactly the family of techniques used by JPL's stereo-NavCam → digital elevation map pipeline feeding GESTALT/ENav traversability analysis, and by the GNC architecture's upper "Full Navigation" level (NavCam stereo + SLAM producing a local elevation map several meters ahead). In modern robotics the off-the-shelf version is robot-centric elevation mapping (e.g., ETH's GridMap / `elevation_mapping`, GPU variant `elevation_mapping_cupy`).

## v1 → v2: from sim lookup to real perception

The privileged sim lookup is a stand-in for the pipeline above, and closing that gap is precisely the **v2** step in the plan: replace the privileged height-scan with an **onboard depth camera** and train a **student** policy (distilled from the v1 "teacher") that must *estimate* the local 2.5D heightmap from camera + proprioception rather than being handed it. So "how do we actually build this heightmap" is a sim array-query in v1 and a stereo/SLAM perception problem in v2 — same 2.5D representation, very different cost to obtain.
