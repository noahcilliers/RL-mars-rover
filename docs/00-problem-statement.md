# Problem Statement

**Project:** Learned vs. Planned Navigation for Mars Surface Traversal

## The problem

The bottleneck in Mars rover navigation is not safety failure — it is **how little ground a rover can cover in a given amount of time.** Rovers explore slowly, and the thing that makes them slow is the **cost of being confident that the next move is safe.** Two scarce resources drive that cost: **onboard compute** and **assurance**. Compute is severely limited, and the algorithms used to assure a safe path are expensive to run. Expensive assurance on a small compute budget means the rover spends most of its time deciding rather than driving — and that directly reduces how much it can explore in the same window of time.

In short: **small compute + expensive assurance → slow decisions → less exploration per sol.** That is the problem this project addresses.

## Why rovers are slow

A rover's onboard processor is radiation-hardened and, by Earth standards, very weak (Perseverance-class hardware runs on the order of a few hundred MHz). When the rover drives itself, the heavy part of each step is not moving — it is the safety evaluation: assessing the terrain ahead, checking whether the vehicle body can cross it without tipping, getting stuck, or damaging a wheel, and rejecting candidate paths that fail. That assurance step is computationally costly, and on such limited hardware it dominates the driving cycle: stop, sense, evaluate, inch forward, repeat.

Communication latency compounds this. The Earth–Mars round trip is on the order of tens of minutes, with only a few contact windows per sol, so a rover cannot be driven in real time. Where the rover cannot decide for itself cheaply, humans on Earth must plan and vet routes against carefully prepared maps — a slow, ground-in-the-loop cycle. The rover's caution is therefore not a free safety margin; it is paid for in speed and in dependence on human oversight.

## What is *not* the problem

It is worth stating plainly, because it is easy to assume otherwise: rovers tipping over is **not** the bottleneck. No Mars rover has ever rolled over. They avoid it precisely *by* being slow and conservative and by relying on human route approval. The real terrain hazards that have actually hurt missions are **immobilization** (Spirit became permanently embedded in soft soil; Opportunity was stuck in a dune for weeks) and **wheel damage** (Curiosity's wheels were punctured by sharp rocks, forcing slower, more cautious driving). All of these are managed today by spending more time and more caution — that is, by exploring less.

## What this implies

If the limiting factor is the cost of assured navigation decisions, then the lever is to make those decisions **cheap and fast** while keeping them trustworthy. A learned policy is attractive on exactly this axis: it produces a decision in constant time (a single forward pass) rather than running an expensive search-and-check loop, and it can act on **local sensing** rather than requiring a high-quality global map prepared and uploaded from Earth. If such a policy can decide as safely as a classical planner while costing far less per decision and depending far less on prior maps, the rover can drive faster, more autonomously, and cover more ground in the same time.

That is the question this project investigates: **can a learned, locally-sensing navigation policy deliver the same confidence as classical planners at a fraction of the assurance cost — and so let a rover explore more in the time it has?**
