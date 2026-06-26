"""Sanity-check the skid-steer mapping with scripted commands.

Headless (prints pose diagnostics, optional PNG):
    uv run python scripts/manual_drive.py
    uv run python scripts/manual_drive.py --out /tmp/drive.png

Interactive viewer with a fixed command:
    uv run python scripts/manual_drive.py --interactive --v 1 --w 0
"""

from __future__ import annotations

import argparse
import math

import mujoco
import numpy as np

from marsnav.control import SkidSteer, heading
from marsnav.model import load_model


def run(model, data, ctl, action, seconds):
    n = int(seconds / model.opt.timestep)
    x0, y0 = float(data.qpos[0]), float(data.qpos[1])
    h0 = hprev = heading(data, model)
    hacc = 0.0  # accumulate per-step heading change so we can exceed +/-180 deg
    for _ in range(n):
        ctl.apply(data, action)
        mujoco.mj_step(model, data)
        h = heading(data, model)
        hacc += (h - hprev + math.pi) % (2 * math.pi) - math.pi
        hprev = h
    dx, dy = float(data.qpos[0]) - x0, float(data.qpos[1]) - y0
    # forward / lateral displacement relative to the *start* heading
    fwd = dx * math.cos(h0) + dy * math.sin(h0)
    lat = -dx * math.sin(h0) + dy * math.cos(h0)
    return dict(fwd=fwd, lat=lat, dist=math.hypot(dx, dy), dheading_deg=math.degrees(hacc))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--v", type=float, default=1.0)
    ap.add_argument("--w", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=4.0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    model = load_model()
    data = mujoco.MjData(model)
    ctl = SkidSteer(model)

    print(f"geometry: wheel_radius={ctl.wheel_radius:.3f} m  track_width={ctl.track_width:.3f} m")
    print(f"limits  : v_max={ctl.v_max} m/s  w_max={ctl.w_max} rad/s\n")

    if args.interactive:
        from mujoco import viewer

        action = (args.v, args.w)
        with viewer.launch_passive(model, data) as v:
            while v.is_running():
                ctl.apply(data, action)
                mujoco.mj_step(model, data)
                v.sync()
        return

    # Test 1: drive straight (action = [1, 0])
    mujoco.mj_resetData(model, data)
    r1 = run(model, data, ctl, (1.0, 0.0), args.seconds)
    print(f"STRAIGHT [v=1, w=0] for {args.seconds}s:")
    print(f"  forward={r1['fwd']:.3f} m  lateral={r1['lat']:+.3f} m  "
          f"heading_drift={r1['dheading_deg']:+.2f} deg")
    print(f"  (ideal forward ~ {ctl.v_max * args.seconds:.2f} m)\n")

    # Test 2: turn in place (action = [0, 1])
    mujoco.mj_resetData(model, data)
    r2 = run(model, data, ctl, (0.0, 1.0), args.seconds)
    print(f"TURN-IN-PLACE [v=0, w=1] for {args.seconds}s:")
    print(f"  heading_change={r2['dheading_deg']:+.2f} deg  drift_dist={r2['dist']:.3f} m")
    print(f"  (ideal heading ~ {math.degrees(ctl.w_max * args.seconds):+.1f} deg)\n")

    # Test 3: gentle arc (action = [1, 0.5])
    mujoco.mj_resetData(model, data)
    r3 = run(model, data, ctl, (1.0, 0.5), args.seconds)
    print(f"ARC [v=1, w=0.5] for {args.seconds}s:")
    print(f"  forward={r3['fwd']:.3f} m  lateral={r3['lat']:+.3f} m  "
          f"heading_change={r3['dheading_deg']:+.2f} deg")

    if args.out:
        renderer = mujoco.Renderer(model, height=540, width=720)
        renderer.update_scene(data, camera="chase")
        from PIL import Image

        Image.fromarray(renderer.render()).save(args.out)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
