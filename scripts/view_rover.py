"""Inspect the rover model.

Interactive (opens a MuJoCo window):
    uv run python scripts/view_rover.py

Headless sanity render (no display; writes a PNG + prints diagnostics):
    uv run python scripts/view_rover.py --headless --out /tmp/rover.png
    uv run python scripts/view_rover.py --headless --drive 8   # drive forward
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np

from marsnav.model import load_model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--headless", action="store_true", help="render to PNG instead of a GUI")
    ap.add_argument("--steps", type=int, default=400, help="sim steps before rendering")
    ap.add_argument("--drive", type=float, default=0.0, help="constant wheel velocity (rad/s)")
    ap.add_argument("--out", default="rover_view.png", help="output PNG (headless)")
    args = ap.parse_args()

    model = load_model()
    data = mujoco.MjData(model)

    if not args.headless:
        from mujoco import viewer

        if args.drive:
            data.ctrl[:] = args.drive
        viewer.launch(model, data)
        return

    if args.drive:
        data.ctrl[:] = args.drive

    z0 = float(data.qpos[2])
    for _ in range(args.steps):
        mujoco.mj_step(model, data)

    pos = data.qpos[:3]
    quat = data.qpos[3:7]
    # up-vector z component: 1.0 = perfectly upright, < 0 = flipped
    up_z = float(data.sensordata[model.sensor("chassis_up").adr[0] + 2])
    print(f"steps         : {args.steps}")
    print(f"chassis z     : {z0:.3f} -> {float(pos[2]):.3f}  (settled height)")
    print(f"chassis xy    : ({float(pos[0]):.3f}, {float(pos[1]):.3f})")
    print(f"upright (up_z): {up_z:.3f}  (1=upright)")
    print(f"qpos finite   : {bool(np.isfinite(data.qpos).all())}")

    renderer = mujoco.Renderer(model, height=540, width=720)
    renderer.update_scene(data, camera="chase")
    img = renderer.render()
    try:
        from PIL import Image

        Image.fromarray(img).save(args.out)
        print(f"wrote         : {args.out}")
    except Exception as e:  # pragma: no cover
        print(f"render saved skipped: {e}")


if __name__ == "__main__":
    main()
