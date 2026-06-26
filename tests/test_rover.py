"""Sanity checks for the rover model + skid-steer control.

These encode the verified Phase-0 behaviour and guard against regressions —
notably the heightfield heading-drift bug that sphere wheels fixed.
"""

from __future__ import annotations

import math

import mujoco
import numpy as np
import pytest

from marsnav.control import SkidSteer, heading
from marsnav.model import load_model


def _drive(action, seconds: float = 3.0) -> dict:
    """Run a constant command from a fresh reset; return displacement metrics
    expressed relative to the starting heading."""
    model = load_model()
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ctl = SkidSteer(model)

    n = int(seconds / model.opt.timestep)
    h0 = hprev = heading(data, model)
    x0, y0 = float(data.qpos[0]), float(data.qpos[1])
    hacc = 0.0
    for _ in range(n):
        ctl.apply(data, action)
        mujoco.mj_step(model, data)
        h = heading(data, model)
        hacc += (h - hprev + math.pi) % (2 * math.pi) - math.pi
        hprev = h
    dx, dy = float(data.qpos[0]) - x0, float(data.qpos[1]) - y0
    fwd = dx * math.cos(h0) + dy * math.sin(h0)
    lat = -dx * math.sin(h0) + dy * math.cos(h0)
    return {"fwd": fwd, "lat": lat, "dist": math.hypot(dx, dy),
            "dheading_deg": math.degrees(hacc), "qpos": data.qpos.copy()}


def test_model_loads():
    model = load_model()
    assert model.nu == 4  # four wheel velocity actuators
    for name in ("chassis", "wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr"):
        model.body(name)  # raises if missing
    assert model.opt.gravity[2] == pytest.approx(-3.71, abs=1e-3)  # Mars gravity


def test_settles_upright():
    model = load_model()
    data = mujoco.MjData(model)
    for _ in range(400):
        mujoco.mj_step(model, data)
    assert np.isfinite(data.qpos).all()
    assert 0.15 < float(data.qpos[2]) < 0.25  # rests near spawn height
    up_z = data.xmat[model.body("chassis").id].reshape(3, 3)[2, 2]
    assert up_z > 0.99  # essentially level


def test_drives_straight():
    r = _drive((1.0, 0.0))
    assert r["fwd"] > 2.5               # makes forward progress
    assert abs(r["lat"]) < 0.3          # no large lateral drift
    assert abs(r["dheading_deg"]) < 5.0  # no heading drift (the hfield-contact bug)


def test_turns_in_place():
    r = _drive((0.0, 1.0))
    assert abs(r["dheading_deg"]) > 20.0  # actually rotates
    assert r["dist"] < 0.2                # stays roughly in place


def test_action_mapping_symmetry():
    """Straight command -> equal left/right wheel targets; opposite for spin."""
    ctl = SkidSteer(load_model())
    straight = ctl.action_to_ctrl((1.0, 0.0))
    assert straight[0] == pytest.approx(straight[1])  # fl == fr
    spin = ctl.action_to_ctrl((0.0, 1.0))
    assert spin[0] == pytest.approx(-spin[1])  # fl == -fr
