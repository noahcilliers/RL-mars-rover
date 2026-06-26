"""Skid-steer control: map an abstract [forward velocity, yaw rate] command to
per-wheel angular-velocity targets for the velocity actuators.

This is the v1 action abstraction from the plan: the policy outputs a normalized
[v, yaw] in [-1, 1]^2; we scale to physical limits and convert via differential
drive. Geometry (wheel radius, track width) is read from the model so there is a
single source of truth with the MJCF.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np

# Wheel actuator order used everywhere: front-left, front-right, rear-left, rear-right.
WHEEL_ACTUATORS = ("wheel_fl", "wheel_fr", "wheel_rl", "wheel_rr")
_LEFT = (0, 2)   # fl, rl
_RIGHT = (1, 3)  # fr, rr


@dataclass
class SkidSteer:
    """Differential-drive command mapping for the 4-wheel rover."""

    model: mujoco.MjModel
    v_max: float = 1.0   # m/s   (max forward speed)
    # Max yaw-rate *command*. Skid-steer turn authority is geometry-limited
    # (~0.5 rad/s in place is what this rover actually achieves); 1.0 keeps a
    # little headroom without a large dead zone in the action range. Tunable.
    w_max: float = 1.0   # rad/s

    def __post_init__(self) -> None:
        # wheel radius from the front-left wheel geom (first geom of that body)
        bid = self.model.body("wheel_fl").id
        gid = int(self.model.body_geomadr[bid])
        self.wheel_radius = float(self.model.geom_size[gid][0])
        # track width = lateral distance between left and right wheels
        self.track_width = 2.0 * abs(float(self.model.body("wheel_fl").pos[1]))
        # actuator ids in our canonical order
        self.act_ids = np.array(
            [self.model.actuator(name).id for name in WHEEL_ACTUATORS], dtype=int
        )

    def action_to_ctrl(self, action) -> np.ndarray:
        """Map normalized action in [-1, 1]^2 to a length-4 ctrl vector (rad/s),
        ordered as WHEEL_ACTUATORS."""
        a = np.asarray(action, dtype=np.float64).reshape(2)
        v = float(np.clip(a[0], -1.0, 1.0)) * self.v_max
        w = float(np.clip(a[1], -1.0, 1.0)) * self.w_max

        half = 0.5 * self.track_width
        left_lin = v - w * half
        right_lin = v + w * half
        left = left_lin / self.wheel_radius
        right = right_lin / self.wheel_radius

        ctrl = np.empty(4, dtype=np.float64)
        for i in _LEFT:
            ctrl[i] = left
        for i in _RIGHT:
            ctrl[i] = right
        return ctrl

    def apply(self, data: mujoco.MjData, action) -> np.ndarray:
        """Write the mapped command into data.ctrl at the right actuator slots."""
        ctrl = self.action_to_ctrl(action)
        data.ctrl[self.act_ids] = ctrl
        return ctrl


def heading(data: mujoco.MjData, model: mujoco.MjModel) -> float:
    """Chassis yaw (rad) in the world frame, from the body x-axis."""
    bid = model.body("chassis").id
    rot = data.xmat[bid].reshape(3, 3)
    fwd = rot[:, 0]  # local +x in world coords
    return float(np.arctan2(fwd[1], fwd[0]))
