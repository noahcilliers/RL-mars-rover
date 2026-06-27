"""RoverNavEnv — Gymnasium environment for goal-reaching navigation.

Phase 0/1 scope: flat terrain, local sensing (proprioception + egocentric goal
vector; the local height-scan is added in Phase 2 when terrain gains features).
Action is the abstract [forward velocity, yaw rate] command. Reward is minimal
on purpose — dense progress-to-goal plus a rollover penalty — per the project
plan (efficiency/energy terms are added later as curriculum).

Observation (8-d, all egocentric / body-frame):
    0  goal distance (m, clipped)
    1  cos(bearing to goal)
    2  sin(bearing to goal)
    3  tilt_x   (world-up expressed in body frame, x)
    4  tilt_y   (world-up expressed in body frame, y)
    5  vx       (forward velocity, body frame)
    6  vy       (lateral velocity, body frame)
    7  yaw rate (body frame)
"""

from __future__ import annotations

import math

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from marsnav.control import SkidSteer
from marsnav.model import load_model


class RoverNavEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 20}

    def __init__(
        self,
        *,
        control_hz: float = 20.0,
        max_episode_seconds: float = 30.0,
        goal_distance: tuple[float, float] = (5.0, 15.0),
        success_radius: float = 1.0,
        rollover_tilt_deg: float = 50.0,
        spawn_half_extent: float = 22.0,   # start/goal sampled within +/- this (m)
        bounds_half_extent: float = 28.0,  # out-of-bounds beyond +/- this (m)
        success_bonus: float = 10.0,
        rollover_penalty: float = 10.0,
        oob_penalty: float = 5.0,
        render_mode: str | None = None,
    ):
        super().__init__()
        self.model = load_model()
        self.data = mujoco.MjData(self.model)
        self.skid = SkidSteer(self.model)

        self.decimation = max(1, round((1.0 / control_hz) / self.model.opt.timestep))
        self.max_steps = int(max_episode_seconds * control_hz)
        self.goal_distance = goal_distance
        self.success_radius = success_radius
        self.up_z_min = math.cos(math.radians(rollover_tilt_deg))
        self.spawn_half = spawn_half_extent
        self.bounds_half = bounds_half_extent
        self.success_bonus = success_bonus
        self.rollover_penalty = rollover_penalty
        self.oob_penalty = oob_penalty
        self.render_mode = render_mode

        self._chassis_id = self.model.body("chassis").id
        self._goal_mocap = int(self.model.body("goal").mocapid[0])
        self._spawn_z = 0.20
        self._goal = np.zeros(2)
        self._prev_dist = 0.0
        self._t = 0
        self._renderer: mujoco.Renderer | None = None

        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        high = np.array([60.0, 1.0, 1.0, 1.0, 1.0, 3.0, 3.0, 6.0], dtype=np.float32)
        low = np.array([0.0, -1.0, -1.0, -1.0, -1.0, -3.0, -3.0, -6.0], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    # -- helpers ---------------------------------------------------------------

    def _yaw(self) -> float:
        r = self.data.xmat[self._chassis_id]
        return math.atan2(r[3], r[0])  # atan2(R10, R00)

    def _obs(self) -> np.ndarray:
        pos = self.data.xpos[self._chassis_id]
        r = self.data.xmat[self._chassis_id]
        tilt_x, tilt_y = float(r[6]), float(r[7])  # world-up in body frame (x, y)

        to_goal = self._goal - pos[:2]
        dist = float(np.linalg.norm(to_goal))
        bearing = math.atan2(to_goal[1], to_goal[0]) - self._yaw()

        vel = self.data.sensor("chassis_vel").data   # body-frame linear velocity
        gyro = self.data.sensor("chassis_gyro").data  # body-frame angular velocity

        obs = np.array(
            [
                min(dist, 60.0),
                math.cos(bearing),
                math.sin(bearing),
                tilt_x,
                tilt_y,
                float(vel[0]),
                float(vel[1]),
                float(gyro[2]),
            ],
            dtype=np.float32,
        )
        return obs

    def _goal_dist(self) -> float:
        return float(np.linalg.norm(self._goal - self.data.xpos[self._chassis_id][:2]))

    @staticmethod
    def _yaw_to_quat(yaw: float) -> np.ndarray:
        return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])

    # -- gym API ---------------------------------------------------------------

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)

        start = self.np_random.uniform(-self.spawn_half, self.spawn_half, size=2)
        yaw = self.np_random.uniform(-math.pi, math.pi)
        d = self.np_random.uniform(*self.goal_distance)
        ang = self.np_random.uniform(-math.pi, math.pi)
        goal = start + d * np.array([math.cos(ang), math.sin(ang)])
        goal = np.clip(goal, -self.spawn_half, self.spawn_half)
        self._goal = goal

        self.data.qpos[0:2] = start
        self.data.qpos[2] = self._spawn_z
        self.data.qpos[3:7] = self._yaw_to_quat(yaw)
        self.data.qvel[:] = 0.0
        self.data.mocap_pos[self._goal_mocap] = [goal[0], goal[1], 0.4]
        mujoco.mj_forward(self.model, self.data)

        self._prev_dist = self._goal_dist()
        self._t = 0
        return self._obs(), {}

    def step(self, action):
        self.skid.apply(self.data, action)
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
        self._t += 1

        obs = self._obs()
        dist = self._goal_dist()
        pos = self.data.xpos[self._chassis_id]
        up_z = float(self.data.xmat[self._chassis_id][8])

        reward = self._prev_dist - dist  # dense progress to goal
        self._prev_dist = dist
        terminated = False
        truncated = False
        info: dict = {
            "is_success": False,
            "rollover": False,
            "out_of_bounds": False,
        }

        if dist < self.success_radius:
            reward += self.success_bonus
            terminated = True
            info["is_success"] = True
        elif up_z < self.up_z_min:
            reward -= self.rollover_penalty
            terminated = True
            info["rollover"] = True
        elif abs(pos[0]) > self.bounds_half or abs(pos[1]) > self.bounds_half:
            reward -= self.oob_penalty
            terminated = True
            info["out_of_bounds"] = True
        elif self._t >= self.max_steps:
            truncated = True

        return obs, float(reward), terminated, truncated, info

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=540, width=720)
        self._renderer.update_scene(self.data, camera="chase")
        return self._renderer.render()

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
