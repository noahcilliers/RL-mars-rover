"""Gymnasium environments wrapping the MuJoCo rover model."""

from gymnasium.envs.registration import register

from marsnav.envs.rover_nav import RoverNavEnv

register(
    id="marsnav/RoverNav-v0",
    entry_point="marsnav.envs.rover_nav:RoverNavEnv",
)

__all__ = ["RoverNavEnv"]
