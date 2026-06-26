"""Loading helpers for the rover MuJoCo model."""

from __future__ import annotations

from pathlib import Path

import mujoco

ASSETS_DIR = Path(__file__).parent / "assets"
ROVER_XML = ASSETS_DIR / "rover.xml"


def load_model() -> mujoco.MjModel:
    """Compile and return the rover MjModel."""
    return mujoco.MjModel.from_xml_path(str(ROVER_XML))
