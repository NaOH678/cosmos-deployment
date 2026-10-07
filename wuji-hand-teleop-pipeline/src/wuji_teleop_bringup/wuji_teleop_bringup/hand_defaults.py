"""Dexterous-hand hardware defaults — loaded from wujihand_ik.yaml.

All launch files read hand configuration through this module.
Actual parameters live in: wujihand_output/config/wujihand_ik.yaml

Usage:
    from wuji_teleop_bringup.hand_defaults import (
        LEFT_HAND_SERIAL, RIGHT_HAND_SERIAL,
        LEFT_HAND_NAME, RIGHT_HAND_NAME,
        DRIVER_PUBLISH_RATE, DRIVER_FILTER_CUTOFF_FREQ, DRIVER_DIAGNOSTICS_RATE,
    )
"""

from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory


def _load_hand_config() -> dict:
    """Load hand config from wujihand_output package share directory."""
    config_path = (
        Path(get_package_share_directory("wujihand_output"))
        / "config"
        / "wujihand_ik.yaml"
    )
    with open(config_path) as f:
        return yaml.safe_load(f)


_config = _load_hand_config()

# ===== Exported constants (back-compat; launch files need no changes) =====

LEFT_HAND_SERIAL: str = _config["left_hand"]["serial_number"]
RIGHT_HAND_SERIAL: str = _config["right_hand"]["serial_number"]

LEFT_HAND_NAME: str = _config["left_hand"]["name"]
RIGHT_HAND_NAME: str = _config["right_hand"]["name"]

DRIVER_PUBLISH_RATE: float = float(_config["driver"]["publish_rate"])
DRIVER_FILTER_CUTOFF_FREQ: float = float(_config["driver"]["filter_cutoff_freq"])
DRIVER_DIAGNOSTICS_RATE: float = float(_config["driver"]["diagnostics_rate"])
DRIVER_RECOVERY_DURATION: float = float(
    _config["driver"].get("recovery_duration", 5.0)
)
DRIVER_RECOVERY_TOLERANCE: float = float(
    _config["driver"].get("recovery_tolerance", 0.12)
)
DRIVER_RECOVERY_SETTLE_TIMEOUT: float = float(
    _config["driver"].get("recovery_settle_timeout", 2.0)
)

LEFT_HAND_INITIAL_POSITION: list[float] = [
    float(value) for value in _config["left_hand"].get("initial_position", [])
]
RIGHT_HAND_INITIAL_POSITION: list[float] = [
    float(value) for value in _config["right_hand"].get("initial_position", [])
]

for _side, _pose in (
    ("left", LEFT_HAND_INITIAL_POSITION),
    ("right", RIGHT_HAND_INITIAL_POSITION),
):
    if _pose and len(_pose) != 20:
        raise ValueError(
            f"{_side}_hand.initial_position must contain 20 radians, "
            f"got {len(_pose)}"
        )
