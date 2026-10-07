"""Configuration loading shared by ROS nodes and standalone tools."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml


def default_config_path() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("wuji_data_pipeline")) / "config" / "pipeline.yaml"
    except Exception:
        return Path(__file__).resolve().parents[1] / "config" / "pipeline.yaml"


def load_config(path: Optional[str] = None) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve() if path else default_config_path()
    if not config_path.is_file():
        raise FileNotFoundError(f"pipeline config not found: {config_path}")
    with config_path.open() as handle:
        loaded = yaml.safe_load(handle) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"pipeline config root must be a mapping: {config_path}")
    loaded["_config_path"] = str(config_path)
    return loaded


def section(config: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"config section {name!r} must be a mapping")
    return deepcopy(dict(value))
