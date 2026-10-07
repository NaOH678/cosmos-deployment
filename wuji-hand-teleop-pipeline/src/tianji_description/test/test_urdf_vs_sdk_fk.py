"""Validate the assembled URDF and cross-check its FK against the vendor SDK."""
from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np
import pytest
import yaml


PACKAGE_DIR = Path(__file__).resolve().parents[1]
URDF_PATH = PACKAGE_DIR / "urdf" / "tianji_dual_arm.urdf"
JOINT_MAP_PATH = PACKAGE_DIR / "config" / "joint_map.yaml"
TELEOP_PACKAGE_ROOT = PACKAGE_DIR.parent / "tianji_teleop"

SIDE_DATA = {
    "left": {
        "suffix": "L",
        "init_deg": np.array([55.0, -65.0, -70.0, -60.0, 60.0, 0.0, 0.0]),
        "seed": 696,
    },
    "right": {
        "suffix": "R",
        "init_deg": np.array([-55.0, -65.0, 70.0, -60.0, -60.0, 0.0, 0.0]),
        "seed": 697,
    },
}


def _vector(text: Optional[str], default: str = "0 0 0") -> np.ndarray:
    return np.fromstring(text if text is not None else default, sep=" ", dtype=float)


def _rpy_matrix(rpy: np.ndarray) -> np.ndarray:
    """URDF fixed-axis XYZ rotation: Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    roll, pitch, yaw = rpy
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rx = np.array([[1.0, 0.0, 0.0], [0.0, cr, -sr], [0.0, sr, cr]])
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry @ rx


def _origin_matrix(origin: Optional[ET.Element]) -> np.ndarray:
    transform = np.eye(4)
    if origin is not None:
        transform[:3, :3] = _rpy_matrix(_vector(origin.get("rpy")))
        transform[:3, 3] = _vector(origin.get("xyz"))
    return transform


def _axis_angle_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    one_minus_c = 1.0 - c
    transform = np.eye(4)
    transform[:3, :3] = np.array(
        [
            [c + x * x * one_minus_c, x * y * one_minus_c - z * s,
             x * z * one_minus_c + y * s],
            [y * x * one_minus_c + z * s, c + y * y * one_minus_c,
             y * z * one_minus_c - x * s],
            [z * x * one_minus_c - y * s, z * y * one_minus_c + x * s,
             c + z * z * one_minus_c],
        ]
    )
    return transform


def urdf_fk(target_link: str, joint_positions: dict[str, float],
            stop_link: str = "Link_Base") -> np.ndarray:
    """Compute stop_link -> target_link FK directly from URDF joint definitions."""
    root = ET.parse(URDF_PATH).getroot()
    joint_by_child = {joint.find("child").get("link"): joint
                      for joint in root.findall("joint")}
    chain = []
    link = target_link
    while link != stop_link:
        if link not in joint_by_child:
            raise ValueError(
                f"{target_link} is not connected to {stop_link} (stopped at {link})")
        joint = joint_by_child[link]
        chain.append(joint)
        link = joint.find("parent").get("link")

    transform = np.eye(4)
    for joint in reversed(chain):
        transform = transform @ _origin_matrix(joint.find("origin"))
        joint_type = joint.get("type")
        if joint_type in {"revolute", "continuous"}:
            angle = joint_positions[joint.get("name")]
            transform = transform @ _axis_angle_matrix(
                _vector(joint.find("axis").get("xyz"), "1 0 0"), angle
            )
        elif joint_type != "fixed":
            raise ValueError(f"unsupported joint type: {joint_type}")
    return transform


def _pose_error(reference: np.ndarray, actual: np.ndarray) -> tuple[float, float]:
    position_mm = float(np.linalg.norm(reference[:3, 3] - actual[:3, 3]) * 1000.0)
    relative_rotation = reference[:3, :3].T @ actual[:3, :3]
    cosine = np.clip((np.trace(relative_rotation) - 1.0) / 2.0, -1.0, 1.0)
    orientation_deg = float(np.degrees(np.arccos(cosine)))
    return position_mm, orientation_deg


def _load_joint_map() -> dict:
    with JOINT_MAP_PATH.open(encoding="utf-8") as stream:
        return yaml.safe_load(stream)


def _sdk_or_skip(side: str):
    if str(TELEOP_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(TELEOP_PACKAGE_ROOT))
    try:
        from tianji_teleop.ik.kine import ArmKinematics

        return ArmKinematics(side)
    except (ImportError, OSError) as exc:  # Linux x86-64 .so cannot load on macOS.
        pytest.skip(f"vendor ArmKinematics unavailable: {type(exc).__name__}: {exc}")


def test_urdf_tree_and_mesh_resources() -> None:
    root = ET.parse(URDF_PATH).getroot()
    links = {link.get("name") for link in root.findall("link")}
    joints = root.findall("joint")
    child_links = {joint.find("child").get("link") for joint in joints}

    expected_links = {"Link_Base", "Link_Stand"}
    for suffix in ("L", "R"):
        expected_links.add(f"Base_{suffix}")
        expected_links.add(f"TCP_Link_{suffix}")
        expected_links.update(f"Link{index}_{suffix}" for index in range(1, 8))
    assert links == expected_links
    assert links - child_links == {"Link_Base"}
    assert len(joints) == 19
    assert not ({"left_chest_base", "left_chest", "right_chest_base", "right_chest",
                 "chest"} & links)

    expected_meshes = {"Link_Base.STL", "Link_Stand.STL"}
    for suffix in ("L", "R"):
        expected_meshes.add(f"Base_{suffix}.STL")
        expected_meshes.add(f"TCP_Link_{suffix}.STL")
        expected_meshes.update(f"Link{index}_{suffix}.STL" for index in range(1, 8))
    mesh_uris = {mesh.get("filename") for mesh in root.iter("mesh")}
    assert mesh_uris == {
        f"package://tianji_description/meshes/{filename}" for filename in expected_meshes
    }
    assert {path.name for path in (PACKAGE_DIR / "meshes").glob("*.STL")} == expected_meshes


@pytest.mark.parametrize("side", ["left", "right"])
def test_real_assembly_mounts(side: str) -> None:
    """Mount joints must match the lab CAD assembly (Base_and_Stand_Asm)."""
    suffix = SIDE_DATA[side]["suffix"]
    sign = 1.0 if side == "left" else -1.0
    root = ET.parse(URDF_PATH).getroot()
    by_name = {j.get("name"): j for j in root.findall("joint")}
    stand = _origin_matrix(by_name["Joint_Stand"].find("origin"))
    np.testing.assert_allclose(stand[:3, 3], [0.0, 0.0, 0.981], atol=1e-9)
    mount = _origin_matrix(by_name[f"Joint0_{suffix}"].find("origin"))
    np.testing.assert_allclose(mount[:3, 3], [0.0, sign * 0.037, 0.140], atol=1e-9)
    expected_rot = _rpy_matrix(np.array([-sign * 1.5708, 0.0, 0.0]))
    np.testing.assert_allclose(mount[:3, :3], expected_rot, atol=1e-9)


def test_joint_map_and_flange_offsets() -> None:
    config = _load_joint_map()
    for side in ("left", "right"):
        suffix = SIDE_DATA[side]["suffix"]
        assert config[side] == [f"Joint{index}_{suffix}" for index in range(1, 8)]
        offset = np.asarray(config[f"flange_offset_{side}"], dtype=float)
        assert offset.shape == (4, 4)
        np.testing.assert_allclose(offset[:3, :3].T @ offset[:3, :3], np.eye(3), atol=1e-12)
        assert np.linalg.det(offset[:3, :3]) == pytest.approx(1.0)
        np.testing.assert_allclose(offset[3], [0.0, 0.0, 0.0, 1.0])


@pytest.mark.parametrize("side", ["left", "right"])
def test_urdf_vs_sdk_fk(side: str) -> None:
    sdk = _sdk_or_skip(side)
    config = _load_joint_map()
    joint_names = config[side]
    suffix = SIDE_DATA[side]["suffix"]
    flange_offset = np.asarray(config[f"flange_offset_{side}"], dtype=float)
    init = np.radians(SIDE_DATA[side]["init_deg"])
    rng = np.random.default_rng(SIDE_DATA[side]["seed"])
    configurations = [init]
    configurations.extend(init + np.radians(rng.uniform(-10.0, 10.0, size=7))
                          for _ in range(5))

    rows = []
    failures = []
    for index, joints in enumerate(configurations):
        positions = dict(zip(joint_names, joints))
        # Chain FK in the arm base frame directly: mount-independent invariant.
        link7_in_base = urdf_fk(f"Link7_{suffix}", positions,
                                stop_link=f"Base_{suffix}")
        corrected_urdf = link7_in_base @ flange_offset
        sdk_fk = sdk.fk(joints)
        raw_error = _pose_error(link7_in_base, sdk_fk)
        corrected_error = _pose_error(corrected_urdf, sdk_fk)
        rows.append((index, raw_error, corrected_error))
        if corrected_error[0] >= 20.0 or corrected_error[1] >= 3.0:
            failures.append(
                "\n".join(
                    [
                        f"case={index} q_deg={np.degrees(joints).tolist()}",
                        f"raw_error_mm_deg={raw_error}",
                        f"corrected_error_mm_deg={corrected_error}",
                        f"T_urdf_corrected=\n{corrected_urdf}",
                        f"T_sdk=\n{sdk_fk}",
                    ]
                )
            )

    print(f"{side} FK errors: case, raw(mm/deg), corrected(mm/deg)")
    for row in rows:
        print(row)
    if failures:
        pytest.fail(f"{side} FK exceeded 20 mm / 3 deg:\n" + "\n\n".join(failures))
