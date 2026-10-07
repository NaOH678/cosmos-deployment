"""Build the DAgger FK URDF by matching mount holes, not fitting video pixels.

Assumptions confirmed by the operator: STL units mm; original base and mounting
holes unchanged; camera installed on the same face and in the same orientation.
Other mesh references resolve to the original local description package.
This is a kinematic URDF: the replacement fixed bracket has no estimated inertia.
"""

import argparse
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial.transform import Rotation


def circle(points, normal):
    origin = points.mean(axis=0)
    u = np.cross(normal, [0, 1, 0])
    if np.linalg.norm(u) < 0.1:
        u = np.cross(normal, [1, 0, 0])
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    basis = np.stack([u, v])
    xy = (points - origin) @ basis.T
    fit = np.linalg.lstsq(np.c_[2 * xy, np.ones(len(xy))], (xy**2).sum(1), rcond=None)[0]
    radius = np.sqrt(fit[2] + sum(fit[:2] ** 2))
    residual = np.std(np.linalg.norm(xy - fit[:2], axis=1))
    assert len(points) >= 8 and residual < 0.015, (len(points), residual)
    return origin + fit[:2] @ basis, float(radius), float(residual)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", type=Path, default=Path("/data/shichaojian/wuji-mjlab/marvin_wuji_d435_description"))
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[1] / "assets/dagger_fk")
    args = ap.parse_args()
    src = args.package / "urdf/marvin_wuji_d435_complete.urdf"
    stl = args.package / "urdf/Terrific Amberis (3).stl"
    mesh = trimesh.load(stl)
    points = mesh.vertices
    bottom = []
    fits = []
    # Regions around the four circular holes on the z=0 mounting face (mm).
    for x in (12, 62):
        for y in (32, 52):
            selected = points[
                (abs(points[:, 2]) < 0.002) & (abs(points[:, 0] - x) < 2.1) & (abs(points[:, 1] - y) < 2.1)
            ]
            center, radius, residual = circle(selected, np.array([0.0, 0.0, 1.0]))
            assert abs(radius - 2) < 0.02
            bottom.append(center)
            fits.append(dict(center_mm=center.tolist(), radius_mm=radius, residual_mm=residual))
    bottom = np.array(bottom)
    assert np.allclose(np.linalg.norm(bottom[[2, 3]] - bottom[[0, 1]], axis=1), 50, atol=0.02)
    assert np.allclose(np.linalg.norm(bottom[[1, 3]] - bottom[[0, 2]], axis=1), 20, atol=0.02)
    # Same physical mounting direction. STL +Z -> old bracket +Y, +Y -> -Z.
    align = np.array([[1.0, 0, 0], [0, 0, 1], [0, -1, 0]])
    old_bottom_center = np.array([-1.5, -2.5, 20.0])
    shift = old_bottom_center - align @ bottom.mean(0)
    # Fit the camera contact face. Opposite face is 4 mm away, so exclude it.
    guess = np.array([-np.sin(np.deg2rad(39)), 0.0, -np.cos(np.deg2rad(39))])
    mask = (mesh.face_normals @ guess > 0.999) & (abs(mesh.triangles_center @ guess + 120.8) < 0.2)
    plane_points = np.unique(mesh.triangles[mask].reshape(-1, 3), axis=0)
    _, _, vh = np.linalg.svd(plane_points - plane_points.mean(0), full_matrices=False)
    normal = vh[-1]
    if normal @ guess < 0:
        normal = -normal
    # Design is symmetric along the hole-pair axis Y; eliminate tessellation noise.
    normal[1] = 0
    normal /= np.linalg.norm(normal)
    plane_d = np.median(plane_points @ normal)
    top = []
    for y in (20.5, 65.5):
        selected = points[
            (abs(points @ normal - plane_d) < 0.02) & (abs(points[:, 0] + 7.93) < 1.5) & (abs(points[:, 1] - y) < 1.5)
        ]
        center, radius, residual = circle(selected, normal)
        assert abs(radius - 1.25) < 0.02
        top.append(center)
        fits.append(dict(center_mm=center.tolist(), radius_mm=radius, residual_mm=residual))
    assert abs(np.linalg.norm(top[1] - top[0]) - 45) < 0.02
    old_normal = np.array([-np.cos(np.deg2rad(40)), -np.sin(np.deg2rad(40)), 0.0])
    new_normal = align @ normal
    angle = np.arctan2(np.cross(old_normal, new_normal)[2], old_normal @ new_normal)
    delta = Rotation.from_rotvec([0, 0, angle]).as_matrix()
    old_top_center = np.array([-29.599019, 162.234412, 20.0]) / 1000
    new_top_center = (align @ np.mean(top, axis=0) + shift) / 1000
    tree = ET.parse(src)
    root = tree.getroot()
    root.set("name", "marvin_wuji_d435_dagger_fk")
    mount = root.find("joint[@name='head_d435_mount_joint']/origin")
    old_xyz = np.fromstring(mount.get("xyz"), sep=" ")
    old_rpy = np.fromstring(mount.get("rpy"), sep=" ")
    new_xyz = new_top_center + delta @ (old_xyz - old_top_center)
    # Left multiplication by Rz(delta) simply adds to URDF yaw, including at gimbal lock.
    new_rpy = old_rpy + [0, 0, angle]
    mount.set("xyz", " ".join(f"{v:.12f}" for v in new_xyz))
    mount.set("rpy", " ".join(f"{v:.12f}" for v in new_rpy))
    bracket = root.find("link[@name='head_camera_bracket_link']")
    bracket.remove(bracket.find("inertial"))
    bracket.insert(0, ET.Comment(" Kinematic replacement: mass/inertia not supplied; do not use for dynamics. "))
    args.out.mkdir(parents=True, exist_ok=True)
    mesh.vertices = (mesh.vertices @ align.T + shift) / 1000
    mesh_path = args.out / "head_camera_bracket_dagger_m.stl"
    mesh.export(mesh_path)
    for node in root.findall(".//mesh"):
        filename = node.get("filename")
        if filename.endswith("/head_camera_bracket.stl"):
            node.set("filename", str(mesh_path.resolve()))
        elif filename.startswith("package://marvin_wuji_d435_description/"):
            node.set("filename", str(args.package / filename.split("package://marvin_wuji_d435_description/")[1]))
    root.insert(
        0,
        ET.Comment(
            " DAgger only. Same base mounting and camera mounting side; replacement STL in mm. No video-fitted correction. "
        ),
    )
    ET.indent(tree, space="  ")
    output = args.out / "marvin_wuji_d435_dagger.urdf"
    tree.write(output, encoding="utf-8", xml_declaration=True)
    report = dict(
        source_urdf=str(src),
        source_stl=str(stl),
        source_urdf_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),
        source_stl_sha256=hashlib.sha256(stl.read_bytes()).hexdigest(),
        hole_fits=fits,
        stl_to_bracket_rotation=align.tolist(),
        stl_to_bracket_translation_mm=shift.tolist(),
        old_camera_xyz_m=old_xyz.tolist(),
        new_camera_xyz_m=new_xyz.tolist(),
        old_camera_rpy_rad=old_rpy.tolist(),
        new_camera_rpy_rad=new_rpy.tolist(),
        camera_rotation_change_deg=float(np.rad2deg(angle)),
        old_top_hole_center_m=old_top_center.tolist(),
        new_top_hole_center_m=new_top_center.tolist(),
        output=str(output),
    )
    (args.out / "construction.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
