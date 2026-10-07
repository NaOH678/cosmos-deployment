"""The image-plane projection's arithmetic, and the one mistake that looks plausible.

``tools/render_fk_projection.py`` projects ``anchor + prediction[t]``, which is already
**camera frame**, so it must call ``verify_fk_camera_projection.project``.  The
neighbouring ``render_fk_overlay.project_episode`` takes **base frame** and applies
base→camera itself.  Reaching for the wrong one applies the transform twice: the
result is shifted by the camera's ~1.4 m offset from the base and rotated 180°, which
lands the skeleton *near* the hand rather than nowhere near it -- it reads as a bad
model, not a bad call.  Test 2 is the teeth for exactly that.

Run:  PYTHONPATH=. <venv>/bin/python tools/render_fk_projection_test.py
"""

import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from verify_fk_camera_projection import (  # noqa: E402
    CX,
    CY,
    FX,
    FY,
    IMG_H,
    IMG_W,
    project,
    roll_about_z,
)

# Mirrors render_fk_projection's constants; if those move, this test must move too.
CONTENT_H, CONTENT_W, HEAD_TOP = 716, 544, 308
SCALE = 544 / 640  # compose_views: 640 wide composite resized to 544


def test_project_matches_hand_arithmetic():
    """The published intrinsics, applied by hand, must equal ``project``."""
    cam = np.array([[0.10, -0.05, 0.80], [-0.20, 0.03, 1.50]], np.float64)
    uv, front = project(cam)
    for i in range(len(cam)):
        x, y, z = cam[i]
        assert front[i]
        assert np.isclose(uv[i, 0], FX * x / z + CX, atol=1e-9), uv[i]
        assert np.isclose(uv[i, 1], FY * y / z + CY, atol=1e-9), uv[i]
    print(f"✅ project() 与手算内参一致（{len(cam)} 个点，atol 1e-9）")

    # The optical axis lands on the principal point.
    axis, front = project(np.array([[0.0, 0.0, 0.5]]))
    assert front[0] and np.allclose(axis[0], [CX, CY], atol=1e-9), axis
    print(f"✅ 光轴 -> 主点 ({CX:.2f}, {CY:.2f})")

    # Behind the camera is not projected.
    _, front = project(np.array([[0.0, 0.0, -0.5]]))
    assert not front[0], "points behind the camera must be flagged"
    print("✅ z<=0 的点被标为不可见")


def test_double_transform():
    """Teeth: base-frame points through ``project`` are wrong, and wrong plausibly."""
    import verify_fk_camera_projection as v

    if not Path(v.URDF).is_file():
        print(f"⚠️  跳过双重变换对照：找不到 URDF {v.URDF}")
        return
    roll = roll_about_z(180.0)
    r, t = v.base_to_head_camera(v.URDF)
    r, t = roll @ r, roll @ t

    # A point in front of the head, expressed in the BASE frame.
    p_base = np.array([[0.30, -0.20, 1.20]], np.float64)
    p_cam = p_base @ r.T + t

    uv_correct, _ = project(p_cam)  # what the tool must do
    uv_wrong, _ = project(p_base)  # the double-transform mistake

    shift_px = float(np.linalg.norm(uv_correct[0] - uv_wrong[0]))
    assert shift_px > 50.0, f"the two should differ a lot, got {shift_px:.1f} px"
    print(f"🔬 双重变换对照：同一只手，正确 {uv_correct[0].round(1)} vs 误用 {uv_wrong[0].round(1)}")
    print(f"   相差 {shift_px:.0f} px —— 看起来像'模型偏了'，实际是调用错了函数")


def test_canvas_geometry_is_self_consistent():
    """The composed canvas: wrist stacked over head, then scaled to the content."""
    # right_wrist is 848x480, resized to the shared width 640 -> 362 rows;
    # head is 640x480.  Stacked: 842 rows.  Scaled by 544/640 = 0.85 -> 716.
    wrist_rows = round(480 * (640 / 848))
    composite_rows = wrist_rows + 480
    assert round(composite_rows * SCALE) == CONTENT_H, round(composite_rows * SCALE)
    assert round(640 * SCALE) == CONTENT_W, round(640 * SCALE)
    assert round(wrist_rows * SCALE) == HEAD_TOP, round(wrist_rows * SCALE)
    print(
        f"✅ 画布几何自洽：wrist {wrist_rows} + head 480 = {composite_rows} 行"
        f" × {SCALE} = {CONTENT_H}，head 起始 {HEAD_TOP}"
    )


def test_gt_lands_in_frame():
    """A real recorded case: the GT skeleton must project inside the image.

    This is the offline half of the check the rendered picture makes visually.  It
    skips when no eval output is present rather than pretending to pass.
    """
    runs = Path("/data/shichaojian/runs/cosmos")
    cases = sorted(runs.glob("fk-singlerighthand-edge-9/**/fk_eval/step_*/val_00/prediction.npz"))
    if not cases:
        print("⚠️  跳过真实 case 检查：磁盘上没有找到 eval 产物")
        return
    data = np.load(cases[-1], allow_pickle=True)
    gt_cam = data["anchor"][None] + data["target"]  # [T,21,3] camera metres
    inside = 0
    total = 0
    for t in range(gt_cam.shape[0]):
        uv, front = project(gt_cam[t])
        ok = front & (uv[:, 0] >= 0) & (uv[:, 0] < IMG_W) & (uv[:, 1] >= 0) & (uv[:, 1] < IMG_H)
        inside += int(ok.sum())
        total += len(ok)
    frac = inside / total
    assert frac > 0.9, f"only {frac:.1%} of GT keypoints land in frame -- projection suspect"
    print(
        f"✅ 真实 case ({cases[-1].parts[-3]}/{cases[-1].parts[-2]}): "
        f"GT {inside}/{total} = {frac:.1%} 的关键点落在 640x480 内"
    )


def test_clean_arm_has_no_latent():
    """A case with no generated video must not read its placeholder as a latent.

    ``fk_visualize.write_case`` always writes the ``vision`` key, and the clean-video
    arm writes ``None`` into it -- which ``np.savez`` stores as a **0-d object
    array**, so ``data["vision"] is not None`` is *True*.  Treating that as
    ``[N,C,T,H,W]`` raises IndexError inside ``main`` before a single frame is
    drawn.  That matters because the clean arm is exactly the case anyone runs this
    tool on *first* -- it is where the GT skeleton on the real video gets checked,
    the one panel that validates the projection.
    """
    import tempfile

    from render_fk_projection import stored_latent

    with tempfile.TemporaryDirectory() as tmp:
        clean = Path(tmp) / "clean.npz"
        np.savez_compressed(clean, prediction=np.zeros((4, 3, 3), np.float32), vision=None)
        data = np.load(clean, allow_pickle=True)
        assert "vision" in data.files, "the key is present even with no generated video"
        assert data["vision"] is not None, "guard-by-identity is exactly what fails here"
        assert stored_latent(data) is None, "the placeholder was read as a latent"
        assert stored_latent(data, no_generated=True) is None

        joint = Path(tmp) / "joint.npz"
        real = np.zeros((1, 48, 9, 46, 34), np.float16)
        np.savez_compressed(joint, prediction=np.zeros((4, 3, 3), np.float32), vision=real)
        got = stored_latent(np.load(joint, allow_pickle=True))
        assert got is not None and got.shape == (48, 9, 46, 34), getattr(got, "shape", None)
        assert got.dtype == np.float16
    print("✅ 无生成视频的 case（clean arm）不再被误读为 latent；真有 latent 时形状正确")


def test_contact_sheet_pads_a_half_row():
    """Three panels must still lay out -- the default --stills gives three.

    ``--stills`` defaults to ``1,9,17,32`` while the horizon is 32 steps, so index
    32 never comes up and the sheet has three panels, not four.  Two-up layout then
    leaves a half row, and without padding the rows differ in width and
    ``np.concatenate`` raises -- after the mp4 is already written, so the crash
    looks like a broken run rather than a cosmetic one.
    """
    from render_fk_projection import contact_sheet

    panel = np.zeros((10, 20, 3), np.uint8)

    wanted = sorted({1, 9, 17, 32} & set(range(32)))
    assert wanted == [1, 9, 17], wanted
    sheet = contact_sheet([panel] * len(wanted))
    assert sheet.shape == (20, 40, 3), sheet.shape  # 2 rows of 2, last one padded
    print(f"✅ 默认 --stills {wanted} -> 3 panel，补半行后成 {sheet.shape[0]}x{sheet.shape[1]}")

    for count in (1, 2, 3, 4, 5):
        got = contact_sheet([panel] * count)
        # Every row is padded to two panels, a lone one included, so the width is
        # constant and only the row count varies.
        assert got.shape == ((count + 1) // 2 * 10, 40, 3), (count, got.shape)
    print("✅ 1-5 个 panel 都能拼，宽度恒为两列")


def main():
    test_project_matches_hand_arithmetic()
    test_double_transform()
    test_canvas_geometry_is_self_consistent()
    test_gt_lands_in_frame()
    test_clean_arm_has_no_latent()
    test_contact_sheet_pads_a_half_row()
    print("\nPASS")


if __name__ == "__main__":
    main()
