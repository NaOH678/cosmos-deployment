# 用 dagger 数据渲染 FK-21 overlay（执行文档）

> ⚠️ **从 mano worktree 迁入，前提已被推翻 —— 用之前先读这段。**
>
> 本文档假设 dagger 与 sandwich 共用同一套相机外参（因为 head 相机是同一台，
> serial `147122073219`）。**这个假设不成立**：dagger 的支架不同，用 sandwich
> 的外参渲染会让骨架整体偏低约 200 px。
>
> 正确的做法是 `tools/build_dagger_camera_urdf.py` —— 它按**安装孔位**解出 dagger
> 的相机 mount，而不是去拟合视频像素；先由那份 URDF 重新生成外参
> （`tools/export_fk_camera_extrinsic.py`），再用本文档里的流程。
>
> 保留本文档，是因为它是本仓库唯一的 dagger overlay 渲染流程，**不是因为它现在的
> 相机模型是对的**。

> 目标：把 FK-21 骨架投影叠在**真实 head 视频**上，输出 mp4 + stills。
> 这是 `tools/render_fk_overlay.py` 的**同类方法**，但数据换成 dagger。
>
> **给接手这份文档的对话：先读完 §1 和 §5 再动手。§5 是完整可跑的脚本，§6 是上
> 一次渲染「完全错误、偏离很大」的排查阶梯 —— 大概率问题在那儿。**

---

## 1. 结论先行：什么要改、什么不要改

| | 要不要改 | 为什么 |
|---|---|---|
| **投影数学**（外参、内参、投影公式） | **❌ 完全不用改** | dagger 与 sandwich **是同一台相机**：serial `147122073219`，`fx/fy/ppx/ppy` 逐位相同（见 §2.3），URDF 同一份，机器人同一个 |
| **数据路径** | ✅ 改 | dagger 的 raw / FK 树跟 sandwich 不是一套布局 |
| **FK 文件的读法** | ✅ 改 | sandwich 读 slim 包的 `fk21/chunk-000/episode_%06d.npz` 的 `right_positions_abs`；dagger 读**每 episode 的 `annotations/wuji_fk21.npz`** 的 `positions`，还要**自己挑右手** |
| **episode 索引方式** | ✅ 改 | sandwich 用整数 index 查表；dagger 直接用 episode 目录名 |

---

## 2. 输入数据（精确路径与结构）

### 2.1 原始视频

```
/data/shichaojian/raw_data/dropper_dagger/<episode>/videos/head.mp4
                                           /videos/right_wrist.mp4
                                           /auxiliary_camera/{depth.lmdb, metadata.json, head_ir_*.mp4}
                                           /lmdb/  meta_info.pkl  sync_timestamps.json
```
**51 个 episode。**

⚠️ **`dropper_dagger_mix/` 里那 71 个 episode 目录是空的**，别用那个。用 `dropper_dagger/`。

### 2.2 FK 标注

```
/data/shichaojian/raw_data/dropper_dagger_mix_fk21/<episode>/annotations/wuji_fk21.npz
                                                                         wuji_fk21.json
```
**71 个 episode，与 raw 的交集是 51 个。** 只渲染交集。

`npz` 的字段：

| key | 值 / shape | 说明 |
|---|---|---|
| **`positions`** | **`[T, 2, 21, 3]`** float32 | ⚠️ **两个 side 都在里面**，第 0 维是帧，第 1 维是 side |
| `side_names` | `['left', 'right']` | **右手是 index 1** |
| `keypoint_names` | 21 个 | `wrist, thumb_cmc/mcp/ip/tip, index_mcp/pip/dip/tip, middle_*, ring_*, pinky_*` |
| `coordinate_frame` | **`Link_Base`** | 与 sandwich 同一套基座系 |
| `units` | `metre` | |
| `frame_rate` | `30.0` | |

### 2.3 帧对应关系（已实测）

```
positions 帧数 (2251) == head.mp4 帧数 (2251) == right_wrist.mp4 帧数 (2251)   30 fps
```

**所以 `positions[i]` 就是 `head.mp4` 的第 `i` 帧，一一对应、无偏移。**
（注意 `timestamps` 的 dt 是 0.1 s = 10 Hz —— 那是 qpos 源的采样率，`positions` 已经被重采样到
30 Hz 对齐视频了。**别拿 timestamps 推帧号。**）

### 2.4 相机内参（dagger 与 sandwich 逐位相同）

```json
// <episode>/auxiliary_camera/metadata.json
serial = 147122073219      width = 640   height = 480
fx = 605.5706176757812     fy = 604.4129638671875
ppx = 324.4994812011719    ppy = 238.25637817382812
```

这与 `tools/verify_fk_camera_projection.py:41-43` 里写死的 `FX/FY/CX/CY` 完全一致
—— 这就是「不用改」的依据。

---

## 3. 方法（数学，一个字都不用动）

### 3.1 外参

```python
r, t = base_to_head_camera(URDF)          # 从 URDF 链推出 Link_Base -> D435 光心
r = roll_about_z(180.0) @ r               # ⚠️ 绕视轴 180°，URDF 定不了，是实测的
t = roll_about_z(180.0) @ t
p_cam = p_base @ r.T + t
```

两处**不能从 URDF 推出**的东西（`cosmos_framework/data/fk_camera_extrinsic.py` 文件头）：

- URDF 里相机 link 的 **+Y 朝上**，不是图像约定 → 绕视轴的 180° roll 由 URDF 定不了
- `head_d435_link_optical_joint` 的 origin 是 **32.5 mm / 4.3 mm**，不是出厂默认的 0

### 3.2 投影

```
u = FX * x / z + CX        FX, FY = 605.5706176757812, 604.4129638671875
v = FY * y / z + CY        CX, CY = 324.4994812011719, 238.25637817382812
z <= 0  →  不投影（不是钳位）
```

**必须喂相机系绝对坐标。** 输入是 `Link_Base` 系，所以要先做 §3.1 那一步。

⚠️ **不要**用 `tools/render_fk_projection.py` 那套（那个接的是**相机系**坐标、不再做变换）。
两者混用 = 变换做两遍或一遍都不做，结果都会偏到看起来像「模型错了」。

---

## 4. 与 `tools/render_fk_overlay.py` 的三处差异

原工具是给 sandwich 写死的，改这三处即可（**不要**改它的数学）：

| 位置 | 原来（sandwich） | dagger 要改成 |
|---|---|---|
| `render_fk_overlay.py:35-38` `RAW_ROOT` | `raw_data/singlerighthand_sandwich_100` | `raw_data/dropper_dagger` |
| `verify_fk_camera_projection.py:213` `load_fk()` | 读 `SLIM_ROOT/fk21/chunk-000/episode_%06d.npz` 的 `right_positions_abs` | 读 `<FK_ROOT>/<name>/annotations/wuji_fk21.npz` 的 `positions[:, 1]` |
| `load_episode_map()` | 读 `SLIM_ROOT/meta/source_episodes.jsonl` | 直接用 episode 目录名（`raw ∩ fk21`） |

`SLIM_ROOT`（`verify_fk_camera_projection.py:30`）指向 sandwich 的 slim 包，
**dagger 没有对应的 slim 包** —— 这就是为什么不能直接改 `SLIM_ROOT` 了事。

---

## 5. 完整脚本（可直接跑）

存成 `tools/render_fk_overlay_dagger.py`。**不改任何既有文件。**

```python
#!/usr/bin/env python3
"""FK-21 overlay on the dagger episodes' real head video.

Same projection as tools/render_fk_overlay.py -- the camera is the same unit
(serial 147122073219, identical fx/fy/ppx/ppy), so the extrinsics and intrinsics
carry over unchanged.  What differs is only where the data lives and how the FK
file is laid out: dagger keeps one wuji_fk21.npz per episode under a SEPARATE
tree, holding both hands, where sandwich's slim pack had one npz per index
already reduced to the right hand.

    PYTHONPATH=. <venv>/bin/python tools/render_fk_overlay_dagger.py \
        --episode episode_0000_20260830_155455 --out /data/shichaojian/renders/fk_dagger
    PYTHONPATH=. <venv>/bin/python tools/render_fk_overlay_dagger.py --list
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from verify_fk_camera_projection import (  # noqa: E402
    EDGES,
    FINGER_COLORS,
    IMG_H,
    IMG_W,
    URDF,
    base_to_head_camera,
    finger_of,
    roll_about_z,
)

RAW_ROOT = "/data/shichaojian/raw_data/dropper_dagger"
FK_ROOT = "/data/shichaojian/raw_data/dropper_dagger_mix_fk21"
RIGHT_SIDE = "right"  # side_names is ['left', 'right']


def episodes() -> list[str]:
    """The episodes that have BOTH the video and the annotations.

    Not every annotated episode has raw video: the fk21 tree is a superset
    (71 vs 51), and the extra ones are the 20260903 batch.  Intersecting here
    means --list shows exactly what can be rendered.
    """
    if not os.path.isdir(RAW_ROOT) or not os.path.isdir(FK_ROOT):
        raise SystemExit(f"missing data root: {RAW_ROOT} / {FK_ROOT}")
    raw = {d for d in os.listdir(RAW_ROOT) if os.path.isdir(os.path.join(RAW_ROOT, d))}
    fk = {d for d in os.listdir(FK_ROOT) if os.path.isdir(os.path.join(FK_ROOT, d))}
    both = sorted(n for n in raw & fk if os.path.isfile(os.path.join(RAW_ROOT, n, "videos", "head.mp4")))
    if not both:
        raise SystemExit("no episode has both videos/head.mp4 and an FK annotation")
    return both


def load_fk(name: str) -> np.ndarray:
    """``[T, 21, 3]`` base-frame metres for the RIGHT hand.

    ``positions`` is [T, 2, 21, 3] -- both sides in one array.  Indexing the
    wrong side yields the left hand, which projects to a plausible-looking
    skeleton somewhere it does not belong; that is the first thing to suspect
    if the overlay is badly off (see the doc's troubleshooting ladder).
    """
    path = os.path.join(FK_ROOT, name, "annotations", "wuji_fk21.npz")
    with np.load(path, allow_pickle=True) as d:
        positions = d["positions"]
        sides = [str(s) for s in d["side_names"]]
        if sides.count(RIGHT_SIDE) != 1:
            raise ValueError(f"{name}: side_names={sides} has no unique {RIGHT_SIDE!r}")
        return positions[:, sides.index(RIGHT_SIDE)].astype(np.float64)


def project_episode(pts_all: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Vectorised base -> camera -> pixel.  Returns (uv[T,21,2], in_front[T,21]).

    Identical to render_fk_overlay.project_episode; repeated rather than imported
    so this file stands alone and cannot drift from it silently.
    """
    r, t = base_to_head_camera(URDF)
    r = roll_about_z(180.0) @ r
    t = roll_about_z(180.0) @ t
    cam = pts_all @ r.T + t
    z = cam[..., 2]
    in_front = z > 1e-6
    zs = np.where(in_front, z, 1.0)
    u = 605.5706176757812 * cam[..., 0] / zs + 324.4994812011719
    v = 604.4129638671875 * cam[..., 1] / zs + 238.25637817382812
    return np.stack([u, v], axis=-1), in_front


def draw(frame, uv, valid, scale, label):
    canvas = cv2.resize(frame, (IMG_W * scale, IMG_H * scale), interpolation=cv2.INTER_LINEAR)

    def pt(k):
        return int(round(uv[k, 0] * scale)), int(round(uv[k, 1] * scale))

    for a, b in EDGES:
        if valid[a] and valid[b]:
            cv2.line(canvas, pt(a), pt(b), (0, 0, 0), 5 * scale, cv2.LINE_AA)
            cv2.line(canvas, pt(a), pt(b), FINGER_COLORS[finger_of(b)], 2 * scale, cv2.LINE_AA)
    for k in range(21):
        if valid[k]:
            cv2.circle(canvas, pt(k), 4 * scale, (0, 0, 0), -1, cv2.LINE_AA)
            cv2.circle(canvas, pt(k), 3 * scale, FINGER_COLORS[finger_of(k)], -1, cv2.LINE_AA)
        else:
            x = int(np.clip(uv[k, 0] * scale, 6, IMG_W * scale - 6))
            y = int(np.clip(uv[k, 1] * scale, 6, IMG_H * scale - 6))
            cv2.drawMarker(canvas, (x, y), (255, 0, 255), cv2.MARKER_TILTED_CROSS, 7 * scale, 2 * scale)
    cv2.rectangle(canvas, (0, 0), (IMG_W * scale, 26 * scale), (0, 0, 0), -1)
    cv2.putText(canvas, label, (6 * scale, 19 * scale),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55 * scale, (255, 255, 255), 1 * scale, cv2.LINE_AA)
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--episode", help="an episode directory name; default: the first")
    ap.add_argument("--list", action="store_true", help="print the renderable episodes and exit")
    ap.add_argument("--all", action="store_true", help="render every renderable episode")
    ap.add_argument("--out", default="/data/shichaojian/renders/fk_dagger")
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--fps", type=float, default=30.0, help="source is 30 Hz; 15 gives half-speed playback")
    ap.add_argument("--stills", default="100,500,1000,1500,2000",
                    help="frame indices for the contact sheet (default keeps them inside the shortest episode)")
    args = ap.parse_args()

    names = episodes()
    if args.list:
        print(f"{len(names)} renderable episodes (raw videos AND FK annotation):")
        for n in names:
            print(f"  {n}")
        return 0

    if args.all:
        todo = names
    elif args.episode:
        if args.episode not in names:
            raise SystemExit(f"{args.episode!r} is not renderable; --list to see what is")
        todo = [args.episode]
    else:
        todo = names[:1]

    os.makedirs(args.out, exist_ok=True)
    stills_wanted = {int(x) for x in args.stills.split(",") if x.strip()}

    for name in todo:
        video_path = os.path.join(RAW_ROOT, name, "videos", "head.mp4")
        pts_all = load_fk(name)
        uv_all, front_all = project_episode(pts_all)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"  {name}: cannot open {video_path}; skipped")
            continue
        n_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # The two are 1:1 in every episode checked (2251 == 2251).  A mismatch is
        # not fatal -- it is truncated to the shorter -- but it means the frame
        # correspondence assumption is wrong for this episode, and every skeleton
        # after the divergence point would sit on the wrong pose.  Say so.
        if n_video != len(pts_all):
            print(f"  {name}: WARNING video has {n_video} frames, FK has {len(pts_all)} "
                  f"-- assuming index i == frame i and truncating to the shorter")
        print(f"  {name}: {n_video} video frames, {len(pts_all)} FK frames")

        out_mp4 = os.path.join(args.out, f"fk_overlay_{name}.mp4")
        writer = cv2.VideoWriter(out_mp4, cv2.VideoWriter_fourcc(*"mp4v"), args.fps,
                                 (IMG_W * args.scale, IMG_H * args.scale))
        if not writer.isOpened():
            raise RuntimeError(f"VideoWriter failed to open {out_mp4}")

        stills, i = [], 0
        while True:
            ok, frame = cap.read()
            if not ok or i >= len(uv_all):
                break
            uv, front = uv_all[i], front_all[i]
            valid = (front & (uv[:, 0] >= 0) & (uv[:, 0] < IMG_W)
                     & (uv[:, 1] >= 0) & (uv[:, 1] < IMG_H))
            canvas = draw(frame, uv, valid, args.scale,
                          f"{name[:26]}  frame {i:4d}   FK-21 right (base->D435 + Rz180)")
            writer.write(canvas)
            if i in stills_wanted:
                stills.append(canvas)
            i += 1
        cap.release()
        writer.release()
        print(f"    wrote {out_mp4}  ({i} frames)")

        if stills:
            cols = 2
            rows = (len(stills) + cols - 1) // cols
            h, w = stills[0].shape[:2]
            sheet = np.zeros((rows * h, cols * w, 3), np.uint8)
            for k, s in enumerate(stills):
                rr, cc = divmod(k, cols)
                sheet[rr * h:(rr + 1) * h, cc * w:(cc + 1) * w] = s
            sheet_path = os.path.join(args.out, f"fk_overlay_{name}_stills.png")
            cv2.imwrite(sheet_path, sheet)
            print(f"    wrote {sheet_path}  ({len(stills)} stills)")

    print("\nCHECK FIRST: is the skeleton on the HAND in the first still?")
    print("If it is not, the projection is wrong and nothing else in the frame means")
    print("anything. See the troubleshooting ladder in docs/fk_overlay_dagger_howto.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

### 跑法

```bash
cd /mnt/afs/WorldAct-cosmos3-edge-droid-sft_mano
V=/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python

PYTHONPATH=. $V tools/render_fk_overlay_dagger.py --list              # 先看能渲染哪些
PYTHONPATH=. $V tools/render_fk_overlay_dagger.py \
    --episode episode_0000_20260830_155455 --out /data/shichaojian/renders/fk_dagger
PYTHONPATH=. $V tools/render_fk_overlay_dagger.py --all               # 全部 51 条
```

**不需要 GPU**（纯 CPU 读取 + 画线）。产物：
`<out>/fk_overlay_<episode>.mp4` + `<out>/fk_overlay_<episode>_stills.png`

---

## 6. ⚠️ 排查阶梯：上次渲染「完全错误、偏离很大」

### 6.0 实测结论（2026-10-02，**先读这条**）

§5 的脚本已经跑通并**对照验证过**，结论**不是**「取错手」这类低级错误：

| 实验 | 结果 |
|---|---|
| 同一份脚本渲染 **sandwich**（`tools/render_fk_overlay.py`，当年验证过的那批） | ✅ 骨架**稳稳落在手套上**（5 张 stills 逐帧确认） |
| 同一份脚本渲染 **dagger** | ❌ 骨架方向正确（手腕右下、手指朝左上）、**跟着手运动**，但整体**向下偏约 200 px** |
| 两台机器的相机 | **同一台**：serial `147122073219`，`fx/fy/ppx/ppy` 逐位相同 |
| 两台机器的视频 | **同样 640×480** |
| 右手 vs 左手（dagger） | 右手 97.7% 落在画面内；左手 **0.0%**（v 范围 [1014, 1079]，全在画面下方）→ **手取对了** |

**所以：代码没问题，`Link_Base → head camera` 这条变换对 dagger 那台机器不成立。**

量级也对不上「残差」：200 px 在 z≈0.5 m 处 ≈ **18 cm** 的三维误差。
而 §7 的已知残差是 **8–20 px（≈1 cm）** —— 差一个数量级。

**为什么**：`tools/calibrate_extrinsic_v2.py` 的文件头自己写着 ——

> The URDF's head-camera mount pose is a **nominal value** (matched to CAD hole
> positions from photographs). **Measurement says it is ~5 cm away from the real camera.**

`calibrate_camera_extrinsic.py` 同样写着 URDF 位姿 **off by O(10 cm)**。

即 **URDF 里的相机安装位姿是标称值、不是实测值**。sandwich 采集于 07-30/31，dagger 采集于
08-30 / 09-03，相隔一个月 —— **相机（或夹爪）中途被重装过**，而 sandwich 的
`base_to_head_camera` 外参是按 sandwich 那次安装算的。

**下一步不是继续调渲染，是给 dagger 重新标定外参**（见 §6⑤）。

### 6.1–6.5 一般性排查阶梯（假设偏移**不**是 6.0 那个量级时用）

**按这个顺序排，前两条最容易中。**

### ① 取错手（最可能）

`positions` 是 `[T, 2, 21, 3]`，**两只手都在里面**。`side_names = ['left','right']`
—— 取 index 0 就是**左手**。左手骨架投到右手画面里，会得到一个"形状对但位置完全不对"
的结果，**看起来像标定坏了**。

```python
sides = [str(s) for s in d["side_names"]]
pts = d["positions"][:, sides.index("right")]     # ← 必须是 right
```

### ② 缺了那 180° 的 roll

外参**必须**是 `roll_about_z(180) @ base_to_head_camera(URDF)` 的组合。
只用 `base_to_head_camera` 的原始结果 → 骨架会**镜像+旋转到画面别处**，
而且**不会报错**（URDF 给不出这个 roll，见 §3.1）。

### ③ 坐标系不是 Link_Base

核查 `d["coordinate_frame"] == "Link_Base"`、`d["units"] == "metre"`。
**这两个字段如果变了，整套外参全废。** 本批数据实测是 `Link_Base` / `metre` ✓。

### ④ 帧号错位

本批实测 `positions` 帧数 == `head.mp4` 帧数（2251 == 2251），**一一对应**。
若你把 `timestamps`（10 Hz）当成帧号用过，会整体错位 —— **别用 timestamps 推帧号**。
脚本里已经有 mismatch 的 WARNING。

### ⑤ 相机被重新安装过 —— **dagger 已实测确认就是这条**

外参 `Link_Base → head camera` 依赖**相机在机器人上的安装位置**，而 URDF 里那个位置是
**标称值**（见 §6.0 的两处引用），实测差 5–10 cm。dagger 与 sandwich 相隔一个月采集，
**相机/夹爪中途重装过**，所以 sandwich 的外参对 dagger 不成立。

**这不是假设，是已经做过对照实验的结论（§6.0）。**

**怎么修 —— 三条路，按代价排序：**

**A. 重新标定外参（正解，但要 GPU 机 + mujoco + 深度图）**

工具已存在，且就是为这件事写的：

```bash
# 需要 mujoco（本 venv 已装：mujoco==3.3.2）
PYTHONPATH=. <venv>/bin/python tools/calibrate_extrinsic_v2.py --help
PYTHONPATH=. <venv>/bin/python tools/calibrate_camera_extrinsic.py --help
```

原理（`calibrate_extrinsic_v2.py` 文件头）：把 **FK 的手部网格**（MuJoCo 按录制的关节角摆姿，
在 `Link_Base` 系）拟合到 **D435 深度的真实手**（按 `metadata.json` 出厂内参反投影，
在 D435 彩色系）。若相机位姿正确，A 变换到彩色系后应落在 B 上。

两个必须带上的守卫（文件头写了上次为什么失败）：
- **20 cm 球**限制目标点集（上次用 22 cm，把桌面吞进去了）
- **硬深度切**（手在 0.7–0.9 m，桌在 1.2 m）
没有这两条，trimmed ICP 会把网格滑到**桌面上**（平面拟合物），并报一个好看的 0.64 cm 残差。

标完用 `tools/export_fk_camera_extrinsic.py` 写新模块（**生成文件，不要手改**），
或在渲染脚本里覆盖 `r, t`。

**B. 手工求一个 2D 平移——只当临时验证，别当结论**

如果偏移**在整段视频里近似恒定**（dagger 看起来就是这样），可以先在渲染脚本里加一个
`--duv dx,dy` 看骨架会不会贴上去。**这只能证明「偏移是常量」，不能证明外参正确**：
真正的相机位姿错误在图像里的位移会随手臂姿态和深度变化。

**C. 暂时别用 dagger 做可视化**

如果只是要"看 FK 长什么样"，用 sandwich（外参对得上）。要分析 dagger 的训练结果，
**在标定之前，任何叠加图都不能用来判断 FK 学得好不好** —— 骨架偏移会淹没模型误差。

### ⑥ 排除了全部 5 条之后剩下的那个偏移

见 §7。

---

## 7. 已知残差（这个不是 bug，别去修）

⚠️ **FK 投影到真实相机画面仍残留约 8–20 px 偏差**，而且**一部分随手臂姿态变化**。
诊断显示**不是**相机滚转问题。

- v1 **接受**它（v1 的目标是跑通链路，不是效果）
- **它会直接进入 FK 模态的三维坐标**，所以 **v1 的训练结果不能用来评判 FK 模态的效果**
- 想根治需要**不依赖 FK 的参考**（人工标注几帧手的位置）—— 提过，一直没做

**所以：如果骨架落在手上、只是有几像素到十几像素的偏移，那是已知残差，不是渲染出错。**

---

## 8. 验证清单

跑完先看 stills，按顺序确认：

| 检查 | 判据 |
|---|---|
| 骨架在**手上**吗 | 不在 → 回到 §6 的阶梯，**别继续往下看** |
| 是**右手**吗 | 骨架应该落在画面里那只实际在动的手上；落在另一只 = §6① |
| 五指是否都在 | 有品红色"十"字 = 该关键点在画面外或相机背后（`z<=0`），这是设计行为，不是 bug |
| 手指朝向对吗 | 拇指应指向手的拇指侧；镜像了 = §6② |
| 帧号对吗 | 在 mp4 里暂停看骨架是否**跟着**手的动作；滞后/超前 = §6④ |
| 静止段（手不动） | 骨架应该也不动；如果静止时骨架在漂 = 残差或标定问题 |

---

## 9. 参考

| 文件 | 作用 |
|---|---|
| `tools/render_fk_overlay.py` | sandwich 版原工具（本文档的模板） |
| `tools/verify_fk_camera_projection.py` | 路径常量 `:26-38`、内参 `:41-43`、`base_to_head_camera` `:100`、`roll_about_z` `:133`、`project` `:143` |
| `cosmos_framework/data/fk_camera_extrinsic.py` | 训练侧用的等价变换（**生成文件，不要手改**） |
| `tools/render_fk_projection.py` | ⚠️ **另一套**：接**相机系**坐标、叠在真实+生成视频上。**别和本文档的混用** |
| `docs/fk_projection_rendering.md` | 投影与可视化的总说明（含双重变换的 343 px 坑） |
| `docs/fk_modality_design.md` | 设计文档；§4 记了那个 8–20 px 残差 |

---

*本文件位于 `WorldAct-cosmos3-edge-droid-sft_mano/docs/`（mano worktree）。*
