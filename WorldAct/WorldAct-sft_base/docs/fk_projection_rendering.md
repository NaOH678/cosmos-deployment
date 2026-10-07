# FK 投影与可视化（相机坐标系）

> 说明对象：**`-101-2`（v1）这一版的 FK 处理链路，以及把骨架投影回图像的可视化工具。**
>
> 与 [`fk_modality_design.md`](./fk_modality_design.md) 的分工：那份是**设计与决策记录**
> （为什么这么设计、走过哪些弯路）；本文是**代码怎么运作、怎么跑、怎么看产物**。
>
> ⚠️ 设计文档 §6.7 结尾曾写「可视化是自足的……**不叠在视频上**」—— **那句已过时**。
> 现在 `tools/render_fk_projection.py` / `render_fk_rollout.py` 正是叠在真实视频上的，
> 以本文为准。

---

## 1. FK 的处理链路：base → 相机 → 模型输入

```
wuji_fk21.npz            base 系（Link_Base），米，[num_frames, 21, 3]
      │
      │  base_to_camera()          cosmos_framework/data/fk_camera_extrinsic.py:43
      │      p_cam = R @ p_base + t
      ▼
camera 系（camera_d435_real），[steps+1, 21, 3]
      │
      ├─ camera[0]                      → anchor          [21, 3]
      └─ camera[1:] - camera[0]         → displacement    [32, 21, 3]   ← 模型学的量
```

**唯一的训练侧调用点**：`cosmos_framework/data/generator/action/fk_source.py:117`

```python
camera = base_to_camera(positions[frame_ids])      # 整个窗口转一次
anchor = camera[0]
displacement = camera[1:] - anchor[None]
```

窗口中点先做**一次**旋转再相减，与先相减再旋转等价（都是常量变换）——
所以位移是在相机系里算的，但锚点本身也在相机系，两者自洽。

**产出的 `coordinate` 标签是 `camera_d435_real`。**

### 1.1 变换常量

`cosmos_framework/data/fk_camera_extrinsic.py` —— **生成文件，不要手改**（第 1 行）：

```python
:22  R = [[ 1.4e-11, -1.0,      -3e-12   ],
          [-0.64278761, -6e-12,  -0.76604444],
          [ 0.76604444,  1.2e-11, -0.64278761]]
:30  t = [0.0325, 1.094353449231, 0.830049026781]
:32  URDF_MD5 = "d687e8f52df18839f4f99efe8c12a909"
:33  ROLL_DEG = 180.0
:37-40  导入时自检：R 是 3×3、正交、det=1
```

```bash
python tools/export_fk_camera_extrinsic.py           # 重新生成
python tools/export_fk_camera_extrinsic.py --check   # 检查漂移
```

**两处无法纯靠 URDF 链式推导**（文件头 docstring `:8-13`）：

- URDF 里相机 link 的 **+Y 朝上**，不是图像约定 → **绕视轴的 180° roll 由 URDF 定不了**，
  是实测定下来的（`ROLL_DEG = 180.0`）
- `head_d435_link_optical_joint` 的 origin 是 **32.5 mm / 4.3 mm**，不是出厂默认的 0

源 URDF（写死在 `:15`）：
`/data/shichaojian/wuji-mjlab/marvin_wuji_d435_description/urdf/marvin_wuji_d435_complete.urdf`

---

## 2. 投影回图像

### 2.1 数学

`tools/verify_fk_camera_projection.py:143`，纯针孔：

```python
:41  FX, FY = 605.5706176757812, 604.4129638671875
:42  CX, CY = 324.4994812011719, 238.25637817382812
:43  IMG_W, IMG_H = 640, 480

u = FX * x / z + CX
v = FY * y / z + CY
z <= 0  →  in_front = False（不投影，不是钳位）
```

### 2.2 ⚠️ 输入必须是**相机系绝对坐标**

```python
gt_cam   = anchor + target          # anchor 是 [21,3]，target 是 [T,21,3]
pred_cam = anchor + prediction
uv, front = project(gt_cam[t])      # project() 直接吃相机系
```

**不要**改用 `tools/render_fk_overlay.py` 里的 `project_episode` —— 那个接的是
**base 系**、自己内部再做一次 base→camera。对相机系坐标用它 = **变换做了两遍**：

```
正确 [812.5, 201.1]  vs  误用 [475.9, 137.5]     相差 343 px
```

而且结果**偏偏落在手附近**（相机的 base 偏移 ~1.4 m + 转 180°），
看起来像"模型偏了"而不是"函数调错了"。`tools/render_fk_projection_test.py:test_double_transform`
就是为这个坑设的牙齿。

### 2.3 帧号契约

- `raw_frame_ids[t] = raw_start + 2*t` —— **30 Hz 原始索引**（源 30 fps，训练 15 fps）
- 视频里 step `t` 的骨架应画在 **`frame_ids[t + 1]`**
  —— 一个窗口观测 `steps + 1` 帧（锚点 + 各步），因果 VAE 也是 `1 + (T_lat-1)*4 = 33` 帧，
  所以索引 `t` 是锚点、早一帧。

---

## 3. 四个渲染器

| 工具 | 输入 | 输出 | 需要 GPU |
|---|---|---|---|
| `tools/render_fk_case.py` | `fk_eval/step_X/<case>/prediction.npz` | `comparison.png` / `error_curve.png` / `comparison.mp4` | 否 |
| `tools/render_fk_projection.py` | `<case>/prediction.npz`（含 `vision` latent） | REAL vs GENERATED 双面板 mp4 + stills png | **是**（VAE decode） |
| `tools/render_fk_rollout.py` | `<episode>/rollout.npz` | 双视图骨架动画 mp4 + 投影 mp4 | 是 |
| `tools/render_fk_joint.sh` | 自动找最新 eval | 包装上面两个 | 是 |

### 3.1 `render_fk_case.py` —— 离线补图

训练时 `FK_EVAL_FIGURES=false`（默认）**只写数据不画图**：一次验证 128 帧 matplotlib
在单卡上就是实打实的时间，画在没人的步上不值。事后补：

```bash
PYTHONPATH=. <venv>/bin/python tools/render_fk_case.py --case-dir <...>/fk_eval/step_0001000/val_00
PYTHONPATH=. <venv>/bin/python tools/render_fk_case.py --eval-dir <...>/fk_eval --steps 1000,2000
PYTHONPATH=. <venv>/bin/python tools/render_fk_case.py --eval-dir <...>/fk_eval --all
```

输出与训练器当时画的**逐字节同布局**（走同一套 `fk_visualize.render_case` / `write_video`）。

### 3.2 `render_fk_projection.py` —— 点画在真实视频 / 生成视频上

```bash
PYTHONPATH=. <venv>/bin/python tools/render_fk_projection.py --case-dir <case>
# --out 默认 /data/shichaojian/renders/fk_projection
```

两块面板，**同一对骨架**（绿=GT，红=预测），只有背景不同：

- **左 REAL**：真实录制视频 —— 回答"骨架在不在手上"
- **右 GENERATED**：模型自己生成的视频 —— 回答"FK 是不是在忠实地追一段本身就不对的视频"

`prediction.npz["vision"]` 是**latent**（不是像素），在这里才 decode。

**`stored_latent()` 的坑**：clean arm 不是不写 `vision` 键，而是写 `None`,
而 `np.savez` 把它存成 **0 维 object 数组** → `data["vision"] is not None` 是 **True**。
按秩（`ndim < 5`）判空，不能按身份。

### 3.3 `render_fk_rollout.py` —— 整段 rollout

```bash
PYTHONPATH=. <venv>/bin/python tools/render_fk_rollout.py --rollout <dir>/rollout.npz
# --out 默认 /data/shichaojian/renders/fk_rollout
```

读 callback 写的 `rollout.npz`，出两个产物：

- `rollout.mp4` —— 1×2 正交视图（x/y 与 x/z，实线 GT、虚线预测、每指一色），
  从单窗口 32 步延到整段
- `projection.mp4` —— 真实头部视频上的 GT + 预测骨架

**窗口平铺规则**：相邻窗口**只共享一帧**（后一个窗口的锚点），
所以每帧恰好被一个窗口预测 → 拼接后没有重叠帧。

`window_frames` 是**从形状推导**的（`prediction.shape[0] // vision.shape[0]`），
不从文件读 —— 文件里那个字段存的是**索引步长 64**，不是每窗口步数 32。

### 3.4 `render_fk_joint.sh` —— 包装

```bash
bash tools/render_fk_joint.sh              # 最新 joint eval，两个 val case
bash tools/render_fk_joint.sh val_01       # 单个 case
ARM=joint bash tools/render_fk_joint.sh    # 换成两段式 arm
ROOT=<eval root> bash tools/render_fk_joint.sh   # 指定某个 eval
```

**刻意不接受路径参数** —— case 目录约 150 字符，粘进终端会折行，bash 把路径尾巴当命令读。
已有的受害者：`$D` 变成 `".../fk_"`，渲染器去找一个从不存在的目录。

---

## 4. 合成画布的几何（生成面板为什么被拉长）

数据集的画面**不是** head 视角，而是**合成画布**：

```
wrist 848×480 ──缩到 640 宽──> 362 行
head  640×480                   480 行
                              ─────────
                     合成        842 × 640
                     │  缩放到 544 宽 → 716 行
                     │  reflection-pad 到 patch 倍数 → 736 行（46 个 latent 行）
                     ▼
        模型实际只被喂【前 44 个 latent 行 = 704 px】
```

**最后一条是实测的，不是推的**：rollout 的 conditioning frame 与 cache 窗口裁到前 44 行
**逐元素完全相同（MAD = 0.000000）**。

后果：`decode_head_view` 拿到的 head 块是 `704 - 308 = 396` 行，而真实映射是 `408` 行，
于是生成面板被**纵向拉伸 `408/396 = 1.0303`**，用真实内参投出来的骨架会往上飘
（画面底部约 14 px）。

`render_fk_rollout.generated_v_scale()` 就是算这个因子给 **生成面板**用的：

```python
v_factor = generated_v_scale(latent_h, latent_w)   # ≈ 1.0303
scaled_v = uv[:, 1] * v_factor
```

**真实面板永远不缩放** —— 它没经过模型。

**裁剪常量是推导的，不是写死的**：`canvas_crop(decoded_hw)` 只从**宽度**推缩放系数
（宽度从没被裁过），高度那侧用 min 会得到 0.836 而非 0.85 —— 因为高度正是被裁的轴。
写死 716/308 的那一版立刻死在 `decoded (704, 544) is smaller than the content 716x544`。

---

## 5. 怎么读这些图

**先看绿色。** REAL 面板里如果 GT 骨架不在手上，**投影就是错的**，红色预测没有任何意义。
这是内建的检查 —— GT 走的是同一条代码路径，只是把 `prediction` 换成 `target`。

| 现象 | 含义 |
|---|---|
| 绿在手上、红偏 | **这才是模型误差** |
| 绿不在手上 | 投影坏了，别读红 |
| 绿红都准 | GT 与预测一致 |
| rollout 每窗口边界**跳一下** | **正常** —— 每个窗口从自己的 GT 锚点重启，是窗口化生成的固有性质，不是渲染 bug，**不要抹平** |

---

## 6. 已知残差与限制

⚠️ **FK 投影到真实相机画面仍残留约 8–20 px 偏差**（诊断显示不是相机滚转，
一部分随手臂姿态变化）。**这个误差会直接进 FK 模态的三维坐标。**

- v1 **接受**它（v1 的目标是跑通链路，不是效果）
- **所以 v1 的训练结果不能用来评判 FK 模态的效果**
- 想根治需要**不依赖 FK 的参考**（人工标注几帧手的位置）—— 提过，没做

其他：

- 外参是**单一常量**，仅在 head 相机刚性固定在 `Link_Base` 上时有效。
  head 关节一旦中间动过，此后每帧都错，**且没有报错**。
- ⚠️ 本文档描述的是 **`camera_d435_real`**（真实 D435 系）。
  **没有做**到 **DA3 系**的那一跳 —— 那是各向异性缩放（横 1.164 / 纵 1.084）
  + 平移 4.5 px，**不是刚体变换**，v1 刻意不做，v2 才需要。见
  `fk_modality_design.md` §1.2 与 §7 末尾的修正。

---

## 7. 验证

```bash
PYTHONPATH=. <venv>/bin/python tools/render_fk_projection_test.py   # 6 项检查
PYTHONPATH=. <venv>/bin/python tools/verify_fk21_allowlist.py       # 10 条 episode 逐点复算
```

⚠️ `render_fk_projection_test.py` 的 `test_double_transform` 在**找不到 URDF 时静默跳过**
（打印 ⚠️ 就 return）。所以它"通过"**不代表测过** —— URDF 路径写错时会给你一个假绿灯。
这条真实发生过：迁移后 URDF 路径指向一个不存在的目录，测试照打 PASS。

`verify_fk21_allowlist.py` 是独立的第二条推导路径（`base_to_head_camera(URDF)` +
`roll_about_z(180°)`），与 `base_to_camera` 数值一致到 **4e-13** —— 这是变换可信的依据。

---

*本文件位于 `WorldAct-cosmos3-edge-droid-sft_mano/docs/`（mano worktree）。*
