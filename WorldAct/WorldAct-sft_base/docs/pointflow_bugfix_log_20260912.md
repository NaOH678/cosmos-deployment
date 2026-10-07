# PointFlow 数据链路修复记录（2026-09-11 ~ 09-12）

**这份文档的性质**：一次调试会话的**缺陷修复记录**。它不是设计文档。

与本目录其他文档的关系：

- `pointflow_ptv3_cosmos_design.md` 及 `pointflow_task1..12_*.md` —— 描述**设计与实现**，日期早于本次会话。**本次修复不推翻它们的结论**，但其中关于 cache、affine 画布、attention 模式的描述已**过时**，以本文件为准。
- `pointflow_dense_fullseq_audit.md` —— 对 dense 数据的检查，与本文件不冲突。
- `cosmos_wam_mode.md` —— WAM（video + action 联合去噪）的原理，与本次修复无关。
- `pointflow_window_latent_cache_20260913.md` —— **2026-09-13 的后续改动，另一件事**：逐窗口 VAE latent 缓存。它**取代**了本文第 4 节中 `vae_latents/*.pt` 的地位（旧缓存保留但已降级为 `elif` 分支）。**本文记录的是 09-11~09-12 的缺陷修复，不要与它混读。**

**时间记录**：会话 2026-09-11 开始，2026-09-12 定稿。所有"实测"数字都是本会话在这台机器上跑出来的，命令见各节。

**结论摘要**：链路里存在 **3 个真实缺陷**，都已修复并锁定。**`runs/cosmos/pointflow_test` 下 8000 步的训练全部作废**，必须从头训。

---

## 0. 背景：哪些产物是"对的"

修复过程中最有价值的一条方法论结论：**盘上有未被污染的对照组**，应该先看它，而不是从第一性原理推。

```
video_frames/ 共 101 个 npy
  其中 10 个 (allowlist) 被本分支用坏代码重生过
  其余 91 个是 2026-08-03 的原始产物 —— 全部 (T, 3, 716, 544)
```

这 91 个文件证明了**原始代码产出的就是 `716×544` 竖版**，本分支的改动是一处**回归**。本次"修复"的实质是**撤回该回归**。

---

## 1. 缺陷一：`anchor_xyz` 被 `normal` 覆盖

**位置**：`cosmos_framework/data/pointflow_batch.py` 的 `_reused_buffer` / `_concatenate`

**机制**：拼接缓冲区按 `(形状, dtype)` 缓存复用。`anchor_xyz` 与 `normal` 都是 `[N,3] float32` → 命中同一个 key → 后写的 `normal` 覆盖了先写的相机坐标。

**为什么必然触发**：`max_samples_per_batch=32`，一个 packed batch 有 32 个样本，全部带 pointflow 标签 → `len(parts) == 32`，一定走缓冲区分支。

**为什么 eval 反而正常**：`fixed_cases` 用 `max_samples_per_batch=1`，`len(parts) == 1` → 提前返回，不碰缓冲区。**训练喂的是法向、评估喂的是真坐标** —— 输入分布不一致。

**实测证据**：

```
anchor_xyz is normal : True
|anchor_xyz| mean    : 1.0        （真实相机坐标应为 1.229，范围 0.14~6.63）
anchor_xyz[0]        : [-0.7101, 0.6864, 0.1571]   ← 单位法向
```

**症状**：`pointflow_eval/step_0007600/val_00/metrics.json`

| | 模型 | "预测零位移"基线 |
|---|---|---|
| all ADE | 48.5 mm | 14.1 mm |
| static drift | 44.2 mm | 3.1 mm |

模型比"什么都不预测"差 3.4 倍。

**修复**：缓冲区 key 改为 `(字段名, 形状, dtype)`；新增 `assert_distinct_buffers()` 在每次组装后校验没有两个字段共用 storage（用 storage 指针比较，因为两个 view 也能共享内存）。

**测试**：`pointflow_batch_test.py` 的 fixture 改为**逐字段不同常数**（原来全是 `np.zeros`，别名完全隐形）；新增 `test_each_field_gets_its_own_buffer`；修掉 `test_future_mask_does_not_change_topology_or_budget`（它同时持有两个 live batch 做比较，而两者本就共用缓冲区，断言恒真）。

**回归验证**：临时把 key 改回 `(形状, dtype)` → 3 个测试失败（含新加的），确认测试有牙齿。

---

## 2. 缺陷二：`attention_mode` 标签与实际执行不符

**位置**：`cosmos_framework/model/generator/mot/unified_mot.py:676`

**机制**：

```python
use_reference_point_attention = os.environ.get("POINTFLOW_REFERENCE_ATTENTION", "false") == "true"
```

默认 false，且**全仓库没有第二处出现这个变量**，launch 脚本也没导出。所以实际走 `dispatch_attention_fn` → `two_way_attention`，所有模态共用完整 mRoPE。但 `attach_point_tokens` 默认把 `attention_mode` 写成 `"pairwise_point_mrope"`，`cosmos3_vfm_network` 照样构建 `pointflow_modalities`，`pointflow_task4_sequence_attention.md:33` 还写着"是默认模式"。**三处元数据都在说这个模式生效了。**

**为什么参考实现是关着的（不是"还没验证"）**：`pairwise_point_attention` 是纯 Python 双层循环（样本 × 128 query 分块），每块物化最多 3 个 `[16, 128, n]` 分数张量。用 run 自己的数字（`vision_token_length=83232`、`action_token_length=1056` → 32 样本/批；point token = 9K，K≈450–900 → n≈7300–11000）：

- 每层每 forward ≈ 1800–2700 次 Python 迭代
- 这些中间量全部留在 autograd graph 里直到 backward → `activation_checkpointing` 也救不了（重算一层时那 1800+ 次迭代的图是一次性建起来的）→ 峰值 ≈ **100 GB**，**必然 OOM**

**语义上有一点要说清**：它是**按 query 分块**，不是按 key 分块，所以每个 query 的 softmax 仍覆盖全部 key —— 设计文档警告的"分块必须用 log-sum-exp 合并"在这里不适用。

**legacy 与设计的差距只有一处**：只有在 action↔point 这一对上多了空间旋转；video–point、point–point、text 本来就是设计要的样子。

**修复**：`pointflow_attention.py` 新增 `reference_attention_enabled()` / `default_attention_mode()` 作为唯一判定入口；`attach_point_tokens` 的模式由它推导；`unified_mot` 用同一个函数；eval 的 `metrics.json` 记录 `attention_mode`。

---

## 3. 缺陷三：视频缓存的画布被转置（回归）

**位置**：`tools/prepare_singlerighthand_video_cache.py:40-42`

```python
target_w, target_h = find_closest_target_size(height, width, resolution)   # 返回 (544, 736)
if str(resolution) == "480":
    target_h, target_w = 544, 736      # ← 交换了
```

以及同处新加的提前补零。

**几何事实（三个尺寸，用"宽×高"）**：

| 对象 | 宽 × 高 | 来源 |
|---|---|---|
| head.mp4 | 640 × 480 | 解码实测 |
| tracker 输入 | 640 × 448 | `int(480*1.0)//64*64` 然后 `cv2.resize`（**压缩非裁剪**） |
| 拼图 | 640 × 842 | `_compose_views`：wrist 640×362 在上 + head 640×480 在下 |
| manifest affine 授权画布 | 640 × 842 | `uv_to_video` |
| 缓存内容（正确） | 544 × 716 | 拼图 × 0.85，**不补零** |
| 画布（补零后） | 544 × 736 | 目标桶 '3,4' |

affine 的三个常数逐一被解释，不是拟合：

```
y' = 1.071429·v + 362.035714
     1.071429 = 480/448        ← 反解 tracker 的压缩
     0.035714 = (480/448−1)/2  ← cv2.resize align_corners=False 的像素中心项
     362      = wrist 视图高度（round(480·640/848)）
```

**证据（三条独立）**：

1. **仓库自带的测试**（`prepare_singlerighthand_video_cache_test.py`，本分支未改动）断言竖版且缓存/在线逐位相同 —— 本分支的改动让它挂了：
   ```
   assert torch.Size([2, 3, 544, 736]) == (2, 3, 716, 544)
   ```
2. **像素级 ground truth**（103436 个真实点，取真实帧颜色）：
   ```
   正确：×0.85 竖版 544×736     中位 ΔRGB   0.0    落在画面内 100.0%
   坏：  ×0.646 横版 736×544    中位 ΔRGB 255.0    落在画面内  22.9%
   ```
3. **91 个未被污染的原始缓存文件**全是 `(T,3,716,544)`

**连带缺陷**：`PointFlowSource.load` 把 `video_size_wh` 覆盖成调用方画布，使 `resize_pointflow_metadata` 的 align 分支看到"recorded == actual"而跳过重基 —— 而那个分支存在的唯一理由就是这种情况。（它的注释写着 "The source canvas and Cosmos output canvas are intentionally different"，下一行代码却把两者设成相等。）

**修复**：撤回覆盖与提前补零（== 恢复原代码）；`load` 保留 manifest 的画布，调用方画布另存为 `returned_video_size_wh`。

**数据动作**（用户执行）：重新生成 allowlist 的 10 个视频缓存 → `(T,3,716,544)`。`vae_latents/` 未动。

---

## 4. 数据侧现状（已核实）

```
video_frames/*.npy       10/10 = (T, 3, 716, 544) uint8   与 manifest 一致
video_manifest.json      schema 1, layout TCHW,
                         padding = deferred_to_ActionTransformPipeline,
                         image_size = [736, 544, 716, 544] × 10
vae_latents/*.pt         10/10 = (1, 48, T, 46, 34)
                         original_size=[716,544] padded_size=[736,544]
```

**latent 缓存是视频缓存的派生物**：`examples/cache_singlerighthand_vae_latents.sh` 的输入就是 `video_frames/*.npy`。（本次会话一度想为此加指纹校验，**已全部撤回**，见第 7 节。）

> **⚠ 已被取代（2026-09-13）**：上表 `vae_latents/*.pt` 是**整集编码 + 重采样**，与在线逐窗口编码实测差 ~0.41 相对误差（≈ 窗口平移一帧的量级），且条件 latent（index 0）身份不对。
> 训练现在读 `vae_window_latents/*.npy`（逐窗口独立编码，逐位一致）。旧缓存和旧代码路径**保留**但不再是首选。
> 详见 `pointflow_window_latent_cache_20260913.md`。`tools/verify_vae_latent_cache.py` 验的是**旧缓存与整集重编码一致**，**不能**用它证明逐窗口缓存正确 —— 那要用 `tools/verify_window_latent_cache.py`。

---

## 5. 本次新增的可复现验证工具

### `cosmos_framework/scripts/validate_pointflow_chain.py`

10 条链路、26 项检查，全部执行真实构造函数、打印所用证据。

```bash
B=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian
.venv/bin/python cosmos_framework/scripts/validate_pointflow_chain.py \
  --raw-root $B/raw_data/singlerighthand_sandwich_100 \
  --cache-root $B/datasets/singlerighthand-sandwich-100-cosmos-cache \
  --dense-root $B/datasets/sandwich_dense_fullseq_10_0298_20260908/outputs \
  --pointflow-manifest pointflow_outputs/task5/mixed_manifest.json \
  --episode-allowlist examples/pointflow_sandwich_10_episodes.txt
```

结果：**26/26 通过 + 3 条 note**。其中两项是**执行验证**（不是推理）：

- 视频 token 的 h/w **从 0 开始**（`h∈[0,21], w∈[0,16]`）→ point 位置与它共用同一坐标系
- 视频 latent 的 mRoPE 时间 = 0, 1.6, 3.2 … 12.8 → 与 point block 同格

### `tools/verify_vae_latent_cache.py`

在 GPU 上重跑 VAE 编码，与缓存逐帧比数值。`--frames` 是**原始帧数**：填 33 只查前 9 个 latent 帧（利用 Wan2.2 的时间因果性，几秒完成）；填该 episode 的完整帧数则查**整条 latent**。

```bash
.venv/bin/python tools/verify_vae_latent_cache.py \
  --raw-root $B/raw_data/singlerighthand_sandwich_100 \
  --cache-root $B/datasets/singlerighthand-sandwich-100-cosmos-cache \
  --vae-path $B/models/cosmos3-edge-droid/vae/Wan2.2_VAE.pth \
  --episode episode_0019_20260731_134304 --frames 1430
```

**已对全部 10 个 episode 按整集核验通过**（2026-09-13）。每个 episode 都是：读完整帧数 → 尾部复制到 1+4k → VAE 编码 → 与缓存逐帧比对，全部 `mean|diff| = max|diff| = 0.000000`。

```
episode_0013 (1192 帧) … episode_0019_20260731_134304 (1430 帧 → 1433 → 359 latent)
10/10 逐位相同
```

核验脚本：`/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/test_data.sh`（也等价于下面这个循环，帧数从 manifest 自动读）：

```bash
CACHE=$B/datasets/singlerighthand-sandwich-100-cosmos-cache
while read -r ep; do
  [[ -z "$ep" ]] && continue
  n=$(python3 -c "
import json
d=json.load(open('$CACHE/video_manifest.json'))
print(next(r['shape'][0] for r in d['episodes'] if r['name']=='$ep'))")
  echo "=== $ep ($n 帧) ==="
  .venv/bin/python tools/verify_vae_latent_cache.py \
    --raw-root $B/raw_data/singlerighthand_sandwich_100 --cache-root $CACHE \
    --vae-path $B/models/cosmos3-edge-droid/vae/Wan2.2_VAE.pth \
    --episode "$ep" --frames "$n" || break
done < examples/pointflow_sandwich_10_episodes.txt
```

---

## 6. 运行时护栏与评估诊断（新增）

### 护栏 `check_points_inside_video_grid`（`pointflow_branch.py`）

point token 的 `(h,w)` 与 video token 共用 mRoPE 轴，落在网格外就没有意义，而 loss 看不到、可视化也不画它。护栏在算完位置后立即检查：

```
POINT_GRID_MIN_INSIDE = 0.75   # 实测：画布对 = 0.969，画布错 = 0.158
POINT_GRID_MIN_TOKENS = 256    # 小样本上统计无意义（CPU 集成测试只有 5 个簇）
```

网格从 `sequence.vision.token_shapes` 读（元数据，**不在热路径做 device 同步**）。触发时报错直接说明病因。

### 评估诊断图 `point_video_grid_diagnostic`（`pointflow_visualize.py`）

现有三件套（comparison / error_map / error_curve）都不经过 `uv_to_video`：GT 直接读 `uv_px.npy`，预测用内参把 3D 重投影。所以**它们无法暴露本缺陷类** —— 这也是为什么当时"图看着没问题"。

新图重放**模型自己的算术**（组合后的 affine × anchor_uv ÷ patch stride），画在**模型实际收到的视频画布**上；越界点钳到边缘画红色。产物 `pointflow_eval/step_XXXXXXX/<case>/position_grid.png`，同时进 W&B。

实测对比：修复前 24.8%（红点堆在边缘）→ 修复后 96.9%（绿点铺在桌面物体上）。

---

## 7. 明确**没有**保留的东西

**指纹校验那一轮已全部撤回**（4 个文件，净变更为零）。以后若在仓库里找不到 `content_fingerprint` / `source_fingerprint` / `_check_latent_provenance`，那是**正常的**，不是文件损坏。

撤回的原因：它把一个"加个校验"的请求扩成了 4 个文件的改动，其中两处是破坏性 CLI 变更（`--source-fingerprint` 变必填；driver 在 manifest 无指纹时 `exit 1`），而当时的 manifest 没有该字段 → **会让缓存脚本直接不可用**。

**仍然成立的风险**（撤回后失去的防线）：`vae_latents/*.pt` 是 `video_frames/*.npy` 的派生物，而 driver 遇到已存在的输出会跳过。所以**重新生成视频缓存后必须手动删掉对应的 `vae_latents/*.pt` 再跑一次**，否则会静默沿用旧的。（本次不需要 —— 两者已验证为同一批像素。）

---

## 8. 已知但**未修**的问题

| 问题 | 量级 | 为什么不修 |
|---|---|---|
| token 网格高 22 vs 画布/32 = 23 | 差 1 行 | `_remove_padding_from_latent` 用整除 `716//16 = 44`，而内容占 44.75 行。补零量 20 不是 16 的整数倍，整行裁剪切不干净。点坐标最大 21.86 < 22，**不越界**。修它要改 Cosmos 公共路径，影响所有配方，应单独立项 |
| tracker focal vs D435 真值 0.818× | 495.2 vs 605.6 | tracker 模型本身的性质，预处理改不了 |
| 相机模型 `fx == fy`（各向异性未建模） | 应为 1.071429 | 同上；且量级小于 focal 那 18% |
| 组合 affine 的宽高比 0.042% | 842×0.85=715.7 进位到 716 | 可忽略 |

---

## 9. 尚**未验证**的（需要 GPU）

1. **transformer 是否按预期消费这些位置** —— 位置 id 的构造已执行验证，但没经过模型。护栏 + 诊断图在首次运行时给出答案。
2. **实际 token 数** —— 推算修复后为 **3366/样本**（= 9 × 22 × 17），**未实测**。跑起来后 `SequencePackingPadding/vision_token_length` 应为 **107,712**（= 32 × 3366）；若为 83,232 说明缓存或代码没生效。

---

## 10. 开训前检查清单

- [ ] 缓存已重生成：`video_frames/*.npy` = `(T,3,716,544)`（已核实 ✓）
- [x] latent 缓存有效：**10/10 episode 整集逐位相同**（`mean|diff| = max|diff| = 0`，2026-09-13）
- [x] 链路审计：`validate_pointflow_chain.py` 26/26（已核实 ✓）
- [x] 测试：42 passed（已核实 ✓）
- [ ] **从头训，不要 resume** —— 视频 token 数变了、旧 checkpoint 是在两个几何缺陷下练的
- [ ] 首跑后核对：`vision_token_length` = 107,712；`position_grid` 图上 ~97% 绿点

---

## 附：本次会话的完整改动清单

| 文件 | 改动 |
|---|---|
| `cosmos_framework/data/pointflow_batch.py` | 缓冲区按字段名分区；新增 `assert_distinct_buffers` |
| `cosmos_framework/data/pointflow_batch_test.py` | fixture 逐字段不同值；新增别名回归测试；修掉一条恒真断言 |
| `cosmos_framework/data/generator/action/pointflow_source.py` | `load` 保留 manifest 画布，调用方画布另存 |
| `cosmos_framework/data/generator/action/pointflow_source_test.py` | 第三条断言改为检验真实契约 |
| `cosmos_framework/model/generator/pointflow_attention.py` | 新增 `reference_attention_enabled()` / `default_attention_mode()` |
| `cosmos_framework/model/generator/pointflow_sequence.py` | 模式由该 flag 推导 |
| `cosmos_framework/model/generator/pointflow_branch.py` | 新增 `check_points_inside_video_grid` 护栏 |
| `cosmos_framework/model/generator/pointflow_branch_test.py` | 护栏单测；该用例自固定 `anchor_uv` |
| `cosmos_framework/model/generator/mot/unified_mot.py` | 用共享 flag；去掉每层都读的 `os` |
| `cosmos_framework/model/generator/mot/cosmos3_vfm_network.py` | 注释说明模态图只在 pairwise 时构建 |
| `cosmos_framework/callbacks/pointflow_eval.py` | 记录 `attention_mode`；传画布；保存/记录 `position_grid` |
| `cosmos_framework/callbacks/pointflow_visualize.py` | 新增 `point_video_grid_diagnostic` |
| `tools/prepare_singlerighthand_video_cache.py` | 撤回转置覆盖与提前补零（== 恢复原代码）+ 文档字符串 |
| `cosmos_framework/scripts/validate_pointflow_chain.py` | **新增**：10 条链路 26 项检查 |
| `tools/verify_vae_latent_cache.py` | **新增**：GPU 上验证 latent 缓存 |
| `examples/cache_singlerighthand_vae_latents.sh` | 本次一度改动，**已完全撤回**（见第 7 节） |
| `tools/cache_wan22_latents.py` | 同上，**已完全撤回** |
