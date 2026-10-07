# 逐窗口 VAE latent 缓存(方案 b)

**日期**:2026-09-13
**状态**:**已实现并验证通过**(2026-09-13):10/10 集逐窗口 latent 与在线编码逐位相同
**代码归属**:标注 `[本仓库]` 的是本仓库自有代码;未标注的 `cosmos_framework/**` 来自 NVIDIA 同步
(`e723d67 Sync NVIDIA cosmos-framework main at 5e67049`),本次**未改动**其逻辑。
**与其它文档的关系**:本文只讲**逐窗口 latent 缓存**这一项改动(数据侧)。
缺陷记录见 `pointflow_bugfix_log_20260912.md`;WAM 原理见 `cosmos_wam_mode.md`;
**位置编码/跨模态桥接的核查**见 `pointflow_alignment_audit_20260913.md`(同一天,另一件事,不重叠);
**运动选点**见 `pointflow_motion_selection_20260914.md`(次日,改的是送进模型的点数,不改 latent)。本文不重复它们的内容。

---

## 1. 要解决的问题

训练时视频 latent 必须和"不带缓存、在线编码"得到的结果**完全一致**,否则:

- 缓存成了训练数据的一部分来源误差 —— 而这种误差**不可见**(loss 照降,曲线好看,模型学的是另一套 latent)
- 更糟的是它专打**条件位置**。条件 latent(index 0)是推理时**唯一**真实输入的视频信息,它错了,训练和推理就接不上

所以判据只有一条:

> **缓存里第 w 个窗口的 latent,必须和在线对同一个窗口的 33 帧跑一次 `encode` 的结果逐位相同。**

不是"接近",不是"分布一致" —— 是逐位。

---

## 2. 旧缓存为什么不行

旧路径 `vae_latents/<episode>.pt` 存的是**整集编码**:把整集源帧(30 fps)一次性编码,再从结果里
按窗口的 15 Hz 时间栅格**重采样**出 9 帧。

对一条 episode 的实测(与在线编码对比,相对误差):

| 取样方式 | 相对误差 |
|---|---|
| `floor` 取整(旧实现) | **0.410** |
| `ceil` 取整 | 0.521 |
| **参照线**:把窗口整体平移一帧的在线编码 | **0.381** |

**0.410 ≈ 0.381** —— 也就是说,旧缓存和"在线但窗口取错了一帧"是同一个量级。

> 记录一个当时的错误判断:我最初猜 `ceil` 比 `floor` 好,实测 4/4 个窗口都是 `floor` 更优(delta=0),
> 猜想被推翻。旧实现选的 `floor` 已经是重采样里最好的那一种 —— **问题是重采样本身**。

### 根因:两种序列根本不是同一个东西

整集编码的序列是**连续的 30 fps**;窗口编码的序列是**每隔一帧取一个**的 33 帧。于是:

| | 整集 | 窗口 |
|---|---|---|
| 序列 | `0,1,2,…,1435` 连续 | `r, r+2, …, r+64` 隔帧 |
| latent k 覆盖的源帧 | `[4k-3, 4k]` | `[r+8k-6, r+8k]`(只取偶偏移) |
| 因果 chunk 边界 | 从帧 0 起每 24 帧 | 从帧 r 起每 24 帧 |

**不同的序列 + 不同的 chunk 对齐 ⇒ 结果不能互相推导。** 所以"从整集编码里读出窗口 latent"这件事
在原理上就不成立,不是调参能修的。

最直接的后果是**条件 latent 变了身份**:窗口的 latent 0 在线等于 `f(第 0 帧)`(因果 VAE 的单帧 prime),
而重采样拿到的是一个 **4 帧源帧块** —— 两者不是同一个量。

---

## 3. 方案:每个窗口独立编码

### 3.1 窗口定义

```
source_stride = round(source_fps / fps) = round(30/15) = 2
observation   = window_offset * sample_stride + arange(chunk_length+1) * source_stride

window_offset=0:  源帧 0, 2, 4, …, 62, 64
window_offset=1:  源帧 1, 3, 5, …, 63, 65
window_offset=2:  源帧 2, 4, 6, …, 64, 66
```

33 帧 @15 Hz,跨度 64 个源帧 = **2.133 s**。起点每次 +1,但采样步长是 2,所以相邻窗口取**不同奇偶**的源帧。

枚举公式与数据集 `singlerighthand_raw_dataset.py` 的 `_window_indices` **逐字一致**
(`tools/cache_window_vae_latents.py::window_plan`)。

### 3.2 完整链路(与在线路径同一套函数)

```
video_frames/<episode>.npy          ← 数据集实际交给模型的那些帧
  → reflection_pad_to_target        ← 真实的 NVIDIA 函数(reflect 填充)
  → uint8 → float32 / 127.5 - 1.0   ← 模型自己的归一化
  → Wan2pt2VAEInterface.encode      ← 模型自己的编码器
  → [48, 9, 46, 34] bfloat16
```

**每一环都调用训练时代码里的同一个函数**,没有重写、没有近似。

### 3.3 三层"chunk"要分清

| 层 | 是什么 | 尺寸 | 影响结果吗 |
|---|---|---|---|
| **数据窗口** | 33 个源帧(步长 2,跨 64 帧) | 2.133 s | 这是**数据定义** |
| **VAE 内部块** | `encode()` 内的计算分块 | 1 + 24 + 8 | **不影响**(`feat_cache` 跨块传递) |
| **latent** | 时间压缩 4 | 9 帧 | —— |

第二层是官方 docstring 写明的工程手段:整集一次算会让最深层的中间张量
(`T×640×368×272 ≈ 9.2e10`)超出 **Triton 的 int32 索引上限**(`2.1e9`)42 倍。
切开后用每层 `CausalConv3d` 的最后 2 帧(`CACHE_T=2`)续上下文,首帧单独 prime —— **数值不变**。
它和数据窗口无关:一个窗口的 33 帧是作为**一整条序列**送进 `encode()` 的。

### 3.4 存储格式

`vae_window_latents/<episode>.npy`,`[N_windows, 48, 9, 46, 34]`,dtype `uint16`:

- VAE 本身在 bfloat16 下计算,模型拿到后也是转 bfloat16 → 存 bf16 **位模式**是**无损**的,且文件减半
- manifest 记录 `storage = "bfloat16_bits_in_uint16"`,读取端**断言**这个字段,布局变了不会静默读成数字
- 读取走文件头 + `seek`,**不加载整个 npy**(单文件可达 2 GB)

单窗口 1,351,296 B(+128 B 文件头)。**10 条 episode 共 13,156 个窗口 ≈ 16.6 GiB。**

---

## 4. 为什么比整集编码慢 31.5 倍

**不是实现问题,是固有的。** 每帧成本完全相同(实测 ~16.8 ms/帧):

```
整集:  13,796 源帧              × 16.8 ms =   232 s  ≈  4 分钟
窗口:  13,156 窗口 × 33 帧      × 16.8 ms = 7,294 s  ≈ 122 分钟
       434,148 帧次 / 13,796 = 31.5 倍
```

**31.5 倍从哪来:overlap。** 窗口以 stride=1 滑动、每个窗口 33 帧,一个源帧 s 属于窗口
`r ≤ s ≤ r+64` 且 `(s-r)` 为偶数的所有 r —— 共 **33 个窗口**。**每个源帧被编码 ~33 次。**

这个冗余**省不掉**:要bit-exact就必须每个窗口独立跑一遍(见 2.2,两种序列不可互推)。

**能省的只有每次 `encode` 的固定开销**(batch 化摊薄,约 30 分钟)。真正的大头是那 31.5 倍的逐帧计算。

> 顺带:整集缓存快 31 倍,**恰恰是因为它没做该做的事**。这个 2.5 小时是"做对"的价格。

---

## 5. 代码改动清单

### 新增 `[本仓库]`

| 文件 | 作用 |
|---|---|
| `tools/cache_window_vae_latents.py` | 生成器:逐窗口编码 → uint16 npy + `window_manifest.json` |
| `tools/verify_window_latent_cache.py` | 验证器:从 **mp4** 独立重跑整条链,要求逐位相同 |

生成器要点:

- **每个 worker 必须自己 pin GPU** —— 子进程里裸 `.cuda()` 会全部落到 device 0。用 `--devices` 显式指定
- 输出先写 `<name>.npy.tmp`,整集跑完才 `os.replace` 改名 → **看到的 `.tmp` 就是未完成**
- `np.lib.format.open_memmap` **预分配整个文件**,所以**文件大小看不出进度**(需按已填充行数探测)
- manifest 在**全部结束后**才写

验证器要点:

- 帧来源是 **mp4 解码**,不是视频缓存 —— 换一个独立来源重跑,才能同时抓住"帧索引错""填充模式错"
  "归一化错""uint16 往返坏"这四类问题
- `--atol 0` 是默认值,**要求精确相等**

### 修改 `[本仓库]`

`cosmos_framework/data/generator/action/datasets/singlerighthand_raw_dataset.py`:

- 新参数 `vae_window_latent_root`
- `_read_window_latent(name, window_offset)`:读 uint16 npy、断言 storage、`.view(torch.bfloat16).unsqueeze(0)`
- `__getitem__` **优先**走窗口缓存;旧的整集路径降级为 `elif`,**保留可用**
- 启动时断言 manifest 的 `fps / chunk_length / sample_stride` 与数据集一致,**不一致直接拒绝启动**
  (缓存按 window_offset 索引,任何改变窗口枚举的改动都会使它失效 —— 宁可报错也不能读错窗口)
- 断言缓存覆盖所有被选中的 episode

`examples/launch_sft_action_policy_singlerighthand_edge.sh`:

- 新增 `SINGLERIGHTHAND_VAE_WINDOW_LATENT_ROOT`,默认 `$CACHE_ROOT/vae_window_latents`,已 export

### 未改动

- **旧 `vae_latents/*.pt` 缓存和旧代码路径全部保留**(按要求),只是不再是首选
- NVIDIA 同步文件的数据处理管线**未动**

---

## 6. 验证

### 6.1 数据链路(不需要 GPU)—— **26/26 通过**

`cosmos_framework/scripts/validate_pointflow_chain.py`,2026-09-13 实测。

> ⚠️ 其中 L7 验的是**旧的整集缓存**与视频缓存的关系,**不覆盖**逐窗口缓存。
> 别把 26/26 当成 §6.2 的通过证明。

### 6.2 逐窗口缓存(需要 GPU)—— **10/10 通过,逐位相同**

```bash
CACHE=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache
RAW=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/raw_data/singlerighthand_sandwich_100
VAE=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/models/cosmos3-edge-droid/vae/Wan2.2_VAE.pth

for ep in $(grep -v '^#' examples/pointflow_sandwich_10_episodes.txt); do
  .venv/bin/python tools/verify_window_latent_cache.py \
      --raw-root "$RAW" --cache-root "$CACHE" --vae-path "$VAE" --episode "$ep" || break
done
```

**结果:10/10 集,每集 5 个抽样窗口,`max|diff| = mean|diff| = 0.000000`。**

抽样窗口按**每集自身范围**均匀取(各集长度不同:1128 / 1147 / 1303 / 1687 / 1360 / 1395 / 1355 / 1137 / 1278 / 1366,合计 13156)。
旧默认值 `--windows 0,300,700,1000,1300` 在 **4/10 集上会越界**,已改为自适应;显式传越界值会直接报错,不再静默。

因为验证器从 **mp4** 重解码、而缓存来自 **`video_frames/*.npy`**,这次通过同时证明了两件事:

1. 逐窗口 latent 缓存 == 在线编码(逐位)
2. 视频缓存 == 新鲜解码 + 合成 + resize

### 6.3 消费路径(读代码确认,未执行)

缓存拿到手后**如何交给模型**,逐行追过:

```
① 数据集 _read_window_latent     view(bfloat16).unsqueeze(0)        [1,48,9,46,34]
② _vfm_inner_collate([s])      不在 list_collate_keys 里
                                → default_collate → stack            [1,1,48,9,46,34]  6-D
③ _split_one                   tensor 走 else: v[0:1]               [1,1,48,9,46,34]
④ _accumulate                  包成 list → 每样本一个                list of [1,1,48,9,46,34]
⑤ omni_mot_model.py:3707       list 分支
                                while ndim>5 and shape[0]==1: squeeze(0)   [1,48,9,46,34] ✓

在线侧: _encode_vision_item → [...,C_latent,T_latent,H_latent,W_latent]
                              list of [1,48,9,46,34] ✓
```

**两边到模型手上形状完全一致 ⇒ 训练行为一致。** 开缓存与不开缓存,模型看到的是同一个东西。

> **埋着的雷(今天走不到)**:`omni_mot_model.py:3701` 的 `ndim == 6` tensor 分支做的是
> `cached_latents[i].squeeze(0)` → **4-D**。当前 collator 产出的永远是 **list**,所以该分支**不可达**;
> 但若将来有人把 collator 改成输出堆叠 tensor,它会**静默给出 4-D** 而不是报错。
> 修法:把那个 `squeeze(0)` 换成与 list 分支相同的 `while` 循环。

### 6.4 生成指令(已完成,记录备查)

```bash
.venv/bin/python tools/cache_window_vae_latents.py \
    --cache-root "$CACHE" --vae-path "$VAE" \
    --episode-allowlist examples/pointflow_sandwich_10_episodes.txt \
    --workers 8
```

实测 ~1.64 s/窗口,8 worker 全程约 45 分钟。

---

## 7. 与官方在线路径的关系

**这个缓存不是在"近似"官方路径,而是在复现它。** 官方在线做的事是:

```
OmniMoTModel._encode_vision_x0_tokens:  对一个样本的 [C, T, H, W] 窗口编码,仅此而已
```

缓存做的是一模一样的事,只是把结果存在盘上避免每个 step 重算 33 帧的 VAE。

**为什么这份缓存对本配方尤其重要**:本机有 CUDA 12.8 runtime 但**没有 nvcc**,所以

```toml
[trainer.callbacks.compile_tokenizer]
enabled = false   # This cluster has the CUDA 12.8 runtime but no nvcc toolkit.
```

`compile_tokenizer` 关着 ⇒ **AOT 编译不可用** ⇒ VAE 每步都真跑 eager。
缓存把这份开销整个从训练循环里拿掉 —— 相当于用"离线批处理"替代"在线编译优化"。

> AOT(`Wan2pt2VAEInterface.compile_encode`)本来是把每块 encode 编译成 `.pt2`、按
> `(T_chunk, H_patch, W_patch, cache_t)` 查表调用,免得训练中途重编译。用不了,就缓存。

---

## 8. 训练前检查清单

- [x] `window_manifest.json` 存在(全部 10 集跑完才会写)
- [x] 无残留 `.npy.tmp`
- [x] `verify_window_latent_cache.py` 10/10 集 `max|diff| == 0`(2026-09-13)
- [x] `cosmos_framework/scripts/validate_pointflow_chain.py` 26/26 通过
- [ ] `SINGLERIGHTHAND_VAE_WINDOW_LATENT_ROOT` 指向 `vae_window_latents/`(启动脚本已默认,起训前确认一次)
- [ ] **从头训练,不 resume**(旧 run 已作废,见 `pointflow_bugfix_log_20260912.md`)
