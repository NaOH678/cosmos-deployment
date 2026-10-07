# FK + PointFlow 四模态合并说明

> **2026-09-30 更新**：base 已接入现有 sandwich 500 点分层缓存，保持 cluster 模式，
> scale 改为 0.0528（launcher 覆盖）。当前默认见 [base_quickstart.md](./base_quickstart.md) 开头；
> 本文 300 点配置及运行测量保留为历史记录。


> 目标：把 **FK（右手 21 关键点）** 和 **PointFlow（点云）** 两个模态合并进同一个 Cosmos，
> 使 `video / action / FK / pointflow` **四个模态独立并行去噪**。
>
> 本文档是这次合并的**唯一依据**（决策记录 + 执行状态）。
>
> **接手起点是 [§E](#e-执行状态截至-2026-09-29本节是接手起点)。** §7 的分步计划是 v1 的，
> 已被 §D v2 取代；§4/§5 的"两条互不通气的推理路径"同样被 §D 推翻 ——
> 这些小节都保留了原文并就地标注了推翻点，**照它们实施会得到错的方案**。
>
> | 同目录文档 | 用途 |
> |---|---|
> | **[`base_quickstart.md`](./base_quickstart.md)** | 开箱：环境 / 数据 / 启动 / **怎么读数字** / 排障 |
> | **[`base_multinode_training.md`](./base_multinode_training.md)** | 多机：注入机制 / HSDP / 提交 / 起后验证 / flash2 / 环境坑 |
> | [`fk_new_cluster_setup.md`](./fk_new_cluster_setup.md) | 新集群迁移：绝对路径重映射、数据拷什么 |
> | **本文件** | 为什么这么改（冲突清单、四条决定 D1–D4）、改到哪了（§E）、集群提交细则（§F） |

---

## 0. 术语与来源

| 代号 | worktree / 分支 | 内容 | 提交状态 |
|---|---|---|---|
| **P** | `WorldAct-cosmos3-edge-droid-sft` / `cosmos3-edge-droid-sft` | PointFlow / 点云 | **全部未提交**（131 个文件） |
| **F** | `WorldAct-cosmos3-edge-droid-sft_mano` / `WorldAct-cosmos3-edge-droid-sft_mano` | FK / 21 关键点 | 已提交。**本文档成文时 `3927e61`；2026-09-30 已前进到 `8e34582`（+5 commit）** —— 见下方 ⚠️ |
| **B** | `WorldAct-cosmos3-edge-droid-sft_base` / `WorldAct-cosmos3-edge-droid-sft_base` | **本次合并的工作区** | ⚠️ **20 个文件未提交**（15 改 + 5 新），HEAD `bae6a43`。**这是合并成果的唯一副本** |

三者共同基线：**`ab82606`**（= `origin/cosmos3-edge-droid-sft`）。

> ⚠️ **F 侧在 2026-09-30 又加了 5 个 commit，`_base` 一个都没有**（`git merge-base --is-ancestor`
> 逐个查过）。它们不改变本合并方案的任何结论，但改动了**两个 `_base` 也从它派生的文件**：
>
> | commit | 动的东西 | 对 `_base` 的影响 |
> |---|---|---|
> | `9b669d7` | `tools/run_fk_101.sh` 重写（+192/−120，"schedule the 101-episode run across nodes"） | `_base` 的 wrapper 是另一支，**不影响** |
> | `629bc86` | `launch_sft_action_policy_fk_singlerighthand_edge.sh` 加固（硬失败取代 `command -v python` 回退、解释器 pin 注释）+ 两个 `script/start_*.sh` | ✅ **`_base` 的 `fk_point` launcher 已有同等的加固**（`:91-123`、`:159-164` 的导入自检），是各自写的，**没有漏** |
> | `4ed4f7a` / `7fc5174` | `run_fk_retrain.sh`、一批 `tools/*.py` 的旧 gpfs 路径重指向 | 与 `_base` 无关 |
> | `8e34582` | `docs/fk_multinode_training.md` | 那是 F 线的多机文档；`_base` 的对应文档是本目录的 `base_multinode_training.md` |
>
> 结论：**这 5 个 commit 不需要往 `_base` 回移**。但 F 会继续往前走，
> 涉及 `omni_mot_model.py` / `fk_sampling.py` 这类**共享文件**的改动**必须**回移 ——
> 那才是本合并的真正风险面。

> ⚠️ **P 的全部工作在只有一份未提交的工作区里，没有远端备份。**
> 合并过程中**不得**对 P 做任何写操作（checkout / clean / stash）。所有拷贝都是**从 P 单向读出**。

---

## 1. 架构：Cosmos 的"模态接入模板"

这次合并在结构上之所以可行，是因为 P 和 F 是**同一个模板的两次实例化**。F 的设计文档
（`fk_modality_design.md` §0）本身就是"先读 P 怎么挂上去的，再照着做"。

一个模态接入 Cosmos 需要填的坑位（两边一一对应）：

| 坑位 | PointFlow (P) | FK (F) |
|---|---|---|
| **总开关**（env，`build_net` 内） | `POINTFLOW_SONATA_CHECKPOINT` | `FK_ENCODER_CHECKPOINT`（⚠️ 名字骗人，**不加载任何 ckpt**） |
| 数据侧开关（env → dataset kwargs） | `POINTFLOW_MANIFEST` | `FK_ANNOTATION_ROOT` |
| 时间契约 | `PointFlowTiming` | `FKTiming`（**两个不同的 frozen dataclass**） |
| 批契约 | `PointFlowBatch` / `PointFlowNoised` / `build_pointflow_batch` | `FKBatch` / `FKNoised` / `build_fk_batch` |
| token 预算 | `pointflow_token_upper_bound` | `fk_token_upper_bound` |
| 标签来源 | `PointFlowSource` | `FKSource` |
| 位置编码 | `point_positions` | `fk_positions` |
| token 挂载 | `attach_point_tokens` | `attach_fk_tokens` |
| 网络分支 | `PointFlowBranch`（Sonata/PTv3 + codec） | `FKBranch`（index-embedding + MLP） |
| 安装函数 | `net.install_pointflow(...)` | `net.install_fk(...)` |
| 加噪 / loss / 指标 | `pointflow_add_noise` / `pointflow_loss` / `pointflow_ade` | `fk_add_noise` / `fk_loss` / `fk_ade` |
| 采样 | `sample_displacement` | `sample_displacement` |
| 训练配置字段 | `pointflow_*`（3 个） | `fk_*`（5 个） |
| 最佳学习率 | `lr_multipliers["pointflow_branch.codec"]=25.0` | `lr_multipliers["fk_branch"]=25.0` |
| eval 回调 | `PointFlowEvalCallback` | `FKEvalCallback` / `FKRolloutCallback` |

### 1.1 两边共同修改的共享结构（合并时的接触面）

```
PackedSequence          +3 字段 (P)  +3 字段 (F)   → 共 6 个，名字无交集
GenerationDataClean     +1 (P)      +1 (F)
GenerationDataNoised    +1 (P)      +1 (F)
SequencePlan            +has_point (P only；F 没加)
all_gen_indexes         两边都在 sound 之后 append
collator/batcher        list_collate_keys +1, sparse_data_keys +1, token 预算 +1 行, backfill shim ×1
```

### 1.2 两种模态的物理差异

| | PointFlow | FK |
|---|---|---|
| 输入 | 每窗口点云（Track4World 3D tracks） | 21 个手部关键点（`wuji_fk21.npz`） |
| 编码器 | vendored Sonata/PTv3（~3.5k 行，**冻结**） | `W_idx·e_index(i) + MLP_xyz(p/s)`（**求和不是拼接**） |
| token 数/样本 | `K × (1 + H/q)`（或 `per_point` 时 `N × (1 + H/q)`） | **189** = `21 × (1 + 32/4)`（固定） |
| 空间位置 mRoPE | 真 `(t,h,w)`，来自 UV→patch 仿射 | **强制 `(t,0,0)`**（借用 action 约定，设计决策 G） |
| 坐标系 | DA3 | 真实 D435 |
| 注意力 | 有 `pairwise_point_mrope`，**默认关** | 无 |
| 位移归一化 scale | 0.0432 / 0.0482 / **0.0740**（随选择规则变） | 0.074450（10 回合）/ **0.083745**（101 回合） |

> FK 设计决策 C **明确排除 PTv3**（2cm 体素化会把相邻指节合并、置换不变性丢掉解剖身份）。
> 这是对 point 方案的有意偏离，**合并时不要"统一"两边的编码器**。

---

## 2. 最重要的共同发现：两条线独立撞上同一个 σ bug

两边各自把自己的模态 σ **硬绑在视频 σ** 上，于是 eval 时"干净视频 + 噪声模态"这个条件
在训练分布里**概率为零**，模型学成"视频干净 ⇒ σ≈0 ⇒ 直接吐回输入"。

| | 字段 | 修复 | 修复前 → 修复后 (ratio_to_zero) |
|---|---|---|---|
| P | `independent_pointflow_schedule` | 从同一 sampler、同 shift **独立抽 σ** | 1.53/3.48/1.93/5.92 → **0.82/0.53/0.84/1.14** |
| F | `independent_fk_schedule` | 同上（代码逐字相同） | 0.65/1.85/0.69/2.59 → **0.11/0.26/0.21/0.55** |

**合并含义**：两边的 σ 处理方式**已经一致**，合并不会出现"一个独立抽、一个跟视频走"的错配。
但**两个修复都要保留**——只移植一个会立刻在那个模态上复现 bug。

---

## 3. 冲突清单（已逐行核实）

### 3.1 假冲突：文本冲突，语义一致（选任意一份）

| 位置 | 为什么是假冲突 |
|---|---|
| `dataloader_val` 定义 | 两边**逐行等价**（只差局部变量名 `val_dataloader` / `_val_dataloader`）。同 `dataset_name`、`max_samples_per_batch=2`、`batch_size=1`、`num_workers=0`、`persistent_workers=False`、`prefetch_factor=None`、`split="val"`、`iterable_shuffle=False` |
| `validation_step`: `pass` → `return self.training_step(...)` | **两边同一处同一改** |
| `utils/misc.py` `TrainingTimer` 除零守卫 | 同上 |
| `callbacks/wandb_log_eval.py` 删 `assert len(dataset_name)==1` | 同上 |
| launcher 删 TorchCodec/NPP 环境块 + `import torchcodec` 预检 | 同上 |
| `examples/pointflow_sandwich_10_episodes.txt` | **两边都作为新文件添加，内容逐字节相同** |
| 纯 import 重排（ruff isort） | `omni_mot_model.py` / `unified_mot.py` / `attention.py` / `packers.py` / `joint_dataloader.py` / `transforms.py` |

### 3.2 真冲突：需要**人为决定**

| # | 键 | P 的值 | F 的值 | 建议 |
|---|---|---|---|---|
| **C1** | `trainer.run_validation_on_start` | `False` | `${oc.decode:${oc.env:FK_VAL_ON_START,true}}` | **取 F 的写法**，但把默认值改成 `false`：`${oc.decode:${oc.env:VAL_ON_START,false}}`。⚠️ **必须保留 `oc.decode`** —— 裸 `oc.env` 返回字符串 `"false"`，在 Python 里是 truthy |
| **C2** | 数据集 kwarg `episode_allowlist` | 由 `POINTFLOW_EPISODE_ALLOWLIST` 喂 | 由 `SINGLERIGHTHAND_EPISODE_ALLOWLIST` 喂 | **统一成一个**（建议 `SINGLERIGHTHAND_EPISODE_ALLOWLIST`，因为 `vae_window_latent_root`/`video_decoder` 已经是 `SINGLERIGHTHAND_*`）。留错了**不报错**，只是筛掉不同的回合集 |
| **C3** | `[model.compile] enabled` | `true`（在 toml 里） | 未设置 | 新 toml 里**建议先 `false`**。`true` 是 P 那条线有名的坑：首次 backward 触发 **Triton codegen OOM**，必须同时导出 `TORCHINDUCTOR_MIX_ORDER_REDUCTION=0` |

### 3.3 真冲突：结构性（推理路径，见 §5）

两个模态的**联合采样**代码都假设"自己是序列最后一个模态"，**互斥**。

### 3.4 会干净合并的

两个 `install_*` 门、`packers.py` 的 `return`、`trainer/__init__.py` 的条件、共享 recipe 的
optimizer/callback 段——都是**并列新增**（不是互斥），语义上必须**两边都保留**，只是文本相邻。

> ✅ 已核实：两个 `install_*` 的**调用位置对称**——都在 `parallelize_vfm_network`（FSDP 包装）
> **之后**（P: `:299`→`:328`；F: `:261`→`:285`）。**不存在安装顺序冲突。**
> （副作用：两个分支都不被 FSDP 分片。这是**两边共同**的既有性质，不是合并引入的问题。）

---

## 4. 训练路径：可以共存 —— ⚠️ **但需要一处修复**（见 §C 问题 1）

> ⚠️ **本节初稿的结论"可以任意顺序叠加，不会互相踩"是错的，已被证伪并修复。**
> 两个 `attach_*` 都**只重映射 vision/action/sound + 自己**，不重映射对方，
> 于是后挂载的一方会让先挂载的一方索引失效。详见 §C 问题 1。
> 下表列出的"同构"事实本身成立，但它们**不足以**推出可叠加。

**结论（修复后）：`attach_point_tokens` 与 `attach_fk_tokens` 可以任意顺序叠加。**

关键证据（两个函数是同一模板）：

| 关注点 | 两边代码 | 后果 |
|---|---|---|
| `position_ids` 重建 | `cat(sequence.position_ids[:, old:old+n], 自己的positions)` | **都保留既有值** → 后执行的不会覆盖先执行的 |
| `split_lens` 扩展 | `splits.extend((sl[2b], sl[2b+1]+len(content)))` | 不变量 `sum(sl[2b:2b+2])==sample_lens[b]` 各自维护一致 → 第二个的守卫必然通过 |
| 序号重映射 | `remap[old]=new`，**只覆盖原有 token** | 互不覆盖 |
| 幂等守卫 | `sequence.point` / `sequence.fk` | 不同字段，互不干扰 |
| `attn_modes` 守卫 | `== ["causal","full"]×N` | 两边一致 |

其余训练路径：

- `_add_noise_to_input` 加两个**独立 kwarg**（`sigmas_pointflow` / `sigmas_fk`），
  两个 `_get_train_noise_level_*` **函数体逐字相同**。兼容。
- `_compute_losses` 两个独立 block，`losses_dict` 键名无交集。兼容。
- `PackedSequence` 6 个新字段名字完全无交集；`to_cuda` 两段并列插入。
- 两个 buffer 池是**各自模块级的独立全局**，且 FK 强制 `fk_` 前缀 → 不会出现 P 那种
  `anchor_xyz` 与 `normal` 共享存储的事故（`fk_batch.py:18-23` 记录了这个真实事故）。
- `optimizer["keys_to_select"]` 两边都是「取列表别名 + 幂等 append」→ **两个分支都拿到梯度**。

---

## 5. 推理路径：唯一需要改代码的互斥点

### 5.1 矛盾

**P 走主推理路径**，并硬编码"自己是尾部"：

```python
# omni_mot_model.py:2427  _get_velocity
# Pointflow is the trailing segment
offset 按 vision → action → sound → point 累积
```

**F 走完全独立的另一条路径**（`_prepare_inference_data` / `_get_velocity` 里**一行 F 代码都没有**），
它自己那套也硬编码"自己是尾部"：

```python
# omni_mot_model.py:3437  _sample_joint
fk_parts.append(result[i][cursor:].view(horizon, int(end - start), 3))
# omni_mot_model.py:3587  _make_joint_velocity.split
fk.append(joint[i][cursor:].view(steps, int(end - start), 3))
```

`cursor` 走完 vision(+action) 之后，`[cursor:]` **把剩下的全吃掉当 FK**。

**→ 两个模态都要求自己是最后一个，只能有一个成立。**

### 5.2 解法：钉死顺序 `[vision | action | FK | pointflow]`

> 🔴 **本节顺序已被 §D3 推翻（2026-09-29）。** 现行顺序是
> **`[vision | action | sound | pointflow | FK]`（FK 挂尾）**，不是本节写的 `FK | pointflow`。
> 下面这段"改 2 行"的做法也已由 §E 第 3、4 项以另一种方式落地（PointFlow 从吃尾部改成按
> `point_spans` 显式切，FK 收尾）。**照本节实施会得到反的顺序。** 保留原文只为记录 v1 的推理。

**F 只需要改 2 行。P 一行都不用改。**

理由：F 那套**已经有显式尺寸表**，只是没用上：

```python
# fk_sampling.py:139
fk_sizes = [(end - start) * 3 * int(horizon) for start, end in spans]
```

把两处 `result[i][cursor:]` 改成 `result[i][cursor : cursor + layout.fk_sizes[i]]` 即可：

- F 不再吃尾部 → 只取自己的段
- **P 仍然在尾部 → 它的 `trailing segment` 假设依然成立，零改动**
- `flatten_pieces`（`fk_sampling.py:144`）本来就在对每个 piece 做 `numel()` 校验，
  尺寸不符会**响亮报错**，不会静默错位

### 5.3 仍然存在的静默风险

| 风险 | 说明 |
|---|---|
| P 的 `_get_velocity` offset 累积**没有边界校验** | 若有模态被追加在 point **之后**，切片会**静默错位**——训练照跑、loss 照降。这也是必须把顺序钉成 `[FK \| point]` 而不是 `[point \| FK]` 的原因 |
| `pairwise_point_mrope` 把 FK token 当"其他" | 它按 0=其他 / 1=action / 2=point 编码，FK 落在 0 → **失去时间轴旋转**，退化成普通全 mRoPE。**不报错**。不过这条路默认关（`POINTFLOW_REFERENCE_ATTENTION` 不设走 `legacy_mrope`），属 v2 议题 |

---

## 6. 必须记住的坑（跨两侧）

1. **`FKTiming != PointFlowTiming` 恒为真**（dataclass `__eq__` 要求同类型），即使三个字段完全相同。
   两边各自在自己的 batch 内比较，实际不会撞——但**不要把一个 timing 对象传到另一侧**。
2. **scale 不匹配没有任何运行时报错**，只是所有预测整体缩放错。两个 scale 都要各自钉 CLI。
3. **两边都设了 `lr_multipliers=25.0`**。F 已实测：单独的 FK 分支让 video loss **全程恒定 +12~15%**
   （机制怀疑是共享 attention 被 25 倍步长梯度持续扰动）。**两个分支同时开时这个代价很可能叠加**
   —— 这是合并后第一个该测的数。

   **已测（2026-09-30），结果与"叠加"不符，但不足以反证。** 取两个 run 的
   `train@2_detail/flow_matching_loss_vision`（都是 16 卡 × batch 16 = 全局 256，量级可比）：

   | 窗口 | 四模态（2 分支） | `fk-101-16gpu-flash2`（1 分支） | 比值 |
   |---|---|---|---|
   | 前 10% | 0.2 | 0.1 | **1.25** |
   | 10–25% | 0.2 | 0.1 | 1.10 |
   | 50–75% | 0.1 | 0.1 | 1.01 |
   | 末 10% | 0.1 | 0.1 | **0.98** |
   | 全均值 | 0.1 | 0.1 | **1.05** |

   即：**早期高 ~25%，随后收敛到持平，不是"全程恒升"**。⚠️ 但混淆很重 ——
   两个 run 长度不同（182 vs 70 个记录点）、采样节奏不同、四模态还多了 PointFlow 数据本身
   改变了 batch 构成；而且数值只有 0.1~0.2，**只有 1~2 位有效数字**。
   所以正确结论是 **"未复现"而非"已排除"**，§6.3 那个问题仍开着。
   要真正定量需要一个"两分支都在、但不含另一模态数据"的对照，**不存在**。
4. **`vae_window_latent_root` 只有一个值**，两个模态都要读窗口 latent → 指向同一份缓存即可。
   注意 P 侧的 `tools/cache_window_vae_latents.py` 是 F 侧数据集代码**引用但缺失**的脚本，
   合并正好补上这个依赖。
5. **`tools/__init__.py`** 由 F 添加（使 `tools` 可导入）。
6. ~~**开发节点无 GPU**：本机只能做导入检查 / CPU 单测；任何训练或渲染步骤都要交给 GPU 机器。~~
   ⚠️ **已过期（2026-09-29）**：开发节点现为 **8× A800-SXM4-80GB（全 NV8 互联）**，就是给验证用的
   —— **该测就测，不要以"没有 GPU"为由跳过**。生产目标是 16 卡（2 节点 × 8）。
   详见 §F.0。旧记忆 `dev-node-no-gpu.md` 已删，现为 `dev-node-gpu.md`。

---

## 7. 分步执行计划

> 每步结束都有**验收标准**。不通过就不要进下一步。

### Step 1 — 拷贝新文件（无冲突）

两边新文件**路径交集只有 `examples/pointflow_sandwich_10_episodes.txt`，且内容相同**。

- 从 **P** 拷：`cosmos_framework/auxiliary/sonata/`、`cosmos_framework/**/pointflow_*`、
  `cosmos_framework/scripts/*pointflow*|*sonata*`、`tools/*pointflow*|latent*|efep*`、
  `examples/*pointflow*`、`docs/pointflow_*`
- 从 **F** 拷：`cosmos_framework/**/fk_*`、`cosmos_framework/data/vae_window_latent_test.py`、
  `tools/*fk*|render_*|calibrate_*|verify_*`、`examples/*fk*`、`docs/fk_*`
- **排除**：`=10.1`、`quota_test.bin`（0 字节的 shell 重定向残渣）

**验收**：`git status` 里新文件数 ≈ 106 + 68 − 1（重复项）。

### Step 2 — 合并 19 个共享文件

顺序很重要：

1. **先跑 `ruff check --fix` / `ruff format`** 消掉纯 import 重排（那批 `[MOVE]-only`）。
2. 再手工处理 §3.2 的三处决定 + §3.4 的并列新增。
3. `packers.py` 的 `return`：保留 F 的 `packed = seq_builder.finalize(...)` 结构，
   把 P 的 `has_point` 校验和 `pointflow_data` 挂载也放进去。
4. 共享 recipe：两边的新 callback / 新 dataset kwargs / `dataloader_val` **都要保留**。

**验收**：`uv run ruff check .` 通过；`python -c "import cosmos_framework"` 成功。

### Step 3 — 单模态冒烟：只开 P

`POINTFLOW_SONATA_CHECKPOINT` 设上、`FK_ENCODER_CHECKPOINT` 不设。

**验收**：P 侧的 CPU 单测全绿（`PYTHONPATH=. <sft venv>/bin/python <test>.py`，**不是 pytest**——
那个 venv 没装 xdist，conftest 会 `PluginValidationError`）。

### Step 4 — 单模态冒烟：只开 F

反过来。

**验收**：F 侧的 5 个 FK 单测全绿（`fk_joint_sampling_test` / `fk_sequence_test` /
`fk_branch_test` / `fk_batch_test` / `fk_eval_test`）+ `validate_fk_network.py`。

### Step 5 — 双模态训练 step（不带联合采样）

两个门都开，只跑训练 step。验证：`attach_*` 叠加正确、两份 loss 都出、**无静默错位**。

**验收**：loss dict 同时含 `flow_matching_loss_pointflow` 与 `flow_matching_loss_fk`；
`all_gen_indexes` 长度 = 四段之和；video loss 的劣化幅度被记录（见 §6.3）。

### Step 6 — 推理路径改造

> 🔴 **已被 §D3 取代（2026-09-29）**：现行顺序是
> **`[vision | action | sound | pointflow | FK]`（FK 挂尾）**，不是本步写的 `FK | pointflow`。
> 本步实际由 §E 第 3、4 项完成 —— 见 §E「部署推理路径四段化」。

按 §5.2 改 F 的 2 处切片，钉死 `[vision | action | FK | pointflow]`。

**验收**：`fk_joint_sampling_test.py` 的 layout 算术对四段仍然成立；
`flatten_pieces` 的尺寸校验在四段下通过。

### Step 7 — 端到端

新 toml（第三个）：**`examples/toml/sft_config/action_policy_fk_point_singlerighthand_edge.toml`**
（`job.name` = `action_policy_fk_point_singlerighthand_edge`，`job.experiment` 仍指向共享 recipe
`action_policy_singlerighthand_edge`）。

> 本节初稿把文件名写成了 `action_policy_fk_pointflow_singlerighthand_edge.toml` ——
> **没有这个文件**。实际落地名把 `pointflow` 缩成了 `point`，且 `job.name` 与目录名一致
> （`tools/run_fk_point_101.sh:27` 的 `REL` 常量就是这么拼的）。

---

## 附录 A — 环境与路径约束

- **两个 worktree 共用一个 venv**：`WorldAct-cosmos3-edge-droid-sft/.venv` 里 `cosmos_framework`
  是 editable 安装，`.pth` **硬编码指向 P**。B 借它靠 `PYTHONPATH=.` 优先。
- ⚠️ **跑任何脚本都必须 `PYTHONPATH=.`**。`python <文件>.py` 的 `sys.path[0]` 是**脚本所在目录**
  而非 CWD，会**静默加载 P 的代码**（症状："我改的代码没生效"）。
- **B 里没有 launcher 的两道守卫**（守卫是随 F 的提交进来的）。新增 launcher 时要带上。
- uv 在 `WorldAct-cosmos3-edge-droid-sft/.venv/bin/uv`（不在 PATH 上）。

## 附录 B — 验收用的判据

- 唯一裁决键：`fk/val_XX/ratio_to_zero` 与 `pointflow/.../ratio_to_zero`，**< 1 才算赢**。
- **`zero_ade_mm` 是纯数据属性**（`trajectory_metrics(zeros, target)`），与模型无关。
  两个 run 的 zero 不同 = 评的不是同一个窗口，**不可比**。
- **跨 scale 的 loss 读数不可比**（scale 变了，"预测零"基线也会变）。只看 ADE / ratio。
- **rollout 的 ratio 不可与固定窗跨比**（zero 量级不同：rollout ~807mm vs 固定窗 20~240mm）。

---

## 附录 C — 执行记录（Step 1–2，2026-09-24）

### Step 1 完成
P 120 个 + F 68 个新文件，共 174 项，零缺失。

> ⚠️ **坑**：`git status --porcelain` 默认把未跟踪**目录**折叠为一条，`cosmos_framework/auxiliary/sonata/` 整个没被拷进去。必须用 `-uall` 取文件级清单。
> 另外 `=10.1` / `quota_test.bin` 是 shell 重定向残渣，已排除。

### Step 2 完成：19 个共享文件，52 处冲突全部解决

| 类别 | 处理方式 |
|---|---|
| 并列新增（两个模态的代码共存） | `both`（~38 处） |
| 需人为决定 | 3 处（见 §3.2 的 C1/C2/C3，均已按建议落地） |
| 手工拼装（直接 `both` 会产生重复定义） | 8 处 |

**已落地的三项决定**：

- **C1** `run_validation_on_start`：合并为**单一** `trainer.update`，取 `oc.decode` 写法、默认值 `false`。
  （FK 原先那个 `trainer.update` 已删除，其注释折进保留的那个。）
- **C2** `episode_allowlist`：统一为 `SINGLERIGHTHAND_EPISODE_ALLOWLIST`。
- **C3** 模态顺序：钉死 `[vision | action | FK | pointflow]` —— `attach_fk_tokens` 在前、point 在后，
  `all_gen_indexes` 顺序一致。**P 侧零改动**，其 `trailing segment` 假设继续成立。
  > 🔴 **这条顺序已被 §D3 推翻（2026-09-29）**：现行顺序是
  > **`[vision | action | sound | pointflow | FK]`（FK 挂尾）**，且 PointFlow **不再**是尾段
  > （它改成按 `point_spans` 显式切）。上面"P 侧零改动"的前提也随之失效 ——
  > P 侧改了 `_get_velocity` 的切片方式，见 §E 第 3 项。

### ⚠️ 最重要的一条经验：0 冲突 ≠ 合并正确

`cosmos_framework/data/generator/action/datasets/action_sft_dataset.py` 在 `git merge-file` 里
报 **0 冲突**，但合并结果**语法就是坏的**：两边在函数签名和调用处**不同位置**各加了同一个
kwarg，git 把两处都并了进去，产生

- **重复形参** → `SyntaxError`
- **重复关键字实参** → Python **不报错**，后者静默覆盖前者

该文件里一共清掉 **6 个重复项**（3 形参 + 3 实参）。其他文件（`collators.py` 的
`sparse_data_keys`、`sequence.py` 的 `to_cuda`）也是同类问题的变体——**重复字典键同样静默覆盖**。

**结论：合并后必须跑 AST 级扫描**（重复形参 / 重复关键字实参 / 重复字典键），
不能只看冲突计数。本项目为此的检查项：

```bash
# 1. 冲突标记
grep -rn '^<<<<<<<\|^>>>>>>>' cosmos_framework examples tools
# 2. AST 扫描：重复形参 + 重复关键字实参
# 3. ruff check --fix + ruff format（消 import 重排噪声）
```

### 验收结果（Step 1–2 后）

| 检查 | 结果 |
|---|---|
| 冲突标记残留 | **0** |
| 全仓 `.py` 语法错误 | **0** |
| 全仓重复形参 / 重复关键字实参 | **0** |
| 19 个合并文件的 `ruff check` | **全过**（6 条 import 排序已 `--fix` + `format`） |
| launcher `bash -n` | **通过** |
| 模块导入冒烟（两侧新模块 + 合并共享模块） | **30/30 成功**，且 `cosmos_framework` 正确解析到本 worktree |

### 一个易被误判的现象

`grep -rc '^<<<<<<<'` 之类的粗筛会给出假阳性/假阴性，别只依赖它。同理，
"某文件 0 冲突" 是最容易让人跳过检查的信号——见上。

### 🔴 复查发现的两个问题（均已修复）

#### 问题 1：跨模态索引失效 —— **合并新造的真 bug**

两个 `attach_*_tokens` 都写于"自己是唯一额外模态"的年代，`dataclasses.replace(...)` 里
**只重映射 vision/action/sound + 自己**，不重映射对方：

```python
# pointflow_sequence.py   —— 没有 fk=
        vision=modality(sequence.vision),
        action=modality(sequence.action),
        sound=modality(sequence.sound),
        point=PointTokenPayload(...),
# fk_sequence.py          —— 没有 point=
```

单独用时无害（对方字段是 `None`）。但**两者同时存在**时，后挂载的一方会把先挂载的一方的
`sequence_indexes` 留在**旧坐标系**里。实测（2 样本、FK 先挂）FK 索引 `119..136`
与 sample 1 的 vision `113..121`、action `122..154` **直接重叠**。

**为什么危险**：`cosmos3_vfm_network.forward` 里

```python
packed_sequence[packed_seq.fk.sequence_indexes] = packed_seq.fk.tokens...
```

会**写到别的模态的位置上**——写入成功、**没有形状错误**，只是激活值错了；
`all_gen_indexes` 也会路由到过期位置。**loss 照降**。

**修复**：两边各加一个 `carried_tokens(payload)` 辅助函数，把对方已挂好的 payload
一并过 remap。对称、各 8 行。

**回归测试**：`cosmos_framework/model/generator/fk_pointflow_compose_test.py`。
已做"有牙"验证——撤掉修复即报
`vision indexes collide with the new modalities: [119, 120, 121]`。

#### 问题 1b：C2 决策只改了 recipe，没改 launcher —— **allowlist 静默失效**

**独立复查发现，我漏了。** C2 把 recipe 的 env 名统一成 `SINGLERIGHTHAND_EPISODE_ALLOWLIST`
（`action_policy_singlerighthand_edge.py:461`），但共享 launcher 仍然只设置/导出
`POINTFLOW_EPISODE_ALLOWLIST`（`launch_sft_action_policy_singlerighthand_edge.sh:23,95`），
P 的三个 wrapper（`launch_pointflow_train_1gpu.sh` / `_labeled29_sandwich.sh` /
`_dropper101.sh`）也是。

**后果**：`episode_allowlist` 解析为空字符串 → 数据集读作"allowlist 关闭" →
在**全部**回合上训练，而不是 launcher 点名的那 10 / 29 / 101 个。**不报错。**

**修复**：4 个 launcher 统一到 `SINGLERIGHTHAND_EPISODE_ALLOWLIST`，
旧名 `POINTFLOW_EPISODE_ALLOWLIST` 保留为**回退**（`${NEW:-${OLD:-default}}`），
所以老命令不会失效。已验证三种设值方式都解析正确。

> **教训**：C2 这类"统一名字"的决策，改的是一条**链**（recipe ← launcher ← wrapper），
> 只改链的两端之一不会报错，只会静默改变训练集。

#### 问题 2：Step 1 漏拷了「单边修改」的已跟踪文件

`git status` 只列**新增**文件，"只有一边改过"的**已跟踪**文件不在其中，第一批拷贝整个漏掉：

| 文件 | 漏掉的后果 |
|---|---|
| `cosmos_framework/data/generator/action/transforms.py` | **`sequence_plan.has_point` 永远不被设置** |
| `cosmos_framework/model/generator/mot/attention.py` | `SplitInfo.pointflow_modalities` 字段缺失 |
| `cosmos_framework/model/generator/mot/unified_mot.py` | pairwise attention 分发分支缺失 |
| `.ruff.toml` | sonata 的 `extend-exclude` 缺失 |
| `.gitignore` | `/pointflow_outputs/` 未忽略 |
| `singlerighthand_raw_dataset_test.py` | LRU 驱逐回归测试缺失 |

F 侧无此问题（它的 68 个改动全是新增文件）。

**教训**：合并的完整性检查必须覆盖 5 类，不能只做 2 类：

```
P 新增 / F 新增 / 两边都改（合并）/ 只有 P 改 / 只有 F 改
```

#### 问题 3：recipe 里 `dataloader_val` 被赋值两次（惰性，已清理）

`val_dataloader`（P 的）与 `_val_dataloader`（F 的）两块**逐行等价**，只是局部变量名不同，
都写 `action_policy_singlerighthand_edge["dataloader_val"]`——同一类**静默重复**，
与 `action_sft_dataset.py` 的重复 kwarg 同源。行为上无害（第二次 deepcopy 产出的对象相同，
复查已确认中间没有任何东西改写 `dataloader_train`），但属死代码，已删掉一份。

> 这是本次合并**第三次**遇到"git 无冲突、但结果重复"的同一类问题。
> 结论：合并后必须扫 **重复形参 / 重复关键字实参 / 重复字典键 / 重复配置键赋值** 四类。

### ✅ 复查验收汇总（Step 1–2 后，全部纯 CPU）

| 检查 | 结果 |
|---|---|
| **覆盖性**：P/F 动过的 **212** 个文件逐一分类断言 | **全部符合预期**（含下方 5 处有意差异） |
| **跨模态组合**：FK+point 两种 attach 顺序 | **27/27** |
| 回归测试反向验证（撤掉修复必失败） | **有牙** ✅ |
| **recipe 实际加载**：两个分支入 `keys_to_select`、`lr_multipliers`、3 个 callback、单一 `dataloader_val`、两个 `independent_*_schedule=True`、44 个 dataset 键 | **通过** |
| 全仓 `.py` 语法 / 重复形参 / 重复关键字实参 | **0 / 0 / 0** |
| 模块导入冒烟 | **30/30** |
| 19 个合并文件 ruff | **全过**（P 原有 3、F 原有 7 被打平） |
| 全仓 ruff 170 条 | **全部落在整份拷贝的文件上**（源 worktree 既有），合并文件 0 条 |
| 代码丢失检查（P/F 每一行新增是否仍在） | **无丢失** |
| 配置字段引用（`rf_cfg.*`）是否都存在 | **8 个字段全在** |

### 有意与源 worktree 不同 的 5 个文件

覆盖性检查会把这 5 个标为"≠ 源"，这是**预期的**——它们只差下方这一处修复，已逐一 diff 确认无夹带：

| 文件 | 差异 |
|---|---|
| `cosmos_framework/model/generator/fk_sequence.py` | 新增 `carried_tokens()` + `point=carried_tokens(sequence.point)` |
| `cosmos_framework/model/generator/pointflow_sequence.py` | 新增 `carried_tokens()` + `fk=carried_tokens(sequence.fk)` |
| `examples/launch_pointflow_train_1gpu.sh` | allowlist env 名统一 + 旧名回退 |
| `examples/launch_pointflow_labeled29_sandwich.sh` | 同上（export + echo） |
| `examples/launch_pointflow_dropper101.sh` | 同上（export + echo） |

**改动这 5 个文件之外的任何内容，都应视为回归。**

## §D v2：真正的四模态融合（用户 2026-09-29 确认，取代 §4/§5 的"两条并行路径"方案）

⚠️ **§4 和 §5 描述的"两条互不通气的推理路径"是 v1 的实现，用户明确否定了它**
（原话：不是要"硬生生地拼起来"）。v2 的目标是 **fk + point + video + action 四段同时进入
Cosmos 主干、在同一个去噪循环里一起去噪**。

### 四条已确认的决定

| # | 决定 | 对 v1 的推翻 |
|---|---|---|
| **D1** | **推理 = 一条循环，四段一起去噪**。原两条推理路径合并成一条，输出四段 | v1 是两个并行机制，各自丢弃对方 |
| **D2** | **撤销** v1 里那三处"剥掉对方模态"的修复，改成**四段都接**（都设上 `*_noised`） | v1 修的是 `_fk_fitting_context` 剥 pointflow、`sample_pointflow`/`pointflow_sigma_scan` 剥 fk、`_prepare_inference_data` 剥 fk |
| **D3** | 扁平向量顺序 = **`[vision \| action \| pointflow \| FK]`**（sound 若存在则在 action 之后） | v1 是 `[vision \| action \| FK \| pointflow]`；**FK 现在是尾段** |
| **D4** | **训练侧也要改**（用户将说明具体内容） | v1 认为训练侧已是四模态联合 |

### D3 的直接后果（这是 v2 的主要工作量）

顺序翻转后，两边"谁在尾部"的假设**全部对调**：

- **FK 的 `[cursor:]` 变成真尾段** —— 前提是 cursor 先跳过 point 段；`_make_joint_velocity`
  目前只拼 `[vision | action | FK]`，**必须插入 point 片**，`joint_layout` 也要加 point 段。
- **PointFlow 的 `_get_velocity`** 硬编码"pointflow is the trailing segment"（offset 累加到
  point 结束）—— 现在 FK 在它后面，**必须改成按显式 offset 切 pointflow**，不能再吃尾部。
- `_prepare_inference_data` 的 parts 顺序、`_get_velocity` 的重拼顺序、两处注释里的
  `[vision | action | sound | pointflow]` 全部要更新。
- 四个 `*_noised`（vision/action/pointflow/fk）都要在同一个 `denoise` 调用里供给。

> `_base` 里已落地但与 v2 冲突、需回退/改写的东西：三处"剥掉对方"修复（D2 撤销）、
> `_sample_joint`/`_make_joint_velocity` 里我加的 `fk_sizes` 定界注释（定界本身可留，
> 但顺序说明要改）、`docs/fk_pointflow_merge.md` §4/§5 的结论。

### D4 已定：训练回到**原生 Cosmos 的全共享 σ**（2026-09-29）

`independent_action_schedule` / `independent_sound_schedule` / `independent_pointflow_schedule` /
`independent_fk_schedule` **全部 False** —— 同一样本内五个模态用**同一个 σ**，
与四模态联合推理的部署条件一致。

recipe 已改：`action_policy_singlerighthand_edge.py:125` 与 `:168` 由 `True` 改为 `False`
（action/sound 本来就没在 recipe 里设置，用原生默认 False）。

**代价（已实测、必须记住）**：条件 arm 重新变成**分布外**。共享 σ 下"干净视频 + 噪声模态"
这个组合在训练分布里概率为零，模型会学到"视频干净 ⇒ σ≈0 ⇒ 输出输入"的捷径 ——
这正是两条线当初各自实测到的失效（PointFlow eval ADE/zero 单调涨到 6.8；
FK 的 21 个点沿初始噪声飞散）。所以：

- **条件 arm 的数字从此不可比**，它本来就是上限、不是部署性能
- 若日后要恢复条件 arm 的诊断价值，需另设一条**独立 σ 的对照 run**，不能在同一次训练里兼得

### 实施顺序（重要：先建路径，再拆拐杖）

那三处"剥掉对方模态"的修复（D2）**必须在四模态联合路径建好之后才能撤** ——
它们的成因是"packed 里带着对方的 payload 但没供给对应的 `*_noised`"。
路径建好后两边都供给 `*_noised`，剥除自然不再需要；**先撤会立刻崩**。

1. **一条联合循环，四段都供给 `*_noised`。** 在联合路径里同时设
   `packed.pointflow_noised` 与 `packed.fk_noised`，一次 `denoise` 产出四段。
2. **扁平向量改为 `[vision | action | sound | pointflow | FK]`**（D3；sound 实际不出段）。要改：
   - `_prepare_inference_data` 的 parts 追加 FK 段
   - `_get_velocity` 的 offset 累加改成**显式 offset 切 pointflow**（不能再吃尾部），FK 收尾
   - `_fk_fitting_context` / `_make_joint_velocity` / `joint_layout` 加 point 段，
     或改为复用同一条联合路径
   - 结果 dict 增加 `"fk"` 键（现在完全没有，通用推理路径发不出 FK）
3. **验证后再撤** D2 的三处剥除。
4. 4 个 `independent_*` 已 False；`_get_train_noise_level_*` 方法保留（flag 门控，无害）。

**已知但未解、不属于本方案的**：加 FK 分支让 video loss 全程恒升 12~15%
（怀疑 `lr_multipliers=25.0` 扰动共享 attention）；四模态下两个分支都 25×，可能叠加。
→ **已初测，未复现"叠加"**（实测比值 1.25 → 0.98，全均值 1.05），但混淆重、仍开着；
完整数据与保留意见见 **§6 第 3 条**。

### 已知但**非**合并引入

- **F 的联合采样尾部切片**（§5.2）：`_sample_joint` / `_make_joint_velocity` 用
  `result[i][cursor:]` 吃尾部。两者同时开时会把 point token 也算进 FK——
  但 `flatten_pieces` 的尺寸校验会**响亮报错**，不会静默错。属 Step 6。
- **`types.py` 里有一份陈旧的 `SequencePlan` 副本**（没有 `has_point`）。
  包 `__init__` 从 `sequence` 导出，运行时用的是正确那份；`types.py` 的那份未被使用。
  这是 base 就有的重复定义，不是合并产物。

### 尚未完成

- Step 3/4：全仓 CPU 单测（后台运行中）
- Step 5：双模态训练 step（**需要 GPU**）
- Step 6：改 F 的 2 处 `[cursor:]` 切片（见 §5.2）
- Step 7：新建第三份 toml


---

## §E 执行状态（截至 2026-09-29，本节是接手起点）

### 已改动的文件（全部在 `_base`，未提交）

**训练 σ —— D4 已落地**
- `cosmos_framework/configs/base/experiment/action/posttrain_config/action_policy_singlerighthand_edge.py`
  - `independent_pointflow_schedule` / `independent_fk_schedule`：`True` → **`False`**（`:125`、`:168`）
  - `pointflow_displacement_scale`：`0.0740` → **`0.075255`**（旧 manifest 的 `pointflow_source` 已是 `null`，跑不了；
    新值 = `sandwich_924_20260928` + `select_top_n=300` + mvs0 的池化 std，n=22,595,703）
- action/sound 的 switch 本来就没设 → 原生默认 `False`。**五个模态现在全共享 σ。**

**四模态联合路径 —— 布局层、场函数、FK 侧调用方全部已接（8 卡冒烟通过）**
- `cosmos_framework/model/generator/fk_sampling.py`
  - `JointLayout` 加 `pointflow_sizes`；`sizes()` 返回 4 元组
  - `segment_sizes(index, *, with_action, with_pointflow=False)`
  - `joint_layout(..., pointflow_shapes=None)` + 数量校验
- `cosmos_framework/model/generator/omni_mot_model.py`
  - `_make_joint_velocity(..., with_pointflow=False)`：`split()` 四段化（**每段按 `layout` 尺寸切，FK 不再吃尾部**）、
    `velocity()` 供给 `packed.pointflow_noised`、`built` 在 action 与 FK 之间插入 pointflow 片、
    新增 **PointFlow/FK horizon 必须相同** 的守卫
  - `_sample_joint` / `_fk_fitting_context` 里那两处 `[cursor:]` 已改为 `[cursor : cursor + fk_sizes[i]]`（防御性）
- `cosmos_framework/model/generator/fk_joint_sampling_test.py`：`sizes()` 解包改 4 元组 + 新增四段布局断言。**PASS**

**§E 第 1、2 项已完成（2026-09-29）—— FK 侧调用方接线**
- `_fk_fitting_context`：**撤掉「剥掉 pointflow」那一段**（三处剥除里的第一处，见下）；`packed.pointflow_data`
  非空时把 `pointflow_shapes` 传给 `joint_layout`、`with_pointflow=True` 传给 `_make_joint_velocity`
- 那条**两段 `velocity(state, sigma)`**：现在也供 `packed.pointflow_noised`（σ=0 = "给定"，与该 arm 的
  视频/action 一致）。不供会撞 `cosmos3_vfm_network.py:1064` 的 "never use GT by default"
- `_sample_joint`：联合向量改 `[vision | action | pointflow | FK]`，切分逐段按 `layout` 尺寸做
- 新增 `_seed_pointflow_from_noise`：照 `_prepare_inference_data` 第 7b 步**纯噪声**起步（pointflow 状态里
  没有 condition 段，无从做 blend），走 `video_seed + 2*MODALITY_SEED_STRIDE` 的独立流
- 新增一行 rank-0 日志，报联合向量实际构成。**原因是"每一段都是推导出来的，payload 没进来会静默退回
  三段、数字看着照样合理"，指标分不出来** —— 这行是把它变成可读的
- ⚠️ **已知缺口（未修，已写进代码注释）**：`vision_sigma` 那条诊断路把视频加噪到探测的 σ，而 pointflow
  仍按 σ=0 给，两者噪级不一致，属分布外。目前无 eval 用它（只有 `fk_velocity_field`）。
  收法：用 `pointflow_add_noise` 按同一 level 加噪

**8 卡冒烟实证**（`tools/run_fk_point_101.sh`，`MAX_ITER=2`、`FK_VAL_ON_START=true` 让第 0 步跑完整 eval、
`FK_EVAL_JOINT=1 FK_EVAL_JOINT_ACTION=1`、`FK_ROLLOUT=0` 省时间）：

```
FK joint arm: flat vector elements (sample 0) = vision 646272 | action 2112 | pointflow 28800 | FK 2016
```

`28800 = 32×300×3`（`select_top_n=300`）、`2016 = 21×3×32` —— **四段全非零、顺序正确**，
即 `packed.pointflow_data` 确实进来了、四段布局确实建起来了。这行日志是判据：若哪天它变成
`pointflow 0` 并带 "(no PointFlow payload packed)"，就是 payload 没进来、静默退回三段了。

两次独立运行（`fk-point-smoke-0929-4mod` / `-4seg`）数字逐位可复现（`train_00` 条件 arm 均为 239.98，
联合 arm 240.50 / 240.49），全程 0 异常。**注意 ratio 1.2~8.6 无意义**：这两次都是从基座起的新 run，
FK 分支是随机初始化的，冒烟验的是管道不是精度。

**A800/多机移植（已完成的另一摊）**
- `examples/_sft_launcher_common.sh`：多机拓扑推导（`SENSECORE_*` + pod 名兜底）
- `cosmos_framework/model/attention/flash2/checks.py`：`COSMOS_FLASH2_VARLEN` 逃生口
- `cosmos_framework/checkpoint/dcp.py`：save 前 `empty_cache()`
- 新增：`tools/check_flash2_varlen.py`、`tools/check_multinode_nccl.py`、
  `examples/launch_sft_action_policy_fk_point_singlerighthand_edge.sh`、
  `tools/run_fk_point_101.sh`、`examples/toml/sft_config/action_policy_fk_point_singlerighthand_edge.toml`
- 修掉两处陈旧路径：recipe 的 `pointflow_manifest` 默认值、两个 launcher 的 `SIBLING_VENV`
  （指到已失效的 GPFS 路径 + 静默回退系统 python）

**§E 第 3、4 项已完成（2026-09-29）—— 部署推理路径四段化**

扁平向量由 `[vision|action|sound|pointflow]` 改为 **`[vision|action|sound|pointflow|FK]`**（FK 挂尾）。
**有四处代码在切这个向量，必须同时改**，否则某段会被当成邻居读，而 `torch.cat` 对长度是宽容的 —— 不报错：

| 处 | 位置 | 改了什么 |
|---|---|---|
| 1 | `_prepare_inference_data` 的 parts 循环 | 新增 7c 步 FK 噪声（纯噪声，照 7b 的 PointFlow）+ FK 段追加在 pointflow 之后 |
| 2 | `_get_velocity` 的 offset 累加 | **PointFlow 从"吃尾部"改成按自己的 `point_spans` 显式切**；FK 才是尾段 |
| 3 | `_get_velocity` 的 velocity 重组 | FK 速度不过 mask（与 pointflow 同理：状态里没有 condition 段）；追加在 pointflow 之后 |
| 4 | `generate_samples_from_batch` 的结果拆分 | 新增 `result["fk"]`，乘 `fk_displacement_scale` 回到米（与 pointflow 同一约定：采样器全程在 model unit 里走） |

配套：
- `_get_velocity` 的 `gen_data_for_packing` 补 `fk=gen_data_clean.fk`（否则扁平向量里有 FK 段、却没有 FK token 供分支写入）
- `_get_velocity` 供给 `packed_sequence.fk_noised`（`FKNoised`，σ = t/num_train_timesteps，与 pointflow 同映射）
- `_can_reuse_inference_pack_templates`：有 FK payload 时返回 False（模板路不thread FK 状态）
- `_prepare_inference_data`：FK 的 GT labels 照 pointflow 那样清零
- **FK 的噪声种子单独偏移 `MODALITY_SEED_STRIDE`**。其余四个模态在这个函数里共享 `seed[sample_idx]`，
  **保持不变** —— 改种子会让所有既有部署的输出逐位变化，不是这次改动有权做的事；FK 是新来的，
  直接按分支建立的规矩走（独立流）
- 新增 rank-0 日志报扁平向量构成（同 `_fk_fitting_context` 那条的理由）

**部署路径实测**（第三次冒烟 `fk-point-smoke-0929-item34`，走
`PointFlowEvalCallback.joint_rollout` → `generate_samples_from_batch`，即第 3 项改的那条路）：

```
inference flat vector: [vision | action | pointflow | FK], 679200 elements (sample 0)
```

`646272 + 2112 + 28800 + 2016 = 679200` —— 与 FK 侧四段**逐位相加相等**。
附带证明：UniPC 拿 `_get_velocity` 的返回值推进状态，**长度不等会直接报错**，
所以采样跑过去即"拼"与"切"长度自洽。

这次冒烟**跑完 exit 0、0 异常**，8 条 FK eval 结果齐全（四段 arm 与条件 arm 都在），
训练 2 步 ~6.8 s/步，跑到的路径包括：`generate_samples_from_batch` ×6（pointflow joint，
`val_stage_fractions` 3 段 × 2 case）、`sample_pointflow` 的 unipc + euler 对照
（即撤了剥除、加了 `_attach_clean_fk_state` 的那两条 arm）、`sample_fk` 的四段联合 arm。

**⚠️ 三处「剥掉对方模态」—— 三处全部已撤**

| 处 | 位置 | 状态 |
|---|---|---|
| 1 | `_fk_fitting_context` 剥 pointflow + 清 `plan.has_point` | ✅ **已撤**（第 1、2 项一并做的，8 卡冒烟已证四段接上） |
| 2 | `sample_pointflow` / `pointflow_sigma_scan` 剥 fk | ✅ **已撤** —— 新增 `_attach_clean_fk_state()`：这两条 arm 把 FK 按**干净态 σ=0** 交出去，与它们本来对待视频/action 的方式一致（也正因如此它们的数是上限不是部署性能）。**副作用：PointFlow 这两条 arm 的数字会变**（多了 FK 这个上下文），这是 D2 的预期后果 |
| 3 | `_prepare_inference_data` 剥 fk | ✅ **已撤**（第 3 项必须同时撤，见上表第 1 处） |

成因原本是"packed 带着对方 payload 但没供给对应的 `*_noised`"。现在每条路都供给了，
剥除自然不需要 —— 但**顺序不能反**：先撤会立刻崩。

**同批完成的四处小改（2026-09-29）**

| # | 文件 | 改动 | 依据 |
|---|---|---|---|
| 1 | `cosmos3_vfm_network.py` `install_fk` | 补一行对称安装日志（PointFlow 早就有，FK 没有），带上**实际生效的 dtype** | §F.5 那条检查原本查不到任何东西。日志立刻有用：它打出的 bf16/fp32 两种值查证后是 `net`(:469) 与 `net_ema`(:475)，**设计如此** |
| 2 | `omni_mot_model.py` `_get_velocity` | 切 action 段补 `sequence_plans[i].has_action` | 同函数内 velocity 重组(:2851)有、**紧挨着的 sound(:2572)也有**，只有它没有 → 是遗漏。混合 batch 下会把后面段的字节当 action 读 |
| 3 | `pointflow_geometry_test.py` | 夹具补 `encoder.per_point = False` | 它用 `__new__` 绕过 `__init__` 手搭 encoder，`per_point` 是后加的特性、加时没同步夹具 → **自那时起在提交基线上一直红**，与合并无关 |
| 4 | `fk_independent_schedule_test.py` | 第 4 组拆成**互补的两条**：① 四个 `independent_*_schedule` **必须一致**；② FK 的值必须等于 §D4 定的（`False`）。失败信息写明两个方向各自的后果 | 该测试的职责是"发现独立 σ 这个修复被关掉"，而 D4 有意关了它 —— 它红得**正确**。直接翻断言会毁掉 canary。**两个方向都守**的含义：任何**单向翻转**（真危险，会造出 §2/§D4 都没描述过的**混合 σ 调度**）和任何**统一翻转**（是决定、不是 bug）都会红；② 的失败信息明确写"这是 §D4 级别的决定，不是接线修复" |

另有三条**测试卫生**修复：`pointflow_branch_test` / `pointflow_per_point_test` / `pointflow_sequence_test`
补上与 FK 侧同形的 `sys.path` 自卫（`python <文件>.py` 会把脚本所在目录放到 `sys.path[0]`，
那里的 `tokenizers/` 会**遮蔽**已安装的 `tokenizers`，而 `transformers` 要的是真的那份）。

**🎉 首次多机（16 卡 / 2 节点）启动全部通过**（2026-09-29 17:39 起，run `fk-point-101-0`）

这是**合并后的四模态 recipe 第一次真正跑在多机上**。逐条核的结果：

| 检查 | 实测 | 判据 |
|---|---|---|
| 两节点组网 | `world_size: 16 and rank: 0..7` | ✅ 不是 8 |
| 拓扑 | `data_parallel_shard_degree: 8` / `replicate_degree: 2` | ✅ wrapper 覆盖生效 |
| 两个分支 | `PointFlow branch installed` + `FK branch installed` 各 2 条（bf16 / fp32 = net / net_ema） | ✅ |
| 配置 | 四个 `independent_*` 全 False、两个 scale 正确、`joint_attn_implementation: two_way` | ✅ D4 落地 |
| 部署推理路径 | `pointflow_eval/step_0000000` 产物落盘、`PointFlow joint dream canvas`（= `generate_samples_from_batch`，§E 第 3 项那条） | ✅ 在真跑 |
| **flash2 训练路径** | `flash2 (sm80, requires_grad=True, causal=True/False, varlen=True)` | ✅ §F.5 的判据 |
| 第 1 步 | 342 s（**含**第 0 步完整验证；冷启动仍只发生一次） | ✅ |
| 第 2 步起 | **冷热交替 6.53~17.27 s**（前几步：6.54 / 6.53 / 13.30 / 17.27 / 10.05 / 9.02） | ✅ 与**单机 60 步全样本**同分布（中位 13.49、均值 13.07、区间 6.47~20.82）—— **不是多机引入的**，是数据 I/O |
| 异常 | `Traceback` / `OutOfMemory` / `NaN` / `skip_nan_step` 全 0 | ✅ |

三处"和文档不一样但不是故障"，值得记住 —— **第二条是我自己先报错了**：
- **第 1 步 342 s**，而 FK 文档 §6 写的是 90~170 s —— 那是 **FK 单模态**的冷启动。四模态多跑
  PointFlow eval + joint rollout，单机那次也有 297 s。
- 🔴 **步速是冷热交替的，不是一条直线。** 我最初只看了 iteration 2、3 两个样本（都是 6.5 s），
  据此说了"~6.6 s/步、墙钟 37.8 h" —— **错了**。回查单机 60 步**全样本**：
  中位 13.49、均值 13.07、区间 6.47~20.82。
  PointFlow 文档 §6 早写了这条（`8s/20s 冷热交替 = 数据 I/O`），我没用上。
  → **最终值（更长的样本）见 §F.10：11.7 s/步、20000 步 ≈ 65 h。**
  本段当时的 "≈73 h" 是只用 8 卡单机 60 步估的，已被 16 卡 579 步的实测取代。
- **16 卡与 8 卡是同分布，不是同数字**：两边都 6.5~20 抖。每卡 batch 固定 16、
  跨节点梯度 allreduce 只多 ~0.1-0.2 s。与 FK 文档 §5"步速对卡数不敏感"一致。

⚠️ 同时发现的**有损日志**问题见 §F.10 阶段 3 的红字 —— 上面有几条是从 `tail -f`
实时抓到的，事后 `grep` 同一个文件会返回 0。

**canary 的"有牙"验证**（本项目标准，撤掉修复必失败）：临时改配方实测两个方向 ——
① 只翻 `independent_fk_schedule`（造出混合 σ）→ **两条都红**，并把四个值原样打出；
② 四个全翻 `True`（统一翻转）→ 一致性那条放行、§D4 那条红并说明"这是 §D4 级别的决定"。
配方已 `diff -q` 确认逐字节还原。

**全量单测：26 PASS / 0 FAIL**（修前 21 PASS / 5 FAIL）。

最后一条 `pointflow_sequence_test` 是**注意力后端**问题、不是测试卫生问题：它真的跑一层
`PackedAttentionMoT`，而 `two_way_attention`（`attention.py:551`）在本环境**没有 CPU 后端**
（`Could not find a compatible Attention backend for this use case / device`），
而它全程 `device="cpu"`。**pre-existing**，先前被 ImportError 盖住，修了自卫才露出来。

修法：`torch.set_default_device("cuda")` + 无 GPU 时 `@skipUnless` 跳过。
用**默认设备**而不是给 ~40 处张量字面量逐个加 `device=` —— 否则序列在 CUDA、
散落的 `torch.tensor(...)` 字面量在 CPU，会以**与本测试无关的原因**失败。
上 CUDA 后 **4/4 通过**（选到 `natten`：这次临时跑没设 `COSMOS_FLASH2_VARLEN`，
与 FK 文档 §10.1"sm80 上 natten 是唯一 varlen 后端"一致）。

### 下一步（严格按序）

1. ~~`_fk_fitting_context` 接 pointflow~~ ✅ **已完成（2026-09-29）**
2. ~~`_sample_joint` 的 split 加 pointflow 段~~ ✅ **已完成** —— 必须与第 1 项**同一次改**：
   联合向量变长而切分仍按旧布局时，`torch.cat` 对长度是宽容的，会**静默**把 pointflow 的字节当 FK 读
3. ~~`_get_velocity` / `_prepare_inference_data` 改序~~ ✅ **已完成（2026-09-29）**
4. ~~撤剩下两处剥除~~ ✅ **已完成**（与第 3 项同一次改：`_prepare_inference_data` 那处不撤，
   第 3 项加进去的 FK 段会被自己丢掉）
5. ~~**8 卡长跑验证**~~ ✅ **已完成（2026-09-29）**：`MAX_ITER=60`、`FK_VAL_ON_START=true`
   （第 0 步跑完整 eval）、`FK_EVAL_JOINT=1 FK_EVAL_JOINT_ACTION=1 FK_ROLLOUT=0`。
   **exit 0、0 异常**，跑到 iteration 60。8 卡、每卡 batch 16、四模态。
   ⚠️ **步速不要读成"~6.5 s/步"** —— 那是只看头两步得出的（本节初稿正是这么写的，错）。
   60 步**全样本**：**中位 13.49、均值 13.07、区间 6.47~20.82 s**，冷热交替 = 数据 I/O。
   健康样本行见 §F.5。输出：`/data/shichaojian/runs/cosmos/fk-point-smoke-0929-long60`
6. 16 卡提交命令 —— **命令与前置条件见 §F.2 / §F.6**（含"`NODE_RANK` 不能传、兜底靠 pod 名正则、
   首次提交前先跑 `check_multinode_nccl.py`"）。交接单已给出

### 尚未验证
- ~~四模态联合路径从未端到端跑过~~ → **已跑过**：2 步冒烟 ×2 次 + **60 步长跑 ×1 次**，
  四段布局有实证、两个模态的 loss 与 ADE 同现（见 §F.5）。仍**未**验的是**精度** ——
  三次跑的都是从基座起的新 run，FK/PointFlow 分支随机初始化，ratio 贴着 1 无意义；
  视频 loss 被两个 25× 分支拖累多少（§6.3）也还没数。
  另：**"多机机制"已经跑通过，没跑过的是"合并后的四模态 recipe 在多机上"** —— 这两件事
  的风险差一个量级，别混为一谈：
  - **机制**（`_sft_launcher_common.sh` 的拓扑推导、HSDP shard/replicate、pod 名兜底、
    `LD_LIBRARY_PATH` / `expandable_segments` / flash2 varlen、`OUTPUT_ROOT` 竞态）
    **两台单模态线各自在该集群上验证过**，而且共用同一套逻辑 ——
    FK 文档 `WorldAct-cosmos3-edge-droid-sft_mano/docs/fk_multinode_training.md`
    开头原话："两边的启动链路和注入机制**完全一样**，差别只在 recipes"。
    FK 那份还记了两次真实提交的翻车与修法（第一次 `OUTPUT_ROOT` 竞态挂掉、`-10-3`
    在 iter 42 OOM、flash2 的 1747 步稳定性边界）。
  - **这个合并 recipe 自己 —— 已实测（2026-09-29 起，run `fk-point-101-0`）**：
    组网 16 ranks ✅、shard=8 / replicate=2 ✅、两条安装日志 ✅、四个 `independent_*`
    全 False ✅、flash2 双向 `requires_grad=True varlen=True` ✅、0 异常 ✅。
    **显存/步速也有了**：11.7 s/步、ckpt 30 G/个，见 §F.10。
    仍未测的只有「32 卡（4 节点）」这一档。
    ⚠️ 仍然成立的差异：它的 `run_fk_point_101.sh` 与 FK 那份 `run_fk_101.sh`
    **不是同一个 wrapper**（步数算法、守卫、可覆盖项都不同，见 §F 的注）。
  - 本节此前写的"真实多机一次都没跑过"是**不准确**的，已按上两条更正
- ~~通用推理路径发不出 FK~~ → **已发出**（第 3 项）。覆盖到什么程度，说清楚：
  - **四处之间是否一致**：有实证。`_prepare_inference_data` 的拼与 `_get_velocity` 的切，
    总长必须相等 UniPC 才能推进（不等直接报错），实测 `679200` 对得上；第四处
    `generate_samples_from_batch` 的拆用的是同一套尺寸并 `.reshape()`，不一致会当场炸。
  - **四处是否都错成同一种**：**没有**任何断言能发现。四处用同一套 `point_spans` 推导、
    同一份字面顺序，所以"一起错"在代码里是自洽的、跑得通、数字也"合理"。
    唯一能看出来的地方就是那行 rank-0 日志打的顺序 —— **这行日志就是为此存在的**。
    另注意四处尺寸两两不同（646272 / 2112 / 28800 / 2016），所以任何一个**排列**上的手误
    都会让某段的 `.reshape()` 失败，不会静默 —— 能静默的只有"顺序本身被改错"。
- `vision_sigma` 诊断路径与 pointflow 的 σ 不一致（见上，已写进代码注释）
- **既有的对称性缺口（不是本次引入）**：`_get_velocity` 切 action 段只看 `has_noisy_actions`（全局），
  而 `_prepare_inference_data` 拼的时候还看 `sequence_plans[i].has_action`（逐样本）。混合 batch 下两边会错位
- ~~`tools/run_fk_point_101.sh` 的 `checkpoints/` 守卫、`RESUME`、`DRY_RUN` 已演练；真实多机未跑~~
  → **真实多机已跑**（见上「首次多机启动全部通过」）。守卫本身仍未在**多机**上实测触发过
- 冒烟产物目录（可删）：`/data/shichaojian/runs/cosmos/fk-point-smoke-0929-{1241,item34}`
  （`4mod`、`4seg` 已删）

### ⚠️ FK 有三个 arm，D4 之后只有一个可读 —— 读数前必看

`tools/run_fk_point_101.sh` 开的是这样一组（逐个查过源码行号）：

| arm | 开关 | 本次状态 | **D4 下可读?** |
|---|---|---|---|
| `fk_eval` condition | `FK_EVAL_JOINT` 未设 | **在跑** | ❌ **OOD** |
| `fk_eval` joint / joint_action | 同上 | **未开** | — |
| `fk_rollout_joint_action` | `run_fk_point_101.sh:151` `FK_ROLLOUT_JOINT_ACTION=1` | **在跑** | ✅ **部署同款** |

- 目录名由 `fk_rollout.py:211` 决定：`joint_action=True` → `fk_rollout_joint_action/`，
  否则 `fk_rollout/`。所以**顶层那个目录才是 FK 的主指标**，不在 `fk_eval/` 里。
- condition arm 在 D4 下 OOD 的理由见 `fk_independent_schedule_test.py:140-145`：全模态共享一个 σ
  时，"干净 video + 带噪 FK"这个组合**训练时从未出现**。所以它是个 ceiling，不是可部署性能。

**实测（本 run `fk-point-101-0`，step 0 = 基座）：**

| 指标 | episode_0033 | episode_0059 |
|---|---|---|
| `all_ade_mm` / zero | 158.39 / 824.40 | 179.40 / 820.81 |
| **`ratio_to_zero`** | **0.192** | **0.219** |

下一次落盘在 **step 1000**（`FK_ROLLOUT_EVERY=2` 数的是验证事件，而 `validation_iter=500`
→ 事件落在步 0/500/1000…，rollout 取偶数次 → **步 0 / 1000 / 2000**）。

**🚫 不要拿 condition arm 跨 σ 设置比。** 记一次已犯过的错：曾用本 run 的 condition arm
（0.486 / 3.207 / 0.485 / 2.494，均值 1.67）去比 `fk-101-16gpu-flash2`（0.099 / 1.031 /
0.166 / 0.673，均值 0.49），差点写成"合并把 FK 弄坏了 3 倍"。两个 run 的 bs(16)、lr(2e-5)、
`fk_displacement_scale`、zero 值（逐字节相同）、窗口**全都一样**，所以数字**同底** ——
但那个 run 是 `independent_fk_schedule=true`（§2 的修复），它的 condition arm 在分布内，
而本 run 的是 OOD。**A 的主场比 B 的客场，A 赢是构造出来的**，这个差距无法归因于合并。
另：全集群 20 个有 `fk_eval` 的 run，`fk_eval` 的 joint arm **一次都没跑出来过**，
所以也不存在 joint-vs-joint 的对照。

**⚠️ condition arm 会随训练"掉头向上" —— 看 fk_eval 的人一定会误判成 bug。** 实测轨迹：

| step | 0 | 500 | 1000 | 1500 | 2000 | 2500 |
|---|---|---|---|---|---|---|
| `fk-point-101-0`（D4） | 4.506 | **1.668** | 2.356 | 2.771 | 3.001 | 3.158 ↑ |
| `fk-101-16gpu-flash2`（`ind_fk=true`） | 4.514 | 0.492 | 0.298 | 0.233 | 0.183 | 0.168 ↓ |

两 run 的 **step 0 几乎相同**（4.514 / 4.506，同一基座），bs/lr/scale 也一样，之后完全分道。同期部署 arm
在改善（rollout `ratio_to_zero` 0.192→0.038），`train/fk_loss` 1.235→0.075。所以**不是接线问题**。

**但不要把它归因成"σ 调度已被证实"** —— 已试过一次证伪而**未通过**：点云的 condition arm 在 D4 下
同样 OOD，实测只呈现**很弱的** U 形（2.347→**0.367**@1500→0.402，回升 10%），而 FK 是 1.67→3.16
（回升 89%）。方向一致、量级差一个数量级。**结论：接线的正确性有证据；FK condition arm 退化的
成因未定，σ 只是部分解释。** 要定因需要一个 `independent_fk_schedule=true` 的四模态对照（不存在）。

---

## §F 训练启动与集群迁移（四模态专属运维篇）

> 环境/数据/worktree 的**通用**说明见同目录的 [`fk_new_cluster_setup.md`](./fk_new_cluster_setup.md)；
> **开箱使用**见 [`base_quickstart.md`](./base_quickstart.md)；
> **多机训练的完整说明**见 [`base_multinode_training.md`](./base_multinode_training.md)。
> 本节只写**四模态（fk+point+video+action）这一条启动链**特有的东西，
> 以及**为什么这么定**（决策记录）。

### F.0 硬件要求（用户 2026-09-29 明确）

> 「现在这个开发机上是 **8 张 A800**，用于给你测试。**测试都通过后 我会放到 16 卡上去跑**。」

| 环境 | 规模 | 用途 |
|---|---|---|
| **开发机**（本节点） | **8× A800-SXM4-80GB**，全 NV8 互联 | **给我做验证**。该测就测，不要再以"没有 GPU"为由跳过 |
| **生产（目标）** | **16 卡 = 2 节点 × 8** | 用户提交，我负责把命令调好交出去 |

> ⚠️ 旧记忆 `dev-node-no-gpu.md` 说本节点没有 GPU —— **已过期并删除**，现为 `dev-node-gpu.md`。

### F.1 启动链

```
tools/run_fk_point_101.sh                                       # 自包含 wrapper：守卫/HSDP/步数预算/dry-run
  └─ examples/launch_sft_action_policy_fk_point_singlerighthand_edge.sh   # 两个模态的 gate + A800 环境 + 输入检查
       └─ examples/_sft_launcher_common.sh                      # torchrun 参数组装（多机拓扑在这里）
            └─ torchrun -m cosmos_framework.scripts.train --sft-toml=...
```

### F.2 提交命令

**16 卡（2 节点）—— 每个节点执行完全相同的一条，不要传 `NODE_RANK`：**

```bash
OUTPUT_ROOT=/data/shichaojian/runs/cosmos/fk-point-101-<日期> NNODES=2 \
  bash /mnt/afs/WorldAct-cosmos3-edge-droid-sft_base/tools/run_fk_point_101.sh
```

**8 卡本地验证（我做的那次）：**

```bash
# 先看计划不启动
OUTPUT_ROOT=/data/shichaojian/runs/cosmos/fk-point-dryrun NNODES=1 DRY_RUN=1 \
  bash tools/run_fk_point_101.sh
# 再跑短程
MAX_ITER=60 OUTPUT_ROOT=/data/shichaojian/runs/cosmos/fk-point-smoke-<日期> NNODES=1 \
  bash tools/run_fk_point_101.sh
```

- `OUTPUT_ROOT` **必填且每次换新**；续跑加 `RESUME=1`（同名重跑）。
- 守卫查的是 **`checkpoints/`**，不是目录本身 —— 目录存在性在多机下是竞态（每个节点并发跑，
  `_sft_launcher_common.sh` 的 `mkdir -p logs` 会让 master 抢先建出目录）。
- 可覆盖：`NPROC_PER_NODE`、`PER_RANK_BATCH`、`MAX_ITER`，以及所有 `POINTFLOW_*` / `FK_*` / `SINGLERIGHTHAND_*`。

### F.3 并行拓扑（自动推导，shard × replicate × CP == WORLD_SIZE）

wrapper 推导 **shard = 每节点卡数、replicate = 节点数**：节点内 FSDP 走 NVLink（每层一次 all-gather），
节点间只做梯度 allreduce。**不要把 shard 设成跨节点**（那会把每层 all-gather 推上 IB）。

| 规模 | shard | replicate | 每卡 batch | 全局 batch |
|---|---|---|---|---|
| 8 卡单机 | 8 | 1 | 16 | 128 |
| **16 卡（2 节点）** | **8** | **2** | **16** | **256** |

> recipe 的 TOML 里 `data_parallel_shard_degree = 1`（单机不分片）—— **16 卡会不满足不变式**，
> 必须由 wrapper 通过 `EXTRA_TAIL_OVERRIDES` 覆盖。wrapper 已做。

### F.4 A800 / cu130 环境（都已在 launcher 里，列出以便排障）

| 变量 | 值 | 为什么 |
|---|---|---|
| `LD_LIBRARY_PATH` | **清空** | NGC 容器的 CUDA 13.2 `libcublasLt`(13.4.0.1) 遮蔽 venv wheel(13.1.0.3)，症状是**第一次 eval** 报 `CUBLAS_STATUS_NOT_INITIALIZED`。逃生口 `KEEP_LD_LIBRARY_PATH=1` |
| `PYTORCH_ALLOC_CONF` | `expandable_segments:True` | 显存碎片（曾出现 `7.07 GiB reserved but unallocated` 而 OOM） |
| `COSMOS_FLASH2_VARLEN` | `1` | 上游禁用 flash2 varlen；sm80 上不开就只剩 natten，**步速差 1.85×**（4.30 vs 7.95 s/步 @ batch16） |

### F.5 起后验证清单

控制台开头（wrapper 回显）：
- `gpus: 8 per node x 2 node(s) = 16 ranks`
- `topology : shard=8 replicate=2`（数字须符合 F.3 表）
- `[topology-env] ...` 平台注入的实际值；兜底生效时有一行 `>>> topology fallback: HOSTNAME=... -> NODE_RANK=...`

训练日志（`<OUTPUT_ROOT>/logs/action_policy_fk_point_singlerighthand_edge_sft.log`）：
- `RankPartitionedDataLoader allocation (16 GPUs)` —— 组网成功
- **两条安装日志都在**：
  `PointFlow branch installed: token_mode=cluster, stage=3, ...` 与
  `FK branch installed: keypoints=21, include_anchor=True, index_dim=64, hidden_dim=2048, dtype=...`

  > ⚠️ **各出现 2 次，不是 8 次**（60 步长跑实测）。`build_net` 每个进程跑两遍：
  > `self.net`（`dtype=self.precision` → **bfloat16**，`omni_mot_model.py:469`）与
  > `self.net_ema`（**float32**，`:475`）。**看到 bf16 / fp32 各一条是对的**，
  > 不是 rank 之间不一致。（同理 `OmniMoTModel: config` 只出现 1 次也不是丢日志 ——
  > 训练日志文件只收 rank 0 的常规行；带 `[RANK n]` 前缀的是 `iter_speed` 那条，
  > 每个 rank 各一条。**两条日志共用 `Iteration N:` 前缀，别混淆**。）
- loss dict **同时**含 `flow_matching_loss_pointflow` 与 `flow_matching_loss_fk`，两者都在下降。
  健康的指标行长这样（`callback.py:518`，每 `logging_iter`=10 步一条、rank 0）：
  ```
  Iteration 60: Total Loss: 14.0387 | Video Loss: 0.2001 | Action Loss: 0.9853 \
    | PointFlow Loss: 1.1963 | PointFlow ADE: 102.23mm (zero 88.48mm) \
    | FK Loss: 0.9888 | FK ADE: 111.72mm (zero 103.05mm)
  ```
  两个模态的 loss 与 ADE **都在这一行里**，一眼可查；ADE 在训练早期贴着 zero 是正常的
  （分支刚初始化）
- 第 1 步很慢（**16 卡四模态实测 342 s** —— 含第 0 步完整验证；inductor 编译 + cuBLAS
  autotune + dataloader 预热），**只发生一次**。
  （FK 文档 §6 写的 90~170 s 是 **FK 单模态**的冷启动，四模态多跑 PointFlow eval +
  joint rollout，单机那次也有 297 s。）
- 第 2 步起**不是一条稳定直线**，而是**冷热交替**：
  **中位 13.49 s、均值 13.07 s、区间 6.47~20.82 s**（单机 60 步全样本）。
  16 卡与 8 卡是**同分布、不同数字**（跨节点梯度 allreduce 只多 ~0.1~0.2 s）。
  **量步速请用时间戳差**（见 §F.10），或 `iter_speed.py:75` 那条**窗口平均**：
  ```bash
  grep -a "iter_speed .* seconds per iteration" <log> | tail -5
  ```
  > ⚠️ **不要用 `Hit counter` 那种单点** —— 单点受冷热交替影响，本节初稿就是被两个
  > 6.5 s 的点骗了（6.6 s/步 → "37.8 h"，真值 ≈65 h）。
  > ⚠️ **也不要拿 4.3 s/步 来比** —— 那是 **FK 单模态**的数
  > （`/data/shichaojian/runs/cosmos/fk-101-16gpu-flash2`，16 卡，实测 4.27~4.28）。
  > 四模态多出的是 PointFlow 的 Sonata/PTv3 编码器成本，不是故障信号。
  > 判断 flash2 有没有生效要看下面那条 grep，**不要看步速**
- **每行日志重复 N 次是正常的**（N = rank 数，每个 rank 各写一条）

**必做的一行检查**（确认 flash2 真的生效）：
```bash
grep -oE "Attention backend selected: [a-z0-9]+ \(sm80, requires_grad=True[^)]*\)" \
  <OUTPUT_ROOT>/logs/action_policy_fk_point_singlerighthand_edge_sft.log | sort -u
```
必须出现 `flash2 … requires_grad=True … varlen=True`；仍是 natten 就停。

**组网失败的判别**：日志里 `world_size 8` 而不是 16，或同一 step 被打印两遍**且内容对不上**
（各自的 loss/迭代号不一致）→ 两节点没组网、各自单机跑了。**杀掉重交**，不要让它跑下去
（共享 `OUTPUT_ROOT` 会在 save 时互相写坏）。

### F.6 多机最大的风险点

> 📖 **权威出处在两份单模态文档里，本节不重抄**（重抄会各自漂移）：
> - FK：`WorldAct-cosmos3-edge-droid-sft_mano/docs/fk_multinode_training.md`
>   —— §2 注入机制与 pod 名推导、§4 `OUTPUT_ROOT` 竞态（含**真实翻车日志**）、
>   §6 起后验证（含抄自真实 run 的日志行）、§7 排障表、§8 A800 vs H200、§9 三个环境坑、
>   §10 flash2 varlen（1.85×、验证 grep、**"1747 步不足以证明长程稳定"的风险边界**）
> - PointFlow：`WorldAct-cosmos3-edge-droid-sft-pointflow/docs/pointflow_multinode_training.md`
>   —— 同题、同机制
>
> 两者的启动链路与注入机制**完全一样**（同一份 `_sft_launcher_common.sh`），差别只在 recipes。
> ⚠️ **但那两份的前提是单模态 recipe**；本节的四模态 wrapper 是**另一支**
> （`run_fk_point_101.sh` ≠ `run_fk_101.sh`：步数算法、守卫、可覆盖项都不同），
> 所以**机制可以照抄，规模相关的数不能**（显存余量、步速、max_iter）。

SenseCore 注入 `NNODES` / `MASTER_ADDR` / `MASTER_PORT` / `RANK` / `WORLD_SIZE`(=节点数)，
**不注入 `NODE_RANK`**。launcher 靠 **pod 名正则** `^(.+)-(master|worker)-([0-9]+)$` 推导它。
若 `HOSTNAME` 带 `.service` 之类后缀 → 正则不匹配 → `--node_rank` 被省略，
而 torchrun 的 `--node_rank` **默认是 0** → **每个节点都以为自己是 rank 0**、各自去当 master
→ 全体卡在 rendezvous 直到超时。**不是响亮报错。**

**首次提交前先跑连通性测试**（每个节点各跑一次，`NODE_RANK` 各不同；只有这个脚本要手填）：

```bash
NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0 ip> PYTHONPATH=. \
  /mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python tools/check_multinode_nccl.py
```
它依次验：握手 → 各节点 world size / 卡数是否一致 → 真实 NCCL all-reduce → 带宽。全对才打 `PASS`。

### F.7 数据路径（2026-09-29 实测全部存在）

| 用途 | 路径 |
|---|---|
| 基底 ckpt | `/data/shichaojian/models/cosmos3-edge-droid-dcp` |
| 原始视频（101 集） | `/data/shichaojian/raw_data/singlerighthand_sandwich_100` |
| 视频/VAE 缓存 | `/data/shichaojian/datasets/singlerighthand-sandwich-100-cosmos-cache`（含 `vae_window_latents/`） |
| **FK 标注** | `/data/shichaojian/raw_data/sandwich_fk21/<ep>/annotations/wuji_fk21.npz`（101/101） |
| Sonata/PTv3 | `/data/shichaojian/checkpoints/ptv3/sonata_small.pth` |
| **PointFlow manifest** | `/mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow/pointflow_outputs/sandwich_924_20260928/manifest.json`（101/101 路径可达） |
| 点云数据本体 | `/data/shichaojian/pf_out/9.24/sandwich/labeled/` |
| **venv** | `/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv`（`_base` 没有自己的；必须 `PYTHONPATH=.`） |

⚠️ 旧集群的 `/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/...` **已失效**。
recipe 的 manifest 默认值与两个 launcher 的 `SIBLING_VENV` 已修正；**但 `launch_pointflow_dropper101.sh`
/ `launch_pointflow_motion_scan.sh` / `launch_sft_action_policy_singlerighthand_nano.sh` 里还残留着**
（不在本任务关键路径，未改）。

### F.8 scale 纪律（这次的硬约束）

`pointflow_displacement_scale = 0.075255` 是**这套选择规则的属性**，不是常量：
`manifest=sandwich_924_20260928` + `select_top_n=300` + `min_voxel_members=0` + `select_min_valid_steps=0`，
池化 std 于 n=22,595,703 个元素。

**换 manifest 或换任一选择旋钮 → 必须重测**，否则整个预测被均匀缩放错，而且**没有任何形状错误**。
`tools/run_fk_point_101.sh` 里有守卫，值与它期望的不一致就拒绝启动。

同理 `fk_displacement_scale = 0.083745`（101 回合值）。

### F.9 未移植（有意）

| 文件 | 为什么不拷 |
|---|---|
| `tools/build_pointflow_window_cache.py` | 依赖 `cosmos_framework/data/pointflow_window_cache.py`，**该模块在 `_base` 里不存在** |
| `examples/launch_pointflow_sandwich101.sh` | 默认值依赖 `select_region_quotas` / `window_cache_root` / phantom-guard 接线，**这些在 `_base` 里都没有** |

即：`_base` 缺的是 PointFlow 线**合并之后**才加的一整套特性。本方案改用 `_base` 能复现的
`top300` 选择，因此 scale 是 0.075255 而非 0.0528。

---

### F.10 16 卡提交：可照做的执行清单

> 机制部分的**权威说明**在两份单模态文档里（指路见 §F.6）；本节只把它们编排成
> **一次提交的操作顺序**，并把四模态特有的数填进去。

**本次的目标规模（用户 2026-09-29 定）**

| 量 | 值 | 怎么来的 |
|---|---|---|
| 步数 | **20000** | 显式给 `MAX_ITER=20000` |
| 全局 batch | 256 | 每卡 16 × 16 卡 |
| 总样本 | **5,120,000** | 20000 × 256 |
| 墙钟 | **≈ 65 h（2.7 天）** | 20000 × **11.7 s/步**。四次实测逐次收敛：单机 49 步均值 13.07 → 16 卡前 9 步 10.96 → iteration ~280 窗口平均 10.2~11.5 → **本 run 步 100→679 用时间戳差算：实耗 1.88 h / 579 步 = 11.72 s/步**（同时段 52→679 得 11.78，两个窗口一致）。这个算法把 eval/checkpoint 的停顿自动摊进去了，比 `iter_speed` 的单窗口均值更可信。⚠️ **不是 6.6 s**，也别拿单点 `Hit counter` 估 |
| 剩余 | **≈ 53 h** | 从步 **3667**（2026-09-30 05:22 实测）起算，× 11.7 s/步。首次 ckpt（步 500）实测 **30 G**，与下行一致 |
| 进度参照 | 步 679 @ 02:23 → 步 3667 @ 05:22（**实耗 8h40m / 2988 步 = 11.6 s/步**） | 这个算法把 eval/ckpt 停顿摊进去了 |
| **checkpoint 占用** | **≈ 1.2 TB** | `save_iter=500` → 40 个 × **实测 30 G/个** |

**阶段 0 · 准备**

```bash
# 0a. 每节点都跑，确认 8 卡
nvidia-smi -L | wc -l                      # → 8

# 0b. 确认输出卷放得下 1.2 TB（见下面的红字）
df -h /data/shichaojian/runs
```

- 选一个**全新的** `OUTPUT_ROOT`（带日期）：`/data/shichaojian/runs/cosmos/fk-point-101-<YYYYMMDD>`。
  多机下**显式给是唯一安全形态** —— wrapper 的"重名就加后缀"是**各节点独立状态**，
  node0 建完目录 node1 会选 `-2`、node2 选 `-3`，**三边各写一个 run、各拿 1/3 数据**，
  而 loss 曲线看起来像训练 bug。
- 记下预期：**16 ranks / shard=8 / replicate=2 / 每卡 batch 16 / 全局 256**。

> 🔴 **磁盘：没有任何自动清理，占用是线性的。**
> toml 里 `save_iter` 的注释曾写着 *"old checkpoints are deleted as the run goes"* ——
> **与代码不符，已改掉**。实测：`cosmos_framework/utils/checkpointer.py` 里删除类调用
> **0 条**；全仓 `rmtree` 只出现在训练路径之外（VAE 缓存 / HF snapshot / export /
> inference fixture）；三个启动器都不删；`fk-101-16gpu-flash2` 跑完实打实留着
> **5 个完整 ckpt**（各 30 G，`model/optim/scheduler/trainer` 齐全）。
> **20000 步 = 40 个 × 30 G ≈ 1.2 TB**，提交前必须按这个数留空间。
> 想省：要么调大 `save_iter`（用崩溃可恢复的粒度换磁盘），要么自己定期删旧的 ——
> 但**删过的 iteration 就 resume 不回去了**（裁剪过的 ckpt 会报 `metadata is None`，
> 而且它的真实原因被吞掉），所以至少留最近一两个。

**阶段 1 · 预检**（跳过它的代价是卡到 rendezvous 超时 30 min）

```bash
# 1a. 打计划、不启动（任一节点）
OUTPUT_ROOT=/data/shichaojian/runs/cosmos/fk-point-101-<date> NNODES=2 MAX_ITER=20000 DRY_RUN=1 \
  bash /mnt/afs/WorldAct-cosmos3-edge-droid-sft_base/tools/run_fk_point_101.sh
#   核对三处：
#     gpus: 8 per node x 2 node(s) = 16 ranks
#     topology: shard=8 replicate=2
#     overrides: ... trainer.max_iter=20000 scheduler.cycle_lengths=[20000] ...
#   ⚠️ 那两个 20000 必须同时出现且相等 —— 只来一个的话调度器会在
#      find_in_interval 返回 None 后崩（cycle_lengths < max_iter 即触发）

# 1b. 连通性测试 —— 每个节点各跑一次，NODE_RANK 取 0 / 1
NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0 ip> PYTHONPATH=. \
  /mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python tools/check_multinode_nccl.py
```

⚠️ **只有 1b 这个脚本要手填 `NODE_RANK`**（它直接开 `init_process_group`，不走 pod 名兜底）；
第二个节点换成 `NODE_RANK=1`。`MASTER_ADDR` 必须是**节点 0 上别的节点能 ping 到的 IP**，
没设它会直接报错退出。必须打 `PASS`。

> 💡 **顺手做一个更有价值的变体**：加 `KEEP_MASTER_PORT=1` 会让它改用**训练要用的那个端口
> （50012）**而不是默认的 50013 —— 这样连通性测试**同时证明那个端口是空的**。
> 提交前占端口是那种"跑到 rendezvous 才炸"的失败，这一步能提前排掉：
>
> ```bash
> NNODES=2 NODE_RANK=0 MASTER_ADDR=<node0 ip> KEEP_MASTER_PORT=1 PYTHONPATH=. \
>   /mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python tools/check_multinode_nccl.py
> ```

**阶段 2 · 提交**

```bash
OUTPUT_ROOT=/data/shichaojian/runs/cosmos/fk-point-101-<YYYYMMDD> NNODES=2 MAX_ITER=20000 \
  bash /mnt/afs/WorldAct-cosmos3-edge-droid-sft_base/tools/run_fk_point_101.sh
```

**两个节点执行完全相同的这一条。**

- **不要传 `NODE_RANK`** —— 传了 pod 名兜底就不触发（它要求变量为空），
  **每个节点都以为自己是 rank 0**，全体卡在 rendezvous 直到超时。
- `MAX_ITER` 必须给：不给就用 toml 的 `10000`（步数减半、总样本 256 万而非 512 万）。
- 续跑：同目录 + 加 `RESUME=1`（`MAX_ITER` 照旧给）。

**阶段 3 · 起后 5 分钟内逐条核，任一条不对就杀掉**

> 🔴 **先读这条：多机日志在启动那一波会丢行，grep 返回 0 不代表没发生。**
>
> 复查后的准确画像（`fk-point-101-0`，2026-09-30 05:22 采；**文件本身没有被截断** ——
> 12248 行、时间跨度 `09-29 17` → `09-30 05` 连续、**0 个 NUL 字节**）：
>
> | 行 | grep 到的条数 | 实情 |
> |---|---|---|
> | `RankPartitionedDataLoader: world_size: 16 and rank: N` | **16 / 16** ✅ | 稳态行，完整 |
> | `RankPartitionedDataLoader allocation` | **16 / 16** ✅ | 同上 |
> | `get_config_module` | **8** | 16 个 rank 只留 8 条 |
> | `FK branch installed` | **0** ⚠️ | 启动时**亲眼读到过** |
> | `PointFlow branch installed` | **0** ⚠️ | 同上 |
> | `Attention backend selected` | **0** ⚠️ | 同上 —— 这是 flash2 的判据行 |
>
> 机制（**推断，未直接验证**）：16 个进程、跨 2 个节点、各自 append 到同一个网络文件系统
> （`/data` = `inspurfs`）上的文件，而 **`O_APPEND` 的原子性只在本地文件系统成立** ——
> NFS 类 FS 上多个客户端并发 append 会互相覆盖。丢失**集中在启动那一波**（16 个进程在几秒内
> 同时打开同一个文件），稳态（每步的 `iter_speed`、每 rank 一次的 dataloader 行）不受影响。
> 本段初稿写成"整个日志有损、中段有 NUL 空洞"——**过强了**，那是读到写中文件的中间态；
> 文件静止后复查是干净的。
>
> **分类对待：**
> - **启动类检查**（分支装没装上、attention 后端）→ **用平台控制台**（每个 rank 各自
>   stdout，不会互相覆盖）**，或启动时就 `tail -f`**。事后 grep 不可信。
> - **稳态类检查**（`world_size` 是不是 16）→ **事后 grep 可靠**，那些行活下来了。
> - **grep 返回 0 一律读作"未知"，不是"不存在"。**

控制台：

- [ ] `gpus: 8 per node x 2 node(s) = 16 ranks`
- [ ] `topology : shard=8 replicate=2`
- [ ] `[topology-env] ...`；兜底生效时应有一行 `>>> topology fallback: HOSTNAME=... -> NODE_RANK=...`

训练日志 `<OUTPUT_ROOT>/logs/action_policy_fk_point_singlerighthand_edge_sft.log`：

- [ ] **`RankPartitionedDataLoader: world_size: 16 and rank: <n>`** —— 比下面那条更早、
      更直接；出现 `world_size: 8` 就是没组网
- [ ] `RankPartitionedDataLoader allocation (16 GPUs)` —— 组网成功
- [ ] 两条安装日志都在（**各 2 条**：`net` bf16 + `net_ema` fp32）⚠️ 见上面的有损日志警告
- [ ] 指标行**同时**含 `PointFlow Loss` 与 `FK Loss`
- [ ] **flash2 grep —— 唯一可靠的判据，不要看步速**：

```bash
grep -oE "Attention backend selected: [a-z0-9]+ \(sm80, requires_grad=True[^)]*\)" \
  <OUTPUT_ROOT>/logs/action_policy_fk_point_singlerighthand_edge_sft.log | sort -u
```

必须出现 `flash2 … requires_grad=True … varlen=True`；若是 `natten` **立刻停**重交
（1.85× 步速差，且漏这个前缀已经被踩过一次）。

**阶段 4 · 前 30 分钟盯**

- 第 1 步慢（**342 s**，冷启动 + 第 0 步完整验证）**只发生一次**。
- 🔴 **第 2 步起步速是「冷热交替」的，不是一条直线 —— 这是正常的，不是故障。**
  实测（单机 60 步全样本，去掉冷启动）：**中位 13.49 s、均值 13.07 s、区间 6.47~20.82 s**。

  ```
  6.47 6.55 12.54 12.33 14.16 14.47 7.86 16.77 11.31 12.28 13.49 11.58 11.38 20.35
  6.59 6.60 15.13 16.75 13.97 16.81 17.84 6.47 12.30 10.90 20.82 12.37 10.16 17.18
  12.73 13.49 13.41 11.64 14.78 11.71 14.31 8.22 17.24 8.10 14.93 11.37 18.97 9.59
  17.73 10.70 15.34 17.82 13.97 13.76 15.23
  ```

  > PointFlow 文档 §6 早就记了这条：**`step 时间 8s/20s 冷热交替` = 数据 I/O，不是网络。**
  > 16 卡实测同样如此（6.53 / 6.54 / 9.02 / 10.05 / 13.30 / 17.27）—— **与单机同分布，
  > 不是多机引入的**。
  >
  > ⚠️ **别拿头两步（6.5 s）去估墙钟** —— 那正是本节初稿犯的错：
  > 6.6 s/步 → "37 h"，而真实均值 13.07 → **73 h**，差 2 倍。
  > 也别拿 **4.3 s/步** 来比 —— 那是 **FK 单模态**的数（单模态数据量小，I/O 压力不一样）。
  >
  > **量步速要用对行**：`iter_speed.py:75` 那条
  > ```
  > 281 : iter_speed 11.50 seconds per iteration | Loss: 2.2867
  > ```
  > 是**窗口平均**，比 `[RANK n] … Hit counter … | Time:` 那种单点可靠得多
  > （单点受冷热交替影响，我最初就是被两个 6.5 s 的点骗了）。
  > ```bash
  > grep -a "iter_speed .* seconds per iteration" <log> | tail -5
  > ```
- NaN、或 `skip_nan_step` **频繁**触发 → 提交命令前加 `COSMOS_FLASH2_VARLEN=0` 回退重交。
  （FK 文档 §10.4 记了这条禁令的风险边界：上游禁用 flash2 varlen 的理由是 "due to instability"，
  实测只贴了 1747 步，**不足以证明长程稳定**。）
- 日志出现 `world_size 8` 而不是 16，或同一 step 打印两遍**且内容对不上**
  → **没组网、各自单机跑了，立刻杀**。共享 `OUTPUT_ROOT` 会在 save 时互相写坏。

**四模态特有、两份单模态文档覆盖不到的**

- ~~这个 recipe 从来没在多机上跑过~~ → **已跑（2026-09-29 17:39 起，run `fk-point-101-0`）**，
  16 卡各项全通过（见 §E「首次多机启动全部通过」），显存/步速/ckpt 占用也都有实测
  （本表 + §F.5）。**仍未测的只有 32 卡（4 节点）这一档。**
- 完整的四模态多机说明（注入机制、拓扑、起后验证、排障、flash2、环境坑）已单独成文：
  **[`base_multinode_training.md`](./base_multinode_training.md)**；
  单机开箱见 **[`base_quickstart.md`](./base_quickstart.md)**。本节与 §F 其余部分是更细的
  决策记录与执行状态，两者互补。
- **`max_iter` 两边的算法不同**：FK 的 `run_fk_101.sh` 按样本预算推（文档 §5），
  四模态的 `run_fk_point_101.sh` **不做推导**，直接用 toml 的 `10000`。
  后果：**8 卡跑同一份 toml 也是 10000 步（=128 万样本），16 卡才是 256 万**。
  想要"每个实验固定样本量"，就得显式传 `MAX_ITER`。
