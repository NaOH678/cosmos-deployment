# pointflow_fit005 训练分析(2026-09-14)

**日期**:2026-09-14
**状态**:**已结案 —— 根因不在本文追的方向上。** 见
**[`pointflow_displacement_scale_20260914.md`](pointflow_displacement_scale_20260914.md)**:
PointFlow 的 clean 样本(点位移 std 0.11 m)没有归一化就加了单位方差噪声(1.0 m),
信噪比 1:9,于是 flow-matching loss 的 90% 在度量噪声而不是位移。**修法是改一个配置值。**

本文保留为**诊断过程记录**:两个采样侧假设的证伪(§6、§8)依然有效且有价值;
§9 的误差形状、§10 的注意力矩阵、§11 的消融方法都可复用。
**§5 的走向(位置编码 / action 桥)作废** —— 在分支能学会运动之前,任何关于它读不读 action
的测量都没有意义。

**范围**:`runs/cosmos/pointflow_fit005` 的诊断(最终读到 step 8300)
**与其它文档的关系**:
- **`pointflow_displacement_scale_20260914.md`** —— **本文的结论**。先读它。
- `pointflow_motion_selection_20260914.md` —— 这次跑用的选点方案(0.05)。**§12.1 的 `valid_fraction` 与它直接相关。**
- `pointflow_alignment_audit_20260913.md` —— 位置编码核查。**与根因无关**。
- `pointflow_bugfix_log_20260912.md` —— 更早的缺陷修复。

---

## 0. 一句话

**分支确实在学**(单步去噪误差已低于 zero 基线 30%),**但采样出的 32 步轨迹从来没有赢过"预测不动"** —— 方向对(余弦 0.83),**幅度大了 4 倍**,预测被一块与 GT 无关的残留噪声主导。

> **本文反复出现的几个词**(模型的任务是:看画面,预测盘子里那簇点接下来 32 帧怎么动):
>
> - **`zero` 基线** —— 把预测换成"一动不动"时的误差。**只取决于数据**,是常数。
> - **`ADE/zero`** —— 预测轨迹的误差 ÷ 上面那个基线。**< 1.0 = 比不动强。** 本文的主判据。
> - **`单步 ADE`** —— 只看**一步**的去噪估计(训练日志里那个)。比 32 步采样**容易得多**,
>   所以它好不代表轨迹好 —— 这正是本文的核心矛盾。
> - **余弦** —— 每个点预测的**方向**对不对,+1 最好。
>
> 完整版(含 `pred/gt`、`共模占比` 等)在
> [`pointflow_displacement_scale_20260914.md`](pointflow_displacement_scale_20260914.md) 的开头。

两个采样侧假设都已实施并复测,**都被证伪**:σ 调度漏 `shift`(§6)、求解器/步数(§8)。
**采样器本身是好的** —— 它收敛、方向正确、对步数响应合理,最多只能解释所需的 5%。

> ### ⚠️ 结论(2026-09-14 晚补)
>
> 当时把线索指向"模型侧有一条结构性的东西没接上",并去做动作消融(§11)。
> **那条线追错了。**
>
> 真正的根因是 **PointFlow 的 clean 样本没有归一化**:位移 std 0.11 m,噪声 std 1.0 m,
> **信噪比 1:9**。loss 因此 90% 在度量"噪声复现得准不准",只有 10% 在度量"点动得对不对"。
> 采样输出的残差(0.37)比信号(0.11)大 3~7 倍 —— 在 `comparison.png` 里就是那把
> **从点簇向外辐射的直线扇子**。
>
> **完整证据链与修法见 [`pointflow_displacement_scale_20260914.md`](pointflow_displacement_scale_20260914.md)。**
>
> **已重训验证(该文档 §9)**:根因确认 —— 每个 eval step 都远好于本文这个 run
> (ADE/zero 12.0 → 1.34 @step 500),画面上的"直线扇子"消失。
> **但没达到 `ADE/zero < 1.0`**(最好 1.343),且有**一个与 scale 无关的遗留退化**——
> 两个 run 共有、train/val 同步(所以不是过拟合),读作训练动态问题。
>
> 本文以下各节作为过程记录保留。**§5 的走向作废**;§6/§8 的证伪结论有效。
> **§9 的误差形状现在有了新读法**:一步估计好、采样轨迹差,这个"缺口"在 scale 修好后**依然存在**,
> 是遗留问题的一部分。

---

## 1. 这次跑的是什么

```
run dir    runs/cosmos/pointflow_fit005/cosmos3_action/action_sft/action_policy_singlerighthand_edge
step       7320
选点       POINTFLOW_SELECT_MOTION_FRACTION=0.05  → 每窗口 410 点
logging    logging_iter = 10
```

关键配置(从 `config.yaml` 读):

```
optimizer.lr                      = 2e-05
optimizer.lr_multipliers          = {action2llm: 5.0, action_modality_embed: 5.0,
                                     llm2action: 5.0, pointflow_branch.codec: 25.0}
rf.loss_scale                     = 10.0      (视频)
rf.action_loss_weight             = 10.0
rf.pointflow_loss_weight          = 1.0       ← 分支只占 Total Loss 的 0.8%
rf.pointflow_displacement_scale   = 1.0       ← ★ 根因。位移 std 0.113 m 却加了 1.0 的噪声
rf.shift                          = {'256': 3, '480': 5, '720': 10}
rf.train_time_video_distribution  = waver
```

---

## 2. 训练侧:**在收敛**(但需要看平均,不要看单条)

```
窗口            PointFlow Loss   Video Loss   单步ADE   zero   ADE/zero
0-500               0.2370        0.1764      434.1   89.6    4.85
500-1500            0.0245        0.1516       96.5   90.8    1.06
1500-3000           0.0185        0.1397       80.0   90.1    0.89
3000-5000           0.0148        0.1247       69.8   88.9    0.78
5000-7300           0.0115        0.1142       63.1   89.9    0.70
```

- PointFlow Loss 降了 **20 倍**,单调
- 单步 ADE 从 **434mm → 63mm**,已经**比 zero 基线好 30%**
- Video Loss 同步改善(0.176 → 0.114),说明分支没有拖坏主干

### ⚠️ 一个记录下来的误判

第一次读日志时我每 50 行抽一条,看到 `PointFlow Loss` 在 0.003–0.06 之间跳,结论是"**震荡了 6000 步没有趋势**"。**这是错的。**

```
逐条记录的变异系数:  PointFlow Loss 0.83   Video Loss 0.15
```

PointFlow 的 loss 单步噪声极大(每个样本的 σ 不同,而 `v = ε − x0` 里 ε 是单位方差噪声)。**必须按窗口取平均才看得到趋势。** 下次分析先平均,不要抽样。

---

## 3. 采样侧:**从来没有赢过 zero**

`moving_ade_mm` / `zero_moving_ade_mm` 的比值(>1 = 比"不动"还差):

```
            step 200  step 1500  step 3000  step 5000  step 7300
train_00     2.38×     1.10×      1.60×      1.77×      2.03×
val_00       8.61×     1.99×      3.28×      3.49×      4.12×
```

最佳点(step ~1500)也只有 **1.10× / 1.99×**。而且**越训越差**。

### 画面上是什么样

`pointflow_eval/step_0007300/train_00/comparison.png`:

- **左列(GT,绿点)**:始终贴着夹爪,跟着它走
- **中/右列(Pred,品红点)**:+10 步就开始**拉出几条直线射向右上**,+21 步散在画面上方,+32 步变成一条横带

典型症状:**预测出恒定速度、积分成直线,位移过头。**

---

## 4. 当时的假设:采样器的 σ 调度和训练分布**不匹配**

> ⚠️ **本节是假设,后经复测被证伪 —— 见 §6。** 保留原文以记录推理过程。

### 4.1 训练用的 σ 分布

PointFlow 用的是**视频的 σ**(`pointflow_add_noise`: "one shared video sigma per sample")。而视频的分布是:

```
train_time_video_distribution = waver
shift['480'] = 5

实际采出:
  p10=0.552  p25=0.757  p50=0.833  p75=0.889  p90=0.953
  σ ∈ [0.75, 0.90] 占 53.8%
  σ < 0.25        只占  2.8%
```

(公式:`t = 1 − u − 1.29·(cos(πu/2)² − 1 + u)`,再过 `σ = shift·t/(1+(shift−1)t)`。)

### 4.2 采样器用的网格

```python
# cosmos_framework/model/generator/pointflow_sampling.py —— 没有 shift 参数
sigma = torch.full((batch_size,), 1.0 - index / steps, ...)
```

16 步就是 **σ = 1, 0.938, 0.875, …, 0.0625** 的均匀网格:

```
σ 区间           训练占比    无shift步数    shift=5 步数
[0.00,0.25)       2.8%         3   ✗         0   ✓
[0.25,0.50)       5.4%         4             2
[0.50,0.75)      15.8%         4             3
[0.75,0.90)      53.8%         3             5   ✓
[0.90,1.00)      22.2%         1             5   ✓
```

**无 shift 时 3/16 步落在 σ<0.25,最后一步 σ=0.0625 —— 模型几乎没见过。**

### 4.3 视频/action 那一路是**有** shift 的

```python
# cosmos_framework/model/generator/diffusion/samplers/fm_solvers_unipc.py:189
sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)
```

而且训练↔推理是对齐的:**训练 `shift['480']=5` ↔ 推理 `ops.sft.shift: 5.0`**
(`examples/deployment/cosmos_singlerighthand_protocol_v2.yaml:67`)。

**只有 PointFlow 的采样器漏了这一步。**

### 4.4 为什么这正好造成"直线飞出去"

低 σ 时 `x_t = σε + (1−σ)x0 ≈ x0`,要从 x_t 反推 ε 需要**除以 σ** ⇒ 误差放大 `1/σ`。
而模型在 σ<0.25 上几乎没训练过。

**最后几步既超出训练分布、又误差放大最狠** ⇒ 速度估偏 ⇒ 位移过头 ⇒ 32 步累积成 643mm 的偏差。

这解释了两个看似矛盾的现象:
- 单步 ADE 好(它是**跨 σ 平均**的,训练覆盖的区间权重最大)
- 多步采样差(它**必须**走完整个 σ 区间,包括没训练过的那段)

---

## 5. 按该假设的修复(已实施,但**没有解决问题**)

### 改了什么

**`cosmos_framework/model/generator/pointflow_sampling.py`**

- `shifted_sigmas(steps, shift, device)` —— 生成 `steps + 1` 个 σ 节点(末节点为 0),
  用 unipc 同一个公式 warp:`σ = shift·t / (1 + (shift−1)·t)`。`shift=None` 退化为均匀网格。
- `training_shift(shift_config, resolution)` —— 按 resolution 解析训练 shift,
  镜像 `OmniMoTModel.set_up_scheduler_and_sampler` 的逻辑(支持 int 或 dict)。
- `sample_displacement(..., shift=None)` —— 新增参数;积分改成**按实际节点间距**步进
  `state += v · (σ_{i+1} − σ_i)`,因为 warp 之后节点不再均匀。

**`cosmos_framework/model/generator/omni_mot_model.py::sample_pointflow`**(NVIDIA 文件,+2 行)

```python
shift=training_shift(self.config.rectified_flow_training_config.shift, self.config.resolution),
```

### 验证(已跑)

```
① 回归:shift=None 与改动前逐位相同          max|diff| = 0.00e+00   ✓
② 积分精确性:v 恒定时 16 步走完 init−v       误差 1.7e-6 / 4.8e-7   ✓
③ σ 节点落点(16 步):
     无 shift   最低 σ=0.062   σ<0.25 有 3 步   σ>0.75 有  4 步
     shift=5    最低 σ=0.250   σ<0.25 有 0 步   σ>0.75 有 10 步   ✓ 对上训练分布
```

测试锁在 `pointflow_sampling_test.py` 新增 4 项(共 8 项全过):
`test_shift_none_keeps_the_uniform_schedule_exactly`、
`test_shift_moves_sigma_nodes_onto_the_training_distribution`、
`test_shift_keeps_euler_exact_for_a_constant_velocity`、
`test_training_shift_resolves_by_resolution`。

### 怎么算修好了

重跑后 `moving_ade / zero_moving_ade` **低于 1**(现在 2–4×)。

### 注意:现有 checkpoint 不能续训

修的是采样器,不影响已训出来的权重 —— 但**评测结果会变**。所以判断"分支行不行",
要用修好之后的 eval 数字,不要拿 7320 步那批 eval 的数字下结论。

---

## 6. 复测:**假设一被证伪**

修复上线(代码 11:15 改、训练 11:26 重启并自动 resume),测到 step 7700。

```
step      采样器    ratio (moving_ade / zero)
7000        旧       3.95
7200        旧       4.04
7400        旧       4.16
7500        旧       4.20
────────── 重启,换成带 shift 的采样器 ──────────
7600        新       4.33
7700        新       4.24
```

趋势是旧采样器每 100 步约 +0.05;7500→7600 按趋势应约 4.25,实测 4.33。**没有改善。**

张量层面也一样:

```
                 step 7300(旧)   step 7600(新)
pred 对 gt 斜率      1.38            1.38      ← 完全相同
|pred| p10          198 mm          206 mm
|pred| std (z 轴)   431 mm          453 mm
```

**修复确实生效**(σ 节点确认是 shift 后的 `[1.0, 0.987, …, 0.417, 0.25, 0.0]`),但**对结果没有任何影响**。

⇒ **"σ 调度不匹配"不是这个现象的原因。§4 的假设被证伪。**

### 6.1 证伪后的真实证据

```
step 32 的预测:
  方向    余弦相似度中位 0.826,72% 正相关      ✓ 方向是对的
  幅度    |pred| p10=206  p50=321  p90=1169 mm
          |gt|   p10= 80  p50=153  p90= 184 mm
          ↑ 最小的预测都比 90% 的 GT 大

  pred 对 gt 的线性斜率 = 1.38,但 std 相差 4 倍
  ⇒ pred = 1.38·gt + 一大块与 gt 无关的分量(残留噪声)
```

采样器从 `ε`(三维单位噪声,期望模长 **1596 mm**)走到真值(约 150 mm):

```
现在:  压到 ~200–300 mm  → 消掉 ~85%
需要:  压到 150 mm 以下  → 需要消掉 ~99%
```

**所以现象是:速度场精度不够,16 步串联的误差累积成为主导项。**
画面上"直线飞出"是这个的表现 —— 沿采样路径的速度几乎不变,点被匀速外推。

### 6.2 一个重要的旁证(**后经修正**)

当时观察到 eval-only 的 16 步与 64 步逐位相同,据此推断"速度项几乎无贡献"。
**那个观察来自坏掉的分支(见 §7),不能用作证据。**

**替代的干净证据**(同一次 eval 内的 16 vs 32 步,见 §8):

```
step 7600:  16步 ADE 347.9  →  32步 334.0    改善 4.0%
step 8100:  64步 ADE 379.5  →  32步 366.7    改善 3.4%
```

⇒ 步数**确实有效但很小**,采样器在收敛。结论方向没变,但依据换成了真的。

---

## 7. eval-only 路径**不可信**

为验证"更多采样步数能不能救",用 `checkpoint.load_path` 起了一个只 eval 的进程
(16 步与 64 步各一次)。**两次的结果都是纯噪声:**

```
                        t=32 余弦    正相关    |pred| 均值
训练内 eval (fit005@7600)  0.826      72%       539 mm     ← 正常
eval-only (16 步)          0.051      52%      1631 mm     ← ≈ 纯噪声
eval-only (64 步)          0.051      52%      1629 mm     ← 与 16 步逐位相同
```

`|ε|` 的理论期望模长是 1596 mm —— 1631 与之吻合。**预测就是初始噪声本身。**

### 7.1 输入完全相同

```
                     fit005@7600        eval-only@7500
point_ids / xyz0 / query_uv   相同 ✓          相同 ✓
flow_gt / valid               相同 ✓          相同 ✓
seed                          1185889342      1185889342    ✓
episode                       episode_0015_…  episode_0015_… ✓
valid_fraction                0.943           0.943         ✓
zero_* 基线                    80.4 / 145.9    80.4 / 145.9  ✓
───────────────────────────────────────────────────────────
all_ade_mm                    347.9           1624.6        ✗
pred_outside_fraction         0.138           0.679         ✗
```

### 7.2 已排除的原因

| 怀疑 | 检查结果 |
|---|---|
| checkpoint 里没有 pointflow 权重 | ❌ 有,`.metadata` 里 1209 处 |
| 加载时被跳过 | ❌ 日志 `kept_keys=851 dropped_keys=851` —— **851 个 `net.*` 全部保留**,只丢了 net_ema |
| `load_ema_to_reg` 把 `net.*` 换成 `net_ema.*` 再全跳过 | ❌ 两边都是 `False` |
| 子串匹配误伤 | ❌ `"net_ema." in "net.pointflow_branch…"` 为 False |
| 配置不同 | ❌ 逐项 diff 过,只差 `load_path` |
| 点分支被静默跳过 | ❌ `cosmos3_vfm_network.py:982` 会 raise,不会静默 |

### 7.3 未定位

**同一份权重、同一份输入、同一个种子,`resume` 与 `warm start` 给出不同结果。**
说明还有某个看不到的状态在两者之间不同。**没有定位到。**

### 7.4 结论

- **不要用 `checkpoint.load_path` 的 warm-start 做 eval-only。**
- 因此 **§7 那次 16 vs 64 步的比较是无效的** —— 两次都在测一个死掉的分支,什么也没测到。
- 要测采样参数,就**重启训练**(训练内 eval 是已知正常的):

```bash
OUTPUT_ROOT=<fit005 同一个目录> \
POINTFLOW_SELECT_MOTION_FRACTION=0.05 \
EXTRA_TAIL_OVERRIDES="++trainer.callbacks.pointflow_eval.sampling_steps=64" \
bash examples/launch_sft_action_policy_singlerighthand_edge.sh
```

两个 eval-only 目录(`runs/cosmos/pointflow_eval_7500{,_s64}`)保留,是排查证据。

---

## 8. 假设二:求解器与采样步数(**已证伪**)

```
视频/action 推理:  unipc(高阶),  num_steps = 4,  shift = 5.0
                  (examples/deployment/cosmos_singlerighthand_protocol_v2.yaml:61-67)
PointFlow eval:    Euler(一阶),  sampling_steps = 16
```

**两处不同:**

**① 求解器。** 仓库里视频/action 一律用 `unipc`(二阶/三阶);PointFlow 的
`sample_displacement` 是**一阶 Euler**。同一步数下,一阶的截断误差大一个量级。

**② 最后一步的跨度。** shift=5 之后两个调度格都是:

```
视频 4 步:  σ = 1.0, 0.9375, 0.833, 0.625  →  0     最后一步跨 0.625
PointFlow 16 步: σ = 1.0, 0.987, …, 0.417, 0.25 → 0  最后一步跨 0.25
```

视频那一步跨度**更大**,但 unipc 是高阶的,扛得住;PointFlow 用一阶线性外推,
而 σ→0 附近速度变化最快(`v ∝ 1/σ`)。

**这很可能解释"直线飞出"里相当大的一部分。** 便宜的验证:
把 `sampling_steps` 提到 64/128 跑**训练内** eval。若 ratio 明显下降 ⇒ 是求解器/步数问题,
正解是给 PointFlow 也换成 unipc(与视频一致)。

---

## 9. 误差形状:**方向对、幅度系统性错 3.1 倍**

step 8100 的 `val_00`(64 步采样,`t=32`):

```
  |pred| = 574 mm      |gt| = 149 mm      余弦中位 0.805     72% 正相关
  纯噪声基准 |ε| = 1596 mm(单位方差三维噪声的期望模长)

  分解:  pred = 3.1 · x0  +  339 mm 与 GT 无关的残差
                 └ 幅度大 3.1 倍    └ 方向对,但残留一块噪声
```

采样器从 `ε`(1596 mm)出发,应走到真值(149 mm),现在走到 574 mm
—— **消掉约 64% 的噪声,而需要 90% 以上**。

### 为什么这个形状很重要

**"学得不够"的误差是各向同性的 —— 方向也会乱。**
现在是**方向精准(余弦 0.805)、幅度一致偏大(3.1×)**。这更像**有一条结构性的东西没接上**,
而不是"还没训够"。

> ### ✅ 已解释(2026-09-14 晚)
>
> 这里的 `x0` **就是采样的初始噪声**。实测 14 个 case:GT 位移 std 0.113,预测 std 0.370 ——
> **0.370 ≈ 1 − 0.63,即"把单位噪声等比缩小 63% 之后剩下的那部分"**。
>
> 所以"方向对、幅度错" = **共模部分学到了,逐点部分就是没消干净的噪声**。
> 形状、倍数、方向余弦三个数一次全对上。
> 详见 [`pointflow_displacement_scale_20260914.md`](pointflow_displacement_scale_20260914.md) §3。
>
> "有一条结构性的东西没接上"这个判断是对的,只是**接错地方了** —— 不在位置编码,在 loss 的尺度。

---

## 10. 基础事实:三个模态的注意力**可见性**

这个结论后面所有关于"桥"的讨论都要用到,先钉死。

### 序列只有两段

每个样本的 pack 是 `attn_modes = ["causal", "full"]`:

```
und 段:  text                           ← modalities.py:218 / sequence.py:235
gen 段:  video latent + action + point   ← sequence.py:737 (finish_sample)
```

point token 明确追加在 full 段(`pointflow_sequence.py:112`):

```python
splits.extend((sequence.split_lens[2*b], sequence.split_lens[2*b+1] + len(content)))
                                                              └ 加在 full 段尾部
```

### 注意力规则

`two_way_attention` 两条路:

```python
causal_res = attention(causal_q, causal_k, causal_v, is_causal=True)   # und → und,因果
full_res   = attention(full_q, get_all_seq(packed_key_normalized), get_all_seq(...))
                              └──────── 全部 token(und + gen)────────┘
```

- **und(text)查询**:只能看到它之前的 text
- **gen 查询**:看到**全部** —— 无条件、非因果

### 实际可见性矩阵

```
   token      能被谁看到?  text      video     action    point
   ─────────────────────────────────────────────────────────────
   text                   ✓(因果)    ✗         ✗         ✗
   video                  ✓          ✓         ✓         ✓
   action                 ✓          ✓         ✓         ✓
   point                  ✓          ✓         ✓         ✓
```

**video / action / point 三者完全双向可见**,没有任何隔离。gen 查询还能单向读到 text。

### 对诊断的意义

**桥的管道是通的。** 如果消融测试显示"模型没用 action",原因**不可能是"它们看不见彼此"**,
只能是下面两条之一:

1. attention **稀释或扭曲**了 action 的信息(例如 `alignment_audit` 发现 1 的虚假空间相位)
2. 模型**学会了忽略** action(权重太低、梯度太弱)

**这两条正好对应文档 §14 待办里排前的两个嫌疑。** 消融能回答"用没用",
但区分不了这两条 —— 那要靠发现 1 的 A/B。

> **顺带一个推论**:这也解释了为什么**方向是对的** —— video 和 point 全双向,
> 模型能直接从画面里读到夹爪往哪走。而"幅度"要靠 action,那一路若被污染,就只剩方向。
> **与观测完全吻合。**
>
> ⚠️ **该推论已作废(2026-09-14 晚)**:"方向对、幅度错"的真实原因是归一化,不是 action 桥。
> 本节关于**注意力可见性矩阵**的实测部分仍然有效,可继续引用;由它推出的这条因果链不成立。

---

## 11. 方案:动作消融诊断

### 为什么要做

PointFlow 的设计前提是 point token 是 **video 与 action 之间的桥**。
现在症状是"方向对、幅度错 3.1×",而**幅度信息只存在于 action 里**。

⇒ 假如 action 没接上,模型只能靠 video 猜 —— 猜得出方向,猜不出行程。
**这正是观测到的样子。** 但这个前提**从未被验证过**。

### 怎么做

同一次 eval,同一个 checkpoint、同一个 case、**同一个 seed**(初始噪声相同),只动一件事:

```
第 1 遍:  data_batch["action"]  原样
第 2 遍:  data_batch["action"]  置 0
```

比两次的预测。其他一切不变 ⇒ 任何差异只能归因于 action。

**加强点**:这个 eval 给模型的是 **action 的真值**(`conditions = "clean GT video/action + ..."`)。
所以问题不是"能不能猜到动作",而是:

> **把正确答案摆在它面前,它的点预测会不会因此改变?**

| 两次预测 | 结论 | 下一步 |
|---|---|---|
| **逐位相同** | 模型**完全没用 action** —— 桥是断的 | 去修 `alignment_audit` 的**发现 1** |
| **明显不同** | action 传到了 point token | 幅度问题不在桥 → 转向 `pointflow_loss_weight` / 训练量 |

### 两个要写进结论的限制

**① 置 0 是分布外输入。** 模型见到 0 可能因为"没见过"而乱输出,而不是因为"依赖 action"。
所以如果结果是"明显不同",还需要更干净的版本:**把 action 换成另一个样本的动作**(分布内),
看预测是否朝那个样本的运动靠拢。

**② state 要不要一起零。** action token 0 是**当前关节状态**(合理条件),
1..32 才是**未来动作指令**。更精确的做法是**只零 1..32、保留 0** ——
测的才是"未来动作计划有没有被用"。

### 结果(已跑,step 8100,14 个 case)

同一次 eval,`sampling_steps=16` 各跑两遍(原样 / action 置零):

```
case       正常ade   置零ade   位移差mm    依赖度        case       正常ade   置零ade   位移差mm    依赖度
train_00    739.9    743.2       8.9     0.013        val_06      394.3    396.0       7.7     0.019
train_01    207.9    213.9       9.6     0.047        val_07      551.8    554.7      10.5     0.020
val_00      385.4    390.7      10.8     0.027        val_08      631.4    628.7       9.5     0.015
val_01      480.5    491.8      16.1     0.034        val_09      624.7    621.2      10.4     0.017
val_02      348.2    353.6       9.9     0.028        val_10      683.0    681.7       8.7     0.013
val_03      360.9    368.7      12.0     0.032        val_11      244.7    250.8      13.5     0.056
val_04      777.7    782.9      16.8     0.025
val_05      455.7    461.7      10.7     0.024        ──────────────────────────────
                                                      均值                 0.026
```

`依赖度 = |预测(置零) − 预测(原样)| / |预测|`。

**读数:2.6%,落在"≈0"那一端。** 按上表的判据,当时读作"模型完全没用 action,桥是断的"。

**这个读数现在作废,原因不是判据错了,是它测错了对象**(见 §0 的结论框):

- 根因是 clean 样本没归一化,分支**根本没学会预测运动**。一个不预测运动的模型,
  当然不会对 action 敏感 —— 这不能推出"桥是断的"。
- 但**这个消融本身仍然可用**,而且值得留成常设诊断:改成 `swap` 变体
  (把 action 换成另一个样本的,而不是置零),在**分支能运动之后**再跑一次,才有判别力。
- 一个不依赖"置零是否 OOD"的旁证:`sampling_difference_mm = 89.6`
  (同一输入,16 步 vs 32 步),而 action 效应是 10.8 mm —— **求解器的数值噪声是 action 的 8 倍**。
  同样地,这也在退化模型上量的。

### 顺带:整条 ADE 曲线

拉的 83 次 eval(14 个 case 的均值;eval case 是钉死的,`mean_zero` 恒为 128.9):

```
 step   mean_ade   vs zero          step   mean_ade   vs zero
  100     1015.5     7.88×          4500      358.2     2.78×
  500      350.8     2.72×          5000      360.6     2.80×
 1000      240.0     1.86×          5500      368.6     2.86×
 1500      210.8     1.64×  ← 最好   6000      379.3     2.94×
 2000      281.7     2.19×          6500      391.0     3.03×
 2500      336.6     2.61×          7000      403.5     3.13×
 3000      351.4     2.73×          7500      430.0     3.34×
 3500      342.1     2.65×          8000      483.0     3.75×
 4000      365.6     2.84×          8300      481.7     3.74×
```

**两个要点:**

1. **从来没有跌破 1.0** —— 最好也只有"预测不动"的 1.64 倍差。
   所以"ade 低于 zero"的印象在这条曲线上不成立(可能是训练侧 loss,或更早的指标)。
2. **step 1500 之后单调变差**(1.64× → 3.75×),而训练侧 `train/pointflow_loss`
   同期从 0.021 降到 0.012,**从没掉头**。

⇒ **train 降 / eval 升**。当时读作过拟合,现在有了统一解释(见
`pointflow_displacement_scale_20260914.md` §4):loss 90% 在度量噪声,训练推进的是那一项,
被推进的那一项对轨迹有害无益。

---

## 12. 另外三个值得记的数字

### 12.1 `valid_fraction = 0.177` ⚠️

那个 eval case 里,**82% 的 (步, 点) 是无效的**。

原因是选点本身:选出的 410 个点是**运动最快**的(夹爪上),而夹爪跟着手臂移动 —— **未来帧里它们经常被遮挡或出画**,于是 `valid` 掩码把大部分监督抹掉了。

⇒ **"信号强 10 倍"是真的,但代价是有效监督只剩 18%。** 这两条要一起看(见 `pointflow_motion_selection_20260914.md` §1 与 §3)。

### 12.2 `zero_moving_ade = 318mm`

有效点的平均位移是 **318mm** —— 目标本身很难,不是"动一点点"。

### 12.3 权重

```
rf.pointflow_loss_weight = 1.0      (loss ≈ 0.01)
rf.loss_scale            = 10.0     (video loss ≈ 0.12 → 加权 1.2)
```

⇒ PointFlow 只占 Total Loss 的 **0.8%**。主干基本是视频在训,point 分支靠自己的 lr(`codec` 25× = 5e-4)往前爬。

**这未必是问题,但决定了"分支学得慢"是预期内的。**

> ⚠️ **2026-09-14 晚**:这条不是根因。同一个数字 `pointflow_displacement_scale = 1.0`
> (上表第 2 行)才是 —— 它让 loss 的 90% 落在噪声项上,所以**这个 loss 本身就没有在度量分支
> 学得好不好**,提高权重只会放大一个错误的信号。先改 scale,再回头看权重。
> 见 [`pointflow_displacement_scale_20260914.md`](pointflow_displacement_scale_20260914.md)。

---

## 13. 复现这套分析

```bash
R=/mnt/shared-storage-gpfs2/ailab-eailabagent-gpfs/shichaojian/runs/cosmos/pointflow_fit005
C=$R/cosmos3_action/action_sft/action_policy_singlerighthand_edge

# ① 训练曲线(必须按窗口平均,不要抽样)
grep -o "Iteration [0-9]*: .*" $R/logs/action_policy_singlerighthand_edge_sft.log | tail -100

# ② 采样轨迹 vs zero 基线
for s in 200 1500 3000 5000 7300; do
  python -c "import json;m=json.load(open('$C/pointflow_eval/step_$(printf %07d $s)/val_00/metrics.json'));\
print('$s', round(m['moving_ade_mm'],1), round(m['zero_moving_ade_mm'],1), round(m['moving_ade_mm']/m['zero_moving_ade_mm'],2))"
done

# ③ σ 分布对照
python -c "
import numpy as np
W=1.29; r=np.random.default_rng(0); u=r.random(400000)
t=1-u-W*(np.cos(np.pi/2*u)**2-1+u); sig=5*t/(1+4*t)
g0=1-np.arange(16)/16; g5=5*(1-np.arange(16)/16)/(1+4*(1-np.arange(16)/16))
print('训练 σ 中位', round(float(np.median(sig)),3))
print('σ<0.25 训练占比', round(float((sig<0.25).mean()),4))
print('无shift 步数', int((g0<0.25).sum()), ' shift=5 步数', int((g5<0.25).sum()))
"
```

### 读 wandb offline history(训练侧曲线的唯一来源)

`WANDB_MODE=disabled` 的 run **仍然把完整 history 写在** `wandb/offline-run-*/run-*.wandb`
里(step 0–7792 的 7748 条)。旧 log 会被下一次启动覆盖,所以这个文件是**唯一**能拿到
早期训练曲线的地方。

```python
from wandb.proto import wandb_internal_pb2 as pb
from wandb.sdk.internal import datastore

ds = datastore.DataStore()
ds.open_for_scan("<run>/wandb/offline-run-XXXX/run-XXXX.wandb")   # 注意:不是 .open()
rows = []
while True:
    data = ds.scan_data()                 # 返回序列化后的 protobuf bytes
    if data is None:
        break
    record = pb.Record()
    record.ParseFromString(data)
    if record.WhichOneof("record_type") != "history":
        continue
    rows.append({("/".join(i.nested_key) if i.nested_key else i.key): i.value_json
                 for i in record.history.item})
```

可用的 key(实测):

```
train/pointflow_loss      train/video_loss      train/action_loss      train/loss
pointflow/val_loss        val/loss              train@2_detail/flow_matching_loss_pointflow
```

**注意**:`pointflow/val_loss` 是**双峰**的(0.003 / 0.05 来回跳),疑似大部分 val step
里没有 pointflow 样本。**不要用它判断泛化。**

### 看画面

```
$C/pointflow_eval/step_0007300/train_00/comparison.png     ← 三格静态图
$C/pointflow_eval/step_0007300/train_00/comparison.mp4     ← 动图
$C/pointflow_eval/step_0007300/train_00/error_curve.png    ← 误差随步数
$C/pointflow_eval/step_0007300/train_00/metrics.json       ← 全部指标
```

---

## 14. 待办

> **优先级已在 2026-09-14 晚重排。** 根因是 clean 样本没归一化(见
> `pointflow_displacement_scale_20260914.md`),下面第 1 条是判定实验,其余顺延。

| # | 事项 | 状态 |
|---|---|---|
| **1** | **训练集实测位移 std,设进 `pointflow_displacement_scale`(现在 = 1.0),重训** | **下一步** —— 判定根因 |
| 2 | 重训后判据换成 **ADE vs zero 基线**,不再看 loss(改后 loss 会上升,这是对的) | 未做 |
| 3 | 检查 §7 的 warm-start 路径是不是同一个根因(当时观察到"预测成为纯噪声",而纯噪声正是失败形态) | **需复核** |
| 4 | 一并把 `sampling_steps` 提到 32/64 —— 步数假说虽已证伪,但改完 scale 后值得重测 | 未做 |
| 5 | 实测 `valid_fraction` 只有 0.18 —— 选点选的是最易失效的点 | 未做(§12.1) |
| 6 | 若 scale 修好后仍不动:提高 `pointflow_loss_weight`(现 1.0,分支只占总 loss 的 0.8%) | 未做(§12.3) |
| 7 | 若仍不动:给 PointFlow 换 **unipc**(与视频一致),而不是一阶 Euler | 未做 |
| 8 | 用 **swap** 变体重跑动作消融(action 换成别的样本,不置零),此时才有判别力 | 未做(§11) |
| 9 | `pointflow_alignment_audit_20260913.md` 的**发现 1** | 回到原定性"设计未实现,危害未证实" |

**已作废的条目**(保留以便追溯):

| 原 # | 事项 | 为什么作废 |
|---|---|---|
| 1 | 给 `sample_displacement` 加 `shift` | ✅ 已做(§5),**复测证明无效**(§6)。问题不在 σ 网格 |
| 3 | 把 `sampling_steps` 提到 64/128 | 实测 16→32 步只改善 5%,最多解释所需的 5%(§8) |
| 5 | 回看"发现 1"(action↔point 虚假空间相位) | 分支还没学会运动,测不了它读不读 action —— 而真正的根因是归一化 |
| 2 | 用 eval-only 测 16 vs 64 步 | eval-only 路径本身可疑(§7),且步数假说已证伪 |
