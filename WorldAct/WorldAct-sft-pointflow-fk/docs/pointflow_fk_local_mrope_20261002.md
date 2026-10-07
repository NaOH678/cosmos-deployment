# PointFlow 与 FK 的局部四轴位置编码

目标是在共同相机坐标系中，让视频侧 PointFlow 与动作侧 FK 的三维相对位置参与
attention 内容匹配，从而检验 video–pointflow–FK–action 桥接是否改善。
首版采用原方案 A 的成对旋转，不新增 MLP、几何 bias 或 token。

状态（2026-10-02）：FK 四模态路径与局部四轴 attention 已接入，采用纯 Flash2 varlen。CPU/CUDA 数值检查及 8 卡 FSDP 冒烟通过，包括完整梯度检查点、torch.compile 三步训练、保存与整轮联合生成（exit 0）。启动入口默认开启 compile；30 步开关对照及短 profile 已完成，典型步耗时 5.24→6.00 s，额外开销主要指向索引及其反向写回。长期吞吐和训练效果待正式实验。投影核验与运行记录见 §7。
原稿为 `/mnt/afs/mRoPE_公式原文.md:113`；本文件明确实现选择并修正频率计算。
这里的“局部”指 point/FK token 对，不是每个簇拥有独立坐标系，也不是空间邻域硬掩码。

## 1 坐标定义

几何 token 记为集合 G，包含 PointFlow cluster 与 FK 关键点的 anchor 和未来块。
每个 token 有两份位置元数据，共享同一份模型投影得到的 Q/K：

$$
p_i=(t_i,h_i,w_i),\qquad \hat p_i=(t_i,h_i,w_i,\zeta_i).
$$

相机坐标记为 $(X,Y,Z)$，其中 $Z>0$，单位米；$\zeta$ 是无量纲逆深度编码，
不要把它与相机 Z、位移归一化 scale 混用。首版候选固定为：

$$
\zeta_i=\frac{1/Z_i-\rho_0}{s_\rho},
\quad \rho_0=1\;\mathrm{m}^{-1},\quad s_\rho=1\;\mathrm{m}^{-1}.
$$

这只是固定单位换算与公共平移，不是按样本或模态拟合的归一化。
PointFlow 和 FK 必须共用参数。公共 $\rho_0$ 在 G×G 相对相位中抵消；
其余 token 对不使用深度相位，所以也不受该平移影响。
非正、非有限 Z 必须显式报错或按定义的有效性机制处理，不静默取倒数或任意 clamp。

| 字段 | PointFlow cluster | FK 关键点 |
| --- | --- | --- |
| anchor 三维位置 | 原始成员 anchor XYZ 等权均值 | FK 经已核验的 base→camera 变换 |
| h/w | 沿用成员 anchor UV 均值和现有视频仿射 | 相机系 XYZ 投影后，使用同一视频仿射 |
| 深度 | cluster XYZ 均值的 Z，再取倒数 | 关键点相机 Z，再取倒数 |
| t | 视频首 latent 的时间原点，加未来块结束时刻 | 与 PointFlow 相同时间格 |

注意：平均 UV 一般不严格等于平均 XYZ 的投影。沿用现有 cluster UV 是为了控制改动，
此时位置是簇的摘要，不是一颗严格重建的物理点。正式接入时应记录簇内深度跨度及
两种投影的差异；若改为投影 cluster XYZ，必须作为独立、显式的坐标规则变更。
当前汇总实现见 `cosmos_framework/model/generator/pointflow_geometry.py:20`。

FK 投影必须使用与 PointFlow 一致的相机模型和原始图像尺寸，再复用 resize/crop/
拼图偏移及 patch 换算。不能直接使用未经变换的相机像素，不能独立给 FK 设原点。
内参、畸变约定和外参投影的真实窗口核验仍是训练接入前检查项；深度范围扫描不替代它。

首版使用固定 anchor 空间位置：8 个未来块沿用 anchor h/w/ζ，仅 t 推进；
未来位移仍在带噪 token 内容中。不使用未来 GT 位置、掩码或轨迹来构建位置元数据。
这编码“从哪里出发”的关系，并不直接编码未来预测位置。动态位置留作后续独立问题。

## 2 通道和频率

Edge head_dim=128，共64个二维旋转对。代码采用 split-half 布局：
第 c 对是向量维度 c 与 c+64，不是相邻的 2c 与 2c+1。
原分配为 t/h/w=24/20/20，H占1:60:3，W占2:60:3，其余为T；
因此尾部60、61、62、63均属时间，不能简单对所有64对循环取模。

原频率应为：

$$
\omega_c=(10^8)^{-2c/128}=10^{-c/8}.
$$

原稿写成 $10^{-c/4}$ 是算术错误。c=33、时间差15000对应约1.125 rad，
不是原稿的约0.000084 rad；当前窗口时间差12.8时约0.000960 rad。
代码依据 `cosmos_framework/model/generator/reasoner/nemotron_3_dense_vl/nemotron_3_dense_vl.py:103`。

候选四轴分配为 **24/16/16/8**：

- 时间全部保留，连同频率不变。
- H中49、52、55、58，W中50、53、56、59改为深度通道。
- 深度通道按索引排序为49、50、52、53、55、56、58、59。
- 其他56对的轴归属和频率完全保留。

选择这8对，是减少几何子块中原h/w相位的改动，不声称这些内容通道没有作用。
三轴路径不改变任何通道。深度频率的首版候选是8个等比间隔值：

$$
\nu_k=0.125\left(\frac{4}{0.125}\right)^{k/7},\quad k=0,\ldots,7.
$$

频率单位为 rad/ζ-unit。这是可检查的实验候选，不是已验证的最优值；冻结后训练
与采样使用同一份配置，并随 checkpoint 记录。不要使用原文未经核算的频率。
关闭实验开关必须恢复原三轴旋转；仅将 ζ 置零并不能恢复原h/w通道，不能用来当baseline。

## 3 成对打分

令 $R_3,R_4$ 分别为以上三轴、四轴旋转；同一个 token 的两套表示共用投影参数：

$$
q_i^3=R_3(p_i)q_i,\quad k_i^3=R_3(p_i)k_i,
\qquad q_i^4=R_4(\hat p_i)q_i,\quad k_i^4=R_4(\hat p_i)k_i.
$$

$$
s_{ij}=\frac{1}{\sqrt{128}}
\begin{cases}
(q_i^4)^\top k_j^4,& i,j\in G,\\
(q_i^3)^\top k_j^3,&\text{其他合法 token 对}.
\end{cases}
$$

所有合法 keys 共用一次 softmax；原有样本隔离和 causal/full mask 保留。
文本 K 对生成 query 的 normalization、GQA 和 packed sample 边界也必须保留。
位置改变匹配方式，不保证注意力随距离单调下降，不强制 FK 骨架与手表面点重合。

复数记法若使用 q·conj(k)，相位差是 query 减 key；原文与其正相位旋转约定不一致。
原型直接使用实数旋转后点积，避免用错误符号构造参考答案。

“局部”限定的是原始 logits 的替换范围。对于几何 query，G×G分数变化会改变
softmax分母，因此它对非几何 keys 的归一化权重也可改变；这不是额外的跨模态相位。
对非几何 query，本层同样输入下其所有分数和输出不受深度参数影响。
多层传播后，其他模态内容当然可能变化，这正是桥接希望产生的影响。

## 4 精确分组计算

非几何 query 对所有合法 keys 使用原三轴路径。
几何 query 的 keys 分成非几何集合 O 与几何集合 G：前者三轴，后者四轴。
分别得到归一化输出 $o_O,o_G$ 与对数归一化常数 $L_O,L_G$，再合并：

$$
o=\alpha_O o_O+\alpha_G o_G,
\qquad (\alpha_O,\alpha_G)=\operatorname{softmax}(L_O,L_G).
$$

这与对两组 logits 一起做 softmax 完全相同，不是两个 attention 输出直接相加。
空 key 组单独处理。不能先完整计算G×G三轴，再把G×G四轴输出直接加回去。

CPU原型为易审查仍生成完整分数，**不是可用于训练的高效后端**。
当前GPU实现直接调用 Flash2 varlen，几何两组由
`cosmos_framework/model/attention/flash2/two_group.py` 合并 LSE；
两次 Flash2 backward 显式接收全局 output/LSE，不调用 NATTEN。
若调整text/full分组，必须重新验证生产后端反向，不能依靠CPU公式代替。

不新增tokens或QKV参数，但有额外旋转、gather/scatter、kernel和归一化开销。
GPU时间/显存不能从几何子块大小直接推断。比如K=30时G=9×(30+21)=459，
这是历史簇数举例而非本次实测；实际packing容量和有效簇数需分开记录。

## 5 原型验证结果

原型：`tools/pointflow_local_mrope_probe.py`。
结果：`pointflow_outputs/local_mrope_probe_20261002.json`。

```bash
LD_LIBRARY_PATH='' OMP_NUM_THREADS=1 \
/mnt/afs/WorldAct-cosmos3-edge-droid-sft/.venv/bin/python \
tools/pointflow_local_mrope_probe.py --scan \
--output pointflow_outputs/local_mrope_probe_20261002.json
```

float64小张量检查通过：4个query heads / 2个KV heads的GQA、两个变长样本、
文本因果mask、空分组、关闭后恢复baseline、时间通道保持、深度敏感性、
非几何token对分数保持、公共逆深度平移不变性、输出及Q/K/V梯度等价。
输出最大误差约1.33e-15，梯度最大误差约1.55e-15。

另按文件名选取8个episode，每个取首/中/末3个可用窗口，只读取anchor几何。
扫描不是按训练split筛选，**不用于拟合训练参数或报告验证性能**，只做量级检查。
Point数据是原始选点而非stage1簇，FK是同源帧的右手21点。

| 量 | p01 | p50 | p99 |
| --- | --- | --- | --- |
| Point相机深度 m | 0.5113 | 0.6728 | 0.8263 |
| FK相机深度 m | 0.6463 | 0.8126 | 0.9178 |
| Point逆深度 1/m | 1.2102 | 1.4863 | 1.9558 |
| FK逆深度 1/m | 1.0895 | 1.2307 | 1.5472 |

两组分布包含不同物理部位，不能由其均值差判断外参偏移或配准正确。
抽样深度全部有限且为正；这不是全量数据质量保证。
此次读取GPU状态时8张A800均约95–100%利用率，因此没有抢占训练资源做基准测试。

## 6 实验约定

1. 在当前cluster stage1 + skip(1,2) + pb4底座组合已有FK实现，保持几何运动融合关闭；
   对照与实验保持相同sigma调度、loss、选点、batch、数据窗口和训练预算。
2. 实装统一FK投影和anchor坐标元数据，核验真实head画面的point/FK叠加。
3. 实装生产分组attention，比较局部四轴与三轴的forward/backward、BF16误差、吞吐和峰值显存。
   包含空Point样本、只有FK、packing多样本、GQA和真实text K normalization。
4. 对照组为“统一FK投影 + 三轴”，实验组为“同样坐标 + 局部四轴”。
   老base的FK仍填h/w=0，只作为更粗基线，不能把投影收益全部归因于z。
5. 两组的训练与eval必须使用同一位置规则；关闭开关恢复三轴检查在生产路径重做。
   评估同一次joint rollout的video、PointFlow、FK与action对应运动的一致性，兼看成本。

本文件没有证明方案有效，也未启动新训练；下面记录接入和测试证据。


## 7 pointflow-fk worktree 接入记录（2026-10-02）

### 已实现

- 迁入已有 FK 数据源、batch、分支、loss、eval 和四模态采样；保留当前
  cluster/skip/pb、逐帧 scale、geometry-motion fusion 开关。默认实验仍为
  stage1 + skip(1,2) + pb4、fusion=false。
- FK 投影直接复用 `_mano/tools/verify_fk_camera_projection.py` 的 `project`，
  外参继续使用已有 `fk_camera_extrinsic.base_to_camera`，未重新拟合。
  `fk_source.attach_fk_projection` 仅做 head 像素到 tracker 像素的适配；
  随后 FK 和 PointFlow 都调用 `video_aligned_point_positions`，共享实际
  tracker→video affine、patch stride、视频时间和空间原点。
- 骨架点可能正常处于画面外，不套用 PointFlow 选点的 75% 在画面内阈值；
  非有限或相机后方的 FK anchor 则拒绝。
- `pointflow_fk_attention.py`：常规路径四个 FlashAttention2 varlen 调用，
  几何 query 的两组结果通过 `flash2/two_group.py` 共用归一化。
  两次 Flash2 backward 显式接收合并后的 output/LSE；不调用 NATTEN，
  不修改上游 attention buffer，兼容 non-reentrant activation checkpointing。
  GQA 不重复 KV heads，不构造完整 N×N 分数矩阵；分组索引每次网络 forward 只建一次。
  空几何、全几何、同一 batch 中只有部分样本缺少非几何 key 均有独立处理。
- 深度频率和逆深度归一化常数是持久化 buffer；不增加学习参数或 token。
  目前限 Edge head_dim=128 / mrope_section=[24,20,20]、CP=1、two-way，
  不支持 memory/control/null-action/cudagraph 路径。
- 合并时另外发现并修复单位边界：条件诊断中的干净 FK/PF 输入需除以各自
  displacement scale；FK 联合采样返回的 PointFlow 需乘回自身 scale。
  主采样路径和逐帧 PointFlow scale 保持兼容。

### 独立投影核验：成立的部分与未成立的部分

工具：`tools/check_fk_pointflow_projection.py`。结果：
`pointflow_outputs/fk_projection_own_intrinsics_20261002/report.json`，叠图：
`pointflow_outputs/fk_projection_own_intrinsics_20261002/anchor_overlays.jpg`。
PointFlow 使用对应 episode、对应源帧的 DA3 内参；FK 使用已有 projector。
叠图绿点由 PointFlow XYZ 经 DA3 重投影得到，不是直接绘制缓存 UV。
抽取前 8 个排序 episode 的首/中/末 anchor，共 24 个窗口；只做实现审计，不作为验证集成绩。

1. 已有 FK projector 与每个抽查 episode 的原始 D435 内参结果逐点相同，最大差 0 px。
   已查看 8 个首帧叠图：骨架整体落在手部，没有重复应用 base→camera 变换的迹象；
   这不意味着每个关节已完成像素级标定。
2. 640×480 head、640×448 tracker、合成画布、544 宽 resize 和 patch 中心的变换有数值测试。
   原 `render_fk_projection` 是可视化工具，裁掉 head 底部再拉伸回 480 高的结果不能直接
   当作训练位置；训练以实际 affine 和 sequence 元数据为准。
3. 当前 labeled PointFlow 没带上游内参，核验从对应的上游 `intrinsics.npy` 读取，
   用 `frame_indices.npy` 匹配源帧。24 个窗口共 12,000 个 anchor 点，DA3 重投影
   相对原 tracker UV 的误差为：中位 0.096 px、P95 0.187 px、P99 0.233 px、最大 0.330 px。
   这说明抽查点的 XYZ、自己的内参和 UV 高度自洽。
4. FK 沿用已有投影，PointFlow 沿用自己的 DA3 内参，各自投影后进入同一 head 图像，
   再使用同一 tracker→video→patch 链路。已查看 8 个首帧叠图，PointFlow 重投影落在
   对应的手、物体、台面区域，FK 骨架落在手部。
5. 结论修正：**各用各自内参时，投影一致性检查通过**。此前把真实 D435 K 套在
   PointFlow XYZ 上的误差，只能说明不能混用投影模型，不能单独作为两者三维空间
   不一致的证据。该交叉内参诊断已从当前检查脚本移除；历史结果仍保留在旧 audit 目录。
   自重投影也不等于完成三维米制配准，后者需要对应点或独立深度证据；不由本项检查断言。
   保持原标签和各自投影，不因跨内参误差重新标定或阻断位置编码实验。

### 测试证据

- CPU 回归：61 项通过（FK data/branch/sequence、PointFlow codec/branch/fusion、
  四轴分组、已有投影工具）；新增像素中心适配后针对相关模块再跑 20 项通过。
- 新增两项采样单位回归通过：干净 FK 输入归一化、联合采样 PF/FK 分别恢复米制。
- 独立 `fk_pointflow_compose_test.py`、`fk_joint_sampling_test.py`、
  `fk_independent_schedule_test.py` 均通过：两种挂载顺序、四段 ragged 布局和共享 sigma。
- 实际单层 Edge 网络（head_dim=128）四模态前向/反向通过；更改 PF/FK 未来 GT 和 valid
  不改变相同 noisy state 的预测。Sonata 使用 CPU surrogate，此项不等于完整 GPU 训练。
- 初版 A800 + FlashAttention2 + NATTEN merge，BF16 对 float64 dense oracle
  （后续训练发现检查点兼容问题，当前已替换为纯 Flash2，见下节）：
  四种分组（普通、无几何、全几何、混合空 key）通过。最大输出误差 0.00741，
  最大 Q/K/V 梯度误差 0.02446；普通三轴 BF16 对照本身也有约 0.01965 最大梯度误差。
  结果保存为 `pointflow_outputs/fk_local_attention_cuda_20261002.json`。
- 真实内核测试发现空 KV 的特殊 LSE 会使混合分组的梯度错误，已改为对这些样本只调用
  单组 attention；不能对空组直接套常规 merge。修复后上述四种情况全部通过。
- 小张量 CUDA 检查峰值约 28.9 MiB，**不是训练峰值**。现有 GPU 均被训练占用，
  未测整模型吞吐；不能据此声称与原三轴等速。compile/FSDP/完整 joint rollout 待短冒烟。

### 启动入口与下一项验证

入口：`examples/launch_pointflow_fk_sandwich101.sh`，默认投影后 FK + 三轴，
500 点、PointFlow scale=0.0528、FK scale=0.083745、共享 sigma、grad_accum_iter=1。
路径从脚本所在 worktree 解析；不需要手动 cd/activate/清空 LD_LIBRARY_PATH。

```bash
# 只打印计划，不启动训练
DRY_RUN=1 OUTPUT_ROOT=/tmp/pointflow_fk_review \
  bash /mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow-fk/examples/launch_pointflow_fk_sandwich101.sh

# 空闲 8 卡后先做短冒烟；这次尚未执行
OUTPUT_ROOT=/data/shichaojian/runs/pointflow_fk_projected_3d_smoke \
NNODES=1 EXTRA_TAIL_OVERRIDES="trainer.max_iter=2" \
  bash /mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow-fk/examples/launch_pointflow_fk_sandwich101.sh
```

四轴组另用全新 OUTPUT_ROOT，设置 `POINTFLOW_FK_LOCAL_ROPE=true`。
两组都保持 `FK_PROJECT_ANCHORS=true`，不把投影改变混入 z 的消融。
最初的 smoke 关闭 compile；当前入口已默认开启，编译联测结果见本节末尾。
现有 PointFlow-only 入口不自动开启 FK 或改变其 sigma 调度。

### 首次训练预热失败与修复（2026-10-02）

`pointflow_fk_local4d_smoke_20261002` 在外层 `JointDataLoader` 预热时失败：
`default_collate` 尝试递归整理 FK payload，无法处理 `FKTiming`。
尚未进入训练 step 或 attention；此前的内层 collator 和模块测试没有覆盖这个入口。

- `joint_dataloader.custom_collate_fn` 将 FK 保留为逐样本列表，保留缺失样本的 None。
- 外层 packing 补入每个 FK 样本的 189 token 预算，以及跨 batch 累积时的缺失占位。
- 新增外层回归：真实 DataLoader（0/2 workers）、样本拆分与重新打包、列式输入、
  稀疏 FK 和 token 预算；与 FK batch 测试共 18 项通过。
  多 worker 测试在沙箱外通过，沙箱内禁止其 tensor 共享所需的进程间 socket。
- 按本次保存配置加载真实训练 raw dataset 的首个样本，经外层 DataLoader 和
  `build_fk_batch` 成功得到 21 个相机坐标 anchor 及对应投影 UV。

本次只验证并修复数据入口，未重启训练；完整 8 卡训练及 joint rollout 仍待重跑确认。

### 反向传播检查点冲突与纯 Flash2 替换（2026-10-02）

第二次启动已通过数据入口，但第一次 backward 在 NATTEN merge 中失败：
它重复读取 `ctx.saved_tensors`，触发 non-reentrant checkpoint 的二次 unpack 错误。
初版 CUDA 检查没有开启 activation checkpointing，遗漏了这个训练组合。
补加 `--checkpoint` 后，用小张量成功复现同一错误。

按当前方案移除局部 attention 的 NATTEN 调用，全部使用 FlashAttention2 varlen：

- causal、非几何 query、纯几何样本调用 Flash2 varlen。
- 混合几何 query 的两组 key 由 `model/attention/flash2/two_group.py` 处理：
  两次 Flash2 forward，FP32 LSE 合并；两次 Flash2 backward 均显式传入
  全局 output 和 LSE，保留共享 softmax 的正确梯度。
- backward 只读取一次 saved tensors，不修改上游输出，不关闭 activation checkpointing，
  不重复 KV heads，不生成 N×N 分数。当前接口限定 head_dim=128、无 dropout，
  使用环境内 Flash2 的 `_wrapped_flash_attn_varlen_backward` 接口。
- 启动入口默认 `I4_ATTN_BACKENDS=flash2`、`COSMOS_FLASH2_VARLEN=1`。

CUDA 回归 `tools/check_pointflow_fk_attention.py --seed 17 --checkpoint` 通过：
普通、无几何、全几何、混合空 key 四组，分别核对 float64 dense oracle，
并对比 checkpoint 开/关的输出和梯度。最大输出误差 0.007402，最大梯度误差 0.024460。
结果：`pointflow_outputs/fk_local_attention_checkpoint_20261002.json`。
5 项 CPU 分组/网络回归通过。

独立 8 卡冒烟输出：`/tmp/pointflow_fk_flash2_smoke_20261002`。
保持 stage1/skip1,2/pb4、500 点、两种 displacement scale、共享 sigma；
FSDP shard=8、activation checkpointing=full、grad_accum_iter=1、compile=false。
已完成两次训练更新并保存 `iter_000000002`，随后停在日志回调，尚未进入联合生成。
`py-spy` 确认各 rank 等待于 `wandb_log._LossRecord.get_stat`：FK sigma 桶只输出非空项，
但旧回调按每个 rank 自己的 key 列表执行 all_reduce，通信数量/语义可能不同。
已改为先同步 key 并集、统一排序；缺失项参与通信但不计入有效均值。
双进程 Gloo 回归覆盖不同 key/顺序、缺失项、下一轮桶消失或更换贡献 rank，全部通过。

终止卡住的自建冒烟后，在 `/tmp/pointflow_fk_flash2_smoke_v2_20261002` 重跑同样两步，
保留 logging_iter=1 以覆盖该汇总路径。**08:19:44 正常退出，exit 0，8 张卡已释放。**
两步更新、日志汇总、DCP 保存、整轮验证均通过：

- checkpoint：`iter_000000002`。
- PointFlow：14 个 joint case，14 段 dream 和 3 段阶段拼接视频，共 17 段，均可读取首帧；
  joint metrics 的 attention_mode 全部为 `local_pointflow_fk_mrope`。
- FK：4 个固定样本完成四模态联合采样并输出指标和预测。
- 核验记录：`pointflow_outputs/fk_flash2_training_smoke_20261002.json`。

此次证明训练/保存/联合采样路径可运行，不作为生成质量或稳定吞吐结论；
compile 仍关闭，长训练效果待正式实验。

### 开启 torch.compile 的联测（2026-10-02）

`launch_pointflow_fk_sandwich101.sh` 默认设置 `compile.enabled=true`、
`compiled_region=language`、`compile_dynamic=true`。
`parallelize_unified_mot.apply_compile` 对每个 Transformer block 使用
`torch.compile(fullgraph=True, dynamic=True)`；packing 在编译区外。
保留 Flash2 varlen、FSDP shard=8、完整 activation checkpointing、grad_accum_iter=1。

- `tools/check_pointflow_fk_attention.py --seed 17 --checkpoint --compile`：
  普通、无几何、全几何、混合空 key 四组前向/反向通过；核对 eager 和 float64 oracle。
  记录：`pointflow_outputs/fk_local_attention_compile_20261002.json`。
- 真实训练输出：`/tmp/pointflow_fk_compile_smoke_20261002`。
  三步参数更新、日志汇总、`iter_000000003` 保存和整轮验证通过，进程 exit 0，显卡已释放。
- 14 个 PointFlow joint case、17 段 dream/阶段拼接视频均产出且首帧可解码；
  4 个 FK case 完成四模态联合采样。
- rank 0 的三步计时为 115.90 / 6.20 / 22.80 s：首步包含启动和编译，
  第三步包含 14.14 s 的 checkpoint 保存。这不是稳定吞吐基准。
- 核验记录：`pointflow_outputs/fk_compile_training_smoke_20261002.json`。

原启动命令直接使用新的编译默认值；如需 eager 对照，可在 `EXTRA_TAIL_OVERRIDES`
显式设置 `model.config.compile.enabled=false`。

### 局部四轴开关性能对照（2026-10-02）

两组各跑 30 步，均正常退出。8×A800 80GB、每卡 batch=16、全局 batch=128、
grad_accum_iter=1、seed=42、compile/full AC、Flash2 varlen；两组均保留 FK 和其投影，
只切换 `POINTFLOW_FK_LOCAL_ROPE`。固定训练 loader 的 `in_order=true`，
logging_iter=10，关闭验证；结束后的 checkpoint 保存不计入计时。
逐项核对保存的 config.yaml，除输出路径外一致（局部四轴开关由环境变量控制）。

取 rank 0 第 11–29 步，共 19 个计时点；第 30 步汇总各卡训练峰值显存。
显存为从训练开始到第 30 步的累计峰值，单位 GiB，表中取 8 卡最大值。

| 项目 | 四轴关闭，保留 FK | 四轴开启，保留 FK |
| --- | ---: | ---: |
| 步耗时中位数 | 5.24 s | 6.00 s |
| 步耗时均值（保留全部波动） | 6.001 s | 5.994 s |
| 步耗时范围 | 5.18–11.94 s | 5.94–6.05 s |
| PyTorch 分配显存峰值 | 32.132 GiB | 32.134 GiB |
| PyTorch 保留显存峰值 | 33.805 GiB | 34.855 GiB |

关闭组第 23–26 步分别为 6.60 / 10.97 / 11.94 / 6.08 s，原因未确定，未剔除。
开启组中位数增加 0.76 s（14.5%），但本轮均值受关闭组慢步影响几乎相同；
不能把中位数差异直接表述为长期吞吐下降 14.5%。两组只是短性能对照，不评判生成质量。
原三步冒烟的 6.20 s 应以这里的完整短跑记录补充。

rank 0 第 11–20 步的 CPU 区段均值：关闭组 forward/backward/optimizer 为
1.717 / 3.063 / 0.410 s，开启组为 1.836 / 3.931 / 0.188 s；取数据均约 0.005 s。
`TrainingTimer` 未对各区段单独同步 GPU，异步工作可能跨区段，不能据此精确归因 GPU 算子。

产物：

- `/tmp/pointflow_fk_perf_rope_off_20261002.log`、`/tmp/pointflow_fk_perf_rope_on_20261002.log`。
- 对应输出根目录为去掉 `.log` 后的路径；各含 config.yaml、W&B 离线记录、30 步 checkpoint。
- 逐步耗时、各卡显存汇总、区段计时：`pointflow_outputs/fk_local_rope_perf_20261002.json`。

复现时使用全新 OUTPUT_ROOT，`MODE` 分别取 `false`、`true`：

```bash
OUTPUT_ROOT="/tmp/fk_perf_rope_${MODE}_fresh" POINTFLOW_FK_LOCAL_ROPE="$MODE" \
POINTFLOW_EVAL_JOINT=true I4_ATTN_BACKENDS=flash2 NNODES=1 NPROC_PER_NODE=8 \
EXTRA_TAIL_OVERRIDES='trainer.max_iter=30 trainer.logging_iter=10 trainer.seed=42 trainer.run_validation=false trainer.run_validation_on_start=false checkpoint.save_iter=100000 dataloader_train.max_samples_per_batch=16 dataloader_train.dataloader.in_order=true trainer.callbacks.device_monitor.every_n=30' \
bash /mnt/afs/WorldAct-cosmos3-edge-droid-sft-pointflow-fk/examples/launch_pointflow_fk_sandwich101.sh
```

#### GPU profile：优先优化索引及其反向写回

另起开关两组，各 13 步，采集第 11、12 步的 rank 0 trace。其他训练设置与上述对照相同；
启用 `trainer.profiling.enable_profiling=true`，profile_freq=12、profile_warmup=1、
profile_active=2、target_ranks=[0]、with_stack=false、with_modules=false。
两组均 exit 0，结束核查 8 张 GPU 均为 0 MiB / 0% 利用率。

| GPU 内核累计时间 / 步 | 四轴关闭 | 四轴开启 | 差值 |
| --- | ---: | ---: | ---: |
| Flash2 | 1.164 s | 1.210 s | +0.046 s |
| 名称含 index / scatter / gather 的非 NCCL 内核 | 0.086 s | 0.678 s | +0.592 s |
| GEMM / CUTLASS（不含 Flash2） | 2.945 s | 2.869 s | -0.076 s |
| NCCL | 0.834 s | 0.661 s | -0.173 s |

这里按 kernel 名称归类并求和，平均到两步；融合内核可能包含其他操作，不同 stream
可重叠，**不是互斥的墙钟耗时拆分**，也不能把通信内核时间下降直接解释为通信量下降。
profile 中的步耗时受 profiler 干扰，不替代无 profiler 的 30 步对照。

开启组的 `indexing_backward_kernel<c10::BFloat16, 4>` 单项累计约 0.292 s/步，
另有多种融合 index / index_copy 内核。结合两组 trace，主要可疑增量是
`pointflow_fk_attention.fused_partition_attention` / `packed_local_attention` 中的
advanced indexing、重排及反向写回，而非 Flash2 核心计算显著变慢。

下一步保持四轴语义及共享 softmax 不变，优先减少重复 gather、复用各组 Q/K/V 视图，
并研究利用组内索引唯一性实现更直接的梯度写回；跨组贡献仍必须正确累加。
优化后重跑现有 dense oracle、checkpoint/compile 回归及同口径性能对照。
本次没有修改 attention 实现，也不承诺尚未实测的提速幅度。

- 原始 trace：`/tmp/pointflow_fk_profile_{off,on}_20261002/cosmos3_action/action_sft/action_policy_fk_point_singlerighthand_edge/torch_trace/iteration_12/rank0_trace.json.gz`。
- 全部 kernel 名称、调用次数、累计时间：`pointflow_outputs/fk_local_rope_profile_20261002.json`。

### 索引路径优化（2026-10-02）

修改 `cosmos_framework/model/generator/pointflow_fk_attention.py`，保持四轴旋转、
Flash2 varlen 分组、共享 softmax 和 normalized text K 的语义：

- 组内索引由 `make_partition` 保证唯一；`_UniqueRows` 的反向在新建梯度张量中
  用 `index_copy_` 直接写回，避免组内通用 scatter-add。不同分组的梯度仍由 autograd 累加。
- 完整 `all_keys` 本来就是 identity，直接传入 K/V；复用几何 V，常见 mixed-only 情况
  直接使用已排序的 q4，省掉重复 gather。
- 各组输出只拼接并恢复顺序一次。正向/逆向置换在每次模型 forward 准备 metadata 时
  生成，所有 Transformer block 复用；恢复顺序的反向直接 gather。

首个候选（前向 `index_select`）测得 5.64 s 中位步耗时；改为 `values[indexes]`
后的 v2 为 5.65 s。两者分配峰值均为 47.22 GiB，所以此次对照不支持
“更换前向索引能降低显存”的推断。当前保留自定义直接写回的反向，后续以
每卡 batch=16 能运行为约束，优先优化时间。
记录分别保存在 `pointflow_outputs/fk_local_rope_optimized_perf_20261002.json` 和
`pointflow_outputs/fk_local_rope_optimized_v2_perf_20261002.json`。
此后又将分组版的多个 Q gather 合成一次置换，新一轮测速见本文末节，中位耗时仍为 5.65 s。

正确性：

- CPU 共 10 项通过：独立 float64 dense oracle 的输出和梯度、样本隔离、无几何、
  共用/独立 K、跨分组重叠索引、非连续梯度、空索引、逆置换梯度，以及四模态网络无未来信息泄漏。
- A800 BF16 + checkpoint + `torch.compile(fullgraph=True, dynamic=True)` 四类用例通过，
  包含全几何、无几何和混合空 key；核对 eager 与 float64 oracle。
  记录：`pointflow_outputs/fk_local_attention_optimized_compile_v2_20261002.json`。
- Ruff lint/format 与 `git diff --check` 通过。

### 160 维单次生成 attention（2026-10-02）

`POINTFLOW_FK_ATTN_IMPL=lifted160` 已实现；仍需同时开启
`POINTFLOW_FK_LOCAL_ROPE=true`。默认 `partition` 保留原分组实现，两个选项均使用
Flash2 varlen，不增加学习参数，也不改变 head=128 的主干投影和 checkpoint 形状。

只在进入 attention 时将 Q/K 扩成 `[三轴128 | 四轴替换通道16 | 三轴替换通道16]`，
非几何 token 的后两块置零，几何 K 的最后一块取负。这样 G×G 的旧三轴贡献被抵消、
替换为四轴贡献，其余 token 对仍用三轴。V 补 32 个零、输出取前 128 维，
显式保留 `scale=128**-0.5`。生成分支一次 Flash2 varlen；文字因果分支仍独立，
生成 query 对文字 K 的独立 normalization 和跨样本隔离也保留。
完整矩阵推导见 [Q/K/V 矩阵文档](./pointflow_fk_qkv_matrix_20261002.md)第 4 节。

验证记录：

- CPU 19 项测试通过，覆盖 float64 dense oracle 的输出及根输入梯度、共用/独立 K、
  无几何、全几何、混合样本与空分组，以及真实四模态网络的输出等价和无未来目标泄漏。
- A800 BF16、checkpoint、`torch.compile(fullgraph=True, dynamic=True)` 四类用例通过。
  相对 float64 dense oracle，输出最大绝对误差 ≤0.00741、梯度 ≤0.02420；
  同时核对 compiled 与 eager。记录 `pointflow_outputs/fk_lifted160_compile_20261002.json`。
- 新分组 Q 置换的相同 CUDA 检查也通过，记录
  `pointflow_outputs/fk_partition_compile_v3_20261002.json`。
- 8 卡性能测试以此前分组优化 v2 的 5.65 s 中位耗时为参考。新分组版的重复测速在编译阶段
  主动停止，优先测 160 维版；不把被停止的运行作为完整性能结果。

160 维版已完成 30 步并于 12:05:18 UTC 正常退出（exit 0）。收到终止要求后检查，
训练进程已自行结束，GPU 无残留计算进程，没有再启动测试。
配置为 8×A800、每卡 batch=16、累积=1、seed=42、in_order=true、compile 开启、
full activation checkpointing，关闭验证。取第 11–29 步共 19 步：

| 实现 | 中位耗时 | 平均耗时 | 最小–最大 | 分配显存峰值 |
| --- | ---: | ---: | ---: | ---: |
| 已有分组优化 v2 | 5.65 s | 5.6442 s | 5.57–5.71 s | 47.22 GiB |
| lifted160 | 6.26 s | 6.2695 s | 6.20–6.35 s | 45.48 GiB |

本次 160 维版平均步耗时增加约 11.1%，没有实现加速，因此保留 `partition` 为默认。
每卡 batch=16 可运行。扩维增加 QK/V 计算与搬运，不能只由调用次数推断速度；
本轮未对 160 维版做 kernel profile，不将具体瓶颈归因视为已证实。
这是复用此前基线的短训练对照，不是长期吞吐或生成质量评估。
完整逐步记录、配置与比较限制：`pointflow_outputs/fk_lifted160_ab_perf_20261002.json`。

### 分组 Q 一次置换补测（2026-10-02）

按要求补测当前 `partition` 实现：将各组 Q gather 合并为一次置换再切分，
反向用逆置换，避免各组分别生成全尺寸梯度再相加。attention 打分、Flash2 varlen
及局部四轴规则不变。先前 CPU/CUDA、checkpoint、compile 数值检查已通过，本轮只测速。

8×A800、每卡 batch=16、累积=1、seed=42、in_order=true、compile 开启、full AC，
关闭验证，30 步，统计第 11–29 步。配置与此前短测一致，首步 95.45 s 的编译耗时排除。

| 分组版本 | 中位耗时 | 平均耗时 | 最小–最大 | 分配显存峰值 |
| --- | ---: | ---: | ---: | ---: |
| 前版优化 v2 | 5.65 s | 5.6442 s | 5.57–5.71 s | 47.22 GiB |
| 当前 Q 一次置换 v3 | 5.65 s | 5.6489 s | 5.60–5.71 s | 40.26 GiB |

中位耗时相同，平均差约 +0.08%，未测出速度收益；显存峰值降低约 6.96 GiB。
当前版相对关闭局部 RoPE 的 base（中位 5.24 s）仍约增加 7.8%。
此结果复用前版基线，未做交替重复运行，不能将毫秒级差异当成确定的性能变化。
记录含逐步耗时、参数、源码 SHA256 和基线来源：
`pointflow_outputs/fk_partition_v3_perf_20261002.json`。

### Selective activation checkpoint 检查（2026-10-02）

现有 MoT 的 `_apply_selective_ac` 支持 per-op SAC：匹配 `save_ops_regex` 的算子
使用 `MUST_SAVE`，其余使用 `MUST_RECOMPUTE`。当前 recipe 默认 `mode=full`，
`save_ops_regex=[fmha]` 在 full 模式下不使用；切到 selective 时，这个模式
匹配不到本路径实际的 `_flash_attn_varlen_forward`，但默认配置仍可直接运行。
若希望显式保存 Flash2 输出，才需要修改保存规则。下面最初两种测试主动覆盖了默认列表；
仓库默认列表的补测另列在末节，不能将这两种自选策略视为默认 selective。

`tools/check_pointflow_fk_attention.py` 新增 `--selective` 与 `--save-ops-regex`，
使用生产 checkpoint 包装器核对 eager、float64 oracle 的输出和梯度，支持 fullgraph
动态 compile。以下两种策略均通过 A800 BF16 的四类数值用例：

- `[mm,_flash_attn.*forward]`：保存矩阵乘法（含 addmm/bmm）及 Flash2 forward 输出。
  数值记录：`pointflow_outputs/fk_partition_selective_compile_20261002.json`。
- `[_flash_attn.*forward]`：只保存 Flash2 forward 输出，其余重算。
  数值记录：`pointflow_outputs/fk_partition_selective_flash_compile_20261002.json`。

整网使用当前分组 Q 置换实现，8×A800、每卡 batch=16、累积=1、compile 开启，
其他配置与前一轮 full 对照相同：

| 策略 | 中位耗时 | 平均耗时 | 分配显存峰值 | 结果 |
| --- | ---: | ---: | ---: | --- |
| full（前一轮） | 5.65 s | 5.6489 s | 40.26 GiB | 30 步通过 |
| selective：矩阵乘法＋Flash2 | — | — | OOM 时约 78.1 GiB | 首步未完成，超出 80GB 卡容量 |
| selective：仅 Flash2 | 5.70 s | 5.7016 s | 33.08 GiB | 30 步通过 |

只保存 Flash2 的计时窗口为第 11–29 步，共 19 步，范围 5.64–5.88 s，均值比 full
增加约 0.93%。本轮没有测出提速，显存节省不构成切换默认值的理由，因此保持 full。
这是短测且复用已有 full 基线，不把不足 1% 的差异解释为确定的长期速度变化。
记录（包括失败策略与日志来源）：`pointflow_outputs/fk_partition_selective_perf_20261002.json`。

如需开启已验证的 selective 策略，在现有启动指令的 `EXTRA_TAIL_OVERRIDES` 中加入：

```text
model.config.activation_checkpointing.mode=selective
model.config.activation_checkpointing.save_ops_regex=[_flash_attn.*forward]
```

它只影响训练的重算/保存策略，不改变局部 RoPE 的数学定义或 checkpoint 参数形状。

### 仓库默认 selective 补测（2026-10-02）

按要求直接使用默认保存列表：只传入
`model.config.activation_checkpointing.mode=selective`，不覆盖 `save_ops_regex`。
运行落盘配置确认为 `mode=selective, save_ops_regex=[fmha]`，模型实现未修改。

8×A800、每卡 batch=16、累积=1、compile 开启、seed=42、in_order=true，
关闭验证，完成 30 步；第 11–29 步共 19 步统计如下：

| 配置 | 中位耗时 | 平均耗时 | 分配显存峰值 |
| --- | ---: | ---: | ---: |
| full（已有基线） | 5.65 s | 5.6489 s | 40.26 GiB |
| selective，仓库默认 `[fmha]` | 6.00 s | 5.9989 s | 25.06 GiB |

默认 selective 可以运行；本轮平均耗时增加约 6.2%，显存下降，但未获得速度收益。
保留 full 作为现有默认，不继续扩展保存策略。比较复用前面的 full 短测基线，
不是交替重复的长时间吞吐实验。
完整记录：`pointflow_outputs/fk_selective_default_perf_20261002.json`。

### Flash2＋Q/K/V/O selective：单卡接入（2026-10-02）

新增可选 `model.config.activation_checkpointing.save_mm_shapes`。它按 `aten.mm.default`
右侧矩阵的 `[输入宽度, 输出宽度]` 选择额外保存的输出；默认空列表保持原有行为。
规则与 `save_ops_regex` 为“或”的关系，因此本方案不再给名称规则添加宽泛的 `mm`。

当前 Edge 的 Q/O 对应 `[2048,2048]`、K/V 对应 `[2048,1024]`；MLP up/down
分别为 `[2048,9216]` 和 `[9216,2048]`，不会命中。规则按形状而非模块名称筛选，
应用到其他架构时需重新确认；`addmm`、`bmm` 不会被此形状规则选中。

在启动脚本的 `EXTRA_TAIL_OVERRIDES` 中加入以下三项即可开启：

```text
model.config.activation_checkpointing.mode=selective
model.config.activation_checkpointing.save_ops_regex=[_flash_attn.*forward]
model.config.activation_checkpointing.save_mm_shapes=[[2048,2048],[2048,1024]]
```

CPU 3 项测试通过，包含 Q/K/V/O 命中与 MLP 排除、默认行为和名称规则兼容、
OmegaConf/TOML 配置解析，以及输入/参数梯度与未 checkpoint 计算一致。
新增字段最初使用嵌套 tuple，OmegaConf 不支持，在训练启动前发现并修为嵌套 list。

按要求仅使用物理 GPU 1（`CUDA_VISIBLE_DEVICES=1`），未使用 GPU 0。
完整模型单卡 batch=1、累积=1、seed=42，局部四轴分组 attention、compile 开启，
完成 3 步训练，分配显存峰值 39.83 GiB。单卡进程的逻辑 rank=0 对应物理 GPU 1。
运行正常退出（exit 0），GPU 1 已释放。显存使用 PyTorch 峰值计数；原有 monitor
按 LOCAL_RANK 索引 NVML，未处理 CUDA_VISIBLE_DEVICES 重映射，其 NVML 读数不用于本次结论。
本轮是接入冒烟，不作为稳定速度或 8 卡 batch=16 的容量结论；后续八卡结果见下节。
记录：`pointflow_outputs/fk_qkvo_selective_gpu1_smoke_20261002.json`。

### Flash2＋Q/K/V/O selective：八卡 batch=16（2026-10-02）

使用上节形状规则，8×A800、每卡 batch=16、全局 batch=128、累积=1、compile 开启、
seed=42、in_order=true，关闭验证，完成 30 步。取第 11–29 步共 19 步，与已有 full
分组 v3 基线比较，未重跑基线：

| 配置 | 中位耗时 | 平均耗时 | 分配显存峰值 |
| --- | ---: | ---: | ---: |
| full | 5.65 s | 5.6489 s | 40.26 GiB |
| selective：Flash2＋Q/K/V/O | 5.53 s | 5.5321 s | 53.97 GiB |

本轮平均步耗时减少约 2.1%，吞吐约 23.14 samples/s；计时窗口范围 5.46–5.69 s。
分配显存峰值为八卡最大值，reserved 峰值 55.54 GiB，监测时 NVML 最大占用 57.61 GiB。
batch=16 可运行；这是小幅短测收益，尚未做交替重复或长时间稳定吞吐验证，也未评估生成质量。
保留现有 full 默认值，形状筛选策略通过上节三个参数显式开启。
完整记录：`pointflow_outputs/fk_qkvo_selective_8gpu_perf_20261002.json`。

### 统一四模态eval (2026-10-02)

FK包装脚本默认开启`POINTFLOW_FK_UNIFIED_EVAL=true`。PointFlow callback统一选case并调用
`generate_samples_from_batch`一次,取同次video/action/PointFlow/FK结果;独立FK callback不再采样。
统一模式不运行conditional、ablation或16步参考采样,避免把不同条件/不同采样的轨迹拼成一组。
PF/FK各自指标目录保留,`joint_sample.json`共享case身份,`joint_prediction.npz`保存四模态输出。
现有joint dream三列MP4增加同次FK骨架(黄色GT/蓝色预测),无HTML。PF仍用DA3内参,
FK复用原projector,通过现有tracker→canvas仿射叠加。

Dropper mix71保持10%验证划分、7条验证episode×3阶段=21case,加2个训练case。
完整8卡eval＋3步训练通过;23对PF/FK预测数组与共同joint输出逐元素一致,
日志23次joint调用、零独立FK/conditional PF调用。启动eval196s,旧分开评估366s,
仅为单次冒烟结果。正式设置启动eval后每500步一次。
详见`pointflow_outputs/dropper_mix_fk_unified_eval_8gpu_20261002.json`及实验队列E6-dropper-mix。
