# Cosmos 单卡 5090 推理加速与异步衔接实验记录

跨仓库调用关系与代码入口见 [WorldAct 部署导航](../../WorldAct/COSMOS_DEPLOYMENT_MAP.md)。

更新：2026-10-07。本文汇总本轮已经落地的推理优化、参考论文的 async+blend、线性／smoothstep 对照及原生／Omni 真机对照。历史尝试不代表当前启用项，离线数值检查不代表任务成功率。

## 当前保留配置

优先保留 **单卡 RTX 5090、Omni、50k-retrain/40000 EMA、async+blend、smoothstep** 作为后续重复实验基线。原生后端也已在同一衔接配置下连续运行，无断流；Omni 的响应和重叠时间余量更充足。用户对 smoothstep 首轮反馈“感觉不错”，没有明确确认该轮整项任务成功，不能写成成功率结论。

| 项目 | 当前值 |
|---|---|
| 权重 | `WorldAct/model_ch/real_ckpt/singlerighthand-edge-droid-50k-retrain-v1-0831/iter_000040000` |
| 模型配置 | 同一 run 的 `config.deploy.yaml`，不是 16n 的配置 |
| 原生/HTTP 推理源码入口 | `WorldAct/WorldAct-sft` |
| 服务配置 | `WorldAct/WorldAct-sft-pointflow-fk/examples/deployment/cosmos_singlerighthand_50k_retrain_v1_edge_protocol_v2.yaml` |
| 客户端配置 | `src/wuji_data_pipeline/config/cosmos_protocol_v2_50k_retrain_async_blend.yaml` |
| 预测 | 32 个未来动作，15 Hz |
| 请求间隔 | `paper_async_stride_steps: 16`，按观测时间原点调度，单在途请求 |
| 混合 | `paper_async_weight_curve: smoothstep`，臂手共用权重 |
| 发布 | 120 Hz，位置线性插值、姿态 SLERP；不是模型每秒推理 120 次 |
| 旧固定锚点混合 | 臂／手均为 0，不叠加历史 8/2 |
| 其他旧机制 | 固定跳步、旧预取自适应、early-splice、客户端 Butterworth、RTC、额外 settle 均不参与 |

服务配置位于 PointFlow/FK 工作树，不表示本轮启用了 PointFlow/FK 联合去噪。本文记录普通 video/action WAM；后续新增模态需另行验证。

## 一、推理加速：做了什么

Omni 路径使用 vLLM-Omni 的 WAM 实现及其依赖；没有把机器人动作生成当成文本 LLM 请求，也没有另加一个文本服务。当前采用 FP8 线性层、cuDNN 注意力、串行 CFG、4 步采样。基础计算 dtype 与量化层不是同一概念，不能称整个模型都是 BF16 或都是 FP8。FP8 与原生数值等价尚未确立。

| 优化 | 状态及作用 |
|---|---|
| 首帧处理 | 已接入：利用只需真实首帧作为视觉条件的路径，减少无效处理；CPU packet 只构建所需真实首帧 |
| CPU domain 验证 | 已接入：在可信 CPU 入口验证，减少重复 GPU 标量检查和同步；不是取消输入验证 |
| FP8 线性层 | 当前 Omni 使用；早期测速显示明显收益，但引入数值差异，不能用速度证明动作等价 |
| cuDNN 注意力 | 当前保留；本机实测 FA4 未更快，未切换 FA4 |
| compile | 保留经过测试的路径；并非 compile 越大越快，全静态实验未获得收益 |
| 完整 GEN CUDA Graph | 已接入 28 层 GEN 网络及尾部 norm 的整段重放；不是把全部预处理和整个去噪循环捕获成一张图 |
| 实际尺寸预热 | head 480×640、right_wrist 480×848；在 ready 前完成实际 token/text shape 的图捕获，避免首请求临时捕获 |
| 合批／双卡 CFG | 本轮当前单卡服务不使用；不能把历史双卡或合批实验收益算到当前配置 |
| GPU memory utilization、H2D overlap | 不能仅凭可配置就宣称获得收益；本轮没有独立证据将其列为已验证加速项 |

首帧处理涉及两个不同层次：模型侧首帧优化和后续 CPU packet 构建优化。数值一致性只适用于各自受控对照，不能外推到整个 Omni 与原生实现之间。

### 速度证据与边界

1. 早期固定观测离线实验：native 完整调用中位数 621.93 ms，Omni BF16 first-frame 590.21 ms，FP8 first-frame 440.54 ms。计时排除了 HTTP/ROS，native 和 Omni 的预处理起点不完全一致，不能当成严格的端到端同比。证据：[speed_summary.json](../../WorldAct/omni-wam-lab/results/speed_summary.json)。
2. 注意力对照：cuDNN dynamic 440.18 ms，FA4 dynamic 448.44 ms；小观测集上的 cuDNN 441.19 ms、FA4 447.94 ms。保留 cuDNN。早期逐层 graph 候选没有可用收益，不能与后来完整 GEN graph 混为一谈。证据：[backend summary](../../WorldAct/omni-wam-lab/backend_experiments/summary.json)。
3. 组合优化 ABBA：5 份观测、每种方案 40 个正式请求，HTTP 中位数 501.21→494.50 ms，P95 513.16→506.75 ms；60 对 wire 动作逐元素相同。约 6.71 ms 的收益属于该组合相对旧 Omni 的增益，不能宣称每项都独立贡献同样的收益。录制关闭、JPEG 编码未计入，尾部最大值没有改善。证据：[组合 summary](../../WorldAct/omni-wam-lab/combined_optimization/full_graph_combo/summary.json)。
4. 生产模块离线接入验证：录制和 latent 开启，20 个正式请求，20 对 wire 输出一致，25/25 观测与 latent 写入，零 drop/error；2 个图在 ready 前捕获，实际请求没有新增冷捕获。该轮没有同条件录制开启的基线，不用于单独证明提速。证据：[integration_summary.json](../../WorldAct/omni-wam-lab/combined_optimization/production_validation/integration_summary.json)。
5. 最后两轮真机：Omni 服务端推理中位数约 411 ms，原生约 591 ms；完整 RTT 约 435／607 ms。与上述离线测试输入、边界和负载不同，必须分表比较。

### 代码与回退

- 生产 Omni 入口：`WorldAct/WorldAct-sft/cosmos_framework/inference/robot_policy/omni_http.py`。
- CPU packet：同目录 `wam_packet.py`；完整 GEN graph 辅助：`WorldAct/omni-wam-lab/wam_gen_graph.py`。
- vLLM-Omni 固定工作树：`WorldAct/omni-wam-lab/vllm-omni-d5a3380103df2fc827f095c60dbfb6e1c5655fd3`。
- `WAM_FULL_GEN_GRAPH=0` 关闭完整 GEN graph；`WAM_CPU_FIRST_FRAME_PACKET=0` 恢复原 CPU packet 路径。服务启动时读取，需要重启服务，attach 不会改变已有进程。
- 新相机尺寸可能触发新图捕获，换尺寸后必须核对 warmup 与实际输入，不应直接沿用旧首请求耗时结论。

## 二、参考论文的 async+blend：实际实现

参考 [World Action Models in Real Time，第 3–4 节](https://arxiv.org/pdf/2608.01880)。对应的是 **async+blend：新块生成后，在时间对齐的重叠区对最终动作加权**，不改变训练和去噪。不是论文的去噪中混合 simple，也不是 RTC infer 或需要训练的 prefix-conditioned 方法。论文未明确给出 async+blend 的唯一权重曲线；我们先用线性，再单变量改成 smoothstep。

### 时间轴

1. 首块保持从首个返回动作执行，不因首次加载／推理延迟跳过首块。首块请求时间表以执行原点建立。
2. 后续观测以最旧有效相机帧为保守原点，用相邻采样的 ROS／monotonic 时钟对转换。记录各相机年龄和相互偏差；这不代表相机与机器人状态完全同步。
3. 下一请求最早在上一观测原点加 `16/15` 秒触发，单在途、单 pending；迟到不补发积压请求。不是从新块激活后再执行 16 步，也不是用固定剩余步数触发旧式预取。
4. 服务端去掉第 0 行条件状态后返回 32 个未来动作，因此返回行 0 对应 `观测原点 + 1/15 秒`。按实际时间采样，不能直接拿请求构建耗时当固定跳步数。
5. 新块返回后保留正在插值到的旧端点；之后在旧 deadline 上采样新轨迹，融合实际已安装的旧计划，包括此前混合过的部分。
6. 重叠长度取实际新旧支持区间交集；旧尾段结束后使用新轨迹。尾段耗尽则沿既有等待路径处理，不外推动作。

### 权重

令 u 为受保护端点到重叠末端的归一化时间，范围 0–1。位置与手指目标：

`a = (1 - w_new) * a_old(t) + w_new * a_new(t)`

- 首轮线性：`w_new = u`。
- 当前 smoothstep：`w_new = 3*u*u - 2*u*u*u`。
- 手和臂共用 w；姿态用相同 w 做 SLERP。
- 旧目标沿旧轨迹移动，不是固定旧锚点；不叠加原来臂 8／手 2 的 smoothstep。
- smoothstep 使权重两端斜率为零，但不保证整个离散目标轨迹速度／加速度连续，也可能将纠偏集中到中段。

### 与论文不同的适配，不能省略

论文实验 H=24、10 Hz、s=4；当前保持模型 H=32、15 Hz，并选择 s=16。线性／smoothstep 曲线、四元数 SLERP、多相机最旧时标、受保护端点和首块启动均为本地实现选择。论文中的 s 与本实现观测原点调度的工程定义必须按上述时间轴理解，不能宣称逐行复现论文全部部署细节。

代码：`deployment_node.py` 中 opt-in paper 分支、`paper_async_blend.py`、`paper_observation_clock.py`，均位于 `src/wuji_data_pipeline/wuji_data_pipeline/`。smoothstep 增加后 92 项离线回归通过；包含两种曲线的实际发布回调模拟、时间轴和端点、共同权重、容量超限安装前拒绝、时标回退与生命周期清理。完整 `test_deployment.py` 因宿主缺 `wujihand_msgs` 未运行，不能称全项目测试通过。

## 三、关键真机会话与结论

会话根目录：`datasets/tianji_wuji/diagnostics/cosmos_runs/`。run ID 使用 UTC；例如 114022Z 对应北京时间 19:40:22。以下均为单卡 5090、50k/40000。

| run ID | 配置/变量 | 推理中位 ms | RTT 中位 ms | 最长发布间隔 ms | >100ms 缺口 | 现场反馈/结论 |
|---|---|---:|---:|---:|---:|---|
| `20261007T105136Z-05bfe0ed` | Omni，24步顺序8/2，settle=0 | 409.59 | 432.42 | 507.40 | 21 | 成功，但关闭100ms settle后差别不明显；顺序等待仍暴露 |
| `20261007T112817Z-54fbff90` | Omni，async+blend，线性 | 412.77 | 434.97 | 17.66 | 0 | “停顿缓解了挺多，但是有一点块状感”；33次交接，无断流／拒绝 |
| `20261007T114022Z-e6651399` | Omni，仅权重改smoothstep | 411.10 | 434.66 | 15.68 | 0 | “感觉不错”；27次交接，无断流／拒绝；保留基线 |
| `20261007T115439Z-3ae236df` | 原生，沿用smoothstep | 590.75 | 607.13 | 22.08 | 0 | 49次交接，无断流／拒绝；尚无用户明确任务结果反馈 |

后三轮没有将用户未明确确认的结果写为“任务成功”。会话数和运行时长不同，也不能用绝对异常次数直接比较成功率。

| 衔接指标 | Omni线性 | Omni smoothstep | 原生 smoothstep |
|---|---:|---:|---:|
| 剩余重叠区中位数 | 0.53 s | 0.53 s | 0.33 s |
| 相机到交接中位数 | 495.74 ms | 490.90 ms | 679.72 ms |
| 交接处目标速度变化P95 | 0.219 m/s | 0.160 m/s | 0.183 m/s |

速度变化是相邻15Hz目标段速度向量之差的模，不是机器人实测速率或加速度。不同轮次的观测、动作、环境、时长不完全相同，不能把指标下降全部归因于权重或数值后端。原生最高 RTT 840 ms 在首块，不能记成运行中的块间断流。

原生与 Omni 使用相同客户端语义配置，差异仅运行记录目录和run ID；Omni录制latent而原生本轮没有，因此也不是严格相同录制负载的性能基准。

### 每轮证据文件

- 顺序：`20261007T105136Z-05bfe0ed/analysis_settle_zero.json`。
- 线性：`20261007T112817Z-54fbff90/analysis_async_blend_first_run.json`。历史轨迹的smoothstep反事实P95约0.153 m/s是离线计算，不是下一轮实测0.160 m/s。
- smoothstep：`20261007T114022Z-e6651399/analysis_smoothstep_first_run.json`。
- 原生对比：`20261007T115439Z-3ae236df/comparison_native_vs_omni_smoothstep.json`。
- 每轮 `deployment.yaml`、`manifest.json`、`client/deployment_trace_*.jsonl`、`client/deployment_state_*.jsonl`、`server/` 保留实际配置、动作和状态；Omni `--record-video` 另保留 `video_latents/`。

新事件 `paper_async_request_clock` 记录时间原点，`paper_async_activate` 记录 old_plan/raw_prediction/installed_plan、权重、采样索引和交接时间，`paper_async_activation_rejected` 记录拒绝，`paper_async_underrun` 记录尾段耗尽。smoothstep 首轮旧版 recording audit 报28次splice缺少旧格式stage snapshots，但对应28个paper事件均有三阶段轨迹；不能直接把该提示当成轨迹未保存。

## 四、启动、对照及回退

在 `wuji-hand-teleop-pipeline` 目录运行，沿用一个参数化脚本：

```bash
COSMOS_DEPLOYMENT_CONFIG=../WorldAct/WorldAct-sft-pointflow-fk/examples/deployment/cosmos_singlerighthand_50k_retrain_v1_edge_protocol_v2.yaml \
./src/scripts/start_local_cosmos_deployment.sh \
  --backend omni \
  --port 18006 \
  --record-video \
  --checkpoint-dir ../WorldAct/model_ch/real_ckpt/singlerighthand-edge-droid-50k-retrain-v1-0831/iter_000040000 \
  --model-config-file ../WorldAct/model_ch/real_ckpt/singlerighthand-edge-droid-50k-retrain-v1-0831/config.deploy.yaml \
  --config /home/wuji/ros2_ws/src/wuji_data_pipeline/config/cosmos_protocol_v2_50k_retrain_async_blend.yaml
```

- 原生对照：改成 `--backend native`，移除 `--record-video`，加 `--model-package ../WorldAct/models/cosmos3-edge-droid`。客户端配置保持不变。当前脚本只支持 Omni latent 录制，原生仍默认保存观测／动作／trace。
- 线性对照：同一客户端 YAML 将 `paper_async_weight_curve` 改为 `linear`；代码缺省值仍为 linear。无需创建额外启动脚本。
- 顺序对照：`--config` 改为 `cosmos_protocol_v2_50k_retrain_single24_sequential_blend8_hand2.yaml` 的同目录容器路径；该文件保持顺序24步、臂8／手2、settle=0。这同时改变执行窗口和衔接方式，不能称只改了权重。
- 服务参数修改需重启服务；客户端调度修改需重启部署。`--attach-existing` 不会改变既有服务后端或为其补开latent录制。

## 五、阶段判断与后续规则

推理加速把本轮模型延迟从原生约591ms降至Omni约411ms；async+blend让两者在上述会话均没有百毫秒级发布缺口。两条优化分别解决耗时和执行重叠，不能合并归因。

先保留 Omni＋smoothstep 重复实验，记录完整任务结果、是否需要人工帮助、失败阶段、回弹与臂手协调。若后续修改，每轮只改变一个可解释因素并保留原配置快照。当前没有足够重复实验估计成功率，也没有证据将所有抓取误差归因于延迟或将其宣称已解决。


## 六、2026-10-07 按 feature 整理提交

提交署名沿用 `shichaojian <shichaojian@pjlab.org.cn>`，仅对本次提交生效，不修改其他工作树的默认身份。目标远端为当前跟踪的 `dexmanip/tianji-wuji-pipeline-refactor`。本节列出功能拆分，实际推送状态以远端为准。

| 提交 | 功能 |
|---|---|
| `78de3ce` | 关联录制、trace 写入状态及归档审计 |
| `e1f9f57` | EEF 衔接模式、独立臂手混合、实验性前缀与可回放动作诊断 |
| `24c674d` | 论文 async+blend、观测时钟、线性/smoothstep 与测试 |
| `2de0fb2` | 参数化本地启动器、历史对照和当前部署 profile |
| `d1d79dd` | 离线时序审计与衔接候选回放 |
| `37dd170` | 推理测速、CFG 对照、profile 工具 |
| `9bc8ff8` | 独立 Lingbot pointflow Dropper step50000 配置 |

文档和跨仓库入口另作独立提交。WorldAct 工作树不在本次 pipeline 提交范围；克隆 pipeline 不等于同时取得那些源码。

整理时额外验证：拆分后的衔接基础中间版本135项测试通过；最终控制相关测试在具备消息依赖的ROS容器中176项通过、1项跳过；FDM相关27项通过。这补齐了早期宿主缺少 `wujihand_msgs` 时未能运行的 `test_deployment.py`，没有启动真机。外部WorldAct录制契约测试在缺少兄弟仓库时明确跳过，避免独立克隆pipeline时收集失败。

本地保留：空文件 `--extra-on-dst`；`src/scripts/benchmark_cosmos_rtc_vjp.sh`（依赖本地 `datasets/tianji_wuji/diagnostics/rtc_offline_20261007/benchmark_exact_vjp.py`，尚未形成自包含工具）；所有数据、权重和运行产物。历史RTC真机包装脚本保留作实验入口，依赖其注释中的本地已预热服务与临时凭据文件，当前推荐仍为通用启动器。

嵌套 `src/wujihandros2` 不提交或推送，已有 `patches/wujihandros2-supervised-enable.patch` 与其当前四个修改文件的完整diff逐字一致，反向应用检查通过。`src/wuji-retargeting` 及其子仓库的本地状态也保留。
