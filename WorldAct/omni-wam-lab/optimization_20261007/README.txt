2026-10-07 三路并行优化实验索引

范围
仅离线推理、profiling、历史trace分析；未修改生产启动链路、控制代码或YAML，未启动ROS/下发动作。原18005服务保持。GPU实验串行；原服务驻留且桌面使用GPU，并非完全独占。固定4w EMA、FP8线性层、BF16 attention、UniPC4、CFG3、shift5。

1. backend_optimization
../backend_experiments/summary.json 保存实际测速、输出比较、限制。
同5份最近真实观测，5 warmup +20次无profiler计时，packet到CPU raw actions，不是完整HTTP RTT：
cuDNN dynamic regional P50/P95 441.19/445.84ms
FA4 SM120 dynamic regional 447.94/453.13ms
cuDNN static reduce-overhead +层外clone CUDA Graph 442.25/452.28ms
图trace两请求448次cudaGraphLaunch，真实逐层回放；并非全去噪图。
图候选与当前dynamic平均/最大位置差3.18/6.54cm；FA4为3.49/9.01cm。这是候选输出差，不是对真实动作正确性的评判。图实验同时改变static/mode/clone，不能把数值变化单归因图。同输入穿插其他输入再重复均maxdiff0，不同输入输出会变化。
保留当前cuDNN。未测出替换收益；不部署图候选。错误注入未生效的reduce_overhead计时、带大量debug日志的图计时均排除。
该Omni版本gpu_memory_utilization仅用于paged_scheduler KV预算；当前WAM路径调0.90到0.95不是算力开关。

2. profile_performance
../profiling/README.txt、profile_summary.json、domain_sync_comparison.json、cpu_packet_comparison.json。
独立基线无profiler P50/P95 441.71/443.83ms；带profiler约500ms，不能混用。
profile每请求GPU kernel累计374.16ms：FP8 GEMM208.65ms、cuDNN attention120.65ms、VAE卷积13.49ms。实际H2D 0.22ms。时间可能与CPU重叠，不能相加或直接减去同步等待。
CPU入口验证domain，移除每请求16次重复GPU布尔检查：437.73/441.40ms，20对完整动作逐元素一致。约4ms改善，非数百ms。
避免CPU完整33帧分配：5份观测各40次交错，CPU P50/P95由6.33/8.09到2.54/4.27ms。首帧/action/image_size/metadata完全一致；两边均排除image_size GPU往返，额外收益未知。仅实验原型，生产应抽共享首帧helper，不照搬源码字符串改写。
两项收益未组合测试，不能直接加总为端到端收益。

3. scheduling_review
../scheduling/REVIEW.txt、boundary_report.json、shadow_gate.py、shadow_replay.json、DIAGNOSTIC_PATCH_REVIEW.txt、diagnostic_snapshot_candidate.py。
当前Omni成功会话20261007T080926Z-4fdfb8bc为24步顺序执行，24次块间发布间隔中位575.46ms。没有预取，不能称异步miss。
8/2混合使索引1手已到新模型目标，臂仍滞后；当前臂目标偏移中位1.223cm，历史手臂不协调会话中位5.752cm。目标偏移不是实测误差，不能单独证明抓取失败原因。
下一控制候选：记录请求时完整计划/实际状态/时标；保护有限旧轨迹前缀；新旧连续完整臂手动作在有限时间窗内兼容才接入，共同推进时间。不要按空间最近点任意跳步，也不要用反向运动一律拒绝正常重抓。现有early_splice已保护一个端点，这不是新发现。
原型尚缺速度/加速度/接触阶段完整约束。旧日志缺承诺计划，严格影子回放全拒绝不构成效果证明。需先采集可验证信息；诊断快照明确非原子，不能把队列全称已承诺轨迹。
root独立运行 python -B -m unittest discover -s ../scheduling -p test_*.py，12项通过。

输入
corpus.json记录最新成功轮5份真实观测来源和哈希；observation_01/07/13/19/25.npz。
生产配置维持现状；本次没有宣称解决回弹或提高真机成功率。
