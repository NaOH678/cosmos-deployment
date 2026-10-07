# 任务 8：PointFlow RF 加噪、监督与采样约定

任务 8 已实现 PointFlow 的 Rectified Flow 状态和 masked loss。Cosmos 的约定是

`x_σ = σ ε + (1−σ) x_clean`

`v_target = ε − x_clean`

PointFlow 使用同一方向，位移先除以 `pointflow_displacement_scale`（默认 1 米/模型单位）。每个样本共享一个 sigma；暂不支持 diffusion forcing 的逐点时间。

`valid` 只决定监督项。每个有效点的 XYZ 均方误差先在点和时间上平均，再在有有效标签的样本上平均。无标注样本和全空样本不改变分母；无有效标签时返回连接计算图的零损失。

训练循环已在 `_add_noise_to_input` 创建 PointFlowNoised，并写入 PackedSequence；`denoise` 将 noisy state 和 sigma 传给网络，`_compute_losses` 累加 `pointflow_loss_weight × flow_matching_loss_pointflow`。正式 RF loss 不读取 future displacement 来选 anchor 或拓扑。

当前训练开关为 `pointflow_loss_weight` 与 `pointflow_displacement_scale`，属于模型配置。PointFlow 分支仍要求任务 7 的 CP=1、two-way attention。

采样的公共 sampler 仍负责 video/action 时间步；PointFlow 状态应与同一个 sampler sigma 一起更新，保留固定 anchor 的簇拓扑，并在每一步把 point noisy state 传入网络。正式的 `_get_velocity` 变长 PointFlow flatten/split 和 CFG/UniPC 输出接线留在下一轮采样工程验证；本任务已完成训练态 RF 数学、mask 和模型调用接口。

验证：`pointflow_training_test.py` 覆盖 sigma=0/1、valid mask、无有效标签和非法 sigma；与任务 7 回归测试一起通过。真实完整训练 loss、优化器和采样尚未启动。
