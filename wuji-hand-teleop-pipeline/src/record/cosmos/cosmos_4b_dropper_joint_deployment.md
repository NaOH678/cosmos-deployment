# Cosmos 4B Dropper Joint 部署记录

## 一、重点配置

模型：

- `model_id`: `singlerighthand-dropper-edge-droid-50k-aot-iter-000030000`
- `task`: `draw liquid from the beaker with a dropper and dispense it into the test tube`
- `action_rate_hz`: `15`
- `publish_rate_hz`: `120`
- `wire_chunk_size`: `32`
- `replanning_horizon`: `16`（预测 32 步，执行 16 步）

Joint 配置：

- `arm_command_mode`: `joint`
- `action_space`: `joint`
- 右臂输入：7D 绝对关节位置，rad
- 右手输入：20D 绝对关节位置，rad
- 右臂输出：`arm_joint_action_right`，7D，rad
- 右手输出：`hand_action_right`，20D，degree
- 插值：`linear_joint`
- 本地 `action_smoothing_method`: `none`
- `boundary_blend_method`: `smoothstep`
- `boundary_blend_steps`: `4`
- `initial_blend_steps`: `0`

Prefetch：

- `prefetch_min_lead_actions`: `7`
- `prefetch_initial_lead_actions`: `8`
- `prefetch_max_lead_actions`: `9`
- `prefetch_latency_window_size`: `100`
- `prefetch_min_latency_samples`: `5`
- `prefetch_safety_margin_actions`: `2`
- `time_alignment`: `true`

通信：

- cameras：`head`、`right_wrist`
- 图像 key：`observation["images"]["head"]`
- 图像 key：`observation["images"]["right_wrist"]`
- `inference_timeout_ms`: `31000`
- `max_observation_age_sec`: `1.06`

ROS 发布：

- topic：`/right_arm/external_joint_target`
- type：`sensor_msgs/JointState`
- position：7D，rad
- joint 顺序：`right_joint_1` ～ `right_joint_7`
- Joint 模式不发布 EEF 控制目标，EEF 仅保留作诊断字段

Joint 位置限制（degree）：

- lower：`[-170, -120, -170, -140, -170, -60, -90]`
- upper：`[170, 120, 170, 78, 170, 60, 90]`

当前本地动态检查：

- `joint_step_velocity_validation_enabled`: `false`
- `acceleration_limit_deg_s2`：未配置，本地加速度预检查关闭
- 仍保留：shape、finite、关节位置限位、inactive-side hold 检查
- 最终发布速度限制：`180 deg/s`
- Tianji 控制器自身的限位和保护没有修改

与 EEF 版本的主要区别：

- EEF 输出 `ee_pos + ee_quat`；Joint 输出 7 维绝对关节目标。
- EEF 使用位置线性插值和姿态 SLERP；Joint 直接对 7 维关节做线性插值。
- EEF 发布 external EEF target；Joint 发布 `external_joint_target`。
- EEF 检查末端位置和姿态跳变；Joint 按关节空间处理。
- Joint 模型不得把前 7 维输出解释成 EEF position + quaternion。

## 二、问题及解决方案

1. **云端握手报 `COSMOS_PROFILE_MISMATCH`**

   原因：本地已经使用 Joint `robot_layout`，但云端仍声明为 EEF。

   解决：云端 manifest、模型输入适配、输出适配和协议响应全部切换为 Joint；响应使用 `arm_joint_action_right`，不再解释为 EEF。

2. **READY 后立即变成 DISABLED**

   原因：`DISABLED` 和 `HAND SAFE` 只是停机结果，真正原因在之前的 trace 中，主要出现过 observation 超时和 Joint chunk 动态检查失败。

   解决：通过 `deployment_trace` 中的 `pending_chunk_activation_failed`、`policy_request_failed`、`TARGET_HOLD` 等事件定位实际错误。

3. **Observation 到达时过旧**

   原因：旧 Prefetch 提前量过大，chunk 激活时 observation age 超过限制。

   解决：`max_observation_age_sec` 调为 `1.06`，Prefetch 改为 `min=7`、`initial=8`、`max=9`。实测激活 observation age 约为 0.59～0.66 秒。

4. **怀疑云端推理过慢**

   现象：实测云端推理约 238～253 ms，总 RTT 约 380～411 ms。

   结论：推理耗时不是本轮立即停机的主要原因，主要问题是跨 chunk 的 Joint 轨迹不连续。

5. **Joint step、速度和加速度反复超限**

   原因：每次重新规划产生的绝对关节轨迹在 chunk 接缝处不连续；单纯把阈值从 5°、8°、12°或 450、700、900 deg/s² 逐步放宽，只会延后报错。

   处理：保留 smoothstep 4 步边界混合；按当前联调要求关闭本地 step/velocity 和 acceleration 的整块预检查。Tianji 控制器保护仍然生效。跨 chunk 连续性仍应由云端轨迹或后续连续轨迹拼接从根本上解决。

6. **旧 EEF 任务没有出现相同报错**

   原因：旧版本只检查 EEF 位置和姿态跳变，不检查关节速度、加速度，因此不能说明旧轨迹在关节空间连续。

   解决：Joint 版本必须使用 Joint action、`linear_joint` 插值和 Joint ROS topic，不能复用 EEF 的校验和执行语义。
