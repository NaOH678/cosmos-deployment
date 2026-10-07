# PI0.5 Dropper Joint 本地部署记录

实现基线：当前分支的 protocol-v2 Joint 动作支持与 PI05 Dropper profile

配置文件：`src/wuji_data_pipeline/config/pi05_protocol_v2.yaml`

启动脚本：`src/scripts/start_pi05_deployment_session.sh`

## 一、重点配置

### 任务与模型

- 任务：Dropper
- checkpoint：`checkpoints/pi05_singlerighthand_dropper_100_joint/dropper_joint_4gpu_v3/30000`
- prompt：`draw liquid from the beaker with a dropper and dispense it into the test tube`
- 协议：`pi_v2`，protocol-v2 HTTP
- API：`POST /v1/robot-policy`
- API key：仅从环境变量 `PI05_HTTP_API_KEY` 读取

### 机器人范围

- `active_arm=right`
- `active_hand=right`
- 左臂在 Recovery 后垂直停放，不执行云端动作；左手不启动
- 右臂命令模式：`joint`
- Tianji 硬件模式：`impedance`
- 阻抗速度比/加速度比：`30% / 30%`

### Joint 动作语义

- `arm_action_right.joint_pos`：7 维绝对关节角，单位 rad
- `hand_action_right`：20 维绝对关节角，单位 degree
- 云端返回 50 步、54 维双侧动作；本地只执行右臂 7 维和右手 20 维，忽略左侧 hold 动作
- `expected_arm_action_space=joint_position`
- 右臂 Joint 目标直接进入 `/right_arm/external_joint_target`，不经过 EEF 和 IK

### 与 EEF 版本的主要区别

| 项目 | EEF | Joint |
|---|---|---|
| 机械臂动作 | 位置 3 维（m）+ 四元数 4 维（xyzw） | 绝对关节角 7 维（rad） |
| 执行路径 | EEF → IK → Joint | 直接 Joint target |
| 插值 | `linear_slerp` | `linear_joint` |
| 主要安全检查 | 位置和四元数跳变 | 关节限位、30 Hz 相邻速度、120 Hz 发布速度 |

右手单位仍为 degree，没有随右臂改成 rad。

### 频率与 Chunk 调度

- 模型动作频率：30 Hz
- 本地硬件发布频率：120 Hz
- Tianji 状态读取/发布频率：500 Hz
- 模型每次返回：50 步
- `open_loop_horizon=30`，即 50/30
- 异步预取：启用
- 初始预取提前量：5 步
- 自适应提前量范围：3～9 步
- P99 RTT 安全余量：2 步
- 预取时间对齐：启用
- Chunk 边界融合：`smoothstep`，6 步，约 200 ms
- 首段融合：0 步
- 插值：`linear_joint`
- Butterworth 低通：关闭
- PCHIP：关闭
- `velocity_continuous` 边界融合：关闭

### 启动交接与恢复

- handoff hold：0.2 s
- handoff ramp：1.0 s
- handoff 超时：3.0 s
- WujiHand deployment 渐入：1.0 s
- Recovery 最大速度：5 deg/s
- Recovery 最大加速度：10 deg/s²

启动采用两次推理：第一段只提供固定接入目标；控制器完成 handoff 后丢弃第一段，再用最新观测推理，第二段才启动 30 Hz 策略时钟。

### Joint 安全值

- 关节下限（deg）：`[-170, -120, -170, -140, -170, -60, -90]`
- 关节上限（deg）：`[170, 120, 170, 78, 170, 60, 90]`
- 30 Hz 轨迹速度上限：每关节 180 deg/s
- 120 Hz 最终命令速度上限：每关节 180 deg/s
- 等效 30 Hz 相邻最大步长：6 deg
- 超限时拒绝整个 chunk，不做裁剪
- 已删除“跟随误差超过 10° 持续 0.25 s 自动 Standby”的附加 watchdog

### 相机与通信

- 相机：`head + right_wrist`
- 相机链路：direct shared memory，不经过 ROS 图像话题
- 网络图像：JPEG，quality=90
- 请求超时：1.0 s
- Observation 最大允许年龄：0.5 s
- Reconnect 间隔：1.0 s
- READY 后策略启动超时：2.0 s

## 二、出现过的问题及处理

1. **EEF 和 Joint 同为 7 维，语义容易混淆**

   Hello 必须协商并锁定 `arm_action_space=joint_position`；Joint 响应只能使用 `joint_pos`，严格检查维度、finite、单位和绝对目标语义，禁止进入 EEF/IK 路径。

2. **使用错误任务、checkpoint 或动作空间后机械臂动作异常**

   本地固定核验 Dropper 30000 的完整 model ID；Hello 和每个 action response 均须保持 `joint_position`。Model ID、action space、session ID 或 request ID 不一致时拒绝动作。

3. **READY 后第一段动作抖动，策略时间轴在底层 handoff 期间提前消耗**

   增加两次推理启动屏障：第一段仅用于 handoff 并丢弃，handoff 完成后用最新观测重新推理，第二段才启动 30 Hz 策略时钟。

4. **第一次抓取经常失败，后续抓取正常**

   原因是 WujiHand 的 5 s 渐入使模型已开始执行而真实手尚未跟上。Deployment 手部渐入已改为 1.0 s，连续测试中首次抓取成功率明显恢复。

5. **Chunk 边界回弹、停顿或过度平滑**

   最终保留 50/30、异步 pending 预取、时间对齐和 6 步 smoothstep；恢复 `linear_joint`，关闭 Butterworth、PCHIP 和 `velocity_continuous`，避免平滑过度。

6. **READY 后突然出现 `DISABLED / HAND SAFE`**

   曾加入的“关节跟随误差大于 10° 持续 0.25 s”不适合实际硬件跟随速度，现已删除。另一次停止由约 0.702～0.703 s 的响应超过 0.5 s observation 年龄上限引起；当前单次过期仍会触发 PI Joint Standby。建议后续改为丢弃过期响应、进入 `TARGET_HOLD` 并立即重请求，连续失败后才 Standby；该项尚未完成。

7. **第二次启动出现静止动作，怀疑沿用上一任务状态**

   云端没有旧 chunk、observation history 或 task progress 缓存，session/request 身份也没有串线；但不同 session 共用 Policy 实例及连续 JAX RNG。由于更高 step checkpoint 未明显复现，30000 checkpoint 的鲁棒性或随机采样更可疑。建议云端按 session 重置 RNG，并记录首次图像哈希、qpos 和前几个 action。

## 备注

“Latent 视频保留已撤掉的道具、bootstrap 图像提前约 38.8 s 上传”属于 LingBot-VA FDM 启动路径，不属于本 PI `pi_v2` Joint 路径，不应混入 PI 问题记录。
