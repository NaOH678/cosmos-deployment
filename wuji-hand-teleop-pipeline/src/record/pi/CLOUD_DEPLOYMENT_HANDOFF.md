# Tianji + Wuji 云端部署交接

> 本文是早期 π0.5 sandwich 部署的历史实现记录，不能原样作为当前 Dropper Joint、LingBot-VA、
> Cosmos 等模型的通用接入契约。新模型先阅读
> [`CLOUD_MODEL_INTEGRATION_GUIDE.md`](../../wuji_data_pipeline/CLOUD_MODEL_INTEGRATION_GUIDE.md)，再把本文
> 作为 50/40、预取、插值和实机诊断的已验证实例参考。

本文记录机器人端当前已经实现的云端策略接口。内容以 2026-08-05 的
`refactor/staged-data-pipeline` 分支为准，用于机器人端和云端定位协议、维度、
时序与安全问题。

## 1. 当前架构

正式云端推理通过在线服务平台暴露的 HTTP/1.1 Service 访问：

```text
三路相机进程 ──共享内存──┐
Tianji 状态 ───ROS 2─────┼→ deployment_node
WujiHand 状态 ─ROS 2─────┘        │
                                  ├─ JPEG + pickle + HTTP POST
                                  ↓
                          π0.5 在线推理服务
                                  │
                                  └─ 50 步 action chunk
                                          ↓
                        时间对齐后顺序执行 40 步
                                          ↓
                         30 Hz 航点 → 120 Hz 插值
                                          ↓
                       EEF 目标 → Tianji IK/阻抗控制
                       手部角度 → WujiHand 驱动
```

云端不访问 ROS 2、Marvin SDK 或 WujiHand SDK。Tianji IK、Recovery、Enable、
watchdog 和最终硬件安全状态均在机器人端。

本地 Replay/LAN 测试仍支持 ZMQ `tcp://`。HTTP 与 ZMQ 使用同一套 protocol-v2
mapping；变化只在传输层。

## 2. 机器人端入口

```bash
cd /home/pjlab/ros2_ws/worktrees/wuji-hand-teleop-pipeline

read -rsp 'PI05 HTTP API key: ' PI05_HTTP_API_KEY
echo
export PI05_HTTP_API_KEY

./src/scripts/start_deployment_session.sh \
  http://SERVICE_HOST \
  right
```

第二个位置参数同时选择机械臂和手。只支持：

- `right`：右臂 + 右手；左臂在 Recovery 时缓慢停到垂直位置；
- `left`：左臂 + 左手；
- `both`：双臂 + 双手。

不支持“双臂 + 仅右手”等混合模式。正式 sandwich 模型需要 `head` 和
`right_wrist` RGB，不能使用 `--no-camera`；该选项只用于 hold/协议检查。

会话按键：

```text
r  Recovery
a  云端握手成功后使能 Tianji 和选中侧 WujiHand
x  关闭手并请求 Tianji standby
q/e/Ctrl+C 关闭手、请求 standby、停止全部会话子进程
```

WujiHand 在 Prepare/Recovery 阶段保持失能。Tianji 到达 READY 后才使能手，
第一条手部命令从实测位置用 5 秒渐入。

## 3. HTTP 服务契约

机器人端向固定路径发送：

```text
POST /v1/robot-policy HTTP/1.1
Content-Type: application/octet-stream
Accept: application/octet-stream
Authorization: Bearer <API_KEY>
X-OpenPI-API-Key: <API_KEY>
```

body 是 `pickle.dumps(mapping, protocol=pickle.HIGHEST_PROTOCOL)`。成功响应必须：

- HTTP 200；
- 提供准确的 `Content-Length`；
- Content-Type 为 `application/octet-stream` 或
  `application/x-python-pickle`；
- body 是完整的 pickle mapping，不能只返回 `action_chunk`；
- 保持 HTTP/1.1 persistent connection。

API key 只从 `PI05_HTTP_API_KEY` 环境变量读取，不写进 YAML、URL 或日志。
客户端不读取系统代理变量、不跟随 3xx，并限制最大响应大小。

## 4. 握手与请求 identity

Enable 之前机器人会发送 `hello`。服务回复至少包含：

```python
{
    "protocol_version": 2,
    "message_type": "hello_ack",
    "session_id": request["session_id"],
    "request_id": request["request_id"],
    "model_id": "checkpoints/pi05_singlerighthand_sandwich_100/sandwich_v1/99999",
    "action_rate_hz": 30.0,
}
```

每个 action 响应也必须原样返回当前请求的 `session_id` 和 `request_id`。机器人会
在提取 action 前严格检查 protocol、identity、model ID 和 action rate。超时或
协议错误后连接会关闭，缓存动作清空，并重新握手；旧连接的响应不会进入新计划。

握手状态检查：

```bash
ros2 service call /wuji_deployment/ready std_srvs/srv/Trigger "{}"
```

运行状态：

```bash
ros2 topic echo /wuji_deployment/status
```

重点字段为 `server_ready`、`last_server_error`、`last_model_id`、`requests`、
`failures`、`last_latency_ms`、`last_observation_age_ms`、
`last_server_inference_ms` 和 `action_plan.buffered_actions`。

## 5. Observation

状态维度与训练数据保持一致：

```text
qpos = arm_left(7) + hand_left(20) + arm_right(7) + hand_right(20) = 54
eef  = eef_left(7) + eef_right(7) = 14
```

手部关节状态单位为 rad；EEF 为对应 chest frame 下的绝对
`[x, y, z, qx, qy, qz, qw]`。单侧模式仍发送双侧协议结构，未接入侧手部状态用
20 维零值补齐。

图像 mapping 按配置顺序包含 `head`、`right_wrist`。相机采集、GUI 和 Recorder
使用共享内存中的原始帧；deployment worker 只对网络副本做 JPEG 编码。云端解码
后得到 OpenCV BGR `uint8`，模型 adapter 必须按训练预处理转换 RGB、resize/crop
和 normalization。当前深度与红外只是数据分析附件，不上传给该 RGB 模型。

## 6. Action

云端固定返回完整 50 步 action chunk。每一步是完整双侧 mapping：

```text
eef_left(7) + hand_left(20) + eef_right(7) + hand_right(20) = 54
```

其中：

- EEF 仍是 chest frame 下的绝对 `xyz + xyzw`；
- 手部 action 单位为 degree；
- 即使模型只预测右侧，云端也必须用当前观测生成左侧 hold action，补齐双侧结构；
- 响应 `action_rate_hz` 必须为 `30.0`；
- 模型输出必须在返回前完成反归一化。

机器人会预先校验全部 50 步的维度、finite、四元数和相邻跳变。当前保留的部署
协议检查包括单步 EEF 位移不超过 0.15 m、旋转不超过 75°、手关节角跳变不超过
75°。这些是云端动作协议检查，不是已从 Tianji 原生遥操路径删除的额外 IK 保护。

## 7. 当前 50/40 执行语义

当前配置：

```text
model action chunk: 50 steps
robot open_loop_horizon: 40 steps (current 50/40 comparison)
action_rate_hz: 30 Hz
Tianji control/publish check: 120 Hz
Tianji SDK state read: 500 Hz
```

机器人每轮执行时间对齐后的 40 步，但 active plan 不会被新响应覆盖。HTTP网络线程维护最多
一个在途请求和一个 pending chunk。预取提前量根据最近100次完整请求的P99计算：

```text
prefetch_lead = ceil(P99_complete_RTT_ms / 33.33 ms) + 2
范围：[3, 9]，累计不足5次样本时使用初始值5
```

完整RTT包括观测快照、JPEG、pickle、HTTP、云端推理、响应解码、50步校验和进入
pending。active 剩余动作达到提前量时用最新观测请求；当前40步继续执行。边界只
允许 pending 接管，且根据 `activation_time - observation_time` 跳过已经过期的
模型前缀，再选择完整40步。新计划首个deadline固定接在旧计划最后一步之后的
33.33 ms，不以HTTP响应到达时间重新计时。

时间对齐完成后，预取chunk的前6步使用上一条已发布目标做200 ms短桥接。配置项
`boundary_blend_method`支持 `smoothstep` 和 `velocity_continuous`。当前 Pi 配置使用
`smoothstep`位置桥接和姿态SLERP。可选的 `velocity_continuous` 从最近两条120 Hz
已发布命令估计旧chunk结束速度，从时间对齐后的新chunk估计接入速度，EEF XYZ与
20维手关节使用cubic Hermite，四元数在anchor局部rotation-vector空间使用Hermite。
桥接在120 Hz发布路径直接采样，第6步精确到达 `new[aligned_index + 5]`，随后按原
时间轴进入后续动作；不会添加6帧延迟。当前50/40配置最多允许裁掉9步，并继续使用
6步边界融合。两种模式都会重新执行finite、维度与相邻跳变检查。

第一段chunk没有旧命令轨迹，可选使用独立的 `initial_blend_steps`。当前设为0，
即不启用首段桥接，第一段从 `action[0]` 开始。若以后配置为正数，机器人端会从
最新实测EEF和启用侧手状态以零速度建立anchor，并连续接入对应模型步骤；初始实测
状态缺失或超过 `max_observation_age_s` 时拒绝激活第一段。云端无需为该逻辑修改
响应格式。

PCHIP不改变30 Hz模型航点本身。HTTP客户端保留可选的三阶零相位Butterworth低通，
但当前 `action_smoothing_method: none`，不对模型航点进行低通处理。

模型动作仍是30 Hz带时间戳的绝对航点，deployment client将其采样为120 Hz命令。
`action_interpolation_method`支持 `none`、`linear_slerp` 和 `pchip_slerp`。当前默认
使用PCHIP对EEF XYZ和20维手关节做保形C1插值；EEF四元数采用最短路径SLERP，ZSP
线性插值后归一化。PCHIP在每个选定chunk激活时构建一次，120 Hz回调只按时间采样；
少于3个航点时自动退回线性插值。PCHIP只负责chunk内部；跨chunk由独立的
`boundary_blend_method`处理。底层Tianji仍以120 Hz执行IK，云端action rate保持
30 Hz。

初始请求或边界预取未及时返回时，机器人保持最后目标；从保持状态重新请求得到的
chunk从第0步开始，不把网络等待误算成机器人已经执行过的模型动作。Replay的
`tcp://`短chunk路径保持原顺序请求，不启用HTTP预取。

这与 OpenPI 的 `ActionChunkBroker` 顺序消费思想一致，但当前系统不是官方 ALOHA
控制链的原样复制：官方示例是关节 action，当前 Tianji 是绝对 EEF action 再做 IK；
当前网络是平台 HTTP；控制频率也不同。

### READY 后的显式 handoff 闸门

Pi HTTP 实机部署不再在 Tianji 刚进入 READY 时立即消费第一段动作。当前启动顺序为：

```text
READY
  -> 第一次推理，仅取 action[0] 作为固定接入目标
  -> 控制器从实测关节位置执行 0.2 s hold + 1.0 s quintic ramp
  -> /tianji_arm/handoff_state = COMPLETE
  -> 丢弃第一次推理的整个 chunk
  -> 使用 handoff 完成后的最新观测重新推理
  -> 第二次响应到达后才启动 30 Hz action clock
```

handoff 期间部署节点以120 Hz重复同一个EEF/手目标，控制器只锁存第一次有效IK关节
解作为ramp终点。因此网络响应时间和第一段后续航点都不会在底层接管期间推进或改变
接入终点。`startup_handoff_timeout_s`超时、控制器报告FAILED、或第二次推理超过
`policy_start_timeout_s`时，部署节点请求Tianji回到standby。该闸门只对
`pi_v2 + HTTP`启用；TCP replay和LingBot-VA FDM不改变。标准
`start_deployment_session.sh`会读取协议配置并把同一个布尔值传给控制器；若绕过会话
脚本直接执行`ros2 launch`，必须显式传入
`startup_handoff_gate_enabled:=true`才能启用这套同步。

可从 `/wuji_deployment/status` 的 `startup_handoff` 字段观察
`WAIT_BOOTSTRAP_CHUNK -> HANDOFF_ACTIVE -> WAIT_FRESH_CHUNK -> RUNNING`，控制器原始
状态在 `/tianji_arm/handoff_state`。诊断JSONL还记录
`startup_anchor_latched`、`startup_bootstrap_discarded` 和
`startup_policy_clock_started`，可验证第一段动作没有在handoff中被消费。

此前实机 50/25 测试比原 50/50 的单次 chunk 边界跳变更小，但边界次数加倍，仍可观察
到段间回弹。此前调度器每次用“实际发送时间 + 33.3 ms”计算下一时刻，ROS timer
的微小迟到会持续累积，实测 action 发布因此只有约 24.8 Hz。当前调度已经改为固定
时间轴：普通 timer 量化误差不再改变后续相位；若 executor 真正落后至少一个完整
action 周期，则整体后移剩余计划一次，避免恢复后以 120 Hz 突发补发。100 Hz 轮询
单元测试下 30 Hz 平均频率已经通过。异步预取加入后，状态话题通过 `prefetch`
字段报告P99、lead、hit/miss、时间对齐跳步数和边界等待，仍需用真机性能日志完成
最终验收。

部署客户端同时把结构化诊断保存为：

```text
/home/wuji/datasets/tianji_wuji/diagnostics/
deployment_trace_YYYYMMDD_HHMMSS_SESSION.jsonl
```

启动日志会打印本次文件的完整路径。JSONL记录握手、请求触发时的剩余动作、完整
RTT与动态lead、pending到达、边界hit/miss、接管等待、时间对齐skip、融合步数和
融合前后边界跳变，以及每个实际发布动作的chunk编号、计划时间、发送时间和EEF/手部目标。文件不包含图像、API key
或机器人SDK状态流。写盘由独立线程完成；控制回调仅做非阻塞入队，队列满时丢弃
诊断事件而不是阻塞机器人。

## 8. 安全行为

- policy hello 未通过时拒绝 Enable；
- Tianji 到达 READY 后 2 秒内没有第一条有效 action，会请求 standby；
- observation 年龄超过 0.5 秒时拒绝响应并重连；
- 状态或必需相机缺失时不创建 observation；
- HTTP预取失败时不覆盖或立即清空已经验证的active plan；pending会被丢弃并重连；
- 初始请求失败、生命周期退出或旧generation响应到达时清空相应动作状态；
- Tianji 进入 standby、故障或会话退出时关闭选中侧 WujiHand；
- Tianji 外部目标超时仍由控制器 watchdog 进入安全状态。

第一次运行新 checkpoint 或修改动作时序后必须保持急停可达，先用 hold 或小动作
验证，再执行完整任务。

## 9. 本地协议测试

本机/LAN 可以启动内置 hold policy：

```bash
source /opt/ros/humble/setup.bash
source /home/wuji/ros2_ws/install/setup.bash
ros2 run wuji_data_pipeline cloud_policy_server \
  --bind tcp://0.0.0.0:5555 \
  --chunk-size 50 \
  --action-rate-hz 30
```

机器人端：

```bash
./src/scripts/start_deployment_session.sh tcp://127.0.0.1:5555 right --no-camera
```

本地 hold 只验证协议、生命周期和退出安全，不代表真实模型或平台 HTTP 已验收。
