# Tianji + Wuji 通用云端模型接入指南

本文面向 LingBot-VA、Cosmos、OpenPI/π0.5 及后续模型的云端部署开发者，说明
机器人端已经固定的数据协议、控制边界和验收方法。目标是：

> 每个模型只在云端实现模型 adapter；机器人端不增加模型专属控制代码。

本文描述的是机器人端 protocol-v2 契约，不描述任何一个模型的网络结构、checkpoint
或训练代码。π0.5 Dropper Joint 当前参数见
[`PI05_DROPPER_JOINT_DEPLOYMENT_NOTES.md`](../record/pi/PI05_DROPPER_JOINT_DEPLOYMENT_NOTES.md)；
早期 sandwich 联调过程见
[`CLOUD_DEPLOYMENT_HANDOFF.md`](../record/pi/CLOUD_DEPLOYMENT_HANDOFF.md)。两者都是
具体模型的部署实例，不是通用协议默认值。

## 1. 总体架构与职责边界

```text
机器人端（固定）
  相机 + Tianji/Wuji 状态
          ↓ protocol-v2 observation
平台 HTTP Service（固定传输层）
          ↓
云端模型 adapter（按模型实现）
  解码 → 预处理 → 推理 → 反归一化 → 统一动作映射
          ↓ protocol-v2 action_chunk
机器人端（固定）
  时间对齐 → chunk 边界融合 → 30→120 Hz 插值
  → Tianji EEF/IK 或 Joint 控制 + WujiHand 命令
```

机器人端负责：

- 相机和机器人状态采集；
- Recovery、Enable、standby、watchdog 和生命周期；
- action chunk 的时序、预取、过期检查、边界融合和插值；
- 按协商结果执行 EEF→IK 或绝对 Joint 目标；
- WujiHand 驱动和 profile 配置的首次命令渐入；
- 最终协议校验和硬件安全停机。

云端负责：

- 加载、预热和运行模型；
- 模型专属的图像、状态、历史和语言预处理；
- 模型输出反归一化；
- 将模型原生动作转换为 profile 协商的绝对 EEF 或 Joint 动作；
- 单侧模型的另一侧 hold 补齐；
- 返回完整、带身份信息的 action chunk。

云端不得直接访问 ROS 2、Marvin SDK 或 WujiHand SDK，也不得绕过机器人端生命周期
直接控制硬件。

## 2. “不改机器人端”的适用条件

一个新模型可以仅修改云端代码，前提是它能由云端 adapter 转换为以下现有契约：

- 输入来自当前 observation 中的 RGB、双臂状态、双手状态；
- 输出能转换为 profile 声明的绝对 EEF 位姿或 7 维绝对关节位置，以及
  20 维手关节角；
- 云端能返回机器人配置要求的 chunk 长度和动作频率；
- 运行模式是 `right`、`left` 或 `both`，即单臂+同侧单手，或双臂+双手。

以下需求不能靠“只改模型推理函数”自动解决：

- 模型必须接收当前未上传的深度、红外、力传感器或其他新传感器；
- 模型要求运行时动态语言指令，但现有 observation 没有该字段；
- 模型只能输出当前协议无法表达的力矩、电流或混合底层命令；
- 模型要求当前协议无法表达的混合硬件模式；
- 模型输出长度或频率无法在云端可靠转换成机器人配置要求。

如果语言指令在一次服务部署中固定，可作为云端服务配置，不必修改协议。如果模型
需要视频历史，云端可按 `session_id` 保存滚动历史，并在 session 变化时清空；不必让
机器人重复上传整段历史。如果需要新增随请求变化的观测字段，应新增版本化协议，
不能复用或覆盖现有字段。

## 3. 当前 HTTP 传输契约

机器人向服务基础 URL 的固定路径发送：

```text
POST /v1/robot-policy HTTP/1.1
Content-Type: application/octet-stream
Accept: application/octet-stream
Authorization: Bearer <API_KEY>
X-OpenPI-API-Key: <API_KEY>
```

请求和响应 body 均为：

```python
pickle.dumps(mapping, protocol=pickle.HIGHEST_PROTOCOL)
```

成功响应必须满足：

- HTTP 200；
- `Content-Length` 准确且 body 完整；
- Content-Type 是 `application/octet-stream` 或
  `application/x-python-pickle`；
- body 解码后是一个 mapping；
- 支持 HTTP/1.1 persistent connection，不能每次响应后主动关闭连接；
- 不使用 3xx 重定向表达正式服务地址。

`X-OpenPI-API-Key` 是当前客户端遗留的兼容请求头，不表示协议只支持 OpenPI。
新服务至少应正确验证 `Authorization: Bearer ...`，可以兼容接收并忽略第二个头。

Pickle 只能用于受控、鉴权的可信服务。服务绝不能把该接口直接暴露到公共互联网，
也不能反序列化来自不可信来源的请求。如果模型服务不是 Python，应在它前面部署一个
可信 Python protocol adapter，再通过内部接口调用模型。

本地 Replay/LAN 测试可使用 ZMQ `tcp://`；它和 HTTP 使用同一套 pickle mapping，
但正式平台部署使用上述 HTTP Service。

## 4. 握手协议

机器人在允许硬件 Enable 前发送：

```python
{
    "protocol_version": 2,
    "message_type": "hello",
    "session_id": "<uuid>",
    "request_id": 1,
    "robot_layout": {...},
    "camera_names": ["head", "right_wrist"],
}
```

服务必须只检查能力和就绪状态，不能在 hello 中执行一次正式模型动作。回复：

```python
{
    "protocol_version": 2,
    "message_type": "hello_ack",
    "session_id": request["session_id"],
    "request_id": request["request_id"],
    "model_id": "<唯一且稳定的模型/检查点标识>",
    "action_rate_hz": 30.0,
}
```

要求：

- `session_id` 和 `request_id` 必须原样回传；
- `model_id` 必须与机器人端部署 profile 中的期望值完全一致；
- `action_rate_hz` 必须与机器人端 profile 一致；
- profile 要求动作空间协商时，`arm_action_space` 必须明确返回
  `eef_pose` 或 `joint_position`，不能根据同为 7 维的形状猜测；
- 模型、权重、归一化统计和相机配置没有就绪时，不得返回成功 hello；
- 新 `session_id` 出现时，云端必须清空旧会话的图像历史、动作缓存和异步结果。

机器人重连后会创建新 generation/session。旧请求即使后来完成，也不能进入新会话。

## 5. Observation 契约

正式 observation 的外层字段为：

```python
{
    "protocol_version": 2,
    "schema_version": 2,
    "message_type": "observation",
    "session_id": "<uuid>",
    "request_id": 12,
    "timestamp": 1785...,          # wall time，秒
    "client_monotonic": 1234.56,   # 仅用于同一机器人进程内时序
    "arms": ["left", "right"],
    "robot_layout": {...},
    "active_hand_sides": ["right"],
    "zero_filled_hand_sides": ["left"],
    "source_timestamps": {...},
    "arm_state_left": {...},
    "hand_state_left": {...},
    "arm_state_right": {...},
    "hand_state_right": {...},
    "images": {...},
}
```

### 5.1 机器人状态

每侧机械臂状态：

```python
arm_state_<side> = {
    "joint_pos": np.ndarray((7,), float32),     # rad
    "joint_vel": np.ndarray((7,), float32),     # rad/s
    "joint_torque": np.ndarray((7,), float32),
    "ee_pos": np.ndarray((3,), float32),        # m
    "ee_quat": np.ndarray((4,), float32),       # xyzw
    "eef": np.ndarray((7,), float32),           # xyz + xyzw
}
```

`left` EEF 是 `left_chest` 下的绝对位姿，`right` EEF 是 `right_chest` 下的绝对
位姿；二者不是同一个全局坐标系。

每侧手状态：

```python
hand_state_<side> = {
    "joint_pos": np.ndarray((20,), float32),     # rad
    "joint_vel": np.ndarray((20,), float32),     # rad/s
    "joint_torque": np.ndarray((20,), float32),
}
```

单侧模式仍保留双侧结构。未接入侧的手状态是 20 维零值，并在
`zero_filled_hand_sides` 中明确标记；双臂的实测状态仍会上传。模型不得把补零侧误判
成一只真实处于零姿态的手。

对应的扁平布局为：

```text
qpos  = arm_left(7) + hand_left(20) + arm_right(7) + hand_right(20) = 54
eef   = eef_left(7) + eef_right(7)                                  = 14
EEF action   = eef_left(7) + hand_left(20) + eef_right(7) + hand_right(20) = 54
Joint action = arm_left(7) + hand_left(20) + arm_right(7) + hand_right(20) = 54
```

adapter 应优先读取请求中的 `robot_layout`，并验证其与模型 profile 一致，不要仅凭
数组长度猜测硬件布局或动作语义。

### 5.2 图像

`images` 是按相机名索引的 mapping：

```python
images["head"] = {
    "codec": "jpeg",
    "shape": [height, width, 3],
    "color_space": "bgr8",
    "timestamp": 1785...,
    "data": b"...jpeg bytes...",
}
```

当前网络图像为 OpenCV BGR `uint8`。模型 adapter 负责：

1. JPEG 解码；
2. 按训练过程转换 BGR→RGB；
3. resize、crop、相机顺序和 normalization；
4. 检查请求中的相机集合与该 checkpoint 的要求完全一致。

不得按 mapping 的偶然遍历顺序给模型分配相机；使用相机名和模型 profile 中的明确
顺序。缺少必需相机时返回错误，不能交换左右腕视角或静默补黑图。

当前 deployment 只上传配置选中的 RGB。数据集中已经保存的深度、红外和头部双目
附加流不会自动进入该协议。

## 6. Action 契约

机械臂动作空间由部署 profile 和握手结果明确决定。EEF 与 Joint 都是每侧 7 个数，
不得按维度自动判断。下面先给出 EEF mapping；Joint mapping 见 6.1。

每次正式推理回复：

```python
{
    "protocol_version": 2,
    "message_type": "action_chunk",
    "session_id": request["session_id"],
    "request_id": request["request_id"],
    "model_id": "<与 hello 完全相同>",
    "action_rate_hz": 30.0,
    "action_chunk": [action_0, action_1, ...],
    "server_timing": {                 # 强烈建议提供
        "received_wall_time": ...,
        "inference_started_wall_time": ...,
        "inference_finished_wall_time": ...,
        "inference_ms": ...,
    },
}
```

每一步必须包含完整双侧动作：

```python
action_i = {
    "arm_action_left": {
        "ee_pos": [x, y, z],            # m，left_chest 下绝对位置
        "ee_quat": [qx, qy, qz, qw],    # xyzw，单位四元数
        # "zsp": [x, y, z],             # 可选，finite 3维
    },
    "hand_action_left": [20 values],    # degree
    "arm_action_right": {
        "ee_pos": [x, y, z],            # m，right_chest 下绝对位置
        "ee_quat": [qx, qy, qz, qw],    # xyzw，单位四元数
    },
    "hand_action_right": [20 values],   # degree
}
```

扁平数组的固定次序是：

```text
[left EEF 7, left hand 20, right EEF 7, right hand 20]
```

注意 observation 的手关节是 rad，而 action 的手关节是 degree。模型输出必须先做
反归一化和单位转换，不能把归一化张量或 rad 直接放入 `hand_action_*`。

### 6.1 Joint 动作

Joint profile 使用每侧 7 维绝对关节位置，机械臂单位始终为 rad，并直接进入
`external_joint_target`，不经过 EEF/IK。不同协议模式的 wire mapping 为：

- PI `pi_v2`：`arm_action_<side>.joint_pos`，手部仍为 degree，并通过
  `arm_action_space=joint_position` 锁定语义；
- Cosmos `protocol_v2`：`arm_joint_action_<side>`，手部仍为 degree，profile 中
  `action_space` 与 `arm_command_mode` 必须同时为 `joint`；
- LingBot-VA FDM：`arm_action_<side>.joint_pos`，机械臂和手部都为 rad，且 hello
  中的 `action_mode` 必须为 `joint`。

云端和机器人端都必须按 profile 校验单位、关节顺序、位置限制和相邻轨迹约束。
Joint profile 不接受只有 `ee_pos + ee_quat` 的响应，也不能把手部单位在不同协议间
混用。

### 6.2 单侧模型补齐

以右臂+右手模型为例：

- 右侧使用模型输出；
- EEF profile 的左臂使用实测左 EEF 生成 hold；Joint profile 使用实测左臂 qpos；
- 左手使用本次 observation 的左手位置，并转换为该 wire profile 规定的单位；
  补零侧因此保持零值；
- 仍然返回完整 54 维/完整双侧 mapping。

机器人只向本次启用侧发布命令，但会先校验完整双侧响应。不要省略未启用侧字段。

### 6.3 时序由模型 profile 决定

Action chunk 的 horizon 和 rate 不是通用协议常量。每个模型必须在自己的机器人端
profile 中声明，并与云端 `hello_ack`/`action_chunk` 完全一致。当前 π0.5 Dropper Joint
实机 profile 是：

```text
云端返回：50 steps @ 30 Hz
机器人选择并执行：时间对齐后的 30 steps
边界桥接：6 steps
机器人发布：Joint 线性插值到 120 Hz
Tianji SDK 状态读取：500 Hz
```

这里的 50/30 只属于 π0.5 Dropper Joint，不是 LingBot-VA、Cosmos 或其他模型必须
遵守的协议长度。LingBot-VA 使用独立 FDM 48 步时间线，Cosmos 当前为 32/16；
不得为迎合另一个 profile 复制末帧、无依据裁剪或重采样。

异步预取配置必须满足：

```text
chunk_size >= open_loop_horizon + prefetch_max_lead_actions + 1
```

具体 horizon 和 lead 应根据该模型完整 HTTP RTT 的 P95/P99 决定，不能照抄其他
profile 的参数。FDM 使用自己的异步时序校验，不应套用同步 protocol-v2 的配置。

机器人会拒绝：

- chunk 数量不是配置值；
- action rate 不匹配；
- 缺少任一侧 arm/hand；
- 维度错误或 NaN/Inf；
- EEF profile 的零四元数、位置/姿态跳变超限；
- Joint profile 的关节位置、step/velocity 或最终发布速度超限；
- 手关节跳变超过对应 profile 的限制。

这些检查也适用于 chunk 第一帧相对上一条已发布目标的边界。

## 7. 推荐的云端代码分层

不同模型不要各自重写 HTTP 协议。推荐分三层：

```text
HTTPProtocolServer（所有模型共用）
  ├─ 鉴权、pickle、hello、identity、错误 envelope
  └─ ModelAdapter（每个模型实现）
       ├─ preprocess(observation)
       ├─ infer(model_input)
       └─ to_robot_action_chunk(native_output, observation)
```

最小 adapter 结构：

```python
class ModelAdapter:
    model_id = "lingbot_va/<checkpoint-or-version>"
    action_rate_hz = 30.0
    chunk_size = 48
    required_cameras = ("head", "right_wrist")

    def __init__(self, config):
        self.model = load_model_once(config)
        self.normalizer = load_matching_statistics(config)
        self.warmup()

    def infer(self, observation):
        self.validate_observation(observation)
        model_input = self.preprocess(observation)
        native_actions = self.model.infer(model_input)
        actions = self.denormalize_and_convert(native_actions, observation)
        return self.fill_inactive_side_and_validate(actions, observation)
```

服务端处理逻辑：

```python
request = pickle.loads(body)

if request["message_type"] == "hello":
    response = hello_ack_with_same_identity(request)
elif request["message_type"] == "observation":
    actions = adapter.infer(request)
    response = action_chunk_with_same_identity(request, actions)
else:
    response = error_envelope_with_same_identity(request)
```

任何异常回复都应尽量保留 `protocol_version`、`session_id` 和 `request_id`，并在
`error` 中给出不含密钥、图像内容和用户隐私的简短原因。

仓库中的 `wuji_data_pipeline/cloud_policy_server.py` 是通用 adapter 接口和 ZMQ hold
参考实现；在线平台的 HTTP server 仍需按本节实现 HTTP wrapper。

## 8. 每个模型必须提交的部署 profile

每个模型/检查点必须提供一份清单，不能只给启动命令：

| 项目 | 必填内容 |
|---|---|
| model_id | 唯一、稳定，并与 hello/action 返回一致 |
| checkpoint | 完整路径、版本或制品哈希 |
| mode | `right`、`left` 或 `both` |
| cameras | 精确名称和顺序，如 `head,right_wrist` |
| state input | 使用哪些 arm/hand 字段及单位 |
| image preprocessing | BGR/RGB、尺寸、crop、normalization |
| language/task | 固定服务配置还是请求字段；当前协议无动态 prompt |
| native action | 原生维度、绝对/增量、坐标系、单位 |
| arm action space | `eef_pose` 或 `joint_position`，以及握手字段 |
| robot action conversion | 如何转换成绝对 EEF/Joint + 20DoF hand |
| native horizon/rate | 模型原生值 |
| wire horizon/rate | 该模型profile声明的值；不得沿用其他模型的固定值 |
| inactive-side fill | 单侧模式的 hold 生成方式 |
| expected latency | warm/cold 的 mean、P95、P99 |
| API key env | 机器人进程读取的环境变量名 |

建议为每个模型保留完整 pipeline YAML profile，只修改 `deployment` 相关项，例如：

```yaml
deployment:
  policy_http_path: "/v1/robot-policy"
  policy_http_api_key_env: "MODEL_POLICY_API_KEY"
  policy_http_expected_model_id: "model/<checkpoint>"
  policy_http_expected_chunk_size: 50
  expected_arm_action_space: "joint_position"
  camera_names: [head, right_wrist]
  action_rate_hz: 30.0
  open_loop_horizon: 30
  prefetch_max_lead_actions: 9  # 50 >= 30 + 9 + 1；仍需按实测P99调整
```

当前 `--config` 接收的是完整 pipeline YAML，不是只含上述字段的局部 overlay。

## 9. 当前机器人端的 OpenPI 兼容请求头

`start_deployment_session.sh` 支持 profile 指定的 API key 环境变量，也支持
`--api-key-env NAME`。HTTP client 为兼容既有服务，除标准
`Authorization: Bearer ...` 外仍附带 `X-OpenPI-API-Key`。该附加 header 只是 wire
compatibility，不表示云端必须使用 OpenPI，也不能据此复用 π0.5 的模型参数。

不同模型必须选择各自完整 YAML profile。不要为 LingBot-VA、Cosmos 分别复制
controller 或 deployment node。

## 10. 云端验收顺序

### 10.1 离线 adapter 测试

- 用一条真实 observation fixture 运行预处理；
- 检查相机名称、色彩空间和尺寸；
- 检查反归一化后的 EEF/Joint/手单位；
- 检查每一步完整双侧 mapping、finite；EEF profile 还要检查四元数范数；
- 检查 chunk 数量和模型输出时间语义；
- 检查单侧 hold 不随模型右侧输出漂移。

### 10.2 HTTP 协议测试

- hello 不触发正式动作推理；
- hello/action 原样回传 session/request；
- 返回正确 model ID、rate 和 Content-Length；
- 同一连接可连续处理多个请求；
- 新 session 清除旧历史；
- 超时、坏输入和模型异常返回可诊断 error envelope；
- 服务端日志不输出 API key 和完整 pickle/image。

### 10.3 机器人安全测试

先启动云端 hold/最小动作模式，再启动机器人会话：

```bash
cd <WUJI_HAND_TELEOP_PIPELINE_ROOT>

read -rsp 'Policy HTTP API key: ' WUJI_POLICY_API_KEY
echo
export WUJI_POLICY_API_KEY

./src/scripts/start_deployment_session.sh \
  http://SERVICE_HOST \
  right \
  --api-key-env WUJI_POLICY_API_KEY \
  --config /home/wuji/ros2_ws/src/wuji_data_pipeline/config/<MODEL_PIPELINE>.yaml
```

检查握手：

```bash
ros2 service call /wuji_deployment/ready std_srvs/srv/Trigger "{}"
```

检查状态：

```bash
ros2 topic echo /wuji_deployment/status
```

必须确认：

- `server_ready: true`；
- `last_model_id` 与 profile 完全一致；
- `last_server_error` 为空；
- `requests` 持续增加，`failures` 不持续增加；
- 首次真机动作只发布到选中的模式；
- 急停全程可达。

每次部署诊断保存在：

```text
/home/wuji/datasets/tianji_wuji/diagnostics/deployment_trace_*.jsonl
```

验收时至少分析完整 RTT、云端 `inference_ms`、pending hit/miss、时间对齐 skip、
边界等待、边界融合前后跳变、30 Hz 航点和 120 Hz 实际发布。仅看到模型服务返回
HTTP 200，不代表控制时序已经验收。

## 11. 新模型交付完成标准

只有同时满足以下条件，才算完成一个新模型的云端接入：

- 没有新增模型专属机器人 controller/deployment node 分支；
- 模型 adapter 和服务 profile 独立、可复现；
- protocol-v2 hello 和 action identity 全部通过；
- 输入预处理、输出反归一化、坐标系和单位有文档及测试；
- 单侧/双侧模式与模型训练配置一致；
- 服务在真实相机数据下满足延迟和稳定性要求；
- hold、小动作和完整任务依次完成真机验收；
- 失败、重连、退出时机器人和手均进入既有安全状态。
