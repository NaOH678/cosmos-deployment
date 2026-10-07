# Tianji + Wuji data pipeline

LingBot-VA's independent asynchronous FDM deployment mode is documented in
[`LINGBOT_VA_FDM_LOCAL.md`](../record/lbva/LINGBOT_VA_FDM_LOCAL.md). Pi deployment uses
`config/pi05_protocol_v2.yaml`; `pipeline.yaml` remains the shared
recording/general configuration.

Model-neutral HTTP protocol, action-space, safety, and acceptance requirements
are documented in
[`CLOUD_MODEL_INTEGRATION_GUIDE.md`](CLOUD_MODEL_INTEGRATION_GUIDE.md). Model-specific
deployment findings are indexed in [`src/record`](../record/README.md).

这个 ROS 2 package 为当前的 HTC Tracker + MANUS 遥操增加两条管线：

- 数采：同步天机双臂、无极双手和三路相机，保存为与
  `dexmanip_tool` 相同的 LMDB scalar key + MP4 组织形式；
- 部署：机器人端使用同一套 observation/action 协议连接 replay server
  或云端 policy server。Replay 不依赖 Tracker 或 MANUS。

控制安全边界没有改变：启动进程不会自动 Recovery 或 Enable。操作者必须在
清空工作区并准备好急停后，显式执行这两个动作。

## 数据定义

默认硬件布局由 metadata 描述，而不是在 reader 中写死：

| 数据 | 顺序 | 维度 | 单位/语义 |
|---|---|---:|---|
| `qpos` | 左臂 7 + 左手 20 + 右臂 7 + 右手 20 | 54 | 全部为 rad，实测关节位置 |
| `qvel` | 与 `qpos` 相同 | 54 | rad/s；驱动未提供速度时由相邻同步实测位置计算 |
| `effort` | 与 `qpos` 相同 | 54 | 驱动实测值；硬件未提供的分量为 0 |
| `action` | 左 EEF 7 + 左手 20 + 右 EEF 7 + 右手 20 | 54 | EEF 为实际执行轨迹（m + quaternion xyzw），手为目标角度 deg |
| `eef` / `action_eef` | 左 EEF 7 + 右 EEF 7 | 14 | 实际执行的 EEF |

`action` 使用实际 EEF 而不是 Tracker 原始目标，这是 `dexmanip_tool`
录制器的既有语义，也能避免 replay 一个机器人当时没有真正到达的轨迹。
上游目标 EEF、天机 IK 关节命令和 ZSP 另存于 `/diagnostics/*`。

为分析遥操链路，recorder 还会按同一个训练帧 anchor 旁路记录可选的
`/teleop/*` 数据。它不参与开始录制前的 source readiness，也不改变 30 Hz
训练帧的同步和丢帧逻辑：诊断样本在 100 ms 内取最近邻，缺失时仍正常写入
原来的训练帧，只把对应 `available` / `valid` 写成 0。

- `/teleop/tracker/raw_pose`：5 个 Tracker 的 OpenVR 原始位姿，形状 `N×5×7`；
- `/teleop/tracker/corrected_pose`：现有 wrist offset/坐标修正后的位姿，`N×5×7`；
- `/teleop/tracker/{linear_velocity,angular_velocity,valid,...}`：OpenVR 原始状态；
- `/teleop/manus/{left,right}/raw_node_pose`：MANUS 原始 25 节点位姿，`N×25×7`；
- `/teleop/manus/{left,right}/keypoints_21`：与手部控制节点完全相同的语义映射，
  `N×21×3`；
- 同一 MANUS 消息里的 node id/语义类型编码、ergonomics、raw sensor 数据也一并保存；
- 每个来源都有 `source_timestamp`、`alignment_error_s` 和显式有效标记。

为避免更改 MANUS 消息 ABI 和既有 OpenVR TF 协议，`source_timestamp` 是 recorder
收到消息时的 ROS 时钟；OpenVR raw state 直接复制自生成现有 corrected TF 的
同一次 SDK 读取，不会额外轮询 OpenVR。这个时间语义会写入 episode metadata，
分析时可以直接使用 `alignment_error_s` 筛选样本。
语义类型采用紧凑整数编码以避免重复保存字符串，完整 codebook 写在 episode
metadata 的 `teleop_diagnostics.manus_type_codebooks` 中。

Tracker 顺序固定为 `chest, left_wrist, right_wrist, left_arm, right_arm`。
旧 episode 没有这些可选 key 时仍可读取。Replay 默认消费 `action` 中的 EEF
pose 和手指目标；使用 `--arm-command-mode joint` 时机械臂改为消费实测 `qpos`
电机关节角。两种模式都不会读取 `/teleop/*`。

相机语义固定为三路：`head`、`left_wrist`、`right_wrist`。阶段D默认由独立
Camera Manager按RealSense序列号或稳定udev路径直读设备，并通过
`/dev/shm/wuji_camera_v1`有界环提供给GUI、Recorder和Deployment；图像不经过
ROS 2。`camera_transport=ros`只保留为迁移对比。

## 构建

Dockerfile 已加入 `python3-lmdb`、`python3-zmq` 和固定版本
`pyrealsense2`。旧镜像需要重建一次：

```bash
cd /home/pjlab/ros2_ws/worktrees/wuji-hand-teleop-pipeline/docker
docker compose build
docker compose up -d
docker exec -it wuji-hand-teleop bash
```

容器内构建工作区：

```bash
cd /home/wuji/ros2_ws
source /opt/ros/humble/setup.bash
colcon build --symlink-install --packages-select camera controller wuji_data_pipeline
source install/setup.bash
```

OpenVR 诊断 publisher 位于现有 `openvr_input` 源码中；独立 OpenVR 容器使用源码
挂载，因此构建主工作区后，下一次按原文档重启该容器就会以 30 Hz 发布
`/openvr/tracker_diagnostics`。原 TF 仍按 120 Hz 原路径发布。

## 当前无相机数采

控制链路严格遵循仓库根目录的 `TIANJI_VIVE_TELEOP_RECORD.md` 和
`MANUS_WUJI_INTEGRATION_RECORD.md`。先按 Tianji 文档 5.2 节启动独立的
`wuji-openvr-input` 容器；数采 launch 不会在主容器内重复启动 OpenVR。

相机尚未装好时，宿主机运行以下脚本。它会检查 OpenVR 容器的 Domain/RMW，
再按文档参数启动静态 TF、Tianji controller、单侧 MANUS/Wuji 链路和 recorder：

```bash
cd /home/pjlab/ros2_ws/worktrees/wuji-hand-teleop-pipeline
./src/scripts/start_record_session.sh right
```

当前只接入一只 Wuji Hand 时，用 `--active-hand` 明确指定物理手。未接入侧的
`qpos`、`qvel`、`effort` 和 `action` 手部 20 维全部写零，整体仍保持 54 维；
episode metadata 会记录 `zero_filled_hand_sides`。例如只接左手：

```bash
./src/scripts/start_record_session.sh left
```

只接右手则把 `left` 改为 `right`。已启用侧仍要求 state/command topic 持续
新鲜；运行中掉线不会自动降级成零值。

交互键：

- `r`：天机 guarded Recovery（会移动机器人，一次一条手臂）；
- `a`：Recovery 成功后先 Enable Tianji，READY 后使能所选 WujiHand；
- `x`：先关闭所选 WujiHand，再 Disable Tianji 并请求 standby；
- `s`：开始新 episode，仅在天机 lifecycle 为 `READY` 时允许；
- `z`：停止并保存；
- `e` / `q`：放弃尚未通过 `z` 保存的 active episode，请求 standby 并退出；
- `Ctrl+C`：与 `q` 完全相同，不调用保存、不等待 recorder；先关闭所选手并
  Disable 天机，读回双臂 `state=0`，再关闭本次 launch 的全部子进程。程序只允许在交互式
  TTY 中启动，防止键盘控制进程消失后留下孤儿 launch。独立运行的
  `wuji-openvr-input` 容器仍按 Tianji 文档 5.2 节单独管理。

默认输出目录是容器内的 `~/datasets/tianji_wuji`，对应宿主机仓库旁的
`datasets/tianji_wuji`（Compose 已做持久化挂载），可在
`config/pipeline.yaml` 的 `recording.output_dir` 修改。若没有成功同步到任何
frame，目录会保留为 `.inprogress`，不会伪装成可 replay 的完整 episode。

相机安装完成后，去掉 `--no-camera`：

```bash
ros2 run wuji_data_pipeline record_session
```

正式相机模式在开始episode时冻结当前在线相机子集，并以其中配置顺序第一路作为
anchor；通常为`head`。至少一路相机在线即可开始，未连接相机不阻塞其余相机。
本条episode选定的相机与所有机器人scalar数据仍必须新鲜并在容差内匹配。

检查保存结果：

```bash
ros2 run wuji_data_pipeline inspect_episode \
  ./datasets/tianji_wuji/episode_0000_YYYYMMDD_HHMMSS
```

## Replay / 云端部署

先在 policy/replay 机器上只做数据校验，不打开端口：

```bash
ros2 run wuji_data_pipeline replay_server \
  --episode-dir /path/to/episode_0000_YYYYMMDD_HHMMSS \
  --dry-run
```

仅当需要把 Replay server 独立放在另一台机器做协议测试时，再单独启动：

```bash
ros2 run wuji_data_pipeline replay_server \
  --episode-dir /path/to/episode_0000_YYYYMMDD_HHMMSS \
  --bind tcp://0.0.0.0:5555
```

正式本机 Replay 使用受控会话，不要分别手动启动 controller 和手部驱动：

```bash
./src/scripts/start_replay_session.sh right \
  /home/wuji/datasets/tianji_wuji/TASK/episode_0000_YYYYMMDD_HHMMSS
```

上面默认使用 EEF pose。需要对比电机关节角直放时显式增加开关：

```bash
./src/scripts/start_replay_session.sh right \
  /home/wuji/datasets/tianji_wuji/TASK/episode_0000_YYYYMMDD_HHMMSS \
  --arm-command-mode joint
```

Replay 默认使用位置模式。需要在阻抗模式下验证 EEF Replay 时增加：

```bash
./src/scripts/start_replay_session.sh right \
  /home/wuji/datasets/tianji_wuji/TASK/episode_0000_YYYYMMDD_HHMMSS \
  --arm-command-mode eef --arm-hardware-mode impedance
```

Replay 把“命令来源”和“Tianji 硬件模式”作为两个正交参数：

| 命令来源 | 硬件模式 | 数据路径 | 用途 |
|---|---|---|---|
| `eef` | `position` | `action EEF -> IK -> state=1` | 默认 Replay |
| `joint` | `position` | `qpos arm -> state=1` | 对比实测电机关节轨迹 |
| `eef` | `impedance` | `action EEF -> IK -> state=3` | 对比阻抗跟随效果 |
| `joint` | `impedance` | `qpos arm -> state=3` | 关节目标的阻抗模式诊断 |

`joint` 模式要求 episode 中存在完整 `qpos`，并强制关闭 Cartesian rebase；
`eef` 默认把录制首帧刚体 rebase 到 Enable 时的实际 EEF。四种组合都只能在
`control_source=external` 下使用，Tracker 遥操的默认硬件模式仍是阻抗。

Replay 中几个“频率”不能混为一谈：

| 频率 | 默认值 | 含义 |
|---|---:|---|
| `--playback-rate-hz` | 6 Hz | 每秒推进多少个录制源帧；30 Hz 数据下等于 0.2x |
| `--action-rate-hz` | 30 Hz | Replay server 输出给 deployment 的 waypoint 时间轴 |
| deployment publish / Tianji control | 120 Hz | 对 30 Hz waypoint 插值并发送机械臂目标 |
| Tianji state read | 500 Hz | 读取并缓存 SDK 实测状态，不是命令发送频率 |

通常只调整 `--playback-rate-hz`。`--action-rate-hz` 必须和
`config/replay.yaml` 的 `deployment.action_rate_hz` 一致，不应作为调速旋钮。

`active_hand` 可设为 `left`、`right` 或 `both`。单手模式只启动并发布到实际接入
的 Wuji Hand；缺失侧不会被等待或发布命令，但发往 replay/policy server 的
observation 会用 20 维零值补齐，因此 LMDB 和网络协议仍保持双臂双手 54 维。

该会话统一启动 replay server、选定的手部驱动、external 模式的天机控制器和
deployment client，不会启动 Tracker、MANUS 或手部 retargeter。Wuji 手在 Prepare
阶段保持失能，并在 Tianji 到达 `READY` 后由会话统一使能。

部署 client 在 Tianji lifecycle 为 `READY` 或短暂进入 `TARGET_HOLD` 时运行。
Replay 默认使用录制 EEF pose，并将首帧刚体 rebase 到 Enable 时的实际 EEF；
控制器完成 IK 后，以 120 Hz 向 Tianji 发送关节目标。可选 joint 模式直接使用
录制 `qpos` 且不做 rebase。两种 Replay 命令源默认使用 6 Hz 录制轨迹时间轴，
再采样为 30 Hz waypoint；硬件默认切到 SDK `state=1` 位置模式，也可显式选择
`state=3` 阻抗模式。
外部双臂目标断流超过 1 秒时天机控制器请求 standby。

本地 Replay 固定使用 `config/replay.yaml`，不再继承 π0.5、LingBot-VA 等云端
模型配置中的 horizon、预取、边界融合或平滑参数。

Pi 0.5 部署配置位于 `config/pi05_protocol_v2.yaml`；通用数采配置仍使用
`config/pipeline.yaml`。`deployment.launch.py` 默认选择 Pi 专用配置，其他模型必须
显式传入自己的 `pipeline_config`。

Replay 和真实云端策略均返回有界 action chunk。当前 π0.5 HTTP 配置要求模型返回
完整 50 步，机器人每轮执行时间对齐后的 40 步。active plan 剩余动作到达动态
P99 提前量时，唯一网络线程使用最新观测异步请求下一块；响应只进入单一 pending
槽位，禁止覆盖 active。边界切换时按 observation 时间跳过已经过期的前缀，并接在
上一动作的固定 30 Hz 时间轴上。Replay 的 `tcp://` 短 chunk 路径不启用该预取。
旧 lifecycle generation 的响应以及重连前动作都不会在恢复后补发。

使用 `x` 停止并进入 standby，或使用 `q/e/Ctrl+C` 让会话按顺序关闭 Wuji、
Tianji 和所有子进程。

protocol-v2 observation/action mapping 仍使用 Python pickle。`tcp://` endpoint
保留 ZMQ REQ/REP，供本地 Replay 和受信内网使用；`http://`/`https://` endpoint
通过在线推理平台的 HTTP/1.1 Service 发送相同 pickle，并使用 API key 双请求头鉴权。

## 云端 Policy 部署

### 架构约束

- 云端只进行模型预处理、推理和 action chunk 生成，不连接 ROS 2、Marvin SDK
  或 WujiHand SDK；
- 云端统一输出左右 EEF 绝对位姿和左右 20 维手部角度，共 54 维；Tianji IK、
  ZSP、阻抗控制和硬件 watchdog 始终留在机器人端；
- 相机原始帧仍通过共享内存提供给 GUI/Recorder。部署 worker 只对上传副本进行
  JPEG 编码，不让图像进入 ROS 2；
- `deployment.camera_names` 的名称和顺序必须与模型训练数据完全一致。模型需要
  三路相机时缺一路就拒绝推理，不能动态交换或静默补零；
- 当前深度/红外是分析附加数据，不会上传给只使用 RGB 训练的模型。

### 构建

修改本包后在现有容器中构建一次：

```bash
docker exec -it wuji-hand-teleop bash -lc '
source /opt/ros/humble/setup.bash
cd /home/wuji/ros2_ws
colcon build --packages-select wuji_data_pipeline --symlink-install
'
```

### 本地 ZMQ hold 联调

云端或同机另一个终端先启动内置 hold policy。它只返回机器人当前 EEF/手部位置，
用于验证协议、JPEG、时延和 chunk，不能替代真实模型：

```bash
source /opt/ros/humble/setup.bash
source /home/wuji/ros2_ws/install/setup.bash
ros2 run wuji_data_pipeline cloud_policy_server \
  --bind tcp://0.0.0.0:5555 \
  --chunk-size 8 \
  --action-rate-hz 30
```

机器人端一条命令启动受控部署会话：

```bash
./src/scripts/start_deployment_session.sh tcp://POLICY_SERVER_IP:5555 right
```

### 在线 π0.5 HTTP 推理

在线服务固定接口为 `/v1/robot-policy`。API key 只通过进程环境传递，禁止写入
YAML、URL 或命令历史：

```bash
cd /home/pjlab/ros2_ws/worktrees/wuji-hand-teleop-pipeline

read -rsp 'PI05 HTTP API key: ' PI05_HTTP_API_KEY
echo
export PI05_HTTP_API_KEY

./src/scripts/start_deployment_session.sh \
  http://s-20260804175737-hs9pc.ailab-eailabagent.pjh-service.org.cn \
  right
```

HTTP client 使用标准库直连，不读取 `http_proxy/https_proxy`，不会经过本机代理；
不跟随 3xx。它要求 HTTP 200、合法 `Content-Length`、pickle Content-Type、匹配的
protocol/session/request/model，并且每次连接重建后必须重新完成 hello。在线模型
必须上传 `head` 和 `right_wrist`，所以正式运行不能加 `--no-camera`。

每次 transport 重连都会创建新的 session generation 和 UUID。响应先与发送时保存的
不可变 request identity 比较；真正的服务端错配记录为
`SERVER_IDENTITY_MISMATCH`。若响应与旧请求匹配、但旧请求已经被新 generation
淘汰，则只记录 `STALE_GENERATION_DISCARDED`，不执行旧 action，也不破坏新会话的
ready 状态。

第二个位置参数同时选择 Tianji 和 Wuji 侧。合法模式只有单臂+同侧单手，或
双臂+双手；不支持“双臂+仅右手”之类的混合模式。

会话按键为：`r` Recovery、`a` 同步使能 Tianji/Wuji、`x` 同步关闭、`q/e`
退出。Prepare 阶段 Wuji 手保持失能；只有 Tianji 到达 `READY` 后才使能手，
命令以 5 秒从实测手位姿渐入。退出、硬件故障或 Tianji 进入 standby 时手也会
被关闭。`a` 之前还必须完成云端 `hello/ready` 握手；服务器不可达、协议版本不符
或模型尚未加载时会拒绝硬件 Enable。即使握手后服务器立刻故障、模型推理失败
或模型要求的相机尚未就绪，Tianji 到达 READY 后 2 秒内收不到第一条有效 action
也会自动请求 standby。没有相机的协议测试可在命令末尾加 `--no-camera`。

### 接入真实模型

模型仓库提供一个 `module:factory`。factory 可接收一个配置文件路径，并返回带
`infer(observation)` 方法的对象：

```python
class MyPolicy:
    model_id = "my_policy_checkpoint_100000"

    def infer(self, observation):
        # observation["images"][name]["color"] 是解码后的 BGR uint8
        # 返回 Nx54 ndarray，或 action mapping 的 list
        return actions


def create_policy(config_path):
    return MyPolicy(config_path)
```

启动时指定 adapter：

```bash
ros2 run wuji_data_pipeline cloud_policy_server \
  --bind tcp://0.0.0.0:5555 \
  --adapter my_model.deploy:create_policy \
  --adapter-config /models/my_policy.yaml \
  --action-rate-hz 30
```

协议会携带 `session_id`、`request_id`、各状态/相机时间戳、模型标识和服务端推理
耗时。机器人端会拒绝不属于当前 session、request 序号错误、观测年龄超过
`max_observation_age_s`，或 action horizon 已被网络延迟完全耗尽的响应。运行时
状态可通过以下话题查看：

```bash
ros2 topic echo /wuji_deployment/status
```

其中包含请求成功/失败数、端到端时延、请求字节数、chunk 剩余量、滚动实测动作
频率 `rolling_dispatch_rate_hz`、动作发送迟到量和长暂停后的调度重对齐次数。状态
中的历史丢弃计数字段仅为协议兼容保留；当前HTTP预取会按观测时间裁剪过期的
chunk前缀，但不会因发布周期抖动跳过已选择的动作。30 Hz
动作时钟使用固定相位，普通 ROS timer 迟到不会逐帧累积；executor 落后一个完整
动作周期时会后移剩余计划，避免恢复后突发补发。

每次部署还会生成独立的JSONL诊断文件：

```text
/home/wuji/datasets/tianji_wuji/diagnostics/deployment_trace_*.jsonl
```

具体路径会在客户端启动日志的 `diagnostic_trace=` 字段中打印。文件保存请求RTT、
动态预取lead、pending hit/miss、边界等待与skip、30 Hz模型航点，以及120 Hz实际
插值命令。
诊断写盘使用独立线程，队列满时只丢诊断事件，不阻塞控制。

deployment launch 还会自动启动一个只读的低优先级旁路进程，生成：

```text
/home/wuji/datasets/tianji_wuji/diagnostics/deployment_state_*.jsonl
```

该进程不被 controller 或 deployment node 调用，也不向机器人发布任何消息。它只把
ROS topic 中的双臂实测关节、实际 EEF、最终 external action、controller EEF/IK
关节目标，以及启用侧手部状态/命令采样到统一的120 Hz monotonic时间轴。每个字段
保留独立的源时间戳、接收序号和age；不会把60 Hz EEF或500 Hz关节状态伪造成同频
数据。JSON编码和写盘由旁路进程自己的线程完成。sidecar 会记录本次
`arm_command_mode`：EEF 模式订阅 external EEF/controller EEF，joint 模式改为订阅
external 7D joint target，不会因为命令类型不同而错误报告必需流缺失。

退出时文件末尾写入完整性统计。每次 READY 后先允许最多0.5秒必需流建立；稳定分析
窗口内没有缺失/过期流、writer零丢帧、消息与采样序号正常时，离线报告才显示
`Completeness: PASS`。失败原因会写入 summary 的 `completeness_failures`，并直接
打印在终端。在宿主机生成最近一次报告：

```bash
./src/scripts/plot_latest_deployment.sh
```

脚本默认从 trace 元数据自动识别 `left`、`right` 或 `both`；仅在分析旧 trace
或需要强制选择侧别时，才显式追加 `left|right|both`。

脚本优先从 state trace 内的 `session_id` 和诊断路径精确匹配同一次
`deployment_trace_*.jsonl`，旧schema才退回30秒窗口的启动时间匹配。输出文件使用
同一报告前缀：

```text
*_report.pdf          多页可视化报告
*_report.summary.json 完整性、时延和跟踪误差统计
*_report.eef.csv      external EEF / controller target / actual EEF（EEF模式）
*_report.joints.csv   external joint / controller command / actual joint
```

报告包含EEF状态/动作、姿态误差、7维源目标/控制命令/实测关节、20维手状态/命令、
chunk边界、HTTP RTT、数据age和采样间隔。summary中的跟踪误差只使用每个READY段建立
稳定数据流后的样本，不再把Recovery/Standby或启动缺流混入RMS。若完整性为FAIL，
不能用该次曲线得出控制结论。

HTTP预取chunk在时间对齐后默认进行6步边界桥接。配置项
`boundary_blend_method`支持 `smoothstep` 和 `velocity_continuous`。前者保留原有
位置smoothstep/姿态SLERP行为，也是当前 Pi 配置的默认值。后者是实验选项：它从
最近两条120 Hz已发布命令估计旧chunk结束速度，并从新chunk估计接入速度，XYZ和
手关节使用cubic Hermite，姿态在anchor四元数的局部rotation-vector空间使用Hermite。
桥接在120 Hz发布路径直接采样，第6步精确到达时间对齐后的对应模型动作，不会延迟
整个新chunk；TCP Replay不启用该处理。

第一段策略轨迹没有旧chunk可作为anchor。客户端保留可选的
`initial_blend_steps`首段桥接，但当前设为0，不启用启动桥接。

PCHIP只保证航点之间连续，并不会消除模型30 Hz航点自身的高频摆动。HTTP客户端
保留可选的 `action_smoothing_method: butterworth`，但当前设为 `none`，不对完整
chunk做低通处理。

部署控制输出把30 Hz模型航点重采样为120 Hz。配置项
`action_interpolation_method`支持 `none`、`linear_slerp` 和 `pchip_slerp`；当前默认
使用SciPy PCHIP对EEF XYZ与手关节做不越过航点范围的C1保形插值，四元数使用最短
路径SLERP，ZSP插值后归一化。插值器在chunk激活时一次构建，控制回调只采样；短于
3点的chunk自动退回线性方式。PCHIP负责chunk内部重采样，跨chunk速度连续性由上述
独立边界模式负责。网络断档时不会靠重复发布最后目标绕过Tianji外部命令watchdog。

## 关键配置

所有 topic、帧率、同步容差、相机开关、server 地址和超时集中在
`config/pipeline.yaml`。录制和部署都接受 `--config /absolute/path.yaml`；launch
中则使用 `pipeline_config:=/absolute/path.yaml`。
