# Tianji + Wuji 遥操、数采与 Replay 管线完整重构方案

> 版本：2026-07-24  
> 工作分支：`refactor/staged-data-pipeline`  
> 当前状态：阶段 A、B 已完成；阶段 C 实机验证失败并已回退；阶段 D 已实现，
> 待双腕相机和机械臂性能现场验收

## 1. 文档目的

本文档描述当前完整方案，不只包含下一步的阶段 B。方案覆盖：

- Tianji 双臂 Tracker 遥操；
- MANUS 到 Wuji Hand 的手部遥操；
- 三路相机采集与 GUI 显示；
- 训练数据同步、保存和任务目录管理；
- Tracker/MANUS 原始诊断数据；
- 单手、双手数据兼容；
- Replay 和云端部署；
- GUI、脚踏板和整个会话的启停；
- 控制卡顿、IK HOLD 和系统性能的分阶段治理。

目标不是一次性重写所有代码，而是在每个阶段只改变一个主要变量，完成实机
验证后再进入下一阶段。这样出现问题时能够确定原因，并可随时回退到上一阶段。

## 2. 最终目标

最终系统应满足以下要求：

1. GUI、相机、Recorder 和磁盘速度不能阻塞 120 Hz 机械臂控制；
2. GUI 只读取状态快照，不直接周期性查询 Tianji SDK；
3. Marvin SDK 最终只有一个明确的访问入口，所有读写有序执行；
4. 相机图像最终不经过 ROS 2，直接由独立相机进程提供给 GUI 和 Recorder；
5. GUI 只显示最新图像，显示卡顿时不积压历史帧；
6. Recorder 使用有界队列和后台写入，磁盘变慢时不能反向阻塞控制；
7. 保持现有训练数据格式、任务目录、单手补零逻辑和 Replay 兼容；
8. 保持已经验证的 Recovery、Enable、Tracker 映射和安全保护；
9. IK 无解或跳变仍然 HOLD，不能通过取消保护或盲目放宽阈值掩盖问题；
10. 一个数采会话由一个入口统一启动和关闭，正常退出必须确认 Tianji
    `state=0`。

## 3. 不允许被破坏的现有约束

### 3.1 遥操控制来源

Tianji 双臂必须继续使用当前项目已经验证的 Tracker 遥操实现，并遵循：

- `TIANJI_VIVE_TELEOP_RECORD.md`
- `src/controller/TIANJI_VIVE_SAFETY.md`

MANUS 到 Wuji Hand 必须继续遵循：

- `MANUS_WUJI_INTEGRATION_RECORD.md`

不从 `dexmanip_tool` 的 Tianji 控制代码替换当前控制器，也不引入 PICO。

### 3.2 当前控制与安全参数

后续性能重构不能顺便修改这些参数：

```text
control_rate                     120 Hz
state_publish_rate               500 Hz（先保持，后续单独实验）
teleop_position_scale            1.0
impedance_velocity_ratio         30
impedance_acceleration_ratio     30
handoff_hold_sec                 0.2 s
GUI handoff_ramp_sec             6.0 s
recovery_max_speed_deg_s         5.0°/s
recovery_max_accel_deg_s2        10.0°/s²
impedance_max_drift_deg          3.0°
```

如果以后需要修改控制参数，必须作为独立实验，有修改前后的相同基线，不能和
线程、GUI、相机或 Recorder 重构混在同一个提交中。

### 3.3 数据格式

训练帧继续保持：

```text
qpos:
  arm_left(7) + hand_left(20) + arm_right(7) + hand_right(20)
  = 54

action:
  eef_left(7) + hand_left(20) + eef_right(7) + hand_right(20)
  = 54

eef observation:
  eef_left(7) + eef_right(7)
  = 14
```

其中：

- `qpos` 是机械臂和手的实际状态；
- `action` 是双臂目标末端位姿和双手目标；
- `eef observation` 是双臂实际末端位姿；
- 只有一只物理 Wuji Hand 时，缺失侧 20 维状态和目标都补零；
- Tracker/MANUS 原始数据是分析字段，不参与训练源就绪判断；
- 原始诊断数据缺失时写 `available/valid=0`，不能导致训练帧被跳过；
- 不增加深度图；
- 相机位置为 `head`、`left_wrist`、`right_wrist`，每条轨迹只保存本次实际
  在线的相机子集。

### 3.4 任务与轨迹目录

数据根目录继续使用项目约定的：

```text
datasets/tianji_wuji/<task_name>/episode_xxxx_<timestamp>/
```

- GUI 打开时选择旧任务或创建新任务；
- 任务名只能使用 ASCII 字母、数字、`_` 和 `-`；
- 一个任务对应一个文件夹；
- 一个任务文件夹中保存该任务的多条轨迹；
- 最终数据仍包含 LMDB、视频、`meta_info.pkl` 和
  `sync_timestamps.json`。

## 4. 已确认的当前问题

阶段 A 三组 60 秒实测结果如下：

| 指标 | `arm_only` | `gui_idle` | `gui_recording` |
| --- | ---: | ---: | ---: |
| 实际控制频率 | 119.98 Hz | 113.64 Hz | 113.96 Hz |
| 控制周期最大值 | 13.33 ms | 70.61 ms | 67.07 ms |
| 控制调度间隙 | 1 | 60 | 61 |
| `arm_status` 调用次数 | 0 | 61 | 60 |
| `arm_status` 平均耗时 | 无调用 | 57.20 ms | 57.50 ms |
| TARGET_HOLD 帧数 | 230 | 101 | 2138 |
| TARGET_HOLD 状态切换 | 7 | 8 | 1259 |

已经得到三个结论：

1. GUI 每秒一次的完整 `arm_status` 查询是确定的周期性控制阻塞源；
2. 当前 Recorder 没有表现出明显的额外控制降频，但架构仍需异步化；
3. IK/TARGET_HOLD 是另一条独立的卡顿路径，不能归因于 Recorder。

完整原始结论见 `PERFORMANCE_BASELINE_STAGE_A.md`。

## 5. 目标架构

```text
OpenVR Tracker
      │
      ▼
LatestTrackerTarget ────────┐
                            ▼
                    120 Hz 控制计算
                 映射 → 滤波 → IK → 安全检查
                            │
                            ▼
                     LatestCommand
                            │
                            ▼
                  Marvin SDK 唯一访问线程
                   状态读取 + 最新命令发送
                            │
                            ▼
                    RobotStateCache
                      │      │
                      │      ├── ROS 状态/EEF/关节话题
                      │      ├── Recorder 标量输入
                      │      └── GUI 轻量状态
                      │
                      └── 故障时的详细诊断请求


MANUS SDK/ROS 输入 ──► Wuji Hand 控制
        │                    │
        └────原始数据────────┴──► Recorder 诊断字段


三路相机设备
      │
      ▼
独立 Camera Manager（不经过 ROS 2）
      │
      ├── LatestFrame 共享内存 ──► GUI
      │
      └── 有界帧环形队列 ───────► Recorder


RobotStateCache + 手部状态 + 相机帧 + 原始诊断
      │
      ▼
同步器/有界写入队列
      │
      ▼
后台 Episode Writer
      │
      ├── LMDB
      ├── MP4
      ├── meta_info.pkl
      └── sync_timestamps.json
```

核心原则是：

```text
GUI 是观察者，不是硬件访问者
相机显示只取最新帧
需要保存的历史数据使用有界队列
控制路径永远不等待 GUI、相机编码或磁盘
```

## 6. 分阶段实施方案

## 阶段 A：建立性能基线（已完成）

### 目的

在不改变控制行为的前提下，测量：

- 120 Hz 控制定时器；
- 500 Hz SDK 状态读取；
- 60 Hz EEF 发布；
- SDK 读取、IK、命令发送耗时；
- GUI `arm_status` 耗时；
- TARGET_HOLD 和调度间隙。

### 实现内容

- 增加 `/tianji_arm/performance_metrics`；
- 增加有界内存统计窗口；
- 增加 `collect_performance_baseline.sh`；
- 分别采集 `arm_only`、`gui_idle`、`gui_recording`。

### 已达成效果

- 确认 arm-only 可以稳定达到约 120 Hz；
- 确认 GUI 完整硬件查询每秒阻塞约 57 ms；
- 确认 GUI 场景约 60 次调度间隙与约 60 次查询对应；
- 确认 Recorder 不是当前约 114 Hz 的主要原因；
- 确认录制过程还存在独立的 IK/HOLD 问题。

### 验收状态

已完成。提交和结果记录在 `PERFORMANCE_BASELINE_STAGE_A.md`。

## 阶段 B：GUI 状态缓存与硬件查询分级

### 目的

消除已经确认的“GUI 每秒访问完整 Tianji SDK，导致控制线程停顿约 57 ms”
的问题，同时保留 Recovery 和关闭机器时的硬件安全确认。

### 实现内容

1. 在 Tianji 控制节点中维护 `RobotStatusSnapshot`；
2. 快照只使用控制器已经获得的数据，不为 GUI 新增 SDK 读取；
3. 快照至少包含：
   - 时间戳和序列号；
   - lifecycle；
   - `teleop_status`；
   - 双臂是否 Enable；
   - Tracker 是否连接及数据年龄；
   - 当前 HOLD 原因；
   - SDK 连接/故障锁存状态；
   - 最近一次详细硬件诊断结果及其年龄；
4. 以 2～5 Hz 发布轻量级缓存状态；
5. GUI READY 状态只订阅缓存，不再每秒调用 `/arm_status`；
6. 现有完整 `/arm_status` 保留，用于：
   - Recovery 前检查；
   - Recovery 失败；
   - SDK/伺服故障；
   - Standby 和退出确认；
   - 操作员点击“刷新详细故障”；
7. GUI 不允许重叠查询；上一次手动诊断未完成时拒绝再次发起。

### 预期达成效果

- GUI 打开时控制频率从约 114 Hz 恢复到接近 arm-only 的 120 Hz；
- READY 状态下 `arm_status.calls=0`；
- 不再出现每秒一次、约 57 ms 的控制周期长停顿；
- GUI 仍能显示 READY、HOLD、故障、Tracker 和 Recorder 状态；
- Recovery 和退出仍使用真实硬件状态验证，不降低安全性。

### 验收标准

- `gui_idle` 和 `gui_recording` 实际控制频率不低于 119 Hz；
- 60 秒内 GUI 引起的控制调度间隙不超过 5 次；
- 正常 READY 期间完整 `arm_status` 调用次数为 0；
- Recovery、故障刷新和退出确认测试全部通过；
- 与阶段 A 使用完全相同的三组基线重新对比。

### 已达成效果

- `gui_idle` 控制频率由 113.64 Hz 恢复到 120.00 Hz；
- `gui_recording` 控制频率由 113.96 Hz 恢复到 120.00 Hz；
- 两组 60 秒测试的调度间隙由 60/61 次降至 1/0 次；
- READY 期间完整 `arm_status` 调用次数均为 0；
- 最大控制周期由约 67～71 ms 降至约 11～13 ms；
- 2 Hz 状态快照平均耗时约 0.06 ms；
- Recovery、Enable、录制、保存和退出流程通过实机验证；
- 控制参数、IK 保护、训练数据格式和 Replay 均未改变。

退出时双臂已经确认到达 `cur_state=0` 且 SDK 已执行 `Robot released`，但
控制器进程随后返回 `-11`。该 SDK/Python 资源释放问题不影响本阶段 GUI
实时性能结论，作为阶段 C 的生命周期隔离与退出回归项继续处理。

### 验收状态

已完成。实现和实测记录见 `PERFORMANCE_OPTIMIZATION_STAGE_B.md`。

## 阶段 C：Marvin SDK 单一访问与控制路径隔离

### 当前状态

实验失败，已回退到阶段 B，当前后续阶段不得基于该实现继续开发。

失败版本虽然保持约120 Hz控制和约500 Hz SDK读取，但同时改变了实时IK参考关节、
同步命令发送、关节限幅层数和Executor调度语义。实机出现更严重卡顿以及末端运动
方向偏差，因此不满足“控制结果与改造前数值一致”的验收标准。

以后如重新进入阶段 C，必须拆成单变量A/B实验，不能再次把缓存IK参考、异步命令、
重复限幅和多线程Executor放在同一版本中。

### 目的

保证任何 SDK 调用变慢、错误查询或状态读取抖动时，都不会直接阻塞
120 Hz 的 Tracker 映射、IK 和安全计算。

阶段 A 表明单次 SDK 读取目前通常很快，因此本阶段不是为了盲目降低频率，
而是为了建立可预测、可监控且不会并发访问 SDK 的边界。

### 实现内容

1. 建立唯一的 SDK I/O 工作线程；
2. Marvin SDK 的状态读取、命令发送、Recovery、Enable、错误查询都通过该
   线程有序执行；
3. 控制计算线程不再直接调用 SDK；
4. 控制线程写入深度为 1 的 `LatestCommand`：
   - 新命令覆盖未发送的旧命令；
   - SDK 恢复后只发送最新目标；
   - 不追赶已经过期的历史命令；
5. SDK 线程更新 `RobotStateCache`；
6. IK 使用带时间戳的最新状态快照；
7. 增加状态陈旧 watchdog：
   - 状态超过允许年龄时进入 HOLD；
   - 不能继续使用过期状态发送命令；
8. Recovery/Enable 作为 SDK 线程的独占事务执行，保持现有生命周期；
9. 记录 SDK 队列等待、读取、发送、状态年龄和覆盖次数；
10. 500 Hz 状态读取先保持不变；隔离完成后再分别测试 500、250、200 Hz，
    以数据决定是否降低，不能凭感觉修改。

### 预期达成效果

- SDK 不再被多个 ROS 回调或工作线程并发访问；
- 某次 SDK 查询变慢时，控制计算仍可按时运行；
- 不积压过时的 Tracker 和关节命令；
- GUI、Recorder 和 ROS 发布都读取同一个状态快照；
- SDK 断连或状态陈旧时明确 HOLD/故障，不产生失控命令；
- Recovery、Enable、Standby 行为与现有验证版本一致。

### 验收标准

- 代码层面只有 SDK 工作线程能调用 Marvin SDK；
- 120 Hz 控制路径中没有同步 SDK 调用；
- 人工注入 20～100 ms SDK 延迟时不形成历史命令追赶；
- 状态超时能够稳定进入 HOLD；
- Recovery、Enable、退出和急停恢复回归通过；
- 控制结果与改造前在相同输入下数值一致。

## 阶段 D：相机脱离 ROS 2

### 当前状态

代码实现和单路主相机验证已完成，详见
`PERFORMANCE_OPTIMIZATION_STAGE_D.md`。双腕序列号、三路实机在线子集和带机械臂
性能基线仍需现场验收。

### 目的

避免图像发布、DDS 序列化、复制、JPEG 解码和 GUI 绘制造成 ROS 负载或与
控制资源竞争。最终 GUI 和 Recorder 都不通过 ROS 图像话题取图。

### 实现内容

1. 新建独立 `Camera Manager` 进程；
2. 通过设备序列号或稳定 udev 标识识别：
   - `head`；
   - `left_wrist`；
   - `right_wrist`；
3. 不依赖可能变化的 `/dev/videoN` 顺序；
4. 每个相机在取到帧时立即记录单调时钟和系统时钟；
5. 每路相机提供两条输出：
   - GUI：共享内存 `LatestFrame`，新帧覆盖旧帧；
   - Recorder：有界环形帧队列，正常情况下保存完整 30 Hz；
6. GUI 读取慢时只丢显示帧，不能让相机或 Recorder 等待；
7. Recorder 队列满时记录丢帧数量和时间范围，不允许反向阻塞控制；
8. GUI 根据实际在线相机显示 0～3 路画面，不要求三路全部连接；
9. 主视角固定映射为 `head`；
10. 删除最终运行图中的 ROS 相机图像订阅，但保留一个迁移期开关便于对比；
11. 不采集深度图。

### 预期达成效果

- 相机图像不再经过 ROS 2/DDS；
- GUI 卡顿不会形成未处理图像队列；
- 一个相机离线不影响其余已连接相机显示；
- 图像编码和显示不会阻塞机械臂控制；
- Recorder 仍获得带准确时间戳的 30 Hz 相机帧；
- CPU、内存复制和 ROS 带宽明显下降。

### 验收标准

- 正式运行时不存在 `/cam_*/...image...` 的数据依赖；
- 单路、双路、三路相机分别测试通过；
- 拔掉任意一路相机后其余画面和控制继续运行；
- GUI 停止刷新 10 秒后恢复，不追赶旧图像；
- Recorder 视频帧数、时间戳和实际在线相机列表一致；
- 开关 GUI 预览不改变控制性能基线。

## 阶段 E：Recorder 同步与磁盘写入彻底异步化

### 目的

当前 Recorder 已经是独立 ROS 进程，因此没有直接运行在控制节点内；但同步、
图像解码和 `EpisodeWriter.append()` 仍在 Recorder 的定时器路径中执行。
本阶段要保证磁盘或视频编码变慢时，只影响 Recorder 自己的队列和质量统计，
不能影响控制、GUI或硬件状态。

### 实现内容

1. 将 Recorder 分为三个职责：
   - 输入缓存与时间戳；
   - 30 Hz 同步器；
   - 后台 Episode Writer；
2. 同步器构建完整训练帧后执行非阻塞 `try_push()`；
3. 使用有界队列，记录：
   - 当前深度；
   - 最大深度；
   - 丢帧数；
   - 写入耗时；
   - 队列等待年龄；
4. LMDB、MP4 和元数据写入只在后台写线程/进程执行；
5. 队列满时不能等待磁盘，必须明确丢弃并记录；
6. 使用 `.inprogress` 临时目录；
7. 保存完成后原子地变为正式 episode；
8. 保持现有脚踏板状态机：
   - 踏板 1：开始采集；
   - 再踩踏板 1：结束采集，进入待决定状态；
   - 踏板 2：保存已经结束的轨迹；
   - 未保存时再次踩踏板 1：丢弃上一条并开始新轨迹；
   - 踏板 3：只允许在非录制状态断联/重连；
   - 踏板 3 不能丢弃已经结束但尚未保存的轨迹；
9. `e/q/Ctrl+C` 保持“丢弃未保存轨迹、请求 Standby、关闭全部会话”的语义；
10. 保存过程在 GUI 显示进度，不能让机械臂控制等待；
11. 保持现有 LMDB schema、视频命名、时间戳和单手补零逻辑。

### 预期达成效果

- 磁盘短时抖动不会造成控制卡顿；
- 停止采集可以快速进入待保存状态；
- 保存过程不会阻塞机械臂控制，退出时有明确的完成或失败结果；
- 数据过载时有明确的丢帧和队列指标，而不是系统无响应；
- 旧训练和 Replay 代码仍可读取新数据。

### 验收标准

- 人工限制磁盘速度时控制频率不下降；
- Recorder 队列不会无限增长；
- 每个 episode 的帧数、视频帧数和时间戳可核对；
- 保存、丢弃、重新开始、Ctrl+C 四种状态机测试通过；
- 进程异常退出后 `.inprogress` 不会被误认为有效训练数据；
- 现有两条历史数据和新数据均可通过同一读取器加载。

## 阶段 F：GUI 变为纯观察与会话控制端

### 目的

让 GUI 只负责显示和低频控制命令，不参与高频控制、SDK读取、图像采集、
数据同步或磁盘写入。

### 实现内容

1. GUI 状态全部来自缓存快照；
2. 相机画面来自 Camera Manager 的共享内存；
3. 不在 Qt 主线程中做图像解码、硬件查询或保存；
4. 不同内容使用不同刷新频率：
   - lifecycle/故障：事件驱动；
   - 轻量状态：2～5 Hz；
   - 关节/EEF 显示：10～30 Hz；
   - 相机预览：按最新帧 10～30 Hz；
   - 详细伺服诊断：故障或人工刷新；
5. 上一次操作未完成时不重复发起相同请求；
6. GUI 继续统一启动：
   - OpenVR；
   - Tianji controller；
   - MANUS/Wuji Hand；
   - Camera Manager；
   - Recorder；
7. 默认启用相机和双手，允许选择单手和禁用相机；
8. 保持三个脚踏板接口和对应 GUI 按钮；
9. Recovery、Enable 和 Exit 仍由 GUI 明确按钮触发；
10. GUI 关闭或会话异常时执行统一的受控退出；
11. 日志区只显示去重后的关键事件，高频原始日志写入文件。

### 预期达成效果

- GUI 刷新快慢不会改变 SDK 或控制访问频率；
- GUI 画面卡顿不会导致机械臂卡顿；
- 操作员可以看到在线设备、当前生命周期、录制状态、保存状态和异常原因；
- 单入口启动、单按钮退出，避免多个终端残留；
- Ctrl+C 和 GUI Exit 都能关闭整套会话并确认双臂 Standby。

### 验收标准

- GUI 主线程执行耗时和队列长度可监控；
- GUI 强制暂停 10 秒时控制与录制保持运行；
- 关闭任意相机预览不改变数据采集；
- 重复快速点击按钮不会产生重叠服务请求；
- GUI Exit 后 Tianji 确认 `state=0`，相关子进程全部退出；
- 脚踏板状态机与按钮状态机完全一致。

## 阶段 G：Tracker/IK 只读诊断

### 目的

定位 IK 无解、奇异区域、Tracker 映射或瞬时目标跳变问题，但不改变 Wuji
原始仓库的 IK 下发语义。

### 实现内容

1. 使用已经记录的 Tracker/MANUS 原始数据与训练帧做离线对齐分析；
2. 按左右臂分别统计：
   - IK 无解；
   - 相邻 IK 解变化；
   - 末端不可达；
   - Tracker 数据陈旧；
3. 保存 Tracker 位姿、IK seed、IK成功标志和候选关节解；
4. 保持每侧独立语义：某侧 IK 失败只跳过该侧当帧命令；
5. Clutch 重连时继续重新锚定当前人体和机器人姿态；
6. 不在控制链路增加 Tracker 拒绝、滤波、首帧阈值、相邻解阈值、空间偏移
   阈值或软件关节步长限制。

### 预期达成效果

- 能从原始 Tracker 和 IK 诊断中定位卡顿来源；
- 左右臂 IK 成功率、异常设备和异常姿态可量化；
- 诊断记录不改变控制目标、控制频率或训练数据格式。

### 验收标准

- 使用固定的代表性操作任务重复测试至少 10 条轨迹；
- Tracker 遥操不因软件保护进入 TARGET_HOLD；
- 某侧 IK 无解时另一侧继续发送有效 IK 解；
- 有效 IK 解在 handoff 后不经后加保护直接发送；
- 与当前映射对比确认无坐标系、左右臂或方向回归。

## 阶段 H：数据兼容、Replay 与云端部署验收

### 目的

确保性能和架构重构没有破坏现有训练格式、单手模式、Replay 和云端策略部署。

### 实现内容

1. 对历史 episode 和新 episode 执行统一 schema 校验；
2. 验证：
   - `qpos=54`；
   - `action=54`；
   - `eef=14`；
   - 单手模式缺失侧补零；
   - 三路相机在线子集；
   - Tracker/MANUS 原始诊断有效位；
3. Replay 使用与云端部署一致的请求/响应通道；
4. Replay 在发送第一帧前执行：
   - 数据完整性检查；
   - 当前机器人状态检查；
   - 第一帧距离检查；
   - 必要的 rebase/handoff；
5. Replay 只在 Tianji lifecycle 为 READY 时推进；
6. TARGET_HOLD、SDK 故障或网络超时时停止推进，不能积压动作；
7. 单手数据允许 Replay，缺失手侧保持零值且不能产生硬件命令；
8. 云端策略请求使用有界超时和最新动作语义；
9. 保存端到端版本、配置和数据元信息。

### 预期达成效果

- 现有训练程序不需要因重构修改核心维度；
- 历史数据与新数据可以使用同一工具读取；
- Replay 与云端部署共享同一安全动作入口；
- 单手、双手、不同相机在线组合都可以正确运行；
- 网络或策略服务变慢时不追赶旧动作。

### 验收标准

- 历史 episode、新 episode schema 测试全部通过；
- 单手与双手 Replay 分别完成；
- 第一帧差距过大时明确拒绝，不突然跳动；
- Replay 中断、Ctrl+C 和网络超时均能请求 Standby；
- 完成至少一次本地 Replay 和一次云端部署闭环测试。

## 7. 各阶段之间的依赖关系

```text
阶段 A 性能基线（完成）
        │
        ▼
阶段 B GUI 状态缓存
        │
        ▼
阶段 C SDK/控制隔离
        │
        ├──────────────┐
        ▼              ▼
阶段 D 相机直连     阶段 G IK/HOLD 独立治理
        │
        ▼
阶段 E Recorder 异步化
        │
        ▼
阶段 F GUI 最终集成
        │
        ▼
阶段 H 数据、Replay、云端验收
```

阶段 G 可以在阶段 C 稳定后与 D/E 的非控制工作并行分析，但其控制代码提交和
实机验证必须独立。

## 8. 每个阶段统一执行的验证流程

每阶段都执行：

1. 单元测试和静态检查；
2. Docker/ROS 2 构建；
3. 无硬件或 mock 测试；
4. Recovery；
5. Enable；
6. arm-only 60 秒基线；
7. GUI idle 60 秒基线；
8. GUI recording 60 秒基线；
9. 保存一条测试 episode；
10. Replay 读取或 schema 校验；
11. 正常退出并确认双臂 `state=0`；
12. 结果写入文档后再提交。

如果某阶段控制频率、安全状态或数据兼容性退化，则停止进入下一阶段，回退
该阶段实现并根据基线定位。

## 9. 最终达成效果

全部阶段完成后，系统应呈现以下行为：

```text
机械臂控制稳定运行在接近 120 Hz
GUI 不再造成每秒一次的 57 ms 卡顿
SDK 访问单线程、有序、可监控
Tracker 和关节命令不积压旧值
相机图像不经过 ROS 2
GUI 只读取最新画面
Recorder 和磁盘不阻塞控制
三路相机按在线子集工作
单手数据保持 54 维格式并可 Replay
Tracker/MANUS 原始数据与训练帧一一对应
IK HOLD 原因可分析且不通过取消保护掩盖
GUI、脚踏板、Ctrl+C 都能统一结束会话并确认 Standby
本地 Replay 和云端部署使用同一安全动作管线
```

## 10. 当前下一步

当前控制基线保持阶段 B。阶段 C 已回退，阶段 D 已实现。下一步先完成阶段 D
现场验收：

```text
填写双腕真实序列号
验证1/2/3路在线相机
采集GUI idle和recording 60秒性能指标
保存并检查一条带视频episode
确认正常退出和双臂standby
```

验收通过后进入阶段 E。阶段 C 保持暂停，不能混入阶段 D/E 的修改。
