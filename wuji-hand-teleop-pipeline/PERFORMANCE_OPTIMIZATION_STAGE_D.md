# 阶段 D：相机脱离 ROS 2

> 实现日期：2026-07-24
> 基线：阶段 B（`6a65e54`）
> 状态：代码与单路主相机验证完成；双腕相机和机械臂性能对比待现场验收

## 1. 目的

阶段 D 只改变图像数据通路，不修改 Tianji 控制、IK、Tracker 映射、Recovery、
Enable、关节限幅或 Marvin SDK 调度。

原图像路径为：

```text
相机驱动
  -> ROS Image/CompressedImage
  -> DDS 序列化和复制
  -> Recorder 解码
  -> GUI 解码和显示
```

阶段 D 的正式路径为：

```text
三路物理相机
  -> 独立 Camera Manager 进程
  -> /dev/shm 中每路一个 64 帧有界环形缓冲
       |-> GUI：只读最新帧
       |-> Recorder：顺序取走仍在环中的新帧
       `-> Deployment：只读最新帧
```

图像不再发布到 ROS 2，控制进程不参与取图、图像复制、JPEG 编解码或绘制。

## 2. Camera Manager

入口：

```text
ros2 run camera camera_manager
```

它是由 ROS launch 监督的普通独立进程，但本身不创建 ROS node、publisher 或
subscription。

每个在线相机有一个采集线程。当前支持：

- RealSense `d435/d435i/d405`：通过 `pyrealsense2` 按序列号打开；
- UVC/USB：通过稳定的 udev 设备路径和 OpenCV Video4Linux 打开；
- `head`、`left_wrist`、`right_wrist` 三个逻辑角色；
- 只采集 BGR 彩色图，不采集深度图；
- 可选并排双目 UVC 主相机只取左目作为数据集中的 `head` 主视角。

RealSense 不依赖 `/dev/videoN` 顺序。USB相机如果没有可用序列号，必须提供稳定
udev 链接；系统不会回退到任意 `/dev/videoN`，避免重插后左右相机互换。

## 3. 共享内存协议

每路相机对应：

```text
/dev/shm/wuji_camera_v1/<camera_name>.ring
```

每帧包含：

- 自增 sequence；
- 生产者 `monotonic_ns`；
- 生产者 `system_ns`；
- 宽、高、通道数和字节数；
- `uint8 HxWx3 BGR` 图像。

缓冲默认容量为 64 帧。生产者始终覆盖最老帧，不等待任何消费者。

- GUI慢：只跳过显示帧；
- Recorder短时变慢：继续顺序读取尚未被覆盖的帧；
- Recorder落后超过64帧：记录准确的 overwritten 数量，不阻塞相机和控制；
- Camera Manager重启：原子替换共享文件，消费者按 producer generation 自动重连；
- Camera Manager停止：消费者按单调时间戳将画面判定为过期，不显示旧会话图像。

## 4. Recorder

默认配置：

```yaml
recording:
  camera_transport: "direct"
  camera_shared_memory_dir: "/dev/shm/wuji_camera_v1"
```

Recorder以200 Hz非阻塞检查共享环，只在有新帧时复制图像，并继续使用原有：

- 在线相机子集选择；
- 30 Hz相机anchor；
- 标量/手/Tracker/MANUS时间对齐；
- LMDB、MP4、`meta_info.pkl`和`sync_timestamps.json`格式。

训练数据字段、54维 `qpos/action`、视频名称和Replay读取格式没有变化。

Recorder状态增加：

```text
camera_transport
camera_overwritten_frames
camera_capture_failures
```

episode metadata中的 `camera_layout` 会标记
`transport=direct_shared_memory`、`pixel_format=bgr8`和生产者时钟语义。

## 5. GUI

数采GUI的ROS线程仍只负责生命周期、Recorder状态和轻量硬件缓存。默认不再创建
任何图像topic subscription。

独立预览线程以最多30 Hz读取每路最新共享帧：

- 只处理新的 sequence；
- 不积压历史画面；
- 超过2秒的旧帧不显示；
- 实际连接几路就显示几路；
- 固定显示语义为主视角、左腕、右腕。

独立 Camera Preview GUI也使用相同的三路共享内存路径。

## 6. Deployment

`deployment.launch.py`和`deployment_node`默认使用同一个Camera Manager与最新帧
共享内存，不再依赖ROS图像topic。策略请求仍为约30 Hz，每次只发送当前最新图像。

## 7. 迁移回退开关

正式默认值为：

```text
camera_transport=direct
```

如需与旧实现做临时对比，可以使用：

```bash
WUJI_CAMERA_TRANSPORT=ros ./src/scripts/start_record_gui.sh
```

或：

```bash
./src/scripts/start_record_session.sh right \
  --task exchange_tubes \
  --with-camera \
  --camera-transport ros
```

`ros`只作为迁移期回退模式，不是阶段 D 的正式运行方式。

## 8. 常用启动方式

正常使用方式不变：

```bash
cd /home/pjlab/ros2_ws/worktrees/wuji-hand-teleop-pipeline
./src/scripts/start_record_gui.sh
```

GUI点击“进入准备”后，会统一启动 OpenVR、阶段 B controller、手、Camera Manager
和Recorder。

CLI：

```bash
cd /home/pjlab/ros2_ws/worktrees/wuji-hand-teleop-pipeline
./src/scripts/start_record_session.sh right \
  --task exchange_tubes \
  --handoff-ramp-sec 6.0 \
  --with-camera
```

## 9. 已完成验证

2026-07-24完成：

- 共享环写入、读取、覆盖计数和producer重启测试；
- Camera Manager通过序列号打开主相机 `147122073219`；
- 实测输出约30 Hz、`640x480x3 uint8 BGR`；
- Docker到宿主机 `/dev/shm` 跨进程读取成功；
- direct launch中不存在 `/cam_*` 或 `/stereo/*` 图像topic；
- Camera Manager与Recorder direct模式联合启动成功；
- `camera`、`wuji_data_pipeline`、`wuji_teleop_monitor`测试通过；
- 当前累计结果为150 tests、0 errors、0 failures；
- Docker镜像 `wuji-hand-teleop:latest`重建成功并包含
  `pyrealsense2==2.58.3.10794`。

## 10. 仍需现场验收

当前配置中的两路腕部相机序列号仍是placeholder，所以本次只能验证一路主相机。
阶段 D 的代码实现已完成，但进入阶段 E 前还应执行：

1. 填写左腕、右腕真实RealSense序列号；
2. 分别验证1、2、3路在线子集；
3. GUI idle 60秒性能采样；
4. GUI recording 60秒性能采样；
5. 保存一条带视频episode并核对帧数、视频和时间戳；
6. 正常退出并确认Tianji双臂standby。

阶段 D 不解决已有的IK无解、左臂可达性或 `TARGET_HOLD`。这些属于独立控制问题，
不能通过修改相机代码或放宽安全阈值掩盖。
