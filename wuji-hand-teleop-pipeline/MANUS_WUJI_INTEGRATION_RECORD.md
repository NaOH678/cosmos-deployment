# MANUS 3.1.1 -> WujiHand 集成与排障记录

记录日期：2026-07-15

本文记录 MANUS Metaglove Pro 到 WujiHand 的集成过程、动作异常根因、Python 参考实现、最终修复和常用操作命令。右手已完成真机验证；左手共享映射和专用 retarget 参数已完成对齐，尚待真机验证。

## 1. 当前硬件与软件基线

| 项目 | 当前值 |
|---|---|
| MANUS 手套 | Right Metaglove Pro |
| 右手套 ID | `1742852010` / `0x67E1CFAA` |
| MANUS Dongle ID | `3568420666` / `0xD4B1C73A` |
| MANUS SDK | 3.1.1 Integrated |
| Docker 容器 | `wuji-hand-teleop` |
| ROS 工作空间 | `/home/wuji/ros2_ws` |
| 控制频率 | 约 120 Hz |
| 测试范围 | 右手真机已验证；左手配置已对齐、待验证 |

当前 SDK 文件校验值：

```text
libManusSDK.so
91b42c423e36031c7bade01964c341c9f9423b28aeadd300056792d982c0ccaf

libManusSDK_Integrated.so
0e67141b97b64c089c3bbdab47980ca9822c4de19adea810a2f68722adcb3fe3
```

当前右手标定文件：

```text
文件：manus_ros2/calibration/RightMetaglovePro.mcal
大小：616222 bytes
SHA256：c0e4e80c67db2e55407076cd00f97f4ca5f65d5a5f5a61a30d52bbe88ff65ffe
```

## 2. 最终结论

前后使用的 MANUS SDK 都是 3.1.1。动作怪异不是由 SDK 版本变化直接造成的，而是 3.1.1 数据进入 Wuji retarget 之前的适配层仍然包含旧节点编号和坐标处理假设。

实际数据链路为：

```text
MANUS 3.1.1 原始骨架（25 节点）
    -> manus_ros2/ManusGlove
    -> MediaPipe 风格骨架（21 节点）
    -> wuji_retargeting.Retargeter
    -> WujiHand 20 关节命令
```

主要修复的是 25 节点到 21 节点之间的语义映射、NodeInfo 查询、坐标系处理和 retarget 参数，不是更换 SDK 版本。

## 3. 动作异常根因

### 3.1 NodeInfo 被错误地按 node.id 当数组下标访问

旧代码使用：

```cpp
m_NodeInfo[node.id]
```

但 `node.id` 不是可靠的数组下标。MANUS 3.1.1 的节点 ID 可能从 1 开始，也不保证与 `CoreSdk_GetRawSkeletonNodeInfoArray()` 返回数组的排列位置一致。

结果是一个节点可能读取到另一个节点的：

- `parentId`
- `fingerJointType`
- `chainType`

这会直接造成：

- RViz 手指连线交叉或折返
- 中指在某个弯曲位置突然反向
- 关节语义错位
- 最后一个节点存在越界访问风险

当前实现先建立 `nodeId -> NodeInfo` 映射，再按真实 `nodeId` 查询：

- [ManusDataPublisher.cpp](src/input_devices/manus_input/manus_ros2/src/ManusDataPublisher.cpp)
- [ManusDataPublisher.hpp](src/input_devices/manus_input/manus_ros2/src/ManusDataPublisher.hpp)

### 3.2 Wuji 控制层硬编码了旧 MANUS 数字节点编号

旧控制器通过固定数字 ID 提取 21 个 MediaPipe 节点，例如：

```text
thumb: 22, 23, 24, 25
index: 3, 4, 5, 6
middle: 8, 9, 10, 11
```

这种方式依赖某一版骨架节点布局。MANUS 原始骨架有 25 个节点，非拇指链还包含额外的 Metacarpal 节点，数字编号不能作为稳定接口。

当前控制器改为使用消息中的语义标签：

```text
(chain_type, joint_type)
```

提取规则为：

| MediaPipe 区域 | MANUS 语义节点 |
|---|---|
| Wrist | `Hand / Invalid` |
| Thumb | `MCP, PIP, DIP, TIP` |
| Index | `PIP, IP, DIP, TIP` |
| Middle | `PIP, IP, DIP, TIP` |
| Ring | `PIP, IP, DIP, TIP` |
| Pinky | `PIP, IP, DIP, TIP` |

对应实现：

- [wujihand_node.py](src/controller/controller/wujihand_node.py)

如果 21 个必需语义节点不完整，控制器现在会拒绝该帧，不再用零坐标继续发布机械手命令。

### 3.3 坐标系初始化顺序错误，并额外翻转了 Y 轴

旧 C++ 代码在成员初始化时设置了坐标系，但随后又调用：

```cpp
CoordinateSystemVUH_Init(&m_CoordinateSystem);
```

该初始化会重写结构体内容。旧代码没有在调用后重新写入预期的 `view/up/handedness/unitScale`。

当前顺序为：

```text
CoordinateSystemVUH_Init
    -> 显式设置 XFromViewer / PositiveZ / Right / unitScale=1
    -> CoreSdk_InitializeCoordinateSystemWithVUH
```

同时，旧 Python ROS 控制层还人为执行了：

```python
y = -pose.position.y
```

已验证的 `dexmanip_tool` Python 链路直接使用 MANUS 腕局部坐标，不需要这个额外镜像。当前 ROS 控制器使用原始 `[x, y, z]`。

这部分主要解释了所有手指集体向右偏移或左右镜像的问题。

### 3.4 Retarget 参数放大了错误输入

旧右手配置中的方向权重 `w_dir` 为 `10.0`。当输入节点或坐标有误时，过强的方向项会放大偏差，也可能让优化器在局部解之间跳变。

当前配置与已成功运行的 Python `wuji` 分支所用 Wuji retarget 示例配置对齐：

```yaml
w_dir: 2.0
index: [1.1, 0.989, 1.03]
mediapipe_rotation.z: -15.0
lp_alpha: 0.2
```

对应文件：

- [retarget_manus_right.yaml](src/output_devices/wujihand_output/config/retarget_manus_right.yaml)
- `/home/pjlab/code/dexmanip_tool/third_party/wuji-retargeting/example/config/retarget_manus_right.yaml`

### 3.5 启动缓动不是最终姿态错误的根因

启动缓动只负责从机械手当前关节状态插值到第一帧 retarget 目标。

当第一帧目标本身因为坐标或节点映射错误而向右偏时，缓动会把这个错误表现为“启动后慢慢向右移动”。它不是异常目标的来源，只是让错误过程更明显。

当前右手测试使用：

```text
command_ramp_duration:=0.0
```

这表示第一帧命令立即生效。使用时必须保证机械手周围安全，并在启动前保持手套自然张开。

## 4. Python 版本参考了什么

参考仓库：

```text
/home/pjlab/code/dexmanip_tool
branch: wuji
```

Python 最小链路为：

```text
SDKMinimalClient.out
    -> UDP JSON
    -> ManusInterface
    -> extract_keypoints_21
    -> Wuji 官方 Retargeter
    -> WujiHand 20 关节命令
```

### 4.1 借鉴的方法

最关键的参考是 Python 版本不依赖数字 node ID，而是用：

```python
(chain, finger_joint_type)
```

将 MANUS 25 节点骨架转换成 MediaPipe 21 点。

查看该分支实现：

```bash
git -C /home/pjlab/code/dexmanip_tool show \
  wuji:dexhand_teleoperation/devices/manus_interface.py

git -C /home/pjlab/code/dexmanip_tool show \
  wuji:dexhand_teleoperation/sensors/manus_process.py

git -C /home/pjlab/code/dexmanip_tool show \
  wuji:dexhand_teleoperation/manus_to_wuji.py
```

当前 ROS 实现借鉴了以下行为：

- 使用 chain/joint 语义选择 21 个节点
- 缺少节点时丢弃整帧
- 使用腕局部米制坐标直接进入 Retargeter
- 每侧使用一个有状态 Retargeter
- 使用 Wuji 官方 `retarget_manus_right.yaml`
- Retargeter 输出 20 关节弧度后直接进入 WujiHand 控制层

### 4.2 没有直接照搬的部分

以下修复是 ROS/C++ 路径独立需要的：

- `nodeId -> NodeInfo` 映射
- `CoordinateSystemVUH` 初始化顺序
- ROS2 QoS 和左右手消息过滤
- launch 中只启用右手
- WujiHand 驱动启动缓动参数
- RViz MarkerArray 可视化

## 5. “官方代码”和项目适配代码的边界

### 5.1 MANUS 官方派生部分

`input_devices/manus_input/manus_ros2` 是 MANUS 官方 ROS2 示例派生出来并保存在本项目中的副本。其 `package.xml` 维护者为 MANUS：

- [package.xml](src/input_devices/manus_input/manus_ros2/package.xml)

`ManusDataPublisher.cpp` 中原有的 SDK 初始化、callback、Landscape 和 RawSkeleton 发布框架属于这一层。

这不意味着当前文件与某一版官方发布内容逐字一致；它已经作为项目内副本被集成和修改。

### 5.2 Wuji teleop 项目适配部分

`controller/controller/wujihand_node.py` 中原来的固定 `_MEDIAPIPE_TO_MANUS` 数字映射属于 Wuji teleop 项目的适配代码，不是 MANUS 官方 ROS 代码。

### 5.3 Wuji 官方 Retargeter

`wuji_retargeting.Retargeter` 和 `example/config/retarget_manus_right.yaml` 来自 Wuji retargeting 项目。Python `wuji` 分支和当前 ROS 链路都使用这套 Retargeter。

## 6. SDK 版本与运行时一致性

动作异常即使在日志明确显示 3.1.1 时也存在，所以 3.1.1 本身不是主因。

排障过程中曾有一次进程加载 `CoreLite.Settings.3.0.0.json`，说明容器内一度存在源目录、install 目录和 `/usr/local/lib` 之间的旧动态库混用风险。

当前状态已经统一：

- 编译所用 Integrated SDK 为 3.1.1
- install 目录 SDK 哈希与源目录一致
- `/usr/local/lib/libManusSDK_Integrated.so` 哈希与源目录一致
- `ldd` 当前解析到 `/usr/local/lib/libManusSDK_Integrated.so`
- Docker entrypoint 会在 SDK 文件变化时刷新 `/usr/local/lib`，而不是只在文件不存在时复制

这项修复消除了“同一套命令在不同启动方式下加载不同 SDK”的风险，但它不是中指反折和整体右偏的主要修复。

## 7. 本次验证结果

- 右手套数据约 120 Hz
- RViz 骨架没有观察到手指反向扭曲
- 中指连续弯曲时机械手不再突然反向
- 各手指没有集体向右偏移
- 右手单手启动正常
- 真机抓握和张开测试正常
- `/right_hand/joint_commands` 与 `/right_hand/joint_states` 方向一致
- 左手 `retarget_manus_left.yaml` 已与 Python `wuji` 分支基线对齐
- 左手尚未进行 RViz 连续动作和真机验证

## 8. 标准启动顺序

推荐严格遵守以下顺序：

1. 插入 MANUS Dongle。
2. 打开待测试手套，等待正常无线连接。
3. 确认宿主机没有运行 `SDKMinimalClient.out` 或其他 MANUS Core。
4. 启动 Docker 容器。
5. 二选一：
   - 仅看骨架：手动启动 publisher，再启动 RViz。
   - 控制真机：直接启动 `wuji_teleop_hand.launch.py`，不要另开 publisher。
6. 在第三个终端监控 ROS topic。
7. 停止时在启动终端按 `Ctrl+C`，等待 MANUS Core 正常 shutdown。

同一时间只能有一个 MANUS Core/publisher 访问 Dongle。

## 9. 常用操作命令

以下命令均从宿主机执行。

### 9.1 Docker 日常操作

完整 compose 文件路径：

```text
/home/pjlab/ros2_ws/src/wuji-hand-teleop/docker/docker-compose.yml
```

启动容器：

```bash
docker compose \
  -f /home/pjlab/ros2_ws/src/wuji-hand-teleop/docker/docker-compose.yml \
  up -d
```

查看容器：

```bash
docker ps --filter name=wuji-hand-teleop
```
如果状态是 Up，直接进入；如果容器只是停止了，直接执行：
```bash
docker start wuji-hand-teleop
```
然后继续使用 ```docker exec```。

进入容器：

```bash
cd /home/pjlab/ros2_ws/src/wuji-hand-teleop/src
docker exec -it wuji-hand-teleop bash
```

停止但保留容器内构建结果：

```bash
docker compose \
  -f /home/pjlab/ros2_ws/src/wuji-hand-teleop/docker/docker-compose.yml \
  stop
```

重新启动已停止容器：

```bash
docker compose \
  -f /home/pjlab/ros2_ws/src/wuji-hand-teleop/docker/docker-compose.yml \
  start
```

查看容器日志：

```bash
docker logs -f wuji-hand-teleop
```

### 9.2 完全重建 Docker 镜像

仅在 Dockerfile、系统依赖或 SDK 安装方式变化时使用。普通源码修改只需要在容器内运行 `colcon build`。

```bash
docker compose \
  -f /home/pjlab/ros2_ws/src/wuji-hand-teleop/docker/docker-compose.yml \
  down

docker image rm wuji-hand-teleop:latest

docker compose \
  -f /home/pjlab/ros2_ws/src/wuji-hand-teleop/docker/docker-compose.yml \
  build --no-cache

docker compose \
  -f /home/pjlab/ros2_ws/src/wuji-hand-teleop/docker/docker-compose.yml \
  up -d
```

### 9.3 容器内 ROS 环境

进入容器后：

```bash
source /opt/ros/humble/setup.bash
source /home/wuji/ros2_ws/install/setup.bash
```

也可以从宿主机直接执行单条 ROS 命令：

```bash
docker exec -it wuji-hand-teleop bash -lc '
source /home/wuji/ros2_ws/install/setup.bash
ros2 node list
'
```

### 9.4 仅启动 MANUS 数据 publisher

终端 1：

```bash
docker exec -it wuji-hand-teleop bash -lc '
source /home/wuji/ros2_ws/install/setup.bash
ros2 run manus_ros2 manus_data_publisher
'
```

正常日志应包含：

```text
Manus Core connected.
0xD4B1C73A is connected as MetaglovePro Dongle
0x67E1CFAA is connected as MetaglovePro Glove
Calibration loaded successfully for Right glove
```

### 9.5 启动 MANUS RViz

RViz 不会自动启动 publisher。必须保持 9.4 的终端 1 正在运行。

终端 2：

```bash
docker exec -it wuji-hand-teleop bash -lc '
source /home/wuji/ros2_ws/install/setup.bash
ros2 launch manus_ros2 manus_rviz.launch.py
'
```

### 9.6 监控 MANUS 话题

终端 3：

```bash
docker exec -it wuji-hand-teleop bash -lc '
source /home/wuji/ros2_ws/install/setup.bash
ros2 topic list | grep manus_glove
'
```

查看右手套 topic 的侧别和节点数：

```bash
docker exec -it wuji-hand-teleop bash -lc '
source /home/wuji/ros2_ws/install/setup.bash
ros2 topic echo --once /manus_glove_0 --field side
ros2 topic echo --once /manus_glove_0 --field raw_node_count
'
```

查看频率：

```bash
docker exec -it wuji-hand-teleop bash -lc '
source /home/wuji/ros2_ws/install/setup.bash
ros2 topic hz /manus_glove_0
'
```

如果同时连接两只手套，不要假定 `_0` 永远是右手；使用消息中的 `side` 判断，也要检查 `/manus_glove_1`。

### 9.7 启动右手机械手

启动前：

- 停止手动运行的 `manus_data_publisher`，因为 launch 会自行启动它。
- 保持右手套自然张开。
- 清空机械手周围空间。
- 确认只启用右手。

执行：

```bash
docker exec -it wuji-hand-teleop bash -lc '
source /home/wuji/ros2_ws/install/setup.bash

ros2 launch wuji_teleop_bringup wuji_teleop_hand.launch.py \
  enable_left_hand:=false \
  enable_right_hand:=true \
  command_ramp_duration:=0.0
'
```

立即停止时，在该终端按：

```text
Ctrl+C
```

`command_ramp_duration:=0.0` 表示无启动缓动，第一帧目标会立即发送。若需要保守过渡，可临时改为 `3.0`，但缓动不能修复错误的映射目标。

左手单独启动（配置已对齐，首次真机测试前先用 RViz 检查骨架）：

```bash
docker exec -it wuji-hand-teleop bash -lc '
source /home/wuji/ros2_ws/install/setup.bash

ros2 launch wuji_teleop_bringup wuji_teleop_hand.launch.py \
  enable_left_hand:=true \
  enable_right_hand:=false \
  command_ramp_duration:=0.0
'
```

### 9.8 监控机械手命令和状态

查看目标关节位置：

```bash
docker exec -it wuji-hand-teleop bash -lc '
source /home/wuji/ros2_ws/install/setup.bash
ros2 topic echo --once /right_hand/joint_commands --field position
'
```

查看实际关节位置：

```bash
docker exec -it wuji-hand-teleop bash -lc '
source /home/wuji/ros2_ws/install/setup.bash
ros2 topic echo --once /right_hand/joint_states --field position
'
```

左手目标和实际位置：

```bash
docker exec -it wuji-hand-teleop bash -lc '
source /home/wuji/ros2_ws/install/setup.bash
ros2 topic echo --once /left_hand/joint_commands --field position
ros2 topic echo --once /left_hand/joint_states --field position
'
```

查看频率：

```bash
docker exec -it wuji-hand-teleop bash -lc '
source /home/wuji/ros2_ws/install/setup.bash
ros2 topic hz /right_hand/joint_commands
'
```

### 9.9 检查重复 MANUS 进程

宿主机检查：

```bash
pgrep -af 'SDKMinimalClient|ManusCore|manus_data_publisher'
```

Docker 内检查：

```bash
docker exec wuji-hand-teleop \
  pgrep -af 'manus_data_publisher|SDKMinimalClient|manus_rviz'
```

不要同时运行：

```text
/home/pjlab/code/SDKMinimalClient_Linux/SDKMinimalClient.out
```

和 Docker 内的：

```text
ros2 run manus_ros2 manus_data_publisher
```

### 9.10 检查 USB Dongle

在宿主机执行：

```bash
lsusb | grep -i -E '3325|manus'
```

Dongle 应能被识别。USB 正常但没有手套数据时，优先检查重复 Core/SDK 进程和 publisher 日志。

### 9.11 检查实际加载的 SDK

```bash
docker exec wuji-hand-teleop bash -lc '
ldd /home/wuji/ros2_ws/install/manus_ros2/lib/manus_ros2/manus_data_publisher \
  | grep Manus

sha256sum \
  /home/wuji/ros2_ws/src/input_devices/manus_input/manus_ros2/ManusSDK/lib/libManusSDK_Integrated.so \
  /home/wuji/ros2_ws/install/manus_ros2/lib/manus_ros2/libManusSDK_Integrated.so \
  /usr/local/lib/libManusSDK_Integrated.so
'
```

三个 Integrated SDK 的哈希应一致。

### 9.12 修改源码后的局部重编译

```bash
docker exec -it wuji-hand-teleop bash

cd /home/wuji/ros2_ws
source /opt/ros/humble/setup.bash

colcon build --symlink-install --packages-select \
  manus_ros2 \
  controller \
  wujihand_output \
  wujihand_driver \
  wuji_teleop_bringup

source /home/wuji/ros2_ws/install/setup.bash
```

重编译前停止正在使用相关可执行文件的 ROS 进程。通常不需要重启 Docker 容器。

## 10. 常见现象速查

| 现象 | 优先判断 | 处理方式 |
|---|---|---|
| RViz 打开但没有手 | publisher 没启动 | 先运行 `manus_data_publisher`，再运行 RViz |
| RViz 手指连线交叉 | NodeInfo/parentId 映射错误 | 检查 `nodeId -> NodeInfo` 实现和当前 build |
| 中指在某角度突然反向 | 语义节点错位或 retarget 输入跳变 | 检查 RViz 原始骨架，再检查 25->21 映射 |
| 所有手指集体向右偏 | 坐标系或重复 Y 翻转 | 检查 VUH 初始化顺序，禁止额外 `y=-y` |
| 启动后慢慢向错误方向移动 | 缓动正在插值到错误目标 | 先修映射；缓动不是根因 |
| `Glove data not found` | Landscape 已出现但 RawSkeleton 未到 | 检查手套连接、重复 Core，等待数据流 |
| 话题完全没有数据 | publisher 未运行或 Dongle 被占用 | 检查进程、USB 和 publisher 日志 |
| 白灯闪烁 | 手套处于 pairing 状态 | 这是连接层问题，不是 retarget 问题 |
| 蓝灯正常但 ROS 没数据 | Core/SDK 进程冲突 | 退出所有 MANUS 进程后只启动一个 publisher |
| `no configuration file provided` | 不在 compose 文件目录 | 使用本文完整 `docker compose -f ...` 命令 |

## 11. 后续维护原则

1. 不再通过 MANUS 数字 node ID 选择 MediaPipe 节点。
2. `NodeInfo` 必须通过 `nodeId` 映射查询，不能直接数组索引。
3. 坐标系只在一个明确的位置转换，禁止在多层重复镜像。
4. 修改 SDK 后必须同时检查源目录、install 目录和 `/usr/local/lib` 的哈希。
5. 真机前先用 RViz 验证原始骨架连续性。
6. 真机前先监控 `/right_hand/joint_commands`，确认量级和方向正常。
7. 同一时间只允许一个 MANUS Core/SDK 实例访问 Dongle。
8. MANUS publisher 与真机 launch 二选一启动，避免 launch 内外重复 publisher。
9. 左右手路由使用 `msg.side`，不要依赖 `/manus_glove_0` 和 `_1` 的枚举顺序。
10. Retarget 配置修改应与已验证的 Python 基线做差异比较并重新进行 RViz和真机验证。
