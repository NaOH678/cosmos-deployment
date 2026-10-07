# Marvin M6 + Wuji Hand + D435 完整 URDF

主模型：`urdf/marvin_wuji_d435_complete.urdf`

模型包含：

- Marvin M6 双臂、本体和支架；
- 左右 Wuji Hand，共 40 个手指转动关节；
- 头部安装链：`Link_Stand → 相机底座 → 头部支架 v3 → D435`；
- 左右腕相机链：`TCP → 腕部相机支架 → D435`；
- 左右灵巧手安装链：`TCP → 直连转接器 → Wuji Hand`；
- 一台头部 D435 和左右腕各一台 D435，共三台相机；
- 每台相机对应一个零偏置的 `*_optical_frame` 几何参考坐标系。

## 固定安装位姿

| 固定关节 | 父链接 | `xyz`（m） | `rpy`（rad） |
|---|---|---:|---:|
| `head_camera_base_joint` | `Link_Stand` | `-0.053 0.0325 0.215` | `-1.57079632679 0 -1.57079632679` |
| `head_camera_bracket_joint` | `head_camera_base_link` | `0.0525 -0.0325 0.0685` | `3.14159265359 1.57079632679 0` |
| `head_d435_mount_joint` | `head_camera_bracket_link` | `-0.04878840219 0.14613259516 0.02` | `0 -1.57079632679 0.698131700798` |
| `left_wrist_camera_mount_joint` | `TCP_Link_L` | `0 0 0.01705` | `3.14159265359 0 0` |
| `right_wrist_camera_mount_joint` | `TCP_Link_R` | `0 0 0.01705` | `0 3.14159265359 0` |
| `left/right_wrist_d435_mount_joint` | 对应腕部支架（椭圆板背面） | `0.001408 -0.0561768603411 -0.0273588457057` | `2.96705972839 0 3.14159265359` |
| `left_hand_direct_adapter_joint` | `TCP_Link_L` | `0 0 0.01705` | `0 0 3.14159265359` |
| `right_hand_direct_adapter_joint` | `TCP_Link_R` | `0 0 0.01705` | `0 0 0` |
| `left/right_hand_mount_joint` | 对应直连转接器 | `0 0 0.030` | `0 0 -1.57079632679` |

头部位姿依据实物照片与 CAD 安装孔重新匹配：相机底座从上一姿态绕 `Link_Stand` 的 \(+Z\) 轴逆时针旋转 \(90^\circ\)，两侧腿的孔中心分别落在 \((x,y,z)=(\pm0.0625,\pm0.02,0.155)\ \mathrm{m}\)，与基座每侧四孔中的上排两孔对齐；头部支架再绕机器人 \(+Z\) 轴顺时针旋转 \(90^\circ\)，其 \(50\times20\ \mathrm{mm}\) 四孔与相机底座顶面的 \(20\times50\ \mathrm{mm}\) 四孔逐一重合；D435 背面孔距为 \(45\ \mathrm{mm}\)，相机孔与顶部椭圆板孔同轴，并安装到椭圆板的另一面；相机局部 \(+Y\) 保持向上，两个对称孔互换对应关系，背面与该侧安装面贴合。所有头部网格均保持原始尺寸，没有缩放。

腕部支架网格已转换到 TCP 安装坐标系，其四孔法兰中心位于 TCP 原点。椭圆板两孔孔距为 \(45\ \mathrm{mm}\)，孔轴在支架局部坐标系中为 \((0,-\sin10^\circ,\cos10^\circ)\)；左右 D435 均以相同的 \(+10^\circ\) roll 安装，背面两孔与支架孔同轴并贴合安装面。由于左右 TCP 坐标系镜像，为同时修正支架正反面并保持椭圆板位于手腕下方，左支架绕 TCP 局部 \(X\) 轴旋转 \(180^\circ\)，右支架绕 TCP 局部 \(Y\) 轴旋转 \(180^\circ\)。

左右手均加入 Wuji 官方 `Direct-Adapter-Mount.step` 直连转接器。转接器机械臂侧基准面位于原始 CAD 的 \(z=0\)，CAD 端面位于 \(z=30\ \mathrm{mm}\)；Wuji 手的局部 \(z=0\) 安装平面与该端面重合，避免转接器穿过手掌主体。两侧直连转接器均绕各自机械臂轴（TCP \(Z\) 轴）额外旋转 \(90^\circ\)；手安装关节反向补偿 \(90^\circ\)，从而保持左、右掌心分别朝向对应腕部相机。官方 STEP 未提供材料或惯性参数；当前模型按铝合金密度 \(2700\ \mathrm{kg/m^3}\) 从 CAD 体积计算质量、质心和惯性矩阵。

## Wuji Hand 资源

手部关节拓扑、位姿、限位、惯性参数和 52 个手部网格均来自 Wuji Technology 官方 `wuji-description` 仓库 `v2026.7.14`（commit `02bddb1dba8646c0c90a3e4e85f75af1bc1c8c81`）；直连转接器来自同一版本 `hand/attachment/step` 中的 `Direct-Adapter-Mount.step`。统一使用同一官方版本可避免旧版关节坐标与新版 STL 混用造成的指节错位。官方文件许可证见 `third_party_licenses/wuji-description-LICENSE`。

## ROS 2 使用

将 `marvin_wuji_d435_description` 放入 ROS 2 工作区的 `src` 后：

```bash
colcon build --packages-select marvin_wuji_d435_description
source install/setup.bash
ros2 run robot_state_publisher robot_state_publisher \
  "$(ros2 pkg prefix --share marvin_wuji_d435_description)/urdf/marvin_wuji_d435_complete.urdf"
```

当前 `*_optical_frame` 用作几何参考，未包含相机内参、畸变或现场标定结果。用于视觉定位前，应以实机标定外参替换相应固定关节位姿。

## 重建

`tools/convert_step_meshes.py` 将四个 STEP（含 Wuji 直连转接器）转为米制 STL；`tools/build_complete_urdf.py` 合并并生成最终 URDF；`tools/validate_complete_urdf.py` 执行树结构、惯性和网格校验；`tools/render_complete_preview.py` 生成零位预览。重建工具的 Python 依赖列在 `tools/requirements.txt`。普通加载和显示不需要运行这些脚本。
