# 外部依赖

本仓库保存代码及配置，不保存训练权重、运行环境、真实观测和厂商二进制。完整排除清单见 `source_snapshot.json` 的 `excluded` 字段；其中目录条目表示整棵目录未导入。

| 依赖 | 约定位置 / 做法 |
|---|---|
| 50k DCP checkpoint | `WorldAct/model_ch/real_ckpt/singlerighthand-edge-droid-50k-retrain-v1-0831/iter_000040000`，需要含 `model/.metadata` 和实际 shard |
| 训练 frozen config | 同一 run 的 `config.yaml` / `config.deploy.yaml` 已保留，不可随意换成另一任务配置 |
| Edge 模型包、VAE、tokenizer | `WorldAct/models/cosmos3-edge-droid`；这里只保存源码/配置/许可文档，需补齐完整模型资产、tokenizer.json 等依赖 |
| Omni 转换权重 | `WorldAct/omni-wam-lab/artifacts/4w-ema-omni`；使用同一 DCP 的 EMA 导出并转换，脚本为 `export_wam.py`、`convert_wam.py`，保持 provenance 校验数据 |
| 原生 Python | 默认 `WorldAct/WorldAct-sft-pointflow-fk/.venv`，可用 `COSMOS_PYTHON` 指定；依赖参照该项目安装文档 |
| Omni Python | `WorldAct/omni-wam-lab/.venv`；参考 `requirements.lock.txt` 和固定 Omni 源码；锁文件可能保留原机器 editable 路径，安装时替换成本仓库路径 |
| ROS / 厂商 SDK | 参照 teleop 的 docker 和设备 README；`.so`、`.a`、APK、DEB、ELF 等需要另行安装 |
| wujihandros2 | 外部依赖，不纳入本仓库；按 `wuji-hand-teleop-pipeline/patches/WUJIHANDROS2_SUPERVISED_ENABLE.md` 获取基线并应用已保存的补丁 |
| 机器人网格、USD、STEP、仿真资产 | 本次只导入文本代码和配置，二进制及大资产另行取得；缺资产时不能宣称仿真可直接启动 |
| 相机/标定/设备配置 | 使用原项目模板，针对新机器生成；没有迁入原设备的 live config |
| 观测、视频 latent、解码视频、profiling 原始输出 | 留在原工作区或外部实验存储；本仓库保留实验文档及少量统计结论 |

不要直接把旧 `.venv` 移动到新目录：解释器脚本和 editable 安装会保留绝对路径。也不要把完整 `.git` 或失效的 worktree gitfile 搬进来。

本次整理没有重新安装 CUDA / ROS，没有加载模型或下发动作。旧实验记录中的性能数据来自原工作区。迁移后先核对导入源码路径、容器挂载、模型配置与外部资产，再做部署验证。
