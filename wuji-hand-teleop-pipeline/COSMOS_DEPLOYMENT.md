# Cosmos 部署关联：wuji-hand-teleop-pipeline

更新：2026-10-07。本目录负责：**机器人控制与实验记录**。

当前 async+blend 和 smoothstep 位于本仓库；推理加速还依赖下面三个WorldAct目录。

关键文件：`src/scripts/start_local_cosmos_deployment.sh`；`src/wuji_data_pipeline/wuji_data_pipeline/{deployment_node,paper_async_blend,paper_observation_clock}.py`；`src/wuji_data_pipeline/config/cosmos_protocol_v2_50k_retrain_async_blend.yaml`。

- [跨仓库导航](../WorldAct/COSMOS_DEPLOYMENT_MAP.md)：启动链路、四个目录职责与迁移说明。
- [完整加速和async+blend实验记录](docs/cosmos_5090_optimization_experiments_20261007.md)：当前配置、会话结果、证据、启动和回退。
- [VS Code多目录工作区](../WorldAct/cosmos-deployment.code-workspace)。

完整部署关联 `wuji-hand-teleop-pipeline`、`WorldAct/WorldAct-sft`、`WorldAct/WorldAct-sft-pointflow-fk`、`WorldAct/omni-wam-lab`。详细记录的逻辑路径为 `wuji-hand-teleop-pipeline/docs/cosmos_5090_optimization_experiments_20261007.md`；仅复制本仓库时可能没有兄弟目录，请恢复目录布局或调整链接。

当前基线：单卡5090、50k/40000 EMA、Omni、32步15Hz、stride16、smoothstep臂手同权。参数和实测结果以完整实验记录及每轮配置快照为准，不将本页当成永久锁定的运行配置。
