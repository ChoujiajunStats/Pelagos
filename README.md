# Pelagos

水下洞穴机器人感知、建图、局部规划与控制。

当前仿真链路：**Stonefish → OKVIS2 + LAS2-S → Wavemap → NBVPlanner → JEPA-MPPI**。

## 仿真

在已准备好镜像、权重及 Porth 场景的机器上，于仓库根目录运行：

```bash
./Simulation/demo
```

使用已构建镜像，从起点重开仿真并保留旧记录。RViz：**Explore** 开始，**Pause** 暂停，**Manual** 手动接管，**Auto** 选择自主控制但不启动任务，**Plan Preview** 仅预览。

环境为 Docker / ROS 2 Jazzy，NBV 使用独立 ROS 1 Noetic 容器。构建与操作见 [仿真说明](Simulation/README.md)，验证范围与限制见 [仿真状态](Simulation/STATUS.md)。

## 实机部署

环境：Ubuntu 22.04、ROS 2 Humble、ZED X/SDK。在仓库根目录执行：

```bash
./Deployment/run.sh --help
./Deployment/run.sh build native
./Deployment/run.sh mapping --mapper wavemap --check-only
```

## 环境恢复

本仓库为轻量源码快照，不含权重、镜像、地图或运行日志。依赖与外部资源见：

- [第三方源码与本地控制器](TP/README.md)
- [模型权重路径与 SHA-256](Deployment/artifacts/locks/weights.json)
- [Stonefish 工作区、Porth 场景与基础镜像](Simulation/external-workspace/README.md)
