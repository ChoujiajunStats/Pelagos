# Pelagos

水下洞穴机器人感知、建图、局部规划与控制。

当前仿真链路：**Stonefish → OKVIS2 + LAS2-S → Wavemap → NBVPlanner → JEPA-MPPI**。

## 仿真

在已准备好镜像、权重及 Porth 场景的机器上，于仓库根目录运行：

```bash
./Simulation/demo
```

使用已构建镜像，从起点重开仿真并保留旧记录。RViz：**Explore** 开始，**Pause** 暂停，**Manual** 手动接管，**Auto** 选择自主控制但不启动任务，**Plan Preview** 仅预览。

环境为 Docker / ROS 2 Jazzy，NBV 使用独立 ROS 1 Noetic 容器。构建与操作见 [仿真说明](Simulation/README.md)。

## 部署

环境：Ubuntu 22.04、ROS 2 Humble、ZED X/SDK。在仓库根目录执行：

```bash
./Deployment/run.sh --help
./Deployment/run.sh build native
./Deployment/run.sh mapping --mapper wavemap --check-only
```
