# Stonefish 仿真

OKVIS2 + LAS2-S → Wavemap → NBVPlanner → JEPA-MPPI。

## 环境

- Docker、NVIDIA Container Toolkit、X11 与 `xauth`；主容器 ROS 2 Jazzy，NBV 容器 ROS 1 Noetic。
- 恢复 [依赖](../TP/README.md) 和 [权重](../Deployment/artifacts/locks/weights.json)，准备 Porth 资产及本地基础镜像 `surfslam:stereo-v2`、`stonefish-planner:omniplanner`。
- Stonefish 工作区与 ROS 1 依赖按 [版本清单](provenance/stonefish-workspace-source.json) 恢复，应用 [补丁](provenance/stonefish-workspace-local.patch)，再合入 `external-workspace/` 的源码目录。

## 运行

在仓库根目录执行，路径替换为实际位置：

```bash
export PELAGOS_STONEFISH_WORKSPACE=/path/to/stonefish_slam_ws
export PELAGOS_PORTH_ASSETS=/path/to/porth_sump9_v1
./Simulation/sim build
./Simulation/sim build --planner-only
./Simulation/demo
```

`demo` 每次从起点重开，保留 `Simulation/runs/`。

RViz：**Explore** 开始，**Pause** 暂停，**Manual** 手动，**Auto** 选择自主控制但不开始任务，**Plan Preview** 仅预览。停止：`./Simulation/sim stop`。

RViz 布局参考 GBPlanner（[许可证](rviz_panel/GBPLANNER_LICENSE)）。
