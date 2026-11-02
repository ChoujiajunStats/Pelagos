# Pelagos

水下洞穴机器人感知、建图、局部规划与控制。当前仿真主链为 **Stonefish → OKVIS2 / LAS2-S → Wavemap → NBVPlanner → JEPA-MPPI**，通过英文 RViz 原生工具栏操作。

这是轻量源码快照：保留项目代码、配置、测试、控制器源码及接口修改来源；不包含运行日志、Docker 镜像、地图数据、模型权重或完整第三方仓库。现有本机仿真和实验记录保留在原位置。

## 本机演示

在已经准备好镜像、权重及 Porth 场景的实验机器上：

```bash
./Simulation/demo
```

该命令从起点重开仿真，保留旧实验记录。RViz 工具栏：**Explore** 开始、**Pause** 停车、**Manual** 手柄接管、**Auto** 选择自主控制、**Plan Preview** 仅规划预览。

策略为持续前进，前方已知障碍物净距约 3 m 时制动，由 NBV 选择转向，车头对齐后恢复前进。当前验证范围与限制见 [仿真状态](Simulation/STATUS.md)，详细使用见 [Simulation/README.md](Simulation/README.md)。

## 在其他机器恢复

1. 按 [TP/README.md](TP/README.md) 恢复锁定的第三方源码；`TP/Controller` 中的本地控制器代码已随仓库保留。
2. 按 [权重清单](Deployment/artifacts/locks/weights.json) 将已有权重放回所列路径，并核对 SHA-256。权重不会随 Git 下载。
3. 准备 Stonefish 工作区、Porth 场景和基础镜像。外部工作区的版本、本地源码补丁和必要新增文件见 [恢复说明](Simulation/external-workspace/README.md)。当前构建仍依赖本机的 `surfslam:stereo-v2`、`stonefish-planner:omniplanner`；本仓库没有将它们作为公开可拉取镜像发布。
4. 按 [Simulation/README.md](Simulation/README.md) 执行环境检查和构建。`./Simulation/demo` 使用已构建镜像，本身不执行构建。

因此，仅克隆本仓库还不足以在空白系统上运行演示。第三方版本、补丁和外部资产前置条件均单独记录，避免将本机环境误记为独立发行包。

## 组成

| 模块 | 实现 |
| --- | --- |
| 定位 | OKVIS2 |
| 深度估计 | LAS2-S / LAS2-M |
| 建图 | Wavemap Occupancy / nvblox TSDF、ESDF |
| 路径规划 | 仿真 NBVPlanner；Deployment 保留 Wavestar / OmniPlanner 接口 |
| 控制 | JEPA-MPPI / HMU Bridge |
| 可视化 | RViz2 |

## 工程结构

```text
Simulation/               Stonefish 仿真、NBV 适配、RViz 工具和实验测试
TP/                       控制器源码及第三方依赖恢复清单
tools/                    源码恢复工具
Deployment/
├── run.sh                入口
├── contexts/             采集、定位、深度、建图、规划与控制
├── application/          运行
├── contracts/            通信与数据流
├── interfaces/           可视化接口
├── infrastructure/       依赖构建与 ROS 工作区
├── config/               环境配置与相机标定
├── artifacts/            模型权重与来源锁；Git 仅保留锁文件
└── tests/                模块、接口测试、工具
```

## 使用

在项目根目录执行：

```bash
./Deployment/run.sh --help
./Deployment/run.sh build native
./Deployment/run.sh mapping --mapper wavemap --check-only
```

构建依赖 C++17、CMake、Eigen3、Boost、glog 和 yaml-cpp。

实机 Deployment 使用 Ubuntu 22.04、ROS 2 Humble、ZED X/SDK；当前 Simulation 在 Docker 内使用 ROS 2 Jazzy，原版 NBV 使用独立 ROS 1 容器。两者的运行前置条件请分别参照对应文档。
