# 第三方依赖与控制器

`Controller/` 包含 JEPA-MPPI 和 HMU Bridge 源码；其他依赖按 [锁定清单](dependencies.lock.json) 恢复。

## 恢复

在仓库根目录运行：

```bash
python3 tools/bootstrap_dependencies.py                          # 查看计划
python3 tools/bootstrap_dependencies.py --apply                  # 克隆缺失源码
python3 tools/bootstrap_dependencies.py --apply --only nbvplanner # 指定依赖
```

| 方式 | 依赖 |
| --- | --- |
| 自动克隆（含子模块） | wavestar、OmniPlanner、ZED ROS2、nvblox、nbvplanner |
| 手动恢复并核对 | OKVIS2、Wavemap、LiteAnyStereoV2、OmniPlanner-deps |

已有路径跳过，不校验或覆盖；Git LFS 自动下载关闭。

[模型权重](../Deployment/artifacts/locks/weights.json) 和 [仿真环境](../Simulation/README.md) 需另行准备。
