# 第三方依赖与控制器

`Controller/` 包含 JEPA-MPPI 和 HMU Bridge 源码；其他依赖按 [清单](dependencies.lock.json) 恢复。

## 恢复

在仓库根目录运行：

```bash
python3 tools/bootstrap_dependencies.py                          # 查看计划
python3 tools/bootstrap_dependencies.py --apply                  # 克隆缺失源码
python3 tools/bootstrap_dependencies.py --apply --only nbvplanner # 指定依赖
```
