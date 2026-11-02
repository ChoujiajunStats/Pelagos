# 第三方依赖与本地控制器

轻量私有仓库保留 `Controller` 的代码、配置及必要元数据，其余第三方源码用 [dependencies.lock.json](dependencies.lock.json) 记录。权重、数据、缓存和第三方 Git 历史不上传。

这里记录的是本机能核对的来源，不能据此承诺换机器一键复建。盘点没有联网，也没有修改任何现有 TP 子目录。

## 查看与恢复

在项目根目录运行：

```bash
python3 tools/bootstrap_dependencies.py
```

默认只显示计划。明确要下载缺失的已固定源码时运行：

```bash
python3 tools/bootstrap_dependencies.py --apply
# 也可以只恢复某个条目：
python3 tools/bootstrap_dependencies.py --apply --only nbvplanner
```

脚本仅克隆缺失目标，检出完整 commit，再递归初始化 submodule；已经存在的目录或符号链接保持原样。失败留下的部分目录也不会自动删除、覆盖或修复。`EXISTS_UNCHANGED` 只表示跳过，不能证明该目录完整、干净或版本正确。

Git LFS 自动下载被关闭。模型权重和测试资产需另外取得；脚本不会安装系统包、构建镜像或启动仿真。`--apply` 遇缺失的手工依赖返回 2，克隆错误返回 1。

## 本次可核对程度

| 条目 | 记录依据与恢复范围 |
| --- | --- |
| wavestar、OmniPlanner、ZED ROS2、nvblox | 本机可用 Git HEAD/origin；可以固定版本克隆 |
| nvblox_core | 由 nvblox 递归 submodule 恢复，并核对记录的 commit |
| nbvplanner、volumetric_mapping | 既有 `Simulation/nbv/source-lock.json` 的 recorded pin；不是当前 Git HEAD。当前裁剪快照中存在的锁定文件均匹配 Git blob；未用演示文件等已缺失 |
| OKVIS2 | origin/ref 可读，但 HEAD 对象缺失，多个 submodule `.git` 为空；本机 dirty 无法确定，须手工补齐并核对 |
| Wavemap、LiteAnyStereoV2 | `.git` 目录存在但 HEAD/config 等为空；保留 `Deployment/artifacts/locks/algorithm-sources.json` 已声明 URL/commit，未验证本机快照与该版本等价，仍须手工恢复/核对 |
| OmniPlanner-deps | `packages_exact.repos` 提供候选来源/版本；本机快照等价性未验证，保持手工要求 |
| Controller | `vendored_local`，无独立 Git 基线，不推测作者或上游仓库；代码随私有仓库保留 |

nvblox 父仓库 4 个、core 子模块 19 个 dirty 普通文件逐一匹配 HEAD 的 Git LFS SHA256/大小，属于 LFS 内容展开，没有导出大体积伪源码补丁。可用 Git 仓库没有发现需要保留的源码 delta；损坏或缺失 Git 基线的快照无法生成可靠上游 diff，恢复仍需本地备份。

控制器 `data/` 下也有 Python 源码，应保留这些代码。JEPA 与 LAS2 所需权重的路径、大小、SHA256 另列在 lock 的 `external_model_artifacts`，没有杜撰下载地址。HMU 包的本机元数据仅指向一个原作者机器的本地目录，不构成可下载的上游。

仿真接口改动继续保存在 `Simulation/patches`、`Simulation/nbv/patch_sources.py`，构建时只修改源码副本。Stonefish 工作区、Porth 资产、基础镜像及 ROS/CUDA/系统依赖不在这个 TP 锁定范围，见 [仿真说明](../Simulation/README.md)。
