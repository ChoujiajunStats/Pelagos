# Wavemap → NBV 地图格式适配

2026-10-31 更新：用户已批准查询地图改为 Wavemap 原生 0.1 m。实时投影在后台使用不可变叶节点快照，最多一个运行任务和一个最新待处理任务；融合线程只遍历复制叶节点，保持 10 Hz。原生 float32 分辨率与十进制 0.1 的容差已验证。下面 0.4 m 数值是首次适配的历史验收。

同日已用 0.1 m 地图在实际车辆起点生成 15 节点 / 50 点 / 0.98 m 的原 NBV 路径；旧 .16 s 控制期限下 20 s 跟踪未到达，52 个观测到的不同求解结果均超时（median .269 / P95 .330 s）。后续时序参数与出生点起步授权见 [起步记录](STARTUP_ADAPTATION.md)。

用户于 2026-10-30 要求只建一份地图，2026-10-31 要求继续完成适配。Wavemap 是唯一测量融合器；NBV 保留 OctoMap 查询表示，以继续使用锁定 ETHZ RRT、增益和碰撞查询。没有用 occupied 点云重新建图，也没有将未知填成自由。

## 接口与来源

- 上游 NBV commit `5c07657af18f967d7049d187db0607024746a42c`，原始文件由 `source-lock.json` 校验。
- `TP/Planner/nbvplanner/volumetric_mapping/octomap_world/src/octomap_world.cc` 的 `setOctomapFromMsg` / `setOctomapFromFullMsg` 是已有完整地图替换接口；`getCellStatusPoint`、扫掠盒碰撞查询和 RRT 使用同一替换后的树。
- `Simulation/nbv/patch_sources.py` 仅在构建临时副本给 `nbvPlanner` 和 `OctomapManager` 加 `pelagosDisableMapInputs()`，关闭三路 planner 点云，以及 manager 的点云、视差、相机参数和外部原生地图订阅。TP 原文件不改，算法不改。
- ROS2 Wavemap 适配器增加 `octomap_msgs/Octomap` 完整概率快照发布，0.4 m 取自源 `flat_exploration.launch`，现有 Wavemap 最小格为 0.1 m。投影与原地图同一进程同一时刻读取，继续沿用 5 Hz 发布周期；没有第二个测量积分器。
- 分类来源：`TP/Mapping/wavemap/library/cpp/include/wavemap/core/utils/query/occupancy_classifier.h` 及 `impl/occupancy_classifier_inl.h`。默认 log-odds 阈值 0，未观测 `abs(value)<1e-3`。网格几何来自 Wavemap `index_conversions.h`：叶节点边长 `2^height × min_cell_width`，索引乘边长为最小角。
- 投影头文件 `Deployment/infrastructure/ros2_ws/src/underwater_wavemap_ros/include/underwater_wavemap_ros/octomap_projection.hpp` 同时用于实时节点、离线转换器和语义检查。一个目标格有占据子体积就占据；只有自由体积完全覆盖才自由；其余不分配为 OctoMap 节点。负坐标向下取整；多分辨率叶节点按实际相交体积计数。
- 输出保留聚合后的 log-odds（占据取最强证据，自由取最弱证据）；完整 `.ot` 与 `binary=false` 消息。OctoMap 导入树的默认占据阈值 0.5 与 Wavemap 的零 log-odds 分类一致；旧 NBV 深度融合的 0.7 阈值不用于此树。橙色 0.70 是原有显示阈值，不参与转换。此处语义转换并非完全等价于旧的独立 OctoMap 传感器模型。
- ROS2→ROS1 沿用原 Unix socket / 36 MiB 上限；CRC32 检查、完整消息类型/分辨率/坐标系校验、递增来源版本。只保留最新待发快照，不积压队列、不裁图；每次整棵替换，清空也删除旧格。
- `world` 与 `rov01/odom` 仅在 TF 为单位变换时重标记；任意非单位变换拒绝，未实现旋转重采样。不引入 GT 可视化对齐；历史地图不自动装入新估计 epoch。
- `inspect` 通过源 `getOctomapFullMsg()` 复制树再保存证据，避免源 `writeOctomapToFile()` 内部 `writeBinary()` 的最大似然化/剪枝改变实时地图。

运行 provenance 保存适配器、分类器和源地图 API 的 SHA256。`map_source: native_depth` 只保留作历史对照；默认与本次验收均为 `wavemap`。

## 可复现检查

镜像内 `octomap_projection_check` 覆盖完整自由、63/64 自由与未知混合、薄占据格优先、负坐标、剪枝后源未知保持；测试构图采用与融合器一致的 additive API 并核验每次赋值。当前 Wavemap checkout 的 `setCellValue` 对已有叶节点从父尺度系数计算增量，未用于生成测试未知格，也未为此修改源库。

`wavemap_to_octomap INPUT.wvmp OUTPUT.ot 0.4` 可离线转换完整原生地图。
`tools/check_wavemap_nbv.py --run-dir <专用离线后端目录> --source-run <旧地图目录>` 验证真实全图导入、序列化概率逐字节一致、只读检查不改地图、禁用深度融合、错误输入拒绝、旧版本不覆盖、全图清空/恢复及历史车辆起点规划。专用离线后端没有仿真车辆和控制进程，禁止对实时会话运行此清空检查。

实时验收与离线结果另存于 `Simulation/build/wavemap-nbv/` 和实验运行目录；地图接入通过不等于车辆局部导航或 JEPA 跟踪通过。
