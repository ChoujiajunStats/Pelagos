# 2026-10-30 本机闭环实验

状态：已完成控制复现、手动接管接口和原版 NBV 组件对照。地图尺度窗口下的虚拟起点、以及用户手动建图后的真实车辆起点均已生成原版路径；真实路径尚未执行跟踪，路径点导航和连续探索控制闭环尚未通过。最新证据与会话见文末和 EXPERIMENT_HANDOVER.md。

## 环境与保留证据

- Docker 29.9.2、RTX 5090、X11 :1 可用；接管时两个项目容器均未运行。
- 最新历史会话 `20261029T155304Z-18478392` 保留未覆盖。
- `./Simulation/sim build` 与 `build --planner-only` 成功；后者 14 个包构建成功，日志为 `build/nbv-build.log`。
- 原 NBV 源码通过 `nbv/source-lock.json` 校验，未修改树、增益、原迭代/退出条件或采样空间。

## 实际运行

| 会话 | JEPA 采样数 | 观测到的求解结果* | 超时 | 求解最小 / 中位 / 最大 | 结果 |
| --- | ---: | ---: | ---: | --- | --- |
| `20261030T064446Z-67d66a1b` | 256 | 163 | 162 | 0.1491 / 0.2154 / 0.2721 s | 扫描停在索引 1，已暂停 |
| `20261030T064853Z-9d751820` | 128 | 199 | 180 | 0.1351 / 0.1886 / 0.2523 s | 扫描仍在索引 1，已暂停 |

*统计从任务开始至暂停的 `controller-evidence.jsonl` 中按 `timing_seconds` 去除重复状态；不是把每条控制状态计为一次求解。第二轮另有 18 次 accepted、1 次 no predicted improvement；只观察到 10 个有可执行计划的状态样本。

两次任务均经 `sim navigate --route .../acceptance-route.json` 与任务协调器应答启动。短路线包含两个相距 0.30 m 的目标，先执行原初始化扫描，尚未进入路线跟踪。两轮未触发 Guard 故障；到点标准、求解期限 0.16 s、计划有效期 0.20 s 均未放宽。

真实历史达到检查点要求的 26 个样本，模型频率为 10 Hz，GPU 无输出预热完成。控制状态来源为 OKVIS2 原生机体里程计；VIO 输入审计 `unexpected_truth_inputs=[]`。模型/原生方向由实际分配矩阵得出 +1；横滚通道真实应用反馈跟随原生输出，但四轴连续运动与完整横滚受扰闭环还未验收。

每轮保留 `acceptance-*.json`、`planner-status-paused.json`、`control-status-paused.json`、路线、任务事件及完整控制证据。首轮地图通过原生保存服务成功保存。

## 参数授权

用户在本次会话明确批准：仅将 sample_count 从 256 改为 128，其他参数保持不变。128 的来源为本机 Stonefish 工作区 `controller/config/jepa_mppi.yaml`；没有一并采用其中的时延补偿覆盖值。

覆盖位于 `config/nbv.yaml` 的 `jepa_mppi_overrides`，由 `tools/prepare_scene.py` 冻结进每次运行的 `jepa-settings.yaml`；`autonomy-provenance.json` 保存覆盖及授权依据。TP 中原 JEPA 配置不改。

## 求解耗时诊断

新增 `tools/profile_jepa.py`：仅订阅本次真实 OKVIS2、IMU、压力与实际推进器反馈；要求任务 DISARMED，没有控制输出发布者。它使用同一冻结参数、同一检查点与真实因果历史，保存 `jepa-profile/jepa-real-history.npz` 和原生求解诊断。离线目标用于测量计算，不作为成功控制证据。

128 采样下同一输入窗口的诊断：直接调用 8 次中位 0.1688 s；通过工作线程并同时处理 ROS 回调的 8 次中位 0.1955 s。核心优化约占 0.1584 / 0.1774 s，后处理约占 0.0104 / 0.0156 s。说明实际计算已接近或超过期限，回调并发还会增加耗时；该小样本诊断不能证明单改线程即可解决。

`adapters/estimated_control.py` 仅为原 `_solve` 增加真实输入快照及 timing 字段记录，未修改求解输入、结果选择、期限或推进器指令。依据为本机 `controller/uw_controller/checkpoint.py::CheckpointController._solve` 及 `TP/Controller/disaware_learning/control/jepa_mppi.py::optimize` 返回的原生诊断。

## DDS 报错定位

第二轮 `dds_probe.py` 对 NBV 的各个订阅使用 raw CDR 抓取并调用原反序列化器核验。在 40 秒中：

- `/stereo/depth` 783 帧，其中 3 帧不能解码；坏包及头部保存在 `dds-bad-*.cdr` 和 `dds-probe.json`。
- 坏包声明 data 长度 1,228,800 字节、头部偏移 64 字节，应有 1,228,864 字节，实收 1,228,804，缺少 60 字节。
- 同期时钟 3994、机体状态 998、Guard 状态 1997、控制器状态 1997、定位状态 798、导航状态 798 条全部可解码。
- ROS 图的 Odometry、Clock、String 发布/订阅类型哈希一致；只有一个 LAS2 depth 发布者。
- 同环境标准 Image 独立序列化/解码 1500 次均正常；两个独立进程的原生发布与显式序列化发布各 1000 帧均正常。因此不能把“改成显式序列化”宣称为已证实修复。

结论限于：真实 LAS2 深度链上存在少量截短 CDR 消息。传输/发布内部根因仍待定位；未据此改地图、填充缺失深度或更换 DDS 参数。

接口参考与源码来源：

- [ROS2 rmw_fastrtps Jazzy TypeSupport 实现](https://github.com/ros2/rmw_fastrtps/blob/jazzy/rmw_fastrtps_shared_cpp/src/TypeSupport_impl.cpp)，区分 ROS message 与 serialized CDR 发布路径；只读副本在 `build/ros-source/`。
- 容器安装的 `/opt/ros/jazzy/lib/python3.12/site-packages/rclpy/publisher.py` 支持正常消息与 bytes 两条发布路径。
- 容器版本：FastCDR 2.2.8、FastDDS 2.14.6、rmw_fastrtps_cpp 8.4.4、sensor_msgs 5.3.8。

## UI

按用户指定 [oil-ui skill](https://github.com/oil-oil/oil-ui/blob/main/SKILL.md) 整理现有 Qt 面板，保留 RViz 和原任务协议。方向与截图/评审过程在 `.oil-ui/mission-control/`，实际 UI 实现与验收另行更新。

已完成 Qt 面板更新，并重新打开真实连接的任务面板：编号路线表、任务页签、固定暂停栏、中文摘要、折叠诊断、可滚动故障详情。任务初始化未完成时明确显示“待执行”，不把路线索引冒充到达进度。未完成路线暂停后突出“继续路线”，不改变任何按钮的授权条件。

验证包含一轮独立视觉评审、一轮独立真实 UX 走查。后者从实际界面自主走通添加两点、保存、清空、导入，没有触发运动；发现高 DPI 控件挤出按钮和旧保存提示滞留。修正使用逻辑像素字体/输入框尺寸及成功操作反馈，保留原输入和请求协议。主 Agent 随后在真实 X11、真实 ROS、720×660 窗口复测通过：编辑按钮完整可见、当前机体位姿可添加、路线准确保存读回、清空/导入提示更新、切页不激活任务。保存失败保留输入亦通过 Qt 检查；静态编译检查通过。

最终真实窗口截图：`runs/20261030T064853Z-9d751820/ui-live-720x660.png`、`ui-live-explore-720x660.png`；主 Agent 验证结果 `ui-final-check.json`。故障状态预览使用构造状态，不能作为本轮真实故障证据。独立 UX 报告为 `.oil-ui/mission-control/ux/ux-review.md`。

当时（256/128 两轮结束时）的交接状态：任务暂停、Guard DISARMED；64 采样尚待用户回复。此段是早期记录，后续已按继续复测要求运行 64；最新纯规划和 RViz 结果见下节。


## 后续纯规划与 RViz（本次接管最新结果）

用户要求停止无目的控制试验、先拆解 NBV 条件，已切到纯规划。`20261030T112044Z-dbb95ef1` 同一已观测快照上的结果：

| 起点 | 采样范围 | 有效树节点 | 正增益 | 原生路径点 | 已知自由空间检查 |
| --- | --- | ---: | --- | ---: | --- |
| 虚拟组件起点（旧真实 body 向前 1.2 m） | X/Y ±115，Z −5..6 | 9 | 是 | 0 | 无路径 |
| 同一虚拟起点 | 用户批准 X 0..5.2，Y ±3.2，Z ±2.8 | 15 | 是 | 49 | 通过 |
| 旧真实 body 起点 | 同一批准窗口 | 0 | 否 | 0 | footprint 6 处未知，0/26 自由扩展 |

没有发送任何运动。原版 extension 1 m、overshoot 0.5 m、15/200 条件、gain、未知空间拒绝均保留。上游 center/bounds box 查询为 free 不足以保证 swept footprint 安全；后者的起始采样点包含未观测区域。这个结果支持先改善真实起点覆盖，再验证局部采样，不能用虚拟起点的 49 点声称车辆可执行。A/B 使用原随机源的单次顺序调用，不是统计基准。

文件 `planner-map-ab.json` 保存快照 SHA、精确范围、根、完整实际树及每次结果。快照真实边界比最早记录多一个体素，因此采样操作已固定为批准的精确数值，默认配置未覆盖。接口导出及 GBPlanner RViz 适配来源见 `rviz_panel/PROVENANCE.md`。

按用户要求于 2026-10-30 11:42:36 UTC 重置为 `20261030T114236Z-353c4200`，新初始估计原点与地图会话独立。当前 `planning_only=true`，运动相关进程未启动；真实起点第 1 次预览在 map revision 163 仍为 0 节点。旧 Wavemap 已保存为 `map.wvmp`。

GBPlanner 式 RViz 面板已真实加载，显示真实协议结果与白底蓝色布局。局部独立只读 UX 检查发现：无路径原因不够具体、预览地图与当前地图版本混淆、图层名称截断。已增加 footprint 未知/占据数和扩展方向中文说明、两个地图版本，并缩短关键图层/视角名称。该检查未点击运动控件，不把静态检查作为运动验收。新增插件已编译并打包进仿真镜像；Python 和 shell 静态检查通过。

组件对照图：`runs/20261030T112044Z-dbb95ef1/planner-map-ab.png`，三个面板使用同一真实地图的三维结果在 XY 上投影，明确标注虚拟/真实起点，不接入新会话的坐标系。最终 RViz 截图：`runs/20261030T114236Z-353c4200/rviz-gbplanner-final.png`。


## 用户手动建图后的真实起点规划成功

会话 `20261030T115426Z-88ab2f8b`，由用户操作真实 BEITONG 手柄，控制为已确认的 `manual` / body PI，反馈是 OKVIS2 body。随后按用户要求暂停手柄输入并从当前位置做纯规划；故障及状态先保存到 `before-current-position-preview.json`，用源 CLEAR_FAULT 将已撤权终端恢复为 DISARMED，整个预览没有 ARM 或 NavigationGoal。

真实请求位置为 `[12.785832, -0.816176, 1.257216]` m，native OctoMap revision 569，边界约 X −5.6..18、Y −6..5.6、Z −3.2..3.2 m。机器人 footprint 未知/占据采样均为 0，26 个源扩展+overshoot 方向中 23 个 free。原版树接受 20 个节点，正增益，返回 50 个路径点，best edge 长 0.98 m；原生 `getLineStatusBoundingBox` 验证通过。

本次仍使用已批准全局采样 X/Y ±115、Z −5..6，未采用局部采样覆盖，未改增益、节点预算、未知空间或碰撞检查。该结果确认实际起点观测覆盖问题可以通过真实驾驶采图消除；不能把单次规划成功等同于 controller 跟踪或连续自主探索成功。

用户已确认在 RViz 看见蓝色路径。固定证据是 `planner-preview-actual-success.json`、`planner-status-actual-success.json` 和 `nbv-actual-success.png`。`nbv-actual-success-rviz.png` 是当时真实窗口截图；备用观察 preset 已生成但没有覆盖用户当前视角。当前自主任务关闭、Guard DISARMED，未执行轨迹。

手柄另有原生终端 `original_request_expired` 重复故障，已用只读 `control-timing-trace.jsonl` 捕获一次，后续仍需定位；本次没有放宽任何超时。该手柄问题与本轮原版规划通过分别记录。


## 当前位置原版 NBV 规划与 JEPA 短段跟踪尝试

用户明确要求“在现在这个位置再规划一次，然后尝试 controller 跟踪”。在同一现有会话 `20261030T123131Z-21d420df` 重新调用原版规划：23 个有效节点、50 点、长度 0.98 m，源 OctoMap 检查通过；冻结 `controller-trial-plan.json`。

为保留用户已建地图并直接尝试本段，新增有限实验入口 `tools/track_nbvedge.py`。它调用现有 Guard 的 navigation 授权，经原有 NavigationGoal → navigation tracker → JEPA-MPPI → actuator/terminal；没有直接发布速度或电机，没有替换成 PI，没有伪造 source 初始化完成。以实际 body 启动前及地图变更后对目标连接做原生 validate，操作员新指令/模式变化、定位或路径失效时停止；本次最长 20 秒后显式 DISARM。来源接口哈希保存在结果文件。

结果：目标未到达。20 秒中 371 个记录样本处于导航 TRACKING，但 JEPA 76 次求解全部 deadline exceeded，中位 0.25054 s；可执行模型计划样本为 0。原生推进器最大绝对设定约 0.001466（姿态稳定量），不是有效的路径跟踪。实际估计 XYZ 变化约 [−0.0055,+0.0104,+0.0839] m，目标距离由 1.2372 增为 1.3004 m。

准备执行期间车辆被水流/浮力带动，实际启动位置与本次规划根差约 0.518 m。因此记录同时保留原蓝色 NBV 段与启动时重新验证的直接目标接入段，没有把两者冒充同一条实际执行轨迹。跟踪失败结论也受 76/76 求解超时直接支持；未将被动漂移计作成功前进。

参数保持 64 采样、deadline 0.16、plan age 0.20、latency 0.16，以及原增益/速度/Guard 阈值。此前三项时限建议尚未获明确批准，本轮未改变。结束已读取真实终端确认 DISARMED、六个 applied setpoint 全为 0。

证据：`nbv-edge-tracking-test.json`、`nbv-edge-tracking-summary.json`、`nbv-edge-tracking-result.png`，均在本次运行目录。
