# Pelagos 仿真实验接管

用户要求：在当前 Porth / Stonefish 仿真中调通两种任务——指定路径点自动导航（核心 CLI JSON routes），以及 ETHZ 原版 NBVPlanner 的局部自主探索；实际执行必须使用 JEPA-MPPI。用户最新要求全英文 RViz 五项原生工具栏，移除 custom mission panel 与独立 UI；`sim ui` 仅兼容打开 RViz，构建与实际 runtime 分开记录。

## 一键演示

在本机桌面终端执行：

```bash
cd /home/hong/Desktop/Pelagos; ./Simulation/demo
```

入口先检查环境和已准备好的主镜像/native 镜像，再重置本项目的两个仿真容器，从 spawn 启动新的自主会话；**每次重新开始，但保留 `Simulation/runs/` 的旧记录**。它不构建、不改参数。RViz 中点 **Explore** 开始，**Pause** 暂停任务，**Manual** 手柄接管，**Auto** 只选择自主所有权，**Plan Preview** 只预览；`sim ui` 只是同一 RViz 的兼容入口。

## 授权和约束

- 已授权操作本项目的 `pelagos-stonefish`、`pelagos-nbvplanner` 容器，包括构建、启动、暂停、停止、读取 ROS 和日志；不要操作其他容器或真机。
- 参数必须与用户商量；未另行批准的算法参数沿用源仓库。不要自行构建 domain 或车辆 XYZ 围栏。
- 复用 Stonefish 官方实现、本机既有适配和算法源码，减少无意义测试。
- 原版 NBV 固定版本在 `Simulation/nbv/source-lock.json`；树、增益、原生迭代预算与退出条件不改。相机标定 75° / 59.840444° / 0° 已批准；完整 Porth 尺度推导的采样范围 X/Y ±115 m、Z −5..6 m 已获用户授权，它不限制车辆运动。
- Wavemap 融合 10 Hz / 发布 5 Hz 已批准。保留完整 Porth 原图叠加、红色 GT / 绿色估计轨迹和历史相机锥体、原生特征点及实时深度对比。
- 自主控制使用 OKVIS2 原生机体状态；GT 仅用于显示/评估，不输入规划或控制。不要把左相机位姿当作机体状态。

## 2026-11-02 当前复测：至少四轮转向后续航已核对，任务继续

用户终端启动的 **`20261102T031510Z-29de60ff`** 已由实际 RViz Explore 请求 **`131bc3e71324477ab34eda78d8cdfd12-1`** 开始。主执行没有重启该会话或代发 Explore。[独立四轮有限窗口审计](build/wavemap-nbv/map-stale-turn-live-audit.json) 截止 **sim 209.59 s / task 203.278 s**，记录 **enabled=true / Guard ARMED / generation 1**；主执行随后仍看到滚动 cruise。任务保持运行，不等 observer 360 s，也不因观察到期停止；这里不表述长期稳定。

| 已核对轮次 | 当前图方向提交 map | 水平 occupied 净距 | 恢复巡航至下次触发的实际投影 |
| --- | ---: | ---: | ---: |
| 1 | 258 | 7.57846 m | 4.51782 m |
| 2 | 349 | 4.29969 m | 1.29643 m |
| 3 | 499 | 4.72833 m | 1.72983 m |
| 4 | 570 | 3.15170 m | 0.12758 m |

四轮均有真实源 `reached` → 更新地图碰撞/水平净距复验 **>3 m** → 方向提交 → 滚动巡航，未等待 endpoint/observe。第 4 条方向净距只略高于原 **3 m** 门槛，短段是再次遇障触发，不是距离 cap 或精确到点等待。

已开始的转向中，**13 个 map-gap 快照均保持 index1 / XYZ 参考 0 / Guard ARMED**；四轮分别捕获 **211 / 240 / 868 / 138** 个已开始转向的导航样本，均未插入 neutral0。首次转向仍要新图，最终提交与任何前进仍要新鲜当前图和实际碰撞验证；机体、Guard、授权、租约保持严格。零 XYZ 是速度意图，不能当作水流中位置完全静止的保证。

**四轮只按真实源 Tracker 的采样 reached/dwell 判据确认。** 独立 raw 近 1 s 窗口中第 1/2/4 轮采样均低于源运动门槛，第 3 轮有短暂 **0.04471 m/s** 峰值高于 **0.04 m/s**；源 20 Hz 消费判定与独立约 25 Hz raw 抽样不同，不能写成四轮所有 raw 样本连续 1 s 均过门槛。该有限窗口内 raw 约 **24.7 Hz**、最新 body 的诊断 lag 最大 **60 ms**，仍不替代精确 Guard 测量或长期验收。

main **efeb177d…**、native **0302d2bb…** 保留；运行前桥 **2117e537…**、**68 host** 与既有 **44 source** 回归来源见 [桥 provenance](build/wavemap-nbv/turn-cruise-bridge-provenance.json)。本次仅已开始的纯转向可跨 map gap；首次转向、最终方向提交和前进不豁免新图检查。数值、gain、15/200 与 Guard 期限未改。

另已完成 [三层 RViz 显示去重](build/rviz-native-tools/render-display-uniqueness.json)：仅移除 Sensors 下 NBV path / NBV OctoMap / Wavemap occupied 的重复 alias，三个 topic 各保留一个原显示实例，完整地图和数值不变。无需 build，**当前 RViz 未 reload/restart**，下次打开配置才生效；它不属于本轮四次闭环的现场加载结果，也不证明旧 GL 异常根因。详见 [UI provenance](build/rviz-native-tools/ui-native-tools-provenance.json)。

## 历史：`030254` 有限 VIO 窗口通过；终态 RViz −6 / Stonefish 0

**`030254` 实际已结束，不能继续视为当前运行。** [演示 supervisor 日志](build/wavemap-nbv/presentation-demo-live.log) 记录 RViz **−6**，随后 Stonefish **0** 结束；[RViz 原文](runs/20261102T030254Z-ff077951/logs/rviz.log) 为 `Cannot create GL index buffer` / Ogre exception。旧缓存 enabled/ARMED 只是退出前快照，不是存活证据。`run_pipeline.py` 在该轮之前已包含 RViz 非零退出后 continue 的分支；现有日志不能证明 Stonefish 为什么返回 0，或 RViz 异常是否与它存在因果关系。主执行查阅当地 11:10–11:16 kernel 未见 NVRM/Xid/oom，LAS2 未见 CUDA error；不能据此推断 GPU OOM、GPU 硬件故障或 RViz 触发 finally，也不归因为 VIO/Guard。

一键入口启动的 **`20261102T030254Z-ff077951`** 已由用户实际 RViz Explore workflow 开始（request **`e7e53fbf…-1`**）。[独立有限窗口审计](build/wavemap-nbv/okvis-ingress-live-audit.json) 截止 source clock **453.24 s / task 449.516 s**，任务 **enabled=true / Guard ARMED / generation 1**，超过旧故障 **245.24 s / task 228.703 s** 窗口，该审计时任务仍在运行，随后终态以上述 RViz/Stonefish 退出日志为准。

在旧故障点之后的 **245.24–453.24 s** 区间，raw body **24.698 Hz**、最大采集间隔 **60 ms**，视觉更新 **15.886 Hz**。**91** 个 native 成功日志与 **10,138** 对 raw/estimated 状态证明原采集 stamp 保留；terminal 最近接收 body 的诊断延迟中位 **20 ms**、最大 **70 ms**（采样时钟局限见审计，不替代 Guard 精确测量）。这是本轮负载下 raw/visual 解耦和新鲜度的有限窗口验证，未宣称所有 VIO 故障已根治。

本轮前三次转向后分别实际续航 **4.699 / 1.067 / 7.924 m**，不等待 endpoint/observe。**审计时第四次 pure turn 尚未完成：已持续 232.93 s**；最近 30 s 的最大实际速度 **0.07129 m/s**、yaw rate **0.13408 rad/s**、yaw 误差 **0.18519 rad**，仍有振荡，并有 **103 次 neutral index0 / 482 次 turn index1** 样本；这些现象不能仅凭计数认定唯一原因。该轮完整导航顺畅性仍未验收，不把 Guard 未 fault 当成到达或转向成功。

当时 main 镜像 **efeb177d…** 已 fresh 加载 ingress 修复与 Guard/Task 英文反馈源码，native **0302d2bb…**。**41978a32…** 仅是反馈的独立构建/15 项协议来源，不是本轮主镜像，也不回填旧 `023622` runtime。[OKVIS ingress provenance](build/wavemap-nbv/okvis-ingress-provenance.json) 记录无参数接口修复、**5 source PASS**、完整 OKVIS/clients/下游构建，以及 native SIGINT/SIGTERM 均 **exit 0**；视觉算法、预算、频率、真实 stamp、Guard 阈值保持。旧 `023622` 两轮闭环和时间戳 FAULT 保留下方历史。

## 历史：`023622` 两轮 reactive 闭环与时间戳故障

旧会话 **`20261102T023622Z-181e7b43` 已停止：enabled=false / Guard FAULT / state_clock_mismatch**。它已完成两轮用户要求的 **cruise → braking → plan turn → pure turn → continuous cruise**，第三轮转向时被原有 Guard 撤权。不可把旧 ARMED/gen1 快照当作当前仍运动，也不自动清 fault 或重启。用户要求持续自主和 `--leave-running` 保留，不能覆盖真实安全撤权。

任务来自 RViz Explore workflow 的真实 `operator_command/start_exploration`，request **`5281870bf95c43ca88f6c5e9e86fc696-1`**、accepted wall **46015068645558**；这才是完整任务起点。observer 在第一次 turning 才开始，完整依据是 [任务事件](runs/20261102T023622Z-181e7b43/mission-events.jsonl) 与 [全程 trace](runs/20261102T023622Z-181e7b43/control-timing-trace.jsonl)，不推定具体谁点击，也不把 observer 窗口当全程。

[turn-cruise-live-audit.json](build/wavemap-nbv/turn-cruise-live-audit.json) 独立复核两轮：

| 轮次 | 首次遇障净距 | 方向提交 map / 净距 | 新水平巡航方向 | 至下次触发实际投影 |
| --- | ---: | --- | --- | ---: |
| 1 | 2.996309 m | 413 / 8.808866 m | [0.142032, −0.989862, 0] | 6.123853 m |
| 2 | 2.979310 m | 639 / 3.341820 m | [−0.874895, −0.484313, 0] | 0.337485 m |

两轮均为真实源树分支、pure-turn 实际停稳、最新图方向提交，再恢复滚动巡航；pure-turn 平移参考全部为零。转向前 1 s 原生样本最大 `||v|| / |w_z|` 第一轮 **0.029471 / 0.027762**、第二轮 **0.017243 / 0.026357**，保留源门槛 **0.04 / 0.04 / dwell 1 s**。没有真实 `nbv_edge_reached` 或 observe，第一条新方向实际走过原约 0.98 m 首边，证明不再卡在 endpoint 精确到点。第二条只走约 **0.34 m** 即再次触发，原因是提交时净距 **3.34182 m** 只比原 **3 m** 门槛大约 0.34 m；不能说全局 route 已优化或长期探索通过。

第三次 trigger 净距 **2.994802 m** 后进入转向；accepted 后 **228.703412756 s**，Guard fault wall **46243772058314**，clock **245.239999999 s** 与 body stamp **244.979999999 s** 相差 **0.26 s**，超过保留的源 **0.25 s** 检查。body 消息接收年龄 **0.107761849 s**、estimator 状态接收年龄 **0.016558539 s**，仍不能覆盖原始采集时钟检查。原生状态 **TRACKING / native_loss_evidence=null**，原生 tracking-failure 日志数 **0**；终态审计中的 VIO 匹配误差约 **0.030 m**。这不是原生 Lost、no-candidate 停止或已证实的 GPU 故障；不归因 CPU/VIO 算法，也不放宽阈值。

全英文原生 RViz 已实际加载，无 custom mission panel / 独立 web 或 Qt editor，五项工具 **Explore / Plan Preview / Manual / Auto / Pause** 可见：[本轮真实窗口截图](build/rviz-native-tools/rviz-live-english.png)。该轮 [12 项协议检查](build/rviz-native-tools/mission-flow-check.log)、5 plugins 加载及 [UI 生命周期来源](build/rviz-native-tools/ui-native-tools-provenance.json) 的真实 RViz 窗口 **12.012242 s / exit 0**，以及 toolAdded 后再刷新 QAction / 关闭时停止 ROS timer 的实际竞态修复。首个 `023034` 在任务开始前遇到 RViz 初始化 **−11**，随后正常停止清理，不当成本轮运动故障。实际镜像 **main b29260c04d30c6dff84656065677513de0b0a1cc666ca49c75bbb5d76329330f / native 0302d2bbaaf64aeaebc6f6b31dc79418848b5716e146bb1667fd022e721f85ce**。

桥当前使用 `mission_id='nbv-reactive'`：reserved **0** 是完整 neutral，reserved **1** 每源 tick 替换最新估计 XYZ、保留 yaw 的外部 pure turn；源到达需匹配当前 run/mission/generation/index1、晚于转向起点且 <250 ms。完成后用缓存的原源首边几何请求 `set_cruise_direction`，重验当前地图和现有 3 m 条件，恢复 **1 m** 滚动 current-heading cruise。首个 carrot 计算期间续 neutral，避免旧 200 ms 意图过期；Preview 已统一 `plan_turn`，转向黄色箭头显示当前 body/yaw。`nbv-reactive` real index≥12 直接用原 source Tracker，不进入旧 endpoint/observe wrapper；原 wrapper 仅为其他 legacy mission 兼容保留。

[64 host/桥检查](build/wavemap-nbv/turn-cruise-bridge-tests.log)、[44 真实 source 检查](build/wavemap-nbv/turn-cruise-navigation-tests.log)（31 navigation / 9 sensor / 4 clock）与 [plan_turn native 构建](build/wavemap-nbv/plan-turn-provenance.json) 通过。完整来源/当前桥 hash/终态见 [turn-cruise-bridge-provenance.json](build/wavemap-nbv/turn-cruise-bridge-provenance.json)、[turn-cruise-navigation-provenance.json](build/wavemap-nbv/turn-cruise-navigation-provenance.json)。数值 **0.1 m map / 3 m obstacle / JEPA 64 / .36 deadline / .60 age / .36 latency / .48 horizontal cap**、源 gain/yaw 抽样与 **15/200**、碰撞/unknown/Guard/终端期限均未改。候选无 known occupied 命中不等于 observed free；当前执行不走 sampled-view yaw，原 gain 预测不是已执行相同 gaze。两轮局部策略实测通过，**持续长期/全局探索仍未验收**。

### 已结束的 heading 阶段试验

`140825` 当时使用 adapter **37d482…**：实际先转头停稳后平移成立，但 generation 恢复重算路径方向导致过冲后掉头，主执行人为 pause，首边未到点，无 Guard fault/native Lost。修复后的 **`20261101T141248Z-ee3b570d`** 当时使用 **fa6260…**：固定方向/phase、source 真实停稳、map-gap neutral、Guard generation 1 已由 [heading-resume-live-audit.json](build/wavemap-nbv/heading-resume-live-audit.json) 验证；该审计 accepted 后 **209.852 s** 快照仍为 translate、0 observe / 0 真实 edge reached，端点振荡没有收敛，不能标闭环成功。

**141248 的最终停因是 RViz 正常 exit 0，随后整条 pipeline 结束**，原文见 [heading-resume-live.log](build/wavemap-nbv/heading-resume-live.log)。容器均已退出；没有把该轮记录成 operator pause、Guard FAULT 或新增 GPU 故障。磁盘缓存 planner enabled / Guard ARMED 不是存活证据。新 turn-only 源码与测试不得回填为这两轮的历史 runtime；旧 [heading provenance](build/wavemap-nbv/heading-navigation-provenance.json) 仅保留对应阶段来源。

## 历史：2026-11-01 制动交接与 GPU 终态

以下保留旧 endpoint/observe 策略的实际证据；旧 [obstacle-cruise provenance](build/wavemap-nbv/obstacle-cruise-provenance.json) 不作为当前 turn-only runtime 来源。

旧策略实现 **cruise → braking → planning**：前方约 3 m 触发后停止续发直行目标，保留主动控制、连续 neutral index 0 零速度意图，等待源 Tracker 原有 `||v_xyz||<0.04 m/s`、`|w_z|<0.04 rad/s`、连续 dwell **1 s**。采用已有用户“直行到约 3 m 再局部规划”的交接实现，没有新增数值条件。它是新鲜估计状态下的停稳判断，非 GT 物理停车证明，也不是固定位置 hold。

制动阶段只允许 neutral 授权，检查 state/authority 新鲜度；source `reached` 必须匹配当前 run、mission、generation、index 0，且 status wall 在本次 braking 开始之后、年龄小于既有 250 ms，拒绝早先 neutral 的残留状态。未满足时不运行局部 RRT；源停稳满足后记录 `obstacle_braking_complete`，再以最新 body/map 规划、碰撞验证。原 `obstacle_triggered_planning` 保留为遇障触发事件，不能把其时间当成实际 RRT 开始。真正路径 index≥12 及 `nbv_edge_reached` 语义保持。

**`20261101T134639Z-86ae3703` 的实际终态已归档，当前不再运行。** 净距 **2.992354573 m** 触发制动；**6.849764524 s** 后产生 `obstacle_braking_complete`。该事件 body 与原生机体 stamp **67.889999999 s** 逐坐标一致；此前完整 1 s 的 26 个 raw 样本最大线速度范数 **0.03742684 m/s**、最大 `|w_z|` **0.03122568 rad/s**，均低于保留的源 0.04 门槛。随后源 NBV 返回第一条 **50 点 / 0.100015578 s** 有效路径，实际 `nbv_edge_reached` 距离 **0.024334155 m**、航向误差 **−0.008615053 rad**；第二条 **50 点 / 0.049999064 s** 已生成并跟踪。见 [braking-live-audit.json](build/wavemap-nbv/braking-live-audit.json) 和 [任务事件](runs/20261101T134639Z-86ae3703/mission-events.jsonl)。这证明了估计状态下停稳交接和一次源标准到点，不是 GT 停车或长期探索验收。

该轮在 **124.871390372 s / 622 样本** 时因 **`estimator_sensor_stale`** 停止；fault clock **138.63 s**、image **138.36 s**、IMU **138.60 s**、body **138.61 s**。当时原生状态仍 `TRACKING`、`native_loss_evidence=null`，不能记成上一轮的 native Lost。随后 supervisor 报 Stonefish exit **−6**，LAS2 日志 UTC **13:49:05.219049** 为 CUDA unspecified launch failure。宿主机上一 boot 的 kernel 同一事件窗口记录当地 **21:49:05.210519 Xid79/GPU has fallen off the bus**、**.211278 Xid154/Node Reboot Required** 和 UVM global fatal **0x60**。完整 [kernel 原文](build/wavemap-nbv/gpu-kernel-20261101-214905-prev-boot.log)、[事件字段](build/wavemap-nbv/gpu-kernel-20261101-214905-events.json) 与 [GPU 审计](build/wavemap-nbv/gpu-fault-audit.json) 已保存。`pid/name=code` 只是一条报告上下文，不能据此认定某应用导致故障；也没有证据把该 GPU 总线丢失归因 CPU 负载、建图或 VIO 算法。Guard monotonic 与 kernel 接收/source 时间字段分开保留，不强行拼接毫秒级因果顺序。

宿主机于当地 **21:49:50** 重启；当前 RTX 5090 / driver **580.178.04** 的 `nvidia-smi` 正常，主执行复查 `sim doctor` 全通过。恢复没有改驱动、源码参数或阈值。**同配置会话 `20261101T135825Z-55bdb8bc` 已为用户新导航要求主动暂停**：实际事件记录先完成停稳、第一条 50 点到点（距离 **0.036390264 m**、航向误差 **0.018533127 rad**），再生成第二条 50 点规划，最后 `explicit_stop`。终态 **enabled=false / Guard DISARMED / healthy=true**，没有把该暂停记成 fault。该会话当时执行旧导航模式；后续同路径 Python 文件被修改，当前文件 hash 不能替代它的历史 runtime 语义。重启会重置 monotonic 时基，禁止跨两次 boot 直接相减 `wall_ns`。48 项 [host](build/wavemap-nbv/braking-tests.log) 与 22 项 [源接口](build/wavemap-nbv/braking-source-tests.log) 回归保留为该轮证据。Stonefish/UI **`sha256:592ff91ab2efb904aece0acc8a08f35d2fac711315820082779fb388fc659f92`**、native `b85cf92…` 保留为该轮来源；新阶段按上方独立来源/构建记录区分。[provenance](build/wavemap-nbv/obstacle-cruise-provenance.json) 保存冻结设置、构建和终态证据。

保持明确批准值 **64 / deadline 0.36 / maximum age 0.60 / latency 0.36 s / 水平 cap 0.48 m/s**。源 yaw 抽样/gain、碰撞/unknown、地图、motor、Guard 和原始租约/输出期限不变；不因本次改动恢复旧 5 m、body-known/free-exit 停车条件。用户要求持续自主，观察窗口用 `--leave-running`；pause/manual/Lost 等真实安全撤权保持。

[fast-vio-loss-audit.json](build/wavemap-nbv/fast-vio-loss-audit.json) 记录 native 特征支持降为零时数据仍活跃，首次 Lost 在遇障约 5.957 s 后、减速和转向期间；首次失败前没有 IMU/camera queue drop。旧交接没有完整停稳，控制证据见 [fast-control-transition-audit.json](build/wavemap-nbv/fast-control-transition-audit.json)。这些证据**没有隔离速度/转弯/画面哪一项导致 Lost**。`134639` 的停稳与一次到点已验证，但被独立记录的宿主 GPU 故障中断，不能宣称视觉根因已修复或持续探索已验收；下方 `133950` 的 FAULT 保留为历史。

## 历史：2026-11-01 `133950` 原生 VIO Lost 撤权

最新会话 **`20261101T133950Z-15f9e2bc`** 已停止：观察 **57.580241 s / 288 样本**，Guard **FAULT**、任务 **enabled=false**。原生证据 `W20261101 13:41:05.071417 ... ViSlamBackend.cpp:467] TRACKING FAILURE: quality=0` 触发 `estimator_LOST:okvis_native_tracking_lost_restart_required`，Guard fault wall **281545799601777**，停止事件 **281545841997771**；地图 `localization_lost=true`、562 积分帧、融合降为 0 Hz。约 30 s 的健康 ARMED 仅为故障前快照。此前已实际行进投影 **12.944951 m**，在 **wall 281539842153172 / map 272 / 净距 2.807716644 m** 交接，源 NBV 返回 **50 点 / 0.099275215 s**，本轮没有 `nbv_edge_reached`。此轮故障是明确 native Lost，不能沿用旧租约到期作为当前原因。用户已明确批准**只将最大计划年龄 0.40→0.60 s**，已在本轮 [jepa-settings.yaml](runs/20261101T133950Z-15f9e2bc/jepa-settings.yaml) 和 controller 实际 report 核实：**64 / 0.36 deadline / 0.60 age / 0.36 latency**。水平导航上限 0.48 m/s、Guard requested X/Y 0.48；垂向/yaw、增益、1.2 m/s 实测包络、motor cap 和所有 200/250 ms 原始授权及 watchdog 保持。

当前 neutral 接口与源契约：

- 地图版本更新而运动路径尚未重验证时，不续发旧运动目标；正常短查询保留尚有效的授权，不立即把 MPPI 目标切成零。仅当旧 goal 剩余时间≤`2/navigation_rate_hz`（冻结 20 Hz 派生 **100 ms**）时，续发 reserved waypoint **0** 的 neutral。它是新明确零速度意图，不给旧运动意图伪造最新时间。
- 等待下一条规划/hold 阶段持续 neutral；地图流不健康而 body 健康时先发 neutral 再 DISARM，body 本身不健康则直接撤权。原 NavigationGoal 200 ms、Guard 250 ms 和终端输出 200 ms 检查不放宽；运动目标只在最新地图验证成功后恢复。
- `estimated_navigation.py` 仅在原 source Tracker tick 内把 index0 目标临时替换为最新真实状态 pose，经原 guide 输出零速度，并恢复原 goal；身份、generation、sequence、epoch、deadline 检查保持。neutral 的 `reached=true` 不得计作路径到点；真实 NBV index≥12，桥还核对 generation/index/current map，新增 `nbv_edge_reached` 事件。状态现保留 navigation generation/waypoint_index，UI 的“切换时障碍净距”明确为交接历史值。

本轮证据：[planner-status](runs/20261101T133950Z-15f9e2bc/planner-status.json)、[control-status](runs/20261101T133950Z-15f9e2bc/control-status.json)、[mission-events](runs/20261101T133950Z-15f9e2bc/mission-events.jsonl)、[controller-evidence](runs/20261101T133950Z-15f9e2bc/controller-evidence.jsonl)、[完整控制 trace](runs/20261101T133950Z-15f9e2bc/control-timing-trace.jsonl)。`--leave-running` 只取消观察时长造成的停车，真实安全故障仍撤权。该 observer 在异步停止确认前记录了 `task_left_running=true`，但当前实际 disabled/FAULT；不能据此标为仍自主运动。保留用户持续任务要求，恢复需要健康估计会话，不自动循环清 fault/ARM。不要暂停整个容器。

Stonefish/UI **`sha256:2b9073578ba437b21055fe3d38e51245ddade5216b5fe8e428ccd5b825cfa608`**，native **`sha256:b85cf92da2a03ab4357175e759570e5143a97af4d926df4b3b9a78562d15c919`**；[UI 构建](build/wavemap-nbv/image-build-neutral-ui.log) 完成。[42 项核心/状态回归](build/wavemap-nbv/obstacle-cruise-tests.log)、[8 neutral + 9 sensor + 4 clock 回归](build/wavemap-nbv/neutral-alignment-tests.log) 通过。[provenance](build/wavemap-nbv/obstacle-cruise-provenance.json) 保存当前源码/批准/构建与配置；本轮已因原生 Lost 结束，长期 neutral/租约表现尚不能验收。

前一 `132233` 会话在约 12.256 m / 2.998 m 遇障交接后完成真实局部跟踪、两次源到点与重复规划，随后第四条路径阶段停于 **Guard `original_request_expired`**。完整 [审计](build/wavemap-nbv/second-cruise-fault-audit.json) 证明最新原始 issued/deadline 不再前进，approved 后续序号仍复用同一旧 deadline；Guard 和 native 各自到期，相差 60,987 ns。不是仅靠 prefix 推断的单级传播，也没有证据表明更新的 authorization 在下游丢失。精确 bridge map-wait 决策未被当时 trace 保存，不能把所有内部因果都说成已证明。该旧会话已停止；以下里程碑内容按历史阅读。

## 历史：2026-11-01 13:22 会话的交接/到点及其后故障

以下保留该轮早期与 350 s 观察记录；其后的第四路径原始请求到期已在本节上方说明。旧“当前”“等待批准”均指当时，现有 0.60 s 已明确获批。

用户最新明确要求：启动后持续直行，到前方障碍约 3 m 才局部规划，无 5 m 或试验时长停车条件；允许丢弃不适用的旧门槛。**保持当前自主任务运行，不因观察窗口结束暂停。** 本节覆盖后文旧“结束测试后暂停”的默认流程；用户暂停/手动接管和真实安全故障仍撤权，查看器保持打开，禁止 Docker pause 整容器。

当前运行 **`20261101T132233Z-e81ce9d6`** 已实现完整策略交接：

- 沿初始方向持续直行约 **215 s**、投影进度 **12.256182 m**；`wall_ns=280661516835168`、map **968** 的真实 occupied 净距 **2.9982492296 m** 触发 `obstacle_triggered_planning`。清除直行目标/hold/token、DISARM，再做原版 NBV；没有 known/free-exit 提前交接或 5 m cap。
- 首个真实 NBV callback 返回 **50 点 / 0.049403084 s / map 968**、`valid=true`，车辆实际转向并由 JEPA 跟踪；后续第二、第三条分别为 **50 点 / 0.049927055 s / map 1240**、**50 点 / 0.099904684 s / map 1379**。首段有直接源到点证据：观察 **290.500159 s**、`wall_ns=280736945344386`，`reached=true`、距离 **0.049342332 m**、yaw error **−0.012436413 rad**，同次采样已请求下一段。第二段在观察 **329.364027 s**、`wall_ns=280775809213067` 也明确 `reached=true`、距离 **0.055013710 m**，随后进入第三段。已验证局部执行、两次源标准到点及重复规划；**长期持续探索仍在观察**。
- 本轮定位无 native Lost、匹配误差约 **0.03–0.04 m**，尚无 `terminal:original_request_expired`。频繁重新授权/`plan_expired` 仍在审阅，不自行改阈值。滚动直行查询不计 RRT；源事件和完整 callback 见 [mission-events.jsonl](runs/20261101T132233Z-e81ce9d6/mission-events.jsonl)、[planner-execution-results.jsonl](runs/20261101T132233Z-e81ce9d6/planner-execution-results.jsonl)。动态状态见 [planner-status.json](runs/20261101T132233Z-e81ce9d6/planner-status.json)、[control-status.json](runs/20261101T132233Z-e81ce9d6/control-status.json) 与 [观察过程](runs/20261101T132233Z-e81ce9d6/startup-exploration-trial.json)。

用户已明确批准**请求速度 4 倍**。源导航水平 `0.12 × 4 = 0.48 m/s`，本轮 [navigation-parameters.yaml](runs/20261101T132233Z-e81ce9d6/navigation-parameters.yaml) 与 [guard-parameters.yaml](runs/20261101T132233Z-e81ce9d6/guard-parameters.yaml) 已核实：Guard requested X/Y 仅由 0.4 提升至 0.48，保留 Z/yaw 请求限值 0.2/0.4；导航垂向 0.06、yaw 0.12、源增益、1.2 m/s 实测速度包络、motor cap 与 watchdog 均不变。**实际稳态直行约 0.05–0.06 m/s，并非实测速度提高 4 倍。** 0.40 s 计划年龄已包含约 0.26 s 求解，导致计划可用窗口偏短；仅把最大计划年龄 **0.40→0.60 s** 的具体提议等待用户答复，尚未应用。已批准配置仍为 **64 采样 / deadline 0.36 / plan age 0.40 / latency compensation 0.36 s**，禁止自动扩大。

### 当前接口与已验证修正

`obstacle_triggered` 是 CLI/RViz/enable 两种探索开始动作的统一入口。旧 body-known/free-exit、5 m 起步 cap 和额外 first-edge view/yaw 门槛已被最新策略取代；源全 360° yaw 单次抽样、gain、15/200 与 extension 1 m / overshoot 0.5 m 保留，旧 ±90° 映射提议没有实施。源 `.5×.5×.3 m` footprint 与 occupied voxel 扩张后的 slab 相交给出车体前端净距，不再减半车长；它是规划 footprint 代理，不是精确物理 hull 测量。无 occupied 命中明确为 `none_in_current_map`，不表示 unknown 已观测自由。直行允许 unknown，但不写 free，完整路径/overshoot 仍拒绝 occupied。局部 RRT 路径仍保留原生碰撞、显式端点及当前地图验证；迟到 ARM 不能恢复已删除目标，manual/pause 受 task revision 保护。

地图接收及真实源帧进度分别使用派生预算 **250 ms + 1/5 Hz = 450 ms**，其中 5 Hz 来自本轮冻结 `wavemap.yaml`。严格前进的原 `source_revision` 才更新进度墙钟，重复 ready 心跳无法掩盖停滞；绝对采集年龄仅报告，原采集时间戳不替换，Guard/机体状态/原授权/输出期限不变。传感对齐 helper 保留源 SensorWindow append、64 样本及 **0.15 s** 限制，用最新接受流接收年龄核验 liveness，再选择请求采集时刻或更早的真实因果值；不取未来值、不改传感值。实际 postfault 5 s probe 的 103 个 body 状态证实旧历史样本 receipt-age 与当前 live stream 不一致，但不能倒推上一轮终端故障因果。

[来源与验证记录](build/wavemap-nbv/obstacle-cruise-provenance.json) 包含当前代码/授权/冻结参数、[21 项状态测试](tests/test_obstacle_cruise.py)、[两项 C++ helper](build/wavemap-nbv/obstacle-cruise-native-checks.log)、[9 sensor + 4 clock 回归](build/wavemap-nbv/sensor-alignment-tests.log) 及 native `sha256:b85cf92da2a03ab4357175e759570e5143a97af4d926df4b3b9a78562d15c919` [构建](build/wavemap-nbv/nbv-build-cruise-timing.log)。0.1 m 查询图、独立 mapping-depth 与原显示 12 m 契约保持；原生 max-range 8 m 保留粗层跨界支持，不能称为硬体素裁剪。

观察器必须使用 **`--leave-running`**。本轮 350 s 记录已结束，观察文件明确 `finite_trial_complete / task_left_running=true`；这是记录窗口而非策略停车时间，任务继续自主。新会话 [逐级控制 trace](runs/20261101T132233Z-e81ce9d6/control-timing-trace.jsonl) 记录 approved→controller_output→native_input→terminal；保留任何新 fault 的原始上下文，不自动循环清故障。

### 历史：本策略第一轮故障及早期新鲜度误用

`20261101T130928Z-9880a2b4` 的 [观察](runs/20261101T130928Z-9880a2b4/startup-exploration-trial.json) 已结束：**176.510572 s / 878 样本 / 6.907320 m / 3335 滚动请求 / 0 RRT**，因 `terminal:original_request_expired` 停止，当时 VIO TRACKING、无 Lost。[故障审计](build/wavemap-nbv/obstacle-cruise-fault-audit.json) 证明 native 缓存的原始请求 deadline 到期，而优先检查的 Guard lease/output 年龄尚未超限；Guard 收到故障时最新授权仍有效。因没有保存 native fault_ns 和逐级消息序列/原始 deadline，不能定位最早失败 hop，也不能仅凭后来的查询 timing 宣称该次 RPC 堵塞已证实。之后完成查询开销优化、map 周期预算和传感因果对齐，当前健康新运行的实际结果见本节上方。

更早 `20261101T130454Z-ef922ff8` 因新加绝对地图采集年龄 <250 ms 门槛不适配正常约 0.61 s 管线而未 ARM；厘米级自然漂移不计执行。该门槛已移除。旧 Stonefish/UI `7f5bf8…`、NBV `1d58c1…` 的构建记录属于该阶段，不替代 provenance 中当前 native 镜像与实际源码接口记录。

## 历史：2026-11-01 远距链路验收后、持续直行策略重建前

以下完整保留当时记录；“当前”“等待批准”“待验证”指当时状态。后续已由上方用户新策略取代，旧半平面 yaw 采样提议本身未实施。

当前查看会话为 **`20261101T123613Z-37c94312`**，使用 Stonefish 镜像 **`sha256:2e1b40267d9c5d1fef44c26b4ac39de910da1a96b407be0edf38719743e86910`**。任务已停止，实际停后状态为 **Guard DISARMED / ready=true**；地图继续约 10 Hz 融合，VIO 无原生 Lost、位置误差厘米级（固定停后快照约 1 cm，最新读取约 3 cm）。证据 [after-far-ray-trial-health.json](runs/20261101T123613Z-37c94312/after-far-ray-trial-health.json)。保留查看器打开；任务暂停用 `sim pause`，**不要 Docker pause 整个 Stonefish 容器**。

批准值保持 [config/nbv.yaml](config/nbv.yaml) 的 0.1 m 查询地图、JEPA 64 采样、deadline 0.36 s / 最大计划年龄 0.40 s / 延迟补偿 0.36 s、起步上限约 5 m、固定初始方向约束。已批准的完整远距策略现已实现并经过原生对照和实时链路检查：独立 `/stereo/mapping_depth` 保留同一真实视差换算的有限远距值（含 >12 m），原 `/stereo/depth` 显示/有效性 0.3–12 m 契约保留；原生 `max_range=8 m`、测量模型、源碰撞/未知检查、增益/预算和控制阈值未改。8 m 是原生更新范围参数，粗体素可跨越边界，不能称为逐体素硬裁剪。

### 最新实时结果与剩余门槛

- [wavemap-nbv-live-audit.json](runs/20261101T123613Z-37c94312/wavemap-nbv-live-audit.json) **12 项全部通过**：单一 LAS2 mapping-depth 源、Wavemap 正确订阅、投影/导入前进、NBV 不订阅深度且原生深度融合关闭、分辨率一致，检查期间任务暂停、Guard DISARMED。启动位置的 [纯预览](runs/20261101T123613Z-37c94312/planner-preview.json) 仍有 raw footprint **245 unknown / 0 occupied**，源 accepted=0；这是相机近盲区起步条件，随后才执行有限前进。
- [trial-summary.json](runs/20261101T123613Z-37c94312/trial-summary.json) 的观察窗口为 **9.229 s / 47 次采样**，起步进度 **0.363652 m** 后发现 1 个原生前向自由扩展，自动交接 NBV；交接时 raw root 仍有 **193 unknown / 0 occupied**，车体查询例外没有把原地图改成 free。
- 本次实时 [planner-execution-results.jsonl](runs/20261101T123613Z-37c94312/planner-execution-results.jsonl) 明确记录首个 NBV callback 已达 **15 accepted 节点、有正增益**，方向 selector **实际运行**，以 **`no_eligible_forward_node`** 退出。15 个节点均共享首边：前进投影 **+0.997865 m**，但首边偏航 **−106.494°**、观察方向点积 **−0.28343**，不满足已有前向观察条件。因此本轮**没有原生 NBV 路径输出，也没有跟踪 NBV 边**；实际运动仅为起步段。该阻塞与修复前 `maximum_failed_iterations / accepted=0 / selector 未运行` 是不同条件。
- 33 次 JEPA 求解中 **32 accepted、1 no predicted improvement**，max **0.270806 s**，无 deadline 超限；VIO 无 Lost。原始观察器在任务停用到异步 DISARM 回应之间读到 `final_guard=ARMED`，不是最终运动状态；[trial-summary.json](runs/20261101T123613Z-37c94312/trial-summary.json) 已同时保留早读状态和实际停后 **DISARMED**。观察工具 [watch_exploration.py](tools/watch_exploration.py) 现等待异步确认，再标记最终状态；没有修改任何控制期限或 Guard 阈值。

下一步 [forward-yaw-proposal.json](build/wavemap-nbv/forward-yaw-proposal.json) **尚未批准、尚未应用**：仅对已批准的 `initial_heading` 前进任务，把源 `rrt.cpp:347` 的全 360° 均匀偏航样本映射为初始朝向前半平面 **±90°**，位置在原单次角度随机抽样之后、构造 state/计算 gain 之前，不增加随机抽样。保留位置采样、源增益、15/200、extension/overshoot、方向分支检查、occupied/unknown 和 JEPA/Guard 参数。该提议减少后向视点覆盖以配合持续前进任务，不能保证下一棵树成功。等待用户答复，当前任务保持停止。

### 远距策略的原生验证与限制

[far-ray-proposal.json](build/wavemap-nbv/far-ray-proposal.json) 保存用户“同意，修正远距射线后先对照再规划”的授权；[far-ray-provenance.json](build/wavemap-nbv/far-ray-provenance.json) 保存实际接口、固定来源和哈希。15 项深度/输入接口测试见 [far-depth-tests.log](build/wavemap-nbv/far-depth-tests.log)，完整新镜像构建见 [image-build-far-rays.log](build/wavemap-nbv/image-build-far-rays.log)。

[同帧捕获 manifest](runs/20261101T122444Z-959ce4ad/far-ray-capture/manifest.json) 保留 10 对 640×480 float32 深度/视差与采集时刻插值相机位姿，无 GT、无外推。两张空地图的 [原生 A/B](runs/20261101T122444Z-959ce4ad/far-ray-replay.json) 各按原 10 Hz 节流融合 5 帧：free native cells **53,088→147,338**，**94,376 unknown→free**，原始显示深度与重算重叠像素误差为 0；该短批次前向自由扩展仍为 **0→0**，不能用地图覆盖增量宣称路径/控制通过。保留远距后既有边缘过滤器会遇到更多有效邻居，实际地图变化不全是增加；occupied/free/unknown 转换完整保留在 A/B 中。

原生合成用例全部通过：全无效输入不更新，4 m 表面保留 occupied，12 m 与刚超过 8 m 的有限远距输入产生 free、没有人为 occupied 截止面。原生粗体素支持也已实测：12 m 用例在名义 8 m 外仍有 **16,384** 个已观测投影格中心，最远 **9.55 m**，属于保留的上游粗层支持行为，未另加裁剪。此策略恢复测量语义，远距低视差仍可能不准确；不等于已证明所有历史空洞根因或持续自主探索成功。

### 今日修复前历史：起步、失效与深度定位

- [20261101T121834Z-7d3d69a8/trial-summary.json](runs/20261101T121834Z-7d3d69a8/trial-summary.json)：44.115 s、1.855 m、3 个可用前向扩展后交接，随后通用 `no_forward_known_free_edge` 停止，试验内无 Lost。后续 [纯预览](runs/20261101T121834Z-7d3d69a8/planner-preview.json) 的 13 accepted（不足源 15）及 0 个合格分支是另一 callback，不能断言初次实时 callback 的具体退出原因。
- [20261101T122444Z-959ce4ad/trial-summary.json](runs/20261101T122444Z-959ce4ad/trial-summary.json)：观察 37.586 s / 188 次采样，命令后约 39.49 s、1.654 m、2 个可用扩展时交接；完整 [源执行记录](runs/20261101T122444Z-959ce4ad/planner-execution-results.jsonl) 为 `maximum_failed_iterations`、accepted=0、selector=false。148 次求解为 143 accepted / 5 no improvement，max 0.28607 s，无 deadline 超限。该健康会话随后正常结束以构建远距修复，未加载其旧地图到当前估计会话。
- 第一轮之后曾错误暂停整个 Stonefish 容器，RViz 无响应。[Docker 事件](build/wavemap-nbv/stonefish-pause-events.jsonl) 记录 40.431 s pause→unpause；恢复后有 1,834 次 IMU 队列丢弃、一条原生 Lost 和最大匹配误差 5.528 m。[vio-resume-audit.json](build/vio-resume-audit.json) 记录 Lost 健康锁存/地图融合冻结实际生效；发生时任务已停、Guard 已 DISARM，ARMED 时 Lost 撤权仍未验收。具体内部因果链未完全证明，冻结前已可能融合失准状态，故旧地图不保证干净，采用全新重置。
- [forward-unknown-trace.json](build/wavemap-nbv/forward-unknown-trace.json) 固定旧 `.ot` 快照复现 +X 阻塞：209 个 unknown key 中 203 个当前几何可见、6 个当前近盲区，全部曾处于历史几何视场；几何可见不等于收到有效深度。早批次 [depth-gap-probe.json](runs/20261101T122444Z-959ce4ad/depth-gap-probe.json) 只有深度、未配视差；后续另一批 [depth-disparity-gap-probe.json](runs/20261101T122444Z-959ce4ad/depth-disparity-gap-probe.json) 才确认 768,000 个 stride-2 像素中 95,337 个 NaN 都由正视差换算 >12 m 产生，另有 160,448 个 8–12 m 测量被旧入口丢弃，实际非正/非有限视差为 0。两批帧不混合计数，也不逐格外推所有旧缺口。

### 既有保护与夜间历史证据

[root-collision-checks.json](runs/20261101T121334Z-root-check/root-collision-checks.json) 六项原生检查（含 unknown/occupied 端点拒绝）通过；[regression-20261101.log](build/wavemap-nbv/regression-20261101.log) 15 项覆盖 VIO 健康、地图版本、时限、目标租约及起步交接版本。车体例外仅影响碰撞查询，地图不写 free；当前地图未验证时不续发目标、不替代旧 hold；交接要求响应仍是当前地图版本。源实现/早期镜像关系见 [forward-health-provenance.json](build/wavemap-nbv/forward-health-provenance.json)、[source-lock.json](nbv/source-lock.json) 和 [jepa-timing-provenance.json](build/wavemap-nbv/jepa-timing-provenance.json)。NBV 诊断镜像仍为 `f9a356ecd0b5ecb456796a397453faa38f576ef4462ac537ad26e43bb64ba273`；新增远距链路在 Stonefish `2e1b4026…` 中，角度采样尚未改动。

2026-10-31 夜间 [5 m 距离试验](runs/20261031T132715Z-b02e3f40/continuous-5m-trial.json) 持续 111.426 s、前进 4.987 m，footprint unknown 245→74，因 5 m 上限仍未知而停止。后续从已观测位置开始的 [151.424 s 试验](runs/20261031T132715Z-b02e3f40/startup-exploration-trial.json) 生成三条 50 点原生 NBV 边并实际 JEPA 跟踪、重复规划，至少一端点明确到达（0.0492 m），最后一边主动暂停时尚在跟踪。568 次求解 547 accepted / 21 no improvement，无 deadline 超限，但 1,514 条控制状态中 625 条 `plan_expired`。这是历史短时闭环证据，持续探索、长期计划可用性和到点停稳仍待验收。

## 历史：2026-10-31 较早阶段的地图适配验收

**2026-10-31 较早阶段：Wavemap 单一融合 → NBV 地图适配已验收。** 当时会话 `20261031T121526Z-1369c30e`，`--autonomy` 启动，任务 enabled=false、Guard DISARMED，RViz 正常。默认 `config/nbv.yaml` 为 `map_source: wavemap`：只由 Wavemap 融合 LAS2 深度，发布完整概率 OctoMap，原 NBV 整树替换其查询副本。NBV ROS2 桥不再订阅深度；原 NBV 三路点云及 OctomapManager 点云/视差等输入订阅均在构建临时副本关闭。TP 原算法文件未修改，RRT / gain / 15/200 / controller 参数不调。

0.1→0.4 m 转换：任一占据子格优先；自由必须完全已观测覆盖；混合自由与未知仍未知。分类继承 Wavemap 原生 OccupancyClassifier 默认零 log-odds / `1e-3` 未观测阈值，完整概率输出；导入 OcTree 的默认 0.5 阈值与之匹配，旧独立 NBV 深度融合的 0.7 不参与。橙色显示依然 >0.70，所以蓝色占据投影可能更厚；不是完全复制原独立融合器的传感模型。没有用占据点云重打射线，没有 setFree/GT/位置裁剪，也没有将历史地图装入新估计坐标系。

实际验证：旧手动地图 `20261030T123131Z-21d420df/map.wvmp` 转换约 15.24 ms，7,821 free / 9,937 occupied / 1,283 混合未知目标格；完整消息约 79.8 KB。离线专用后端 `20261030T131747Z-wavemap-import` 完成 13 项导入/替换/概率逐字节一致/输入拒绝检查，原生 depth insertions=0。五类 C++ 投影语义检查、两项并发地图版本回归通过。实时 audit 九项全部通过：3 s 新增 30 帧融合、16 次导入，转换约 4–5 ms，唯一投影发布者，NBV 无深度订阅，DISARMED。

规划结果必须区分：旧地图历史真实起点离线规划仍因 footprint 2 处未知、0/26 自由扩展而返回 0 点；新实时会话的真实起点预览 map 345 有 18 处未知、0 占据、0/26 自由扩展、0 点。这说明地图传入成功但当前起点条件仍不足，**本轮没有执行 controller 跟踪或宣称自主避障成功**。源地图接口与哈希见 `nbv/WAVEMAP_ADAPTATION.md`、`build/wavemap-nbv/provenance.json`；验收汇总 `build/wavemap-nbv/REPORT.md`，详细 `import-checks.json` / 当前 `wavemap-nbv-live-audit.json`。

## 以下为 2026-10-30 及此前历史记录

最新短段尝试：用户要求当前位置重新规划并跟踪。当前会话原版 NBV 再次返回 23 有效节点 / 50 点 / 0.98 m。通过新增只用现有 Guard + NavigationGoal 的 `tools/track_nbvedge.py` 做了 20 s JEPA 跟踪；目标未到达，76/76 求解超时、median 0.25054 s、无可执行计划。原来 .16/.20/.16 三项时限和 64 采样保持。最终真实终端 DISARMED，六路输出 0。证据 `runs/20261030T123131Z-21d420df/nbv-edge-tracking-test.json` 及 summary/plot；不要声称 controller 已跟踪成功。准备期间车辆漂移使实际启动点偏离规划根约 0.518 m，启动前重新做了实际起点到目标的原生自由空间检查，报告保留此限制。


最新活动会话 `20261030T123131Z-21d420df`。用户澄清“位姿不确定”实际指起点周围地图未知，要求移除此起步门槛并重置再试；本轮没有做位姿协方差/Good-Marginal-Lost 改造。起步新增固定初始 self-intersecting OctoMap cells 的 unknown 豁免，只用于向前起步，不写 free、不接受 occupied、不豁免外部未知；源距离 1 m + overshoot 0.5 m 保留。`sim start-preview` / `sim explore-forward` 已实现。源接口、测试与限制见 `nbv/STARTUP_ADAPTATION.md`。

本轮实际起步段已规划和进入 JEPA，但未达到终点；317 次预热后求解 median 0.2021 s / P95 0.2417 s，现有 0.16 s 期限仍主要阻断运动。已向用户提出 deadline 0.26 / plan age 0.32 / latency 0.26，但用户只表示手动新地图可以使用，没有明确批准这三个数值，**不要据此擅自改时限**。手动控制的积分改为源 TinyMPC 同样的仿真时钟逻辑，真实 feedback age 和原授权期限保留，四项回归通过；出处哈希记录在 build/forward-start-provenance.json。

用户随后又手动建图，并允许使用新地图。已选择自主控制来源但未启动任务，调用新的标准 `sim plan-preview`，结果见本轮 `planner-preview-after-user-map.json`。当前运动状态重新操作前读取 planner-status/control-status；不要把界面旧 map 外接框当作 fence。蓝色大框是最近规划/检查时的 NBV OctoMap bounds；蓝色小立方体为 0.4 m 原生 OctoMap occupied，橙色是并行 Wavemap，**没有 Wavemap→OctoMap 转换**。


**最新里程碑：真实车辆起点已出轨迹。** 用户手动建图后，按其“在现在的位置直接做轨迹”要求，从真实 OKVIS2 body `[12.785832, -0.816176, 1.257216]` m 请求原版 NBV，地图版本 569：footprint 未知 0、占据 0，23/26 个原生扩展方向自由；有效树节点 20（含根 21）、正增益，返回 50 点 / 0.98 m 的源 best edge，原生已知自由空间检查通过。采样仍为全局 X/Y ±115、Z −5..6，没有改采样范围、增益、15/200 条件或未知检查。用户已在 RViz 亲眼确认蓝色路径。此结果是真实起点规划通过，尚未执行 JEPA 跟踪或闭环换段。

证据：当前会话 `20261030T115426Z-88ab2f8b/planner-preview-actual-success.json`、`planner-status-actual-success.json`、`nbv-actual-success.png`、`nbv-actual-success-rviz.png`。为本次只规划，已保存再次出现的手柄 fault 并暂停手柄输入流，经源 CLEAR_FAULT 后 Guard DISARMED、自主任务 enabled=false。手柄接收需再次 `sim joystick` 才能接入；不要自动 ARM、不要自动切到自主跟踪。默认 RViz 保持用户已看到的蓝色路径，生成的 `nbv-actual-success.rviz` 只是备用观察配置，没有另行切换用户视角。


手动建图阶段记录：会话 `20261030T115426Z-88ab2f8b` 以 `--autonomy` 启动，再确认切到 `manual` / body PI。真实 BEITONG `/dev/input/js0` 产生 646 次请求和实际推进器输出。随后用户直接要求当前位置规划，结果见上面的最新里程碑；手柄输入流现已停止。主仿真运行会话为 23404。

本轮手动驾驶在仿真约 t=49.3 s 捕到 `terminal:original_request_expired`，定位仍 TRACKING、Guard 最近请求新鲜，不能归因为空间围栏或 JEPA 超时。用户反馈手柄无法控制后，确认中立并通过源 `CLEAR_FAULT` 清除此单次故障（DISARMED、source=cli）；不会自动 ARM 或循环清故障。已启动只读 `tools/trace_control.py` 记录 approved→controller_output→native_input→terminal/guard 的时序，限 120 秒或下一次终端 fault，日志在本次运行 `control-timing-trace.jsonl`。原 deadline/授权/输出过期规则未改。真实手柄按键路径已在本轮确认，既往“未验收”描述是旧记录。故障来源源码：外部 Stonefish 工作区 `vendor/patches/0004-terminal-command-lease.patch`，以及 `guard/uw_guard/node.py`、`controller/uw_controller/node.py`、`simulations/uw_simulations/actuator_node.py`。


此前纯规划会话：`20261030T114236Z-353c4200`（已结束）。按用户要求重置了已漂移的仿真，当前为 `--planning-only`：保留真实 OKVIS2、LAS2、Wavemap 和原版 NBV，不启动 Guard、控制器、导航跟踪器或推进器适配器。原会话 `20261030T112044Z-dbb95ef1` 的地图和证据完整保留。仿真继续运行时仍受原有水流/浮力作用；本轮没有改水流、动力学或加入定点保持。

### 原版 NBV 组件拆解与已批准对照

用户优先要求先得到轨迹，再考虑 controller。纯规划预览 `sim plan-preview` 只显示/保存实际地图上的原版结果，不 ARM、不发 NavigationGoal、不伪造初始化完成。

旧会话中同一冻结真实 OctoMap、同一虚拟测试起点的 A/B 已完成：全局 X/Y ±115 m、Z −5..6 m 得到 9 个有效树节点、有正增益、无路径；仅换用户明确批准的采样窗 X 0..5.2、Y −3.2..3.2、Z −2.8..2.8 m 后达到源 15 节点条件，返回 49 点且通过原生已知自由空间检查。虚拟起点是真实旧机体位置向前 1.2 m 的组件测试点，不是车辆实际位置，没有执行运动。真实旧起点在同一窗口仍为 0 节点，车体 footprint 的 6 处采样未知，26 个源扩展+overshoot 检查均不可通行。说明起点覆盖和采样尺度是分别需要解决的条件，不能宣称实际起点路径或控制闭环已通过。

证据：`runs/20261030T112044Z-dbb95ef1/planner-map-ab.json`、`nbv-frozen.bt`；可复现脚本 `tools/planner_map_ab.py`。快照 SHA256 `c8403621807b9f6bf8d44cbd3e5613578788e833d03f5c46c8a5c3d5f6bc35c9`。该快照在稍后的地图时刻保存，边界是 0..5.6 / −3.2..3.6 / −2.8..3.2；实验仍严格使用用户批准的 0..5.2 / ±3.2 / ±2.8 采样窗，没有随新地图放大。A/B 是单次原生随机结果，未当作统计性能结论。默认 NBV 配置仍保留全局采样范围；实验窗口只存在于纯规划诊断操作。

重置后当前会话已从真实起点复测一次：预览地图版本 163，6 处未知 footprint、0/26 自由扩展、0 节点/0 路径。地图继续更新，界面需同时区分当前地图版本和上次预览版本。

### RViz / 操作接口（2026-10-30 历史）

本段保留旧面板/编辑器的历史来源；这些入口现已移除，当前操作以下方“运行与验收”及 README 原生工具栏说明为准。

按当时用户要求移植本机 GBPlanner 洞穴版 RViz 分组、视角与嵌入式控制面板，改为 ROS2 Jazzy 并保持白底蓝色。`sim rviz` 可重开；`sim ui` 保留完整路径点编辑器。RRT 树、候选视点、采样/观测边界与 footprint 检查来自原版真实节点/地图查询，不画假树；无路径时允许树为空。源文件固定版本/哈希、ROS1→ROS2 话题接口与许可证见 `rviz_panel/PROVENANCE.md`。只读 `pelagosRoot()` 加在构建临时副本，不改 TP 原库。GBPlanner 全局图、返航等功能没有接入。

### 已完成和仍未完成的控制工作

256/128/64 采样实际运行均未完成源初始化。64 是用户在明确提出 128→64 后要求继续复测所采用的仿真覆盖；源 deadline 0.16 s、有效期 0.20 s、所有增益和 Guard 阈值保留。最新配置及授权依据见 `config/nbv.yaml`，不要再次自动减采样或放宽期限。既往 64 运行近期 92 次求解中 88 次超时、中位约 0.20 s；本轮不重复 controller 实验。

已增加自主/手动模式选择：切换先暂停撤权，手动复用源 body PI，返回自主不自动恢复任务。自主运行中的手动反馈使用原生 OKVIS2 body；默认独立手动仿真仍沿用既有 GT 辅助。有限手动速度实验曾前进 0.1964 m、随后撤权归零（`20261030T074752Z-e042cdb4/manual-pi-evidence.jsonl`）。真实物理手柄完整按键使能路径尚未验收。Guard 清故障接口已从错误 RESET 修正为源 CLEAR_FAULT；来源哈希见 `build/takeover-source-provenance.json`。

RViz 曾因 Ogre GL 顶点缓冲错误崩溃；非零退出只停查看器，不再杀地图/估计进程，可用 `sim rviz` 重开。LAS2 深度偶发 CDR 缺少 60 字节已有 raw 证据，根因未证实，未猜测性改 DDS。

## 当前运行与验收

1. 先核对任务/故障终态与已保存日志。按需 `./Simulation/sim build` 更新主镜像、`./Simulation/sim build --planner-only` 更新 native；不把构建通过当作实际已加载。
2. 新健康会话使用 `./Simulation/sim run --autonomy`。RViz 原生五项工具是 **Explore / Plan Preview / Manual / Auto / Pause**，没有 MissionControl panel、路线编辑网页或独立 Qt 编辑器。`./Simulation/sim rviz` 为当前会话打开 RViz，`./Simulation/sim ui` 只是兼容别名。
3. **Explore** 确认 Auto 后开始 turn-only reactive 巡航；**Auto** 只选所有权；**Manual** 暂停自主再接管；**Plan Preview** 只预览；**Pause** 撤销任务运动。终端分别为 `sim explore` / `sim autonomous` / `sim manual` / `sim plan-preview` / `sim pause`。状态使用 `./Simulation/sim planner`。
4. 路径点只通过保留的核心 CLI routes：按 [README JSON 格式](README.md#原生工具栏与-cli-路径点) 填入本次 run_id、`frame_id=rov01/odom`、`epoch=1` 和 `poses`（XYZ+xyzw），再用 `./Simulation/sim navigate --route /本机路径/waypoints.json`。它仍走任务应答、源跟踪与当前图已知自由空间验证；`2D Goal Pose` 已移除，因为旧路线编辑器没有 consumer。不要寻找旧 panel 保存入口。
5. 持续探索观察使用 `--leave-running`；观察到时长不自行停车。用户 Pause/manual 和真实 Guard/Lost 故障仍撤权；正常关闭 RViz 沿用 supervisor 结束 pipeline。只按源停稳/时限与真实状态证据判断，不放宽 Guard、不伪造自由空间；参数变更须先向用户提供具体数值、来源和证据。

`023622` 当时实际加载 **b29260c…** UI；新 **41978a32…** feedback 镜像已构建、**15 protocol/feedback** 检查通过，固定英文提示分别说明 Auto/Manual 所有权、Guard FAULT 与 Task Stopped。新 feedback 未被该旧运行加载；它现已随新 efeb 镜像在 `030254` fresh 加载，不能把旧截图当作新运行证据。原生 VIO ingress/processFrame 的无参数源复制接口修复已完成 5 项检查与完整构建，旧 `030254` 通过有限负载窗口审计，随后日志先记录 RViz GL index buffer 异常/−6，再记录 Stonefish 0；两者因果未定；新 `031510` 已完成上述至少四轮有限窗口核对并继续运行；三层 RViz 显示去重尚未在当前 viewer reload，不混作现场结果。旧 `023622` 的 `state_clock_mismatch` 终态保持。

核心证据：`controller-evidence.jsonl`、`control-fault-evidence.jsonl`、`mission-events.jsonl`、`mission-route.json`、`planner-status.json`、`control-status.json`、VIO 原始状态和 GT 对比。当前来源见 [桥 provenance](build/wavemap-nbv/turn-cruise-bridge-provenance.json) 与 [UI provenance](build/rviz-native-tools/ui-native-tools-provenance.json)。
