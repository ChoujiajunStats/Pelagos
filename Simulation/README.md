# Pelagos Stonefish 仿真

本目录接入本机现有 Stonefish、六推进器 BlueROV2 和 Pelagos 的 OKVIS2、LAS2-S、Wavemap。Stonefish 与算法在同一个 ROS 2 Jazzy 容器中运行；Pelagos 真机部署仍使用原来的 Humble 环境和标定。

## 一键演示

在本机桌面终端执行：

```bash
cd /home/hong/Desktop/Pelagos; ./Simulation/demo
```

入口先检查环境和已准备好的主镜像/native 镜像，再重置本项目的两个仿真容器，从 spawn 启动新的自主会话；**每次重新开始，但保留 `Simulation/runs/` 的旧记录**。它不构建、不改参数。RViz 中点 **Explore** 开始，**Pause** 暂停任务，**Manual** 手柄接管，**Auto** 只选择自主所有权，**Plan Preview** 只预览；`sim ui` 只是同一 RViz 的兼容入口。

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

## 历史：`023622` 两轮转向巡航与第三轮 Guard 撤权

旧会话 **`20261102T023622Z-181e7b43`** 已实际完成两轮用户要求的 **3 m 遇障 → 制动 → NBV 选择转向 → 原地转向停稳 → 当前地图确认方向 → 连续巡航**。不再逐段等待 NBV endpoint 精确到点或 sampled-view observe。任务通过 RViz Explore workflow 的真实请求 `5281870b…-1` 启动，完整起点是 [mission event](runs/20261102T023622Z-181e7b43/mission-events.jsonl) 的 `wall_ns=46015068645558`；observer 到第一次转向才开始，不能用它的起点代替完整任务起点。

[独立 live 审计](build/wavemap-nbv/turn-cruise-live-audit.json) 记录：

| 已完成轮次 | 遇障净距 | 转向后当前图净距 | 恢复巡航至下一次遇障的实际投影 |
| --- | ---: | ---: | ---: |
| 第一次 | 2.99631 m | 8.80887 m（map 413） | 6.12385 m |
| 第二次 | 2.97931 m | 3.34182 m（map 639） | 0.33749 m |

两次 pure-turn 的 XYZ 参考始终为零；独立匹配原生机体数据的转向完成前 1 s 样本，最大三维速度/yaw rate 分别为 **0.02947 m/s / 0.02776 rad/s** 和 **0.01724 m/s / 0.02636 rad/s**，均满足保留的源 **0.04 / 0.04 / 1 s dwell** 条件。随后滚动目标继续在前方约 1 m，**0 endpoint-reached / 0 observe**。第二段只有约 0.34 m，是新方向当时仅比 **3 m** 触发门槛多约 0.34 m 的结果；不是距离 cap 或到点等待，也不代表当前 gain 选择已优化全局路线。

**旧 `023622` 任务已停止：**接受请求后 **228.70341 s**，第三次转向期间 Guard 以 `state_clock_mismatch` 撤权，最终 **enabled=false / Guard FAULT**。故障时 clock **245.24 s**、body **244.98 s**，相差 **0.26 s**；原生估计仍 `TRACKING`、`native_loss_evidence=null`，没有原生 tracking-failure 日志。不能把此前 generation 1 / ARMED 的健康快照当成当前仍运行，也没有证据把这次时间戳不一致归因为 GPU、CPU 或 VIO 算法。用户持续运行要求和 `--leave-running` 保留；真实 Guard 故障仍撤权，没有自动 clear/reARM。两轮局部策略闭环已实测，**长期或全局探索尚未验收**。

全英文 RViz 已实际加载，原生工具栏 **Explore / Plan Preview / Manual / Auto / Pause** 可见，custom mission panel 与独立 web/Qt UI 已移除，见 [真实 RViz 客户区截图](build/rviz-native-tools/rviz-live-english.png)。UI 的 12 项协议检查、真实窗口 **12.012 s / exit 0** 测试及初始化/关闭竞态修复见 [UI provenance](build/rviz-native-tools/ui-native-tools-provenance.json)。本轮加载 main 镜像 **b29260c…**、native **0302d2bb…**。此前 `023034` 在任务开始前发生 RViz 初始化 **−11**，不是本轮导航故障；已用 toolAdded 后延迟刷新 QAction、关闭时停止 ROS timer 修复。

后续 **41978a32…** feedback 镜像已构建并通过 **15 项**协议/反馈检查，新增固定英文 ownership、Guard FAULT / Task Stopped 提示；它尚未由旧 `023622` 会话加载，不能把新反馈界面当作该轮截图或 runtime。

[64 项桥/host](build/wavemap-nbv/turn-cruise-bridge-tests.log)、[44 项真实 source](build/wavemap-nbv/turn-cruise-navigation-tests.log)（31 navigation / 9 sensor / 4 clock）和 [native 检查/构建](build/wavemap-nbv/plan-turn-provenance.json) 通过。桥使用 `nbv-reactive`、neutral index **0**、pure-turn index **1**；转向完成经 `set_cruise_direction` 重验当前图和原源首边几何，再恢复连续 current-heading cruise。首个滚动目标计算期间持续 neutral，避免旧 200 ms 意图到期；Preview 也使用 `plan_turn` 候选条件，黄色转向箭头位于当前机体，不冒充原 endpoint gaze。

来源、当前桥哈希与本轮终态见 [桥接 provenance](build/wavemap-nbv/turn-cruise-bridge-provenance.json) 和 [导航 provenance](build/wavemap-nbv/turn-cruise-navigation-provenance.json)。保持 **0.1 m map / 3 m trigger / JEPA 64 / deadline 0.36 / maximum age 0.60 / latency 0.36 s / horizontal cap 0.48 m/s**，源 gain、yaw 抽样、**15/200**、碰撞/unknown 与 Guard 期限不变。gain 原来评价的 sampled-view yaw 不再执行，不能声称已完成相同 gaze；无 known occupied 命中不表示 unknown 已观测自由。

旧 `140825` 因 generation 恢复重算 bearing 而人为 pause；修复后的 `141248` 验证固定方向和停稳，但 endpoint 振荡未收敛，最后是 [RViz exit 0 后全 pipeline 结束](build/wavemap-nbv/heading-resume-live.log)。旧缓存 ARMED 不代表仍在运行，当前新 hash 不回填旧 runtime。

## 历史：2026-11-01 制动交接、到点与 GPU 故障归档

以下属于旧 endpoint/observe 执行策略；当前 turn-only 巡航以上节为准。[旧 provenance](build/wavemap-nbv/obstacle-cruise-provenance.json) 保留旧来源与证据，不作为新策略 runtime 记录。

旧会话采用交接顺序 **持续直行 → 约 3 m 遇障 → 主动制动并等待源停稳 → 局部规划**。遇障后持续发送 neutral index 0 的零速度意图，等待源 Tracker 的估计三维速度 **<0.04 m/s**、yaw rate **<0.04 rad/s**、连续 **1 s**；只有当前 run/mission/generation、该制动阶段之后的新鲜 index 0 `reached=true` 才进入规划。零速度请求本身不等于已经停住，也不是 GT 停车证明；新路径仍从最新机体位置与当前地图检查。

会话 **`20261101T134639Z-86ae3703` 已停止**，留下明确的 [制动与跟踪证据](build/wavemap-nbv/braking-live-audit.json)：前净距 **2.99235 m** 触发制动，**6.84976 s** 后才交接；交接前完整 1 s 的 26 个原生机体样本均满足源停稳标准，最大速度 **0.03743 m/s**、最大 yaw rate **0.03123 rad/s**。随后第一条 **50 点** NBV 路径实际到点，距离 **0.02433 m**、航向误差 **−0.00862 rad**；第二条 **50 点** 路径已生成并跟踪。

该轮观察 **124.871 s / 622 样本** 后由 Guard 的 `estimator_sensor_stale` 撤权。当时 clock/image/IMU/body 为 **138.63 / 138.36 / 138.60 / 138.61 s**，原生估计仍为 `TRACKING`、`native_loss_evidence=null`，此次不属于原生 Lost。同期宿主机 kernel 在当地 **21:49:05** 记录 **Xid 79: GPU has fallen off the bus**、**Xid 154: Node Reboot Required**；LAS2 在 UTC **13:49:05.219** 报 CUDA launch failure，supervisor 随后记录 Stonefish exit **−6**。原始日志及时间域限制见 [GPU 审计](build/wavemap-nbv/gpu-fault-audit.json)。`pid/name=code` 不是根因归属证据；不能据此归因某个应用、CPU 负载或 VIO 算法。

宿主机已于当地 **21:49:50** 重启，RTX 5090 / driver **580.178.04** 的 `nvidia-smi` 正常，`sim doctor` 复查通过。同配置复测 **`20261101T135825Z-55bdb8bc`** 也记录了源停稳交接、一次真实到点（距离 **0.03639 m**）和第二条有效规划，随后为加载用户新要求而 **operator pause / explicit_stop**。旧会话终态为 **任务 disabled / Guard DISARMED**，这是主动暂停；它当时运行旧导航模式，不能用后来修改的源码回填成新 heading 阶段结果。48 项 [host 回归](build/wavemap-nbv/braking-tests.log) 与 22 项 [源接口回归](build/wavemap-nbv/braking-source-tests.log) 保留为该轮验证；Stonefish/UI `592ff91a…`、native `b85cf92…` 未因重启恢复重调参数。源码、构建与证据哈希见 [provenance](build/wavemap-nbv/obstacle-cruise-provenance.json)。

没有新增或放宽阈值：复用源停稳标准，保持 **64 / 0.36 deadline / 0.60 age / 0.36 latency**、水平请求上限 **0.48 m/s**、源 yaw/gain、碰撞/未知判据及各级租约。用户持续探索要求保留，有限观察仍用 `--leave-running`；真实 Lost 或安全故障仍撤权。

上一轮原生 Lost 的 [视觉审计](build/wavemap-nbv/fast-vio-loss-audit.json) 及 [控制交接审计](build/wavemap-nbv/fast-control-transition-audit.json) 显示数据流持续，首次失败前没有队列溢出，旧交接未完成停稳就开始横移/转向。它们**没有证明速度、转弯或场景中的哪项导致特征关联失败**。`134639` 已验证停稳交接和一次源标准到点，但 GPU 故障中断了持续运行，不能宣称已修复 VIO 根因或通过长期探索验收。宿主机重启前后的 monotonic `wall_ns` 不能跨会话直接相减。

## 历史：2026-11-01 `133950` 的原生 VIO Lost 撤权

最新会话 **`20261101T133950Z-15f9e2bc`** 已在 **57.580 s / 288 样本** 因原生 OKVIS `TRACKING FAILURE: quality=0` / Lost 停止，实际任务 **enabled=false / Guard FAULT**，地图 Lost 锁存、融合冻结为 0 Hz。此前实际直行进度 **12.944951 m**，map 272 测得净距 **2.807717 m** 后切换，生成 **50 点 / 0.099275 s** 的有效 NBV 路径；本轮尚无真实路径到点事件。约 30 s 的 ARMED 检查是故障前历史，不能当作当前状态。用户已明确批准**仅最大计划年龄 0.40→0.60 s**；本轮 [冻结 JEPA 设置](runs/20261101T133950Z-15f9e2bc/jepa-settings.yaml) 和实际 controller evidence 均为 **64 采样 / deadline 0.36 / plan age 0.60 / latency 0.36 s**。水平请求上限 **0.48 m/s**、Guard requested X/Y **0.48**、所有原始授权/Guard/终端期限保持。

本轮同时验证地图重检查期间的 neutral intent：不续发未经当前地图验证的运动目标；正常短查询沿用尚未到期的授权，仅在距原 200 ms deadline 剩 **两个 20 Hz 导航 tick（100 ms）** 时续发保留 waypoint **0** 的零速度意图，避免每个 5 Hz map 更新都重置 MPPI 目标。规划/旧 hold 等待期持续发布 neutral；地图不健康但机体健康时先 neutral 再 DISARM。导航仍用最新真实状态和源 guide，保留原身份/序列/时限检查；真实路径到点仅计 index≥12，新 `nbv_edge_reached` 事件提供明确证据。切换后 UI 显示“切换时障碍净距”，不会把历史 2.998 m 当作当前实时测距。

查看 [planner-status](runs/20261101T133950Z-15f9e2bc/planner-status.json)、[control-status](runs/20261101T133950Z-15f9e2bc/control-status.json)、[任务事件](runs/20261101T133950Z-15f9e2bc/mission-events.jsonl) 和 [逐级控制 trace](runs/20261101T133950Z-15f9e2bc/control-timing-trace.jsonl)。本轮已因真实定位故障撤权；**不宣称修复已通过长期运行验收**。`--leave-running` 仅保证观察时长不停车，不能覆盖真实安全故障；observer 中提前读到的 `task_left_running=true` 不代表最终仍运动，以实际 disabled/FAULT 为准。用户的持续任务要求保留，后续需健康估计会话；不要自行清 fault、重新 ARM 或 Docker pause 整容器。

当前 Stonefish/UI 镜像 **`2b907357…`**、native **`b85cf92…`**。[42 项核心/状态回归](build/wavemap-nbv/obstacle-cruise-tests.log) 与 [21 项 neutral/sensor/clock 回归](build/wavemap-nbv/neutral-alignment-tests.log) 通过；完整哈希、批准来源和本轮配置见 [策略 provenance](build/wavemap-nbv/obstacle-cruise-provenance.json)。前一轮虽已完成 3 m 交接、到点与重复规划，随后第四条路径阶段发生原始请求到期；详见下方历史及 [第二次故障审计](build/wavemap-nbv/second-cruise-fault-audit.json)，不能再把旧会话称为当前持续自主运行。

## 历史：2026-11-01 13:22 会话的策略里程碑与后续租约故障

以下为该轮在 350 s 观察结束时的记录；之后实际发生第四路径阶段的原始请求到期，任务已停止。当前运行和批准参数以上节为准。

当前会话 **`20261101T132233Z-e81ce9d6`** 已验证用户要求的策略：启动后持续直行，不等待周围 known；约 **215 s / 12.256182 m** 后，当前地图 **968** 的真实前方 source-footprint 净距降至 **2.998249 m**，触发撤销直行、DISARM，再执行局部 NBV。首个原生结果为 **50 点有效路径 / 0.049403 s**，车辆实际转向并由 JEPA 跟踪；随后第二、第三条也均为 **50 点**（map 1240 / 1379，0.049927 / 0.099905 s），继续实际跟踪。见 [任务事件](runs/20261101T132233Z-e81ce9d6/mission-events.jsonl)、[原生规划结果](runs/20261101T132233Z-e81ce9d6/planner-execution-results.jsonl)、[跟踪状态](runs/20261101T132233Z-e81ce9d6/planner-status.json) 与 [观察记录](runs/20261101T132233Z-e81ce9d6/startup-exploration-trial.json)。源导航在观察 **290.500159 s** 明确 `reached=true`，首端点距离 **0.049342 m**；第二段在 **329.364027 s** 也有 `reached=true`、距离 **0.055014 m**，随后进入第三段。**已验证遇障交接、真实路径跟踪、源标准到点及再次规划；长期探索仍在观察。** 本轮定位无 Lost、误差约 3–4 cm，尚未再现上一轮 terminal expiry。

用户最新要求保持自主任务持续运行。观察器使用 **`--leave-running`**：本轮 350 s 记录已结束，保存的 `task_left_running=true`；**任务继续，不暂停，也不是时间或距离上限**，不要因观察结束自行 DISARM。用户暂停/手动接管及真实安全故障仍撤权。保留查看器；用户需要停任务时用 `sim pause`，不要 Docker pause 整个容器。

用户另已批准速度上限为原来的 **4 倍**：源导航 `0.12 × 4 = 0.48 m/s`，本轮 [导航冻结参数](runs/20261101T132233Z-e81ce9d6/navigation-parameters.yaml) 与 [Guard 冻结参数](runs/20261101T132233Z-e81ce9d6/guard-parameters.yaml) 已核对，Guard 仅 requested X/Y 上限由 0.4 增至 0.48。垂向 0.06、yaw 0.12、增益、实测速度包络 1.2 m/s、motor cap 和各级 watchdog 保持。**实际直行速度仍约 0.05–0.06 m/s，没有达到 4 倍实测速度。** 现有 0.40 s 计划年龄包含约 0.26 s 求解，`plan_expired` 仍限制计划可用时间；仅把最大计划年龄 **0.40→0.60 s** 的提议等待用户答复，**未应用**。当前 JEPA 仍为 **64 采样 / 0.36、0.40、0.36 s**。

`obstacle_triggered` 已移除旧 5 m cap、body-known/free-exit 提前交接及额外 first-edge view/yaw 拒绝；源全 360° yaw 抽样和 gain 保留，旧 ±90° 提议未实施。3 m 是源 `.5×.5×.3 m` footprint 首次接触实际 occupied voxel 的前净距，已包含车体前端；没有 occupied 命中只表示当前地图缺少证据。直行允许 unknown，原始地图不写 free，完整滚动路径和原 overshoot 仍拒绝 occupied；新局部路径保留源碰撞、端点及当前地图版本验证。

地图新鲜度现按冻结发布率派生：**250 ms 原接收 allowance + 5 Hz 的一个 200 ms 发布周期 = 450 ms**，分别约束状态接收与严格前进的真实 `source_revision`；重复 ready 心跳不能续命。原采集时间戳保留，绝对采集年龄仅报告。传感对齐则区分最新接受流的接收新鲜度与历史因果值选择，继续使用源 **0.15 s** 限制，不取未来值；[13 项 sensor/clock 回归](build/wavemap-nbv/sensor-alignment-tests.log) 通过。来源、21 项状态测试、两项 C++ helper、native `b85cf92…` 构建与实时证据见 [策略 provenance](build/wavemap-nbv/obstacle-cruise-provenance.json)。0.1 m 查询地图及远距 mapping-depth 链路保持；max-range 8 m 不是硬体素裁剪。

此前运行 **`20261101T130928Z-9880a2b4`** 已于 **176.511 s / 878 样本 / 6.9073 m** 因 `terminal:original_request_expired` 停止，不能再称运行中。其 [故障审计](build/wavemap-nbv/obstacle-cruise-fault-audit.json) 证实 native 缓存原始 intent 到期；当时缺少逐级消息 trace，无法确定最早失效传输 hop。随后修正地图周期预算、原生查询开销及传感因果对齐，新会话记录 [完整控制 trace](runs/20261101T132233Z-e81ce9d6/control-timing-trace.jsonl)。更早 `130454` 会话因不适用的绝对采集年龄 <250 ms 门槛未 ARM，正常约 0.61 s 管线延迟不等于源停滞；该门槛已移除。

下文旧的初始化扫描、5 m 起步与 known/free-exit 交接均为历史复现记录；当前入口和持续运行要求以本节为准。

### 历史：2026-11-01 远距修复后、持续直行策略重建前

以下保留当时记录；其中“当前”“等待批准”均指当时状态。旧 yaw 提议已由上方用户新策略取代，采样本身未改变。

当前查看会话 **`20261101T123613Z-37c94312`** 使用 Stonefish 镜像 `2e1b40267d9c5d1fef44c26b4ac39de910da1a96b407be0edf38719743e86910`。已批准的远距射线修复完成原生 A/B 与新实时链路验收，当前剩余阻塞是前向观察条件。任务已停，实际 [停后健康快照](runs/20261101T123613Z-37c94312/after-far-ray-trial-health.json) 为 **Guard DISARMED / ready=true**，VIO 无 Lost、误差厘米级（固定停后快照约 1 cm，最新读取约 3 cm），地图约 10 Hz。查看器保留打开；停止任务用 `sim pause`，**不要 Docker pause 整个 Stonefish 容器**。

新 [live audit](runs/20261101T123613Z-37c94312/wavemap-nbv-live-audit.json) **12 项通过**，Wavemap 单一订阅 `/stereo/mapping_depth`，原 `/stereo/depth` 显示 12 m 契约保留，NBV 只查询导入地图。初始纯预览 raw root 245 unknown / 0 accepted 后，有限 [起步试验](runs/20261101T123613Z-37c94312/trial-summary.json) 观察 **9.229 s / 47 次采样**，前进 **0.363652 m**、发现 1 个前向自由扩展即交接（raw root 仍有 193 unknown）。33 次 JEPA 求解 32 accepted、1 no improvement，max 0.270806 s，无 deadline 超限。

首个实时 [NBV callback](runs/20261101T123613Z-37c94312/planner-execution-results.jsonl) 已有 **15 accepted 节点、正增益**，selector 实际运行，但所有节点共享首边：前进 **+0.997865 m**、偏航 **−106.494°**、观察方向点积 **−0.28343**，因 `no_eligible_forward_node` 停止。**没有 NBV 路径输出/跟踪，运动仅为起步段。** 观察器早读到的 `final_guard=ARMED` 位于异步 DISARM 回应之前；summary 已区分它与实际最终 DISARMED，观察工具现等待确认，控制阈值未改。

[下一步角度提议](build/wavemap-nbv/forward-yaw-proposal.json) **等待用户批准、未应用**：仅在 `initial_heading` 任务把源全 360° 偏航样本映射到初始朝向 **±90°**，不增加随机抽样，保留位置采样、增益、15/200、碰撞/未知检查及 JEPA/Guard 参数。该条件与修复前源树 `maximum_failed_iterations / accepted=0 / selector未运行` 的阻塞分开记录。

[原生同帧 A/B](runs/20261101T122444Z-959ce4ad/far-ray-replay.json) 各融合同批 5 帧，free 格 **53,088→147,338**，94,376 unknown→free；该短批次前向扩展仍为 0→0，不代表规划通过。无效、4 m 表面及有限远距合成检查通过。原生 `max_range=8 m` 保持，但粗体素可跨边界，12 m 合成测量的已观测格中心最远 **9.55 m**；没有把远距值裁成 8 m 占据面。[far-ray-provenance.json](build/wavemap-nbv/far-ray-provenance.json) 保存授权、源接口/哈希、15 项深度/输入测试、构建与最新实时证据。

参数保持 [config/nbv.yaml](config/nbv.yaml) 的 0.1 m 查询地图、64 采样、0.36 / 0.40 / 0.36 s 三项时限。更早的两次起步交接、旧地图可见空洞、40.431 s 整容器暂停后 IMU 溢出/Lost 及全新重置均作为历史保存在 [实验接管说明](EXPERIMENT_HANDOVER.md)、[forward-unknown-trace.json](build/wavemap-nbv/forward-unknown-trace.json)、[vio-resume-audit.json](build/vio-resume-audit.json)。2026-10-31 曾从已观测起点验证短时 NBV+JEPA 重复换段；持续自主探索仍未验收。


## 使用

在具备 Docker、NVIDIA Container Toolkit 和 X11 桌面会话的本机执行：

```bash
./Simulation/sim doctor
./Simulation/sim build
./Simulation/sim run
```

`build` 继承本机已有的 `surfslam:stereo-v2` 镜像，复用其中的 Stonefish、ROS Jazzy 和 CUDA/PyTorch；构建 Pelagos 的 OKVIS2、现有仿真控制接口和深度采集时间戳补丁，并沿用本机 LAS2 运行环境的 `timm==1.0.30`（旧版不提供 `fasternet_t0`）。构建末尾检查 LAS2-S 模型创建，不下载骨干预训练权重。镜像构建可能需要通过 apt 安装缺失的 OKVIS2 编译依赖。默认直接启动 Porth 洞穴，以下命令等价：

```bash
./Simulation/sim run --scene porth
```

加入 Wavemap 后需再执行一次 `./Simulation/sim build`，增量构建复用此前 OKVIS2/LAS2 镜像层，只新增本机 `TP/Mapping/wavemap/library/cpp` 和现有 `underwater_wavemap_ros` ROS2 适配器的编译层，不构建规划器或额外 ROS1 镜像。

构建会直接显示编译进度。请使用 `./Simulation/sim build && ./Simulation/sim run`，在构建成功后再启动。`KeyboardInterrupt` / `context canceled` 表示构建被中断，旧镜像标签可能仍存在；此时 Wavemap 编译步骤需要重新执行，之前完成的镜像层可复用。`run` 会在创建运行目录前检查镜像内是否存在 Wavemap 可执行文件。`stop` 在容器不存在时正常返回。

默认资源位置：

- Stonefish 工作区：`/home/hong/Desktop/MY_Stonefish_ws/stonefish_slam_ws`
- Porth 资产：`/home/hong/.local/share/stonefish-slam/assets/porth_sump9_v1`
- LAS2-S 权重：`Deployment/artifacts/weights/LAS2_S.pth`
- 构建镜像：`pelagos-stonefish:local`

如本机路径或镜像名不同，分别用 `PELAGOS_STONEFISH_WORKSPACE`、`PELAGOS_PORTH_ASSETS`、`PELAGOS_SIM_BASE_IMAGE`、`PELAGOS_SIM_IMAGE` 环境变量覆盖。运行默认使用 ROS 域 42，可用 `--ros-domain` 修改。仿真与 ROS2 算法在一个容器内；自主模式另有原版 ROS1 NBV 容器，使用运行目录中的 Unix socket 通信。容器不接入外部网络。

## 可视化

启动后打开 Stonefish 和 RViz：

- **红色相机三角锥及轨迹**：Stonefish 相机位姿 GT。
- **绿色相机三角锥及轨迹**：OKVIS2 估计的左相机位姿。
- **Camera history**：在红/绿连续轨迹上分别按累计行进距离每 0.75 m 留存一个相机三角锥，不覆盖之前的位姿。它是按距离采样的显示历史，与 OKVIS2 自己选择的关键帧区分；可通过可视化节点参数 `camera_pose_spacing_m` 调整间距。
- **Full Porth cave - global scene GT**：使用现有完整 `collision.obj` 显示整个洞穴的半透明几何轮廓，替换原先半径约 8 个源单位的 `rviz_cutaway.obj` 局部裁剪。这层是完整场景 GT，仿真相机仍使用原始高分辨率纹理模型。
- **Wavemap estimated occupied voxels (orange)**：LAS2-S + OKVIS2 融合得到的占据地图，与半透明 Porth 原图叠加。显示完整累计地图，不做位置裁剪；多分辨率叶节点以实际边长显示。
- 默认 **Vehicle close-up** 查看局部规划；右侧 **Views** 可选 **Local top-down** 或 **Full cave**。采用 GBPlanner 洞穴版分组：PlannerViz / Mission / Mapping；白底蓝色。原先相机、特征和深度证据保留在对应组中，图像可按需启用。
- **LIVE depth comparison**：默认启用的单个图像面板，横向显示 GT、LAS2-S 和误差，并在图内显示 MAE、RMSE、有效像素比例、匹配输出帧率及采集时刻。没有匹配数据或超过 2 秒未更新时，显示等待状态与接收计数。
- **Depth GT** 和 **Depth LAS2-S**：同一采集时刻的光轴 Z 深度，色标均为 0.3–10 m，黑色表示无效像素。
- **Absolute depth error**：绝对误差，色标为 0–2 m；无重叠有效深度时该帧不显示对比。

两条轨迹使用第一对时间接近的 GT/估计相机位姿建立一次固定的 SE(3) 可视化坐标对齐，后续不再用真值修正估计；该对齐方式消除了初始坐标系差异，不是未对齐的绝对定位误差。OKVIS2 和 LAS2-S 的输入仅来自仿真双目和 IMU；GT 只用于可视化、误差计算和下述手动运动辅助。

### VIO 可核对证据

仿真标定会启用 OKVIS2 自带的 `display_matches` 和 `display_overhead`，直接复用其输出：

- **OKVIS2 native LEFT feature matches**：`/okvis/cam0_matches`，原生特征圈、路标 ID、观测点与投影点的连线、关键帧标志和跟踪质量提示。绿色为已初始化路标，黄色为未初始化路标，红色为未关联路标的特征。右目和原生图优化俯视图也在 Displays 中，可按需启用。
- **OKVIS2 native sparse landmarks**：`/okvis/okvis_points_matched`，原生筛选后的三维稀疏路标，不使用 LAS2 或 GT 点云。
- **VIO evidence**：原生图像/里程计/视觉优化输出频率、稀疏路标数量、固定初始对齐后的当前位置/角度误差和累计位置 RMSE。误差匹配使用相差不超过 50 ms 的 GT/估计帧，实际时间差保存在记录中；没有当前匹配时明确显示等待。

当前原版 ROS2 Publisher 的 debug 图像没有采集时间戳，路标点云也可能为零时间戳。因此 debug 更新率使用接收时间统计，不用它们冒充精确的图像时刻。位姿误差使用原生估计的有效采集时间戳。特征图来自 OKVIS2 的实际观测和路标，没有另起一个检测器生成示意点。

每次运行还自动保存以下文件：

| 文件 | 用途 |
| --- | --- |
| `vio-raw-body.jsonl` | 未经 GT 对齐的原生 `world→body` 里程计与机体速度 |
| `vio-optimized-body.jsonl` | 原生视觉优化回调的机体位姿 |
| `vio-alignment.json` | 启动时唯一的一次可视化 SE(3) 对齐 |
| `vio-pose-comparison.jsonl` | 原始/对齐后的相机位姿、GT、时间差及逐帧误差 |
| `vio-input-audit.json` | 运行中观察到的输入订阅、发布者；检查双目/IMU 链路及 GT 订阅情况 |
| `vio-status.json` | 最新输出频率、接收年龄、误差和输入检查结果 |

当前仿真使用精确标定、零镜头畸变和较小 IMU 噪声，属于理想化传感条件；轨迹视觉接近程度不能单独作为 VIO 有效性的证据。手动控制器使用 GT 做速度反馈，而 OKVIS2 估计节点只接收双目和 IMU。GT 的初始对齐仅位于可视化进程，原始估计同时保留，后续不会重新拟合轨迹。

### Wavemap 建图与原图叠加

复用项目现有的 ROS2 适配器和本机 Wavemap 2.2.1 C++ 库；默认手动模式不启动规划器，自主模式复用同一张 Wavemap。输入为 `/stereo/depth` 的 LAS2-S 光轴 Z 深度和 `/odometry/estimated` 的 OKVIS2 左相机位姿。地图在 OKVIS2 的 `world` 中融合；RViz 通过既有的一次初始 `pelagos_view → world` 变换叠到 Porth GT 上。GT 位姿、GT 深度和原始场景模型不输入融合器，后续漂移会表现为两层错位。

起步配置在 `Simulation/config/wavemap.yaml`，每次运行将实际相机内参、坐标名称和输出路径冻结到 `runs/<编号>/wavemap.yaml`：

| 参数 | 起步值 | 调整建议 |
| --- | --- | --- |
| 最小体素 `min_cell_width_m` | 0.10 m | 先观察墙面厚度、内存和耗时；细节需要增加时再用 0.05 m |
| 深度 Z 范围 | 0.30–8.0 m | 远处重影较多时先缩至 6 m；这是单帧传感范围，不限制车辆探索位置 |
| 深度步长 `pixel_stride` | 2 | 640×480 深度使用 320×240 采样，内参同步缩放；需要更密细节时改 1 |
| 融合 / 地图发布 | 10 Hz / 5 Hz | 前者控制深度融合频率，后者控制建图结果发布和可视化更新频率 |
| 深度不确定度 `range_sigma_m` | 0.10 m | 初始经验值，后续按现有同帧深度 GT 误差统计调整 |
| 自由 / 占据更新权重 | 0.20 / 0.40 | 起步较轻自由空间更新；弱纹理空洞保持未知，不补成自由空间 |
| 显示占据概率 | >0.70 | 地图未成熟时可试 0.65；增加此阈值只改变占据层显示与导出筛选，不改变内部证据 |
| 融合线程 | 4 | CPU 建图；不额外占用 LAS2-S 的 GPU 推理资源 |

参数含义和测量模型可参见 [Wavemap 官方参数说明](https://projects.asl.ethz.ch/wavemap/pages/parameters/index.html)。本表数值是针对当前仿真的初始建议，尚未作为调优结论。

融合器在采集时间两侧的相机位姿之间做位置线性插值和旋转 SLERP；不使用未来外推或把最新位姿强配给旧深度。缺少插值区间的深度最多等待 0.3 秒后丢弃；队列有界，保持实时性。无效、越量程和深度突变边缘像素不生成占据或自由空间更新。`prune` 仅压缩地图，不裁剪远处空间。

显示建议：默认保留蓝灰色 Porth 原图（透明度 0.12）和不透明橙色占据体素；未知与自由体积默认不绘制。可以分别关闭原图或估计层比较轮廓。需要看高程时启用 **Wavemap points - height or occupancy**，使用 Z 轴着色；需要看证据强弱时可在该层切换 Intensity，通道选择 `occupancy_log_odds`。此点云颜色通道表示占据证据，不是 LAS2 提供的置信度。定量比较可沿用同帧 GT 深度和 GT 位姿生成参考表面；全局显示中的完整 collision mesh 是简化几何轮廓。

```bash
./Simulation/sim mapping     # 融合帧数、实测 Hz、最近一帧积分耗时、内存和占据叶节点
./Simulation/sim map-save    # 保存 map.wvmp，并执行一次原生重载检查
```

正常关闭时也会保存原生地图；长时间运行建议先显式 `map-save` 再停止。原生地图包含已观测的自由/占据证据，RViz 默认仅展示占据部分。地图保存在本次运行目录的 `map.wvmp`，配置在 `wavemap.yaml`，状态在 `mapping-status.json`，进程日志在 `logs/wavemap.log`。

Stonefish 原生 DepthCamera 与左相机同位置、方向、分辨率和视场。补丁让深度图携带渲染采集时刻，只有与 LAS2 输入帧时间戳完全一致的深度才计算误差；不会用邻近帧冒充同帧。`/pelagos/depth/metrics` 发布有效像素数、MAE 和 RMSE，可在另一个终端查看：

```bash
./Simulation/sim metrics
```

原始估计和真值深度分别为 `/stereo/depth` 和 `/pelagos/depth/ground_truth`，单位为米。每次运行的配置和各进程日志保存在 `Simulation/runs/<运行编号>/`。

## 仿真运动

`run` 默认保持已通过的手动控制模式。自主模式用 `run --autonomy` 单独启动；普通 `build` 构建 ROS2 仿真镜像，`build --planner-only` 构建原版 NBV 镜像。

### 原版 NBVPlanner 与 JEPA-MPPI

首次使用自主模式，在桌面终端执行：

```bash
./Simulation/sim stop
./Simulation/sim build --planner-only
./Simulation/sim run --autonomy
```

如果此前 Wavemap 镜像尚未成功构建，先执行普通 `./Simulation/sim build`。NBV 构建复用本机 `stonefish-planner:omniplanner` 的 Noetic 编译环境及现有依赖源码，编译的是 `TP/Planner/nbvplanner` 的 ETHZ 算法；用 `PELAGOS_NBV_BASE_IMAGE`、`PELAGOS_NBV_IMAGE` 可覆盖基础镜像与输出镜像。构建时关闭上游测试，只做编译兼容适配。启动前检查 NBV 镜像是否对应当前源码，避免使用此前未完成的旧适配镜像。Docker 构建的完整输出同时显示在终端并保存至 `Simulation/build/nbv-build.log`，每次构建覆盖该文件；构建失败时不会继续启动仿真。

仿真运行后，在另一个终端操作：

```bash
./Simulation/sim planner    # 查看 NBV、JEPA 历史/求解状态、Guard 原因和导航误差
./Simulation/sim explore    # 连续巡航，约 3 m 遇障时选择转向并续航
./Simulation/sim pause      # 停止探索并撤销当前运动授权；地图保留
```

当前 `obstacle_triggered / turn_then_cruise` 配置从启动朝向连续巡航，前方实际 occupied footprint 净距约 3 m 时制动、选择转向，真实停稳后按新方向继续。旧源初始化扫描仅保留给 legacy 配置/历史复现，当前探索不等待该扫描，不向地图填造自由空间。

通信链路如下，所有 ROS2 模块使用现有 ROS 域：

```text
双目 + IMU → 原生 OKVIS2 body odometry → /rov01/state/odometry
LAS2-S depth + 采集时刻的 OKVIS2 相机位姿 → Wavemap（唯一深度融合）
Wavemap → 完整 OctoMap 格式投影（当前批准 0.1 m）→ 原版 NBV 查询地图 → NBV best edge
NBV 转向 → pure-turn NavigationGoal(index1) → 原有 navigation tracker
当前方向的动态 1 m 巡航目标 → NavigationGoal(real index) → ControlRequest
Guard → AuthorizedCommand → 原有 JEPA-MPPI → 六推进器终端 → Stonefish
实际推进器反馈 + IMU + 压力 + OKVIS2 → JEPA 检查点历史
Wavemap → RViz 与全局 Porth 叠加
```

NBV 的 ROS1 `/clock` 独立转发，规划计算期间持续更新；目标带有原有运行编号、估计 epoch、授权 token、generation、序列号及有效期。规划请求在后台执行，当前图未验证的运动目标不续发；制动/转向和首个滚动巡航目标等待期间续发明确 neutral，新方向通过当前图检查后再巡航；`pause` 撤销运动授权，状态失效也会停止续发。各边有独立 waypoint 编号，避免上一段的“到达”状态被下一段复用。采集深度匹配两侧估计位姿并插值，绝不使用 GT 位姿、GT 深度或 Porth 模型生成规划地图。控制直接使用原生机体里程计，保留 FLU 机体速度；不会把左相机位姿当成机体状态。原生 OKVIS Publisher 未输出协方差，状态明确记录 `covariance_available: false`，不另外填写虚构的不确定度。

参数以 [ETHZ 原仓库](https://github.com/ethz-asl/nbvplanner) 的固定版本 `5c07657af18f967d7049d187db0607024746a42c` 为准。`Simulation/config/nbv.yaml` 只选择上游配置和你已同意的几何适配；每次运行的 `autonomy-provenance.json` 保存来源、哈希和适配说明。

| 项目 | 实际来源 / 数值 |
| --- | --- |
| NBV 树、信息增益与迭代预算 | 原版 `interface_nbvp_rotors/resource/exploration.yaml`；当前方向筛选不改 gain/采样/15-200 |
| 规划地图分辨率 | 当前批准 0.1 m，与 Wavemap 最小网格一致；原版 `flat_exploration.launch` 的 0.4 m 为历史对照 |
| 建图范围 / 融合频率 | 只使用现有 Wavemap 0.3–8 m / 10 Hz；NBV 原 5 m 传感范围与 0.36 s 点云节流在外部地图模式不参与融合 |
| 相机 | 已批准的仿真标定：水平 75°、垂直约 59.84°、俯仰 0° |
| NBV 采样范围 | 根据完整 Porth 缩放及外参保守计算：X/Y ±115 m，Z −5 至 6 m，位于 OKVIS 起始坐标系 |
| 车辆位置限制 | 不设置 X/Y/Z 围栏；NBV 的采样范围不传入控制模块 |
| JEPA 采样数、代价、时延等 | 本机 `disaware_learning/config/jepa_mppi_feedback_search.json`；当前批准覆盖为 64 采样、deadline 0.36 s、最大计划年龄 0.60 s、延迟补偿 0.36 s；其余代价 / 增益保留，覆盖与依据冻结在每次运行的 provenance 中 |
| 导航跟踪、控制器适配参数 | 读取现有 Stonefish 工作区 `navigation/config/defaults.yaml`、`controller/config/jepa_mppi.yaml`；唯一批准的水平速度倍率为 0.12×4=0.48 m/s，源增益/停稳标准保留 |
| JEPA 历史频率 | 从检查点读取；当前模型 10 Hz。对 OKVIS 反馈插值到该网格，同时处理全部真实传感器和推进器反馈 |

源仓库的 RRT、增益、迭代终止及无增益时退出行为保留。原生 RRT 首边保留已知自由空间检查；连续巡航允许 unknown、拒绝已知 occupied，运动续发仍要求当前图验证。Guard 和终端继续使用原有状态、姿态、速度、授权及命令超时检查。手柄 5 倍速度设置属于手动模式；自主水平 cap 0.48 m/s 是另经用户批准的源值 4 倍，增益和其他阈值保留。

Wavemap 保持已确认的融合 10 Hz / 发布 5 Hz。默认 `config/nbv.yaml` 的 `map_source: wavemap`：Wavemap 发布 `/mapping/octomap` 完整快照，`wavemap_nbv_relay.py` 将其送入原版 `setOctomapFromMsg`，整棵替换 NBV 查询地图。NBV 深度订阅与原生点云/视差融合入口关闭；内存中的 OctoMap 只是格式转换后的查询副本，不再次积分。只有显式选择旧的 `native_depth` 才恢复双建图，用于历史实验复现。

当前投影分辨率为 0.1 m，转换保留 free / occupied / unknown 三态；占据优先，自由必须完全已观测覆盖，混合自由与未知保持未知。分类沿用 Wavemap 自带 `OccupancyClassifier` 的零 log-odds 阈值和 `1e-3` 未观测判据，输出完整 `.ot`/Octomap 概率数据；橙色显示仍沿用 >0.70 的显示筛选，所以蓝色查询占据层可能比橙色更厚。车体自身 unknown 的新增例外仅影响碰撞查询，原地图、增益与可视化证据不改。这是地图格式/分类投影，不能将橙色 occupied 点云直接重打射线作为地图。早期 0.4 m 转换验收来源见 [Wavemap 适配记录](nbv/WAVEMAP_ADAPTATION.md)，当前验证范围见本文开头。

`map-source-status.json` 记录已安装的来源时间戳、NBV 版本、导入耗时和 `depth_fusion_enabled: false`；`mapping-status.json` 记录投影自由/占据/混合未知格数及耗时。RViz 的蓝色地图是该投影的 NBV 查询结果，原来的红绿轨迹、相机历史、特征点、实时深度对比和全局 Porth 保留。

运行文件包括 `nbv.yaml`、`jepa-settings.yaml`、`jepa-algorithm-source.json`、`autonomy-provenance.json`、`planner-status.json`；日志分别为 `logs/nbv_native.log`、`nbv_transport.log`、`nbv_roscore.log`、`nbv_bridge.log`、`body_state.log`、`navigation.log`、`controller.log` 和 `guard.log`。上游原生日志写入 `nbv-data/`。原版规划器退出时会生成 `nbv-backend-status.json`；重新启动仿真可建立新的估计与地图会话。

### 规划预览与历史证据

```bash
./Simulation/sim run --planning-only
./Simulation/sim plan-preview  # 只调用原版规划，不运动
./Simulation/sim rviz          # 为仍在运行的会话打开原生 RViz 工具栏
```

纯规划运行不启动 Guard、控制器、导航跟踪器或推进器适配器。使用原生工具栏 **Plan Preview** 请求预览、**Pause** 取消任务；该会话不接受运动开始或接管。`./Simulation/sim ui` 只是打开同一 RViz 的兼容别名，没有单独编辑器。工具接口与保留的 GBPlanner 布局来源见 [接口来源](rviz_panel/PROVENANCE.md)。

PlannerViz 中 **RRT 树**、**候选视点** 来自本次原版真实节点；**地图与采样窗** 显示观测范围和规划采样范围，不约束车辆；**起点检查** 可开关显示原生 footprint / 扩展检查，绿为 free、橙为 unknown、红为 occupied。**NBV OctoMap** 蓝色体素为当前 0.1 m 查询地图，Wavemap 橙色体素显示同一融合地图的占据层。`./Simulation/sim planner` 与运行目录中的 `planner-preview.json` / `planner-diagnostics.json` 提供原生起点、源树和地图版本证据；显示中的候选视点不表示该 gaze 已执行。

2026-10-30 历史实验先在同一真实地图的虚拟组件起点生成 49 点原版路径；随后用户手动建图，在真实车辆起点使用原有全局采样范围成功生成 50 点 / 0.98 m 的原版 best edge，footprint 无未知、23/26 扩展方向自由，原生路径检查通过。当时轨迹只显示保存。2026-10-31 后续已验证短时 JEPA 跟踪与重复换段，持续探索仍待验收；精确范围见本文开头及 [实验接管说明](EXPERIMENT_HANDOVER.md)。

在 `--autonomy` 运行中，可用原生工具栏 **Manual / Auto** 或 `sim manual` / `sim autonomous` 切换。切到手动先暂停并撤销旧授权，确认后摇杆回中、松开 LB 再重新按住 LB；回自主只选择 JEPA，不自动开始任务。一个控制/输出通道按确认模式工作；纯规划运行不支持运动接管。

### 原生工具栏与 CLI 路径点

自主会话使用 `./Simulation/sim run --autonomy`；需要单独打开当前运行的 RViz 时用 `./Simulation/sim rviz`。`./Simulation/sim ui` 是同一入口的兼容别名，不启动网页、Qt 路线编辑器或 MissionControl panel。只有五个原生任务工具：

- **Explore**：确认 Auto 所有权后发起一次探索任务，按约 3 m 遇障转向/持续巡航。
- **Plan Preview**：暂停状态下请求当前真实树的转向候选预览，不 ARM 或运动。
- **Manual**：先暂停自主任务并切换控制所有权，不自动使手柄运动。
- **Auto**：选择自主所有权，不自动开始任务。
- **Pause**：取消等待中的开始操作并撤销运动授权，保留地图。

任务状态以 `./Simulation/sim planner` 及 `control-status.json` 为准。正常关闭 RViz 沿用 supervisor 的整条 pipeline 退出流程；Pause 是保留查看器和建图的任务暂停。`2D Goal Pose` 已随旧路线编辑器移除，没有对应的编辑器 consumer。

路径点核心能力保留为 CLI JSON routes。先通过 `./Simulation/sim planner` 取得本次 `run_id`，把本次 `rov01/odom` 中的目标位置写入本机 JSON 文件；每个 pose 为 `[x,y,z,qx,qy,qz,qw]`，只指定水平 yaw。例如：

```json
{
  "run_id": "<本次运行编号>",
  "frame_id": "rov01/odom",
  "epoch": 1,
  "poses": [[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]
}
```

用实际目标 XYZ 和本次运行编号替换示例；yaw 的四元数为 `[0,0,sin(yaw/2),cos(yaw/2)]`。在健康、无 Guard fault 的自主会话中执行：

```bash
./Simulation/sim pause
./Simulation/sim autonomous
./Simulation/sim navigate --route /本机路径/waypoints.json
./Simulation/sim planner
```

`navigate` 从本机路径读取 JSON，经同一任务协调器、run/epoch 校验和当前地图检查执行；接受后保存为该轮 `mission-route.json`。路径点是局部已知自由空间中的逐点导航，直连穿墙或进入 unknown 时停止并报告，不是任意目标的全局绕障器。GT 不输入路径点控制；不同 run 的估计原点不能混用。暂停使用 `./Simulation/sim pause`。

控制边界已改为源 JEPA 的归一化横滚 PD 与六电机 mixer，参数由 `BodyTwistPidConfig` 直接读取；不使用通用力矩环的横滚增益。模型/原生电机的统一方向系数从实际分配矩阵推导，本机几何检查结果为 +1，保留现有方向。MPPI 源配置为 256 个采样，随后按会话授权复测 128 / 64；当前仿真批准 64 采样及 0.36 / 0.60 / 0.36 s 三项时限，Guard / 终端阈值保留。首次以真实历史执行一次不下发指令的 GPU 预热。压力深度密度与生成场景使用同一份六推进器配置。

历史：2026-10-30 实际复现 256 / 128 采样两轮，当时初始化扫描因求解超时停在索引 1，尚未通过导航和 NBV 连续换段；详细结果见 [本机实验记录](EXPERIMENT_REPORT_20261030.md)。这不代表当前时限或最新闭环进展，后续结果见本文开头。运行证据包括 `controller-evidence.jsonl`、`solver-evidence.jsonl`、`control-fault-evidence.jsonl`、`mission-events.jsonl` 和 `mission-route.json`。

若当前聊天执行环境无法访问 Docker，可在有 Docker 权限的桌面终端运行 `bash Simulation/control-experiment`。它使用已核对的本机 Codex CLI 选项启动一个新的宿主机权限会话并保留按需审批，自动读取 [实验接管说明](EXPERIMENT_HANDOVER.md)；它不修改当前聊天权限或全局配置。该会话用于实际复现、记录与验收，仍遵守用户的参数约束。

### 手柄控制

已运行的仿真可以直接接入本机手柄，不需要重建镜像，也不需要为容器挂载 USB：

```bash
./Simulation/sim joystick
```

宿主机通过 Linux joystick 接口读取 `/dev/input/js0`，用持续的 `docker exec -i` 将输入送入当前 ROS 域，再复用 Guard → 现有速度控制器 → 六推进器终端。当前手动模式的速度控制反馈仍为仿真 GT，OKVIS2/LAS2-S 继续运行用于比较。读取轴和按键的 Linux 编码，自动兼容北通的 `ABS_Z/RZ` 右摇杆及常见 Xbox 的 `ABS_RX/RY` 右摇杆布局。

| 手柄操作 | 车辆响应 |
| --- | --- |
| 左摇杆上下 | 前进 / 后退，最大 0.75 m/s |
| 左摇杆左右 | 向左 / 向右横移，最大 0.75 m/s |
| 右摇杆上下 | 上浮 / 下潜，最大 0.50 m/s |
| 右摇杆左右 | 向左 / 向右转向，最大 1.25 rad/s |
| 按住 LB | 使能；先让摇杆回中，再按住 LB，然后推摇杆 |
| 松开 LB、按 B、Ctrl+C 或输入断开 | 停止发送速度请求，撤销本手柄的运动授权 |
| LB 松开、摇杆回中时按 Start | 请求清除 Guard 故障，再按住 LB 重新使能 |

默认死区 12%。松开 LB 会停止推进器输出，车辆仍按 Stonefish 的浮力和惯性运动。手柄输入会在 250 ms 内过期；不会把排队的旧输入续期成新的命令。沿用当前 `cli` 授权来源，因此能接入已启动的仿真；每次运行只允许一个手柄接收进程。

上述目标速度上限为原来的 5 倍，手柄输入验证和 Guard 速度请求上限同步调整；模型质量、阻力、推力、推进器限幅和仿真时钟不变，实际速度由物理模型和控制器决定。

```bash
./Simulation/sim joystick --list
./Simulation/sim joystick --inspect       # 仅观察摇杆/按键，无运动指令
./Simulation/sim joystick --device /dev/input/js1
```

终端每秒显示连接、使能、当前 GT 位置/姿态、Guard 越界值及实际推进器输出；记录在当前运行的 `logs/joystick.log` 和 `joystick-status.json`。

**手动仿真不设置 X/Y/Z 位置边界**：生成的 `resolved_config.yaml` 明确记录 `manual_position_limits: disabled`，`guard-parameters.yaml` 的位置上下限使用 YAML 的正负无穷，复用原 Guard 时不再人为限制可探索区域。Stonefish 的实体碰撞保持有效。姿态 25°、实际线速度峰值 1.2 m/s、实际角速度峰值 2.5 rad/s 检查，以及手柄速度上限、松键撤权和命令超时停止仍有效；角速度阈值已匹配加速后的 1.25 rad/s 转向目标。

旧运行曾继承 X/Y 各 ±8 m 的位置框，越界后进入 FAULT 并使六推进器归零。Guard 在启动时读取配置，因此需重启仿真加载本次修正；修改旧运行目录里的 YAML 不会更新已运行的 Guard。`ROS_DOMAIN_ID=42` 是 ROS 通信分组，与空间位置无关。

仿真初始为 DISARMED；需要观察轨迹时，可在另一个终端运行有限速度命令：

```bash
./Simulation/sim status
./Simulation/sim move --velocity 0.05 0 0 0 --seconds 5
./Simulation/sim stop
```

速度依次为机体 FLU 的前、左、上速度（m/s）和偏航角速度（rad/s）。`move` 复用现有 Stonefish 的 Guard、body PI 控制器、授权和推进器终端，超时后自动撤权。该手动运动辅助使用仿真真值作为控制反馈，**不是** Pelagos 自主导航或 OKVIS2 闭环控制；它不改变 OKVIS2 的输入和评估。

## 说明

仿真相机使用现有模型的 640×480、0.145 m 基线及对应外参；启动时自动生成仿真专用 OKVIS2 和 LAS2 标定，不使用真机 ZED 的标定。图像同步、IMU 坐标转换和六推进器模型复用现有 Stonefish 工作区实现。

构建前需要确认本机 `surfslam:stereo-v2` 镜像与对应的 Stonefish 工作区确实可用。若启动失败，先查看 `sim doctor` 输出和运行目录里的进程日志。本文目录中的旧方案文件记录了最初的 Humble/Jazzy 分离设想；当前实现选择让算法在 Jazzy 仿真容器内运行，因此不需要跨 ROS 版本的数据桥。
