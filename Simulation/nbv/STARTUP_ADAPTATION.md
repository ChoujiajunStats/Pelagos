# 起步适配与控制时序接口来源

## 2026-10-31 最新授权（覆盖下面历史起步条件）

用户明确要求“未知的时候直接往前走，机器人周围已知就规划”，并指定约 5 m。当前 `startup_unknown_policy: forward_until_observed`：沿任务开始时的估计航向设置一个约 5 m 的连续前移目标，保留 0.5 m 前方检查；不在中间每米停稳。起步阶段允许地图 unknown，所有已知 occupied 叶节点仍与车辆扫掠盒做精确线段/AABB 检查；不向地图写 free。到达 5 m 目标仍未观测到周围空间时停止；不会无限延长起步。之后 RRT 的 1 m 扩展不改。

每次地图更新检查实际机体周围同一源采样网格；未知与占据均为零时，保持当前位置并切入原 NBV RRT，不伪造源旋转扫描完成。起步路径明确标 `startup_forward` / `forward_start_not_rrt`；原 RRT 的严格已知自由查询、增益、扩展距离和迭代条件保留。`calibrated_initial_blind` 和 `initial_self_only` 只保留为历史策略。

同日用户已明确批准：NBV 查询分辨率 0.4→0.1 m；JEPA deadline .16→.36 s、maximum plan age .20→.40 s、latency compensation .16→.36 s，其余参数保留。Stonefish `controller/uw_controller/checkpoint.py` 自带执行侧 .20 s 模型计划年龄上限；`Simulation/adapters/jepa_timing.py` 仅在明确配置覆盖时将同一批准年龄应用到执行侧，Guard / 终端命令授权期限不改。出处哈希在 `build/wavemap-nbv/jepa-timing-provenance.json`。

源接口调整：`Step.include_map` 只在纯预览请求返回整图蓝色体素，执行请求返回实际路径/诊断而不反复复制整图，避免大地图序列化阻塞 50 Hz 目标租约。地图融合、碰撞查询及规划结果不受此显示字段影响。

用户于 2026-10-30 澄清：问题是起步周围地图未知，并明确要求取消这一整圈自由条件、先向前；随后要求重置复测。原 NBV RRT、增益、采样窗与 15/200 条件保持锁定源。

## 向前起步段

- `native.cpp` 新增 `plan_forward_start` / `validate_forward_start`。它是前移起步段，不标作 RRT 结果。
- 使用源 extension 1 m、overshoot 0.5 m、robot box 0.5×0.5×0.3 m、OctoMap 0.4 m、原版 `RrtTree::samplePath`；没有新增数值调参。
- 一次固定在请求时刻：取初始车体包围盒相交的原生体素，允许其中 unknown；不允许 occupied，不更新地图或 setFree，不随着车辆移动扩大范围。体素取整与物理车体体积略有差别，固定边界单元也在豁免集合内。
- 集合之外的每条 footprint 扫掠射线及终点仍须原生 map 查询为 free；前方扩展+overshoot 在请求时检查，地图更新后持续验证目标段。
- 固定初始真实快照测试：成功返回 50 点；后方未知被拒绝；严格原版 validate 仍拒绝起点未知；把车辆状态后移时豁免不跟随。首帧无 occupied leaves，未以这张首帧假称完成实际 occupied 负例。
- 初始快照证据 `runs/20261030T122137Z-41ea43b3/forward-start-component-check.json`。
- `sim start-preview` 仅显示，`sim explore-forward` 执行这一起步方法再转原 NBV。原 `sim explore` 与原 UI 扫描动作仍保留。

## 控制积分时间

- 来源为本机 Stonefish `controller/uw_controller/node.py::Controller.tick`，其 TinyMPC 分支已有显式仿真时钟积分。修正位于 `adapters/clock_control_tick.py` 和 `SelectableController`，上游文件不改。
- 新鲜但短时保持同一时间戳的 OKVIS 样本可在当前仿真 tick 重新计算控制，保存真实 feedback stamp、反馈年龄、held_feedback 和单独 integration stamp；不伪造新反馈，不复用过期授权。
- 源反馈 250 ms 新鲜度、原始请求期限、所有增益和终端 watchdog 保留；四项回归验证覆盖保持反馈输出、保留新授权、过期/陈旧停车和反向积分拒绝。
- 源文件 SHA 与授权/参考事实在 `build/forward-start-provenance.json`。

## 实际执行结论

`20261030T123131Z-21d420df`：初始前移预览时起点仍有 12 处 unknown，起步段 50 点通过前方检查；真实执行进入 JEPA 跟踪但未走完 1 m。预热后 317 次求解中位 0.2021 s、P95 0.2417 s，0.16 s 期限下大量 discarded/expired。不得宣称已完成自主避障闭环。

已提出（未获明确批准）deadline 0.26、max plan age 0.32、latency compensation 0.26 的配套试验。用户回答是新建地图可用，未明确同意这些数值，因此这三个参数保持原值。随后使用用户地图做新的原版规划预览，证据 `planner-preview-after-user-map.json`。
