# Native RViz mission tools

The current interface uses five `rviz_common::Tool` plugins in RViz's standard toolbar: Explore, Plan Preview, Manual, Auto and Pause. It provides no custom Panel or standalone mission window. The installed ROS Jazzy Tool/ToolManager APIs and pluginlib provide the integration; the command flow is project code.

Every action uses the existing `/pelagos/mission/request` protocol. Explore waits for the selection response and fresh actual autonomous ownership before sending one start request. Pause and Manual cancel a queued start. Plan Preview only requests `plan_preview`; Auto selects ownership without starting motion. All original bridge, authorization and controller checks remain authoritative.

The English white-background/blue-map configuration retains standard Displays, Views, Interact, Move Camera, Select, Focus Camera and Measure tools. The old Set Goal/draft controls were removed with their standalone editor; CLI routes remain available. UI locale is English. RViz exit handling is unchanged.

Machine-readable source hashes, installed RViz versions, protocol checks, build identity and scope are recorded in `Simulation/build/rviz-native-tools/ui-native-tools-provenance.json`.

The earlier layout was adapted from GBPlanner cave_exploration/ui.rviz at commit `f9904bb69392d2d46b2cbe939762f967a65daa14`. Its Axes/Planner/Mapping groups and cave views remain historical layout provenance, and its BSD license remains in `GBPLANNER_LICENSE`. No GBPlanner controller or global planner is introduced by these tools.
