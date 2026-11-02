#!/usr/bin/env python3
import sys
from pathlib import Path as TaskLocation
for _role in ('core', 'adapters', 'services'):
    sys.path.insert(0, str(TaskLocation(__file__).resolve().parents[1] / _role))
"""Prepare an isolated build of the upstream goal-navigation core."""
import ast
import json
import math
import operator
from pathlib import Path
import shutil
import subprocess

import yaml

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent.parent
SOURCE = ROOT / 'TP/Planner/OmniPlanner'
CONTEXT = ROOT / 'build/omniplanner/build-context'
DEPENDENCIES = ROOT / 'TP/Planner/OmniPlanner-deps'


def radians(value):
    if isinstance(value, dict):
        return {key: radians(item) for key, item in value.items()}
    if isinstance(value, list):
        return [radians(item) for item in value]
    if not isinstance(value, str) or not value.startswith('rad('):
        return value
    operations = {ast.Add: operator.add, ast.Sub: operator.sub,
                  ast.Mult: operator.mul, ast.Div: operator.truediv}

    def number(node):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.Name) and node.id == 'pi':
            return math.pi
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -number(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in operations:
            return operations[type(node.op)](number(node.left), number(node.right))
        raise ValueError('Unsupported angular expression')

    return float(number(ast.parse(value[4:-1], mode='eval').body))


def main():
    DEPENDENCIES.mkdir(exist_ok=True)
    for name, temporary in [('catkin_simple', '/tmp/omniplanner-catkin-simple'),
                            ('adaptive_obb_ros', '/tmp/omniplanner-adaptive-obb')]:
        if not (DEPENDENCIES / name).exists():
            shutil.move(temporary, DEPENDENCIES / name)
    if CONTEXT.exists():
        shutil.rmtree(CONTEXT)
    src = CONTEXT / 'src'
    src.mkdir(parents=True)
    for name in ['kdtree', 'planner_msgs', 'planner_semantic_msgs',
                 'planner_common', 'map_manager', 'gbplanner']:
        shutil.copytree(SOURCE / name, src / name)
    shutil.copytree(DEPENDENCIES / 'catkin_simple', src / 'catkin_simple', symlinks=True,
                    ignore=shutil.ignore_patterns('.git', 'test', 'tests'))
    shutil.copytree(DEPENDENCIES / 'adaptive_obb_ros/adaptive_obb', src / 'adaptive_obb')
    manager = src / 'map_manager'
    header = (manager / 'include/map_manager/map_manager.h').read_text()
    header = header.replace('#include "map_manager/map_manager_voxblox_impl.h"',
                            '#include <ros/ros.h>\n#include <std_srvs/Empty.h>\n#include <pcl/point_cloud.h>\n'
                            '#include <pcl/point_types.h>\n#include "esdf_grid.hpp"\n'
                            '#include "dapeng_planning_msgs/EsdfGrid.h"')
    start = header.index(' private:')
    header = header[:start] + '''  uint32_t lastSequence() const { return last_map_sequence_; }
 private:
  void gridCallback(const dapeng_planning_msgs::EsdfGrid::ConstPtr& message);
  ros::NodeHandle nh_, nh_private_;
  ros::Subscriber grid_subscriber_;
  ros::Time last_map_stamp_;
  uint32_t last_map_sequence_ = 0;
  dapeng::EsdfGrid grid_;
  double map_timeout_ = 5.0;
  double robot_radius_ = 0.35;
};
#endif
'''
    (manager / 'include/map_manager/map_manager.h').write_text(header)
    shutil.copy2(HERE / 'adapters/map_export/include/esdf_grid.hpp', manager / 'include/esdf_grid.hpp')
    shutil.copy2(HERE / 'adapters/map_export/src/map_manager.cpp', manager / 'src/map_manager.cpp')
    package = (manager / 'package.xml').read_text().replace('<depend>voxblox_ros</depend>', '<depend>dapeng_planning_msgs</depend>')
    (manager / 'package.xml').write_text(package)
    (manager / 'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.0.2)
project(map_manager)
add_compile_options(-std=c++17 -Wall)
find_package(catkin_simple REQUIRED)
find_package(Eigen3 REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})
cs_add_library(map_manager src/map_manager.cpp)
cs_install()
cs_export(LIBRARIES)
''')
    package = src / 'dapeng_planning_msgs'
    (package / 'msg').mkdir(parents=True)
    (package / 'srv').mkdir()
    shutil.copy2(HERE / 'interfaces/msg/EsdfGrid.msg', package / 'msg/EsdfGrid.msg')
    shutil.copy2(HERE / 'interfaces/srv/Plan.srv', package / 'srv/Plan.srv')
    (package / 'package.xml').write_text('''<package format="2"><name>dapeng_planning_msgs</name><version>0.1.0</version>
<description>Bounded ESDF snapshots for the Dapeng planner adapter</description>
<maintainer email="local@example.invalid">Dapeng deployment</maintainer><license>Apache-2.0</license>
<buildtool_depend>catkin</buildtool_depend><build_depend>message_generation</build_depend>
<depend>std_msgs</depend><depend>geometry_msgs</depend><exec_depend>message_runtime</exec_depend></package>
''')
    (package / 'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.0.2)
project(dapeng_planning_msgs)
find_package(catkin REQUIRED COMPONENTS message_generation std_msgs geometry_msgs)
add_message_files(FILES EsdfGrid.msg)
add_service_files(FILES Plan.srv)
generate_messages(DEPENDENCIES std_msgs geometry_msgs)
catkin_package(CATKIN_DEPENDS message_runtime std_msgs geometry_msgs)
''')
    planner = src / 'gbplanner'
    text = (planner / 'package.xml').read_text().replace('<depend>behaviortree_cpp</depend>', '')
    (planner / 'package.xml').write_text(text)
    shutil.copy2(HERE / 'adapters/map_export/src/planner_node.cpp', planner / 'src/dapeng_planner_node.cpp')
    (planner / 'CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 3.0.2)
project(gbplanner)
add_compile_options(-std=c++17 -Wall -Wno-sign-compare -Wno-switch -Wno-maybe-uninitialized)
find_package(catkin_simple REQUIRED)
find_package(Boost REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(adaptive_obb REQUIRED)
catkin_simple(ALL_DEPS_REQUIRED)
include_directories(${Boost_INCLUDE_DIRS} ${EIGEN3_INCLUDE_DIRS})
cs_add_library(gbplanner src/gbplanner.cpp src/rrg.cpp src/gbplanner_rviz.cpp)
target_link_libraries(gbplanner glog gflags)
cs_add_executable(dapeng_omniplanner_node src/dapeng_planner_node.cpp)
target_link_libraries(dapeng_omniplanner_node gbplanner ${planner_common_LIBRARIES} glog gflags)
cs_install()
cs_export(LIBRARIES)
''')
    shutil.copy2(HERE / 'config/planner.yaml', CONTEXT / 'planner.yaml')
    (CONTEXT / 'checks').mkdir()
    shutil.copy2(HERE / 'adapters/map_export/include/esdf_grid.hpp', CONTEXT / 'checks/esdf_grid.hpp')
    shutil.copy2(HERE / 'adapters/map_export/tests/test_esdf_grid.cpp', CONTEXT / 'checks/test_esdf_grid.cpp')
    # Version provenance is shipped explicitly; archives need no .git directory.
    shutil.copy2(HERE / 'provenance/source-lock.json', CONTEXT / 'source-lock.json')
    shutil.copy2(HERE / 'docker/Dockerfile', CONTEXT / 'Dockerfile')
    print(CONTEXT)


if __name__ == '__main__':
    main()
