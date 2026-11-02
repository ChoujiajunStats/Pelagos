"""Apply transport/build compatibility changes to a disposable build copy only."""
from pathlib import Path
import argparse
import re

parser=argparse.ArgumentParser()
parser.add_argument('--source-root',type=Path,default=Path('/opt/pelagos_nbv_ws/src'))
root=parser.parse_args().source_root
header = root / 'nbvplanner/nbvplanner/include/nbvplanner/nbvp.h'
text = header.read_text()
assert 'bool setParams();' in text
header.write_text(text.replace('bool setParams();',
    'volumetric_mapping::OctomapManager* pelagosMap() { return manager_; }\n'
    '  const Params& pelagosParams() const { return params_; }\n'
    '  const char* pelagosTermination_ = "not_called";\n'
    '  bool pelagosTreeInitialized_ = false;\n'
    '  bool pelagosForwardSelectionRan_ = false;\n'
    '  void pelagosDisableMapInputs() { pointcloud_sub_.shutdown(); '
    'pointcloud_sub_cam_up_.shutdown(); pointcloud_sub_cam_down_.shutdown(); '
    'manager_->pelagosDisableMapInputs(); }\n  bool setParams();'))
# Disconnect every native sensor/map subscription in external-map mode. The
# original source tree is untouched; this method exists only in the build copy.
manager_header=root/'nbvplanner/volumetric_mapping/octomap_world/include/octomap_world/octomap_manager.h'
text=manager_header.read_text()
assert 'void publishAll();' in text
manager_header.write_text(text.replace('void publishAll();',
    'void pelagosDisableMapInputs() { left_info_sub_.shutdown(); right_info_sub_.shutdown(); '
    'disparity_sub_.shutdown(); pointcloud_sub_.shutdown(); octomap_sub_.shutdown(); }\n  void publishAll();'))
# Query-only exception for unknown voxels intersected by the current physical
# robot footprint. Occupancy, point queries and gain/visibility stay unchanged.
world_header=root/'nbvplanner/volumetric_mapping/octomap_world/include/octomap_world/octomap_world.h'
text=world_header.read_text();marker='  bool getOctomapBinaryMsg(octomap_msgs::Octomap* msg) const;'
assert marker in text
text=text.replace(marker,
    '  bool pelagosSelfUnknownEnabled_=false;\n'
    '  octomap::OcTreeKey pelagosSelfLow_,pelagosSelfHigh_;\n'
    '  void pelagosSetSelfUnknown(const Eigen::Vector3d& center,const Eigen::Vector3d& box,bool enabled) { '
    'Eigen::Vector3d low=center-box/2.,high=center+box/2.;pelagosSelfUnknownEnabled_=enabled && '
    'octree_->coordToKeyChecked(octomap::point3d(low.x(),low.y(),low.z()),pelagosSelfLow_) && '
    'octree_->coordToKeyChecked(octomap::point3d(high.x(),high.y(),high.z()),pelagosSelfHigh_); }\n'
    '  bool pelagosIsSelfUnknownKey(const octomap::OcTreeKey& key) const { if(!pelagosSelfUnknownEnabled_)return false; '
    'for(int i=0;i<3;++i)if(key[i]<pelagosSelfLow_[i]||pelagosSelfHigh_[i]<key[i])return false;return true; }\n'+marker)
world_header.write_text(text)
world_source=root/'nbvplanner/volumetric_mapping/octomap_world/src/octomap_world.cc'
text=world_source.read_text();start=text.index('OctomapWorld::CellStatus OctomapWorld::getLineStatus(')
end=text.index('OctomapWorld::CellStatus OctomapWorld::getVisibility(',start)
section=text[start:end];old='    if (node == NULL) {\n      return CellStatus::kUnknown;'
assert old in section
section=section.replace(old,'    if (node == NULL) {\n      if(pelagosIsSelfUnknownKey(key))continue;\n      return CellStatus::kUnknown;')
world_source.write_text(text[:start]+section+text[end:])
# A new operator task needs a fresh execution root, while retaining the measured
# OctoMap. Only clear previous-task memory; tree expansion/gain stays upstream.
rrt_header=root/'nbvplanner/nbvplanner/include/nbvplanner/rrt.h'
text=rrt_header.read_text()
text=text.replace('#include <nbvplanner/tree.h>','#include <nbvplanner/tree.h>\n#include <nbvplanner/pelagos_forward_selection.hpp>')
text=text.replace('virtual void initialize();',
    'const Node<StateVec>* pelagosRoot() const { return rootNode_; }\n  '
    'Eigen::Vector3d pelagosForwardDirection_ = Eigen::Vector3d::Zero();\n  '
    'std::function<bool(const Eigen::Vector3d&)> pelagosTurnDirectionAllowed_;\n  '
    'int pelagosForwardEligible_ = 0;\n  '
    'bool pelagosForwardConstraintActive() const { return pelagosForwardDirection_.squaredNorm()>0. || bool(pelagosTurnDirectionAllowed_); }\n  '
    'void pelagosSetForwardDirection(const Eigen::Vector3d& value) { pelagosForwardDirection_=value; }\n  '
    'bool pelagosSelectForward() { auto* selected=pelagos::selectForwardNode(rootNode_,pelagosForwardDirection_,params_.zero_gain_,&pelagosForwardEligible_,pelagosTurnDirectionAllowed_); '
    'if(!selected)return false; bestNode_=selected;bestGain_=selected->gain_;return true; }\n  '
    'void pelagosResetExecution() { bestBranchMemory_.clear(); '
    'while (!history_.empty()) history_.pop(); iterationCount_=0; }\n  virtual void initialize();')
rrt_header.write_text(text)
forward=Path('/tmp/pelagos-forward-selection.hpp')
if not forward.is_file():forward=Path(__file__).with_name('forward_selection.hpp')
(rrt_header.parent/'pelagos_forward_selection.hpp').write_bytes(forward.read_bytes())
callback=root/'nbvplanner/nbvplanner/include/nbvplanner/nbvp.hpp'
text=callback.read_text()
start=text.index('bool nbvInspection::nbvPlanner<stateVec>::plannerCallback(')
end=text.index('\ntemplate<typename stateVec>',start)
section=text[start:end]
# Diagnostic assignments only: retain upstream conditions, budgets, fallback,
# shutdown and return sites, and the previously installed forward selection.
instrumentation=(
    ('  ros::Time computationTime = ros::Time::now();',
     '  pelagosTermination_="callback_started";pelagosTreeInitialized_=false;pelagosForwardSelectionRan_=false;\n'
     '  static_cast<nbvInspection::RrtTree*>(tree_)->pelagosForwardEligible_=0;\n'
     '  ros::Time computationTime = ros::Time::now();'),
    ('  if (!ros::ok()) {','  if (!ros::ok()) {\n    pelagosTermination_="ros_not_ok";'),
    ('  if (!ready_) {','  if (!ready_) {\n    pelagosTermination_="planner_not_ready";'),
    ('  if (manager_ == NULL) {','  if (manager_ == NULL) {\n    pelagosTermination_="map_unavailable";'),
    ('  if (manager_->getMapSize().norm() <= 0.0) {',
     '  if (manager_->getMapSize().norm() <= 0.0) {\n    pelagosTermination_="map_empty";'),
    ('  tree_->initialize();','  tree_->initialize();\n  pelagosTreeInitialized_=true;'),
    ('    if (tree_->getCounter() > params_.cuttoffIterations_) {',
     '    if (tree_->getCounter() > params_.cuttoffIterations_) {\n      pelagosTermination_="no_gain_cutoff";'),
    ('    if (loopCount > 1000 * (tree_->getCounter() + 1)) {',
     '    if (loopCount > 1000 * (tree_->getCounter() + 1)) {\n      pelagosTermination_="maximum_failed_iterations";'),
)
for old,new in instrumentation:
    assert section.count(old)==1,old
    section=section.replace(old,new)
marker='  // Extract the best edge.\n'
assert section.count(marker)==1
section=section.replace(marker,
    '  auto* constrained=static_cast<nbvInspection::RrtTree*>(tree_);\n'
    '  pelagosForwardSelectionRan_=constrained->pelagosForwardConstraintActive();\n'
    '  if(pelagosForwardSelectionRan_)pelagosTermination_="forward_selection";\n'
    '  if(constrained->pelagosForwardConstraintActive() && !constrained->pelagosSelectForward()) { '
    'pelagosTermination_="no_eligible_forward_node";res.path.clear();return true; }\n'
    '  pelagosTermination_="normal_extraction";\n'+marker)
callback.write_text(text[:start]+section+text[end:])
# Keep planner budgets, shutdown-on-exhaustion and gain/tree logic upstream.
# Redirect its existing logs into the writable run directory.
rrt=root/'nbvplanner/nbvplanner/src/rrt.cpp'
text='#include <cstdlib>\n'+rrt.read_text()
text=text.replace('ros::package::getPath("nbvplanner") + "/data/"',
    '(std::getenv("PELAGOS_RUN") ? std::string(std::getenv("PELAGOS_RUN"))+"/nbv-data/" : ros::package::getPath("nbvplanner")+"/data/")')
rrt.write_text(text)
cmake = root / 'nbvplanner/nbvplanner/CMakeLists.txt'
text = cmake.read_text().replace('find_package(Eigen REQUIRED)',
    'find_package(Eigen3 REQUIRED)\nset(Eigen_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIRS})')
text = text.replace('LIBRARIES nbvplanner ', 'LIBRARIES nbvPlannerLib ')
text = text.replace('add_dependencies(nbvPlannerLib ${${PROJECT_NAME}_EXPORTED_TARGETS})',
    'add_dependencies(nbvPlannerLib ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})\n'
    'add_dependencies(nbvPlanner ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})')
cmake.write_text(text)

# Current PCL headers require C++14; only change build flags, not algorithms.
for cmake in (root/'nbvplanner').rglob('CMakeLists.txt'):
    text=cmake.read_text()
    cmake.write_text(text.replace('-std=c++11','-std=c++14'))

# Noetic rejects catkin_add_gtest itself when testing is disabled. Protect the
# complete upstream test section, including declarations and linking.
for relative in ('minkindr/minkindr/CMakeLists.txt','eigen_checks/CMakeLists.txt'):
    cmake=root/relative
    text=cmake.read_text()
    start=re.search(r'find_package\s*\(Threads\)',text)
    links=list(re.finditer(r'target_link_libraries\([^)]*\)',text))
    assert start and links and 'catkin_add_gtest(' in text[start.start():links[-1].end()],relative
    first,last=start.start(),links[-1].end()
    text=(text[:first]+'if(CATKIN_ENABLE_TESTING)\n'+text[first:last]+
          '\nendif()\n'+text[last:])
    cmake.write_text(text)

# Reuse the toolchain's system gflags instead of downloading/building it again.
(root/'gflags_catkin/CMakeLists.txt').write_text('''cmake_minimum_required(VERSION 2.8.3)
project(gflags_catkin)
find_package(catkin_simple REQUIRED)
catkin_simple()
find_path(GFLAGS_INCLUDE_DIR gflags/gflags.h)
find_library(GFLAGS_LIBRARY gflags)
if(NOT GFLAGS_INCLUDE_DIR OR NOT GFLAGS_LIBRARY)
  message(FATAL_ERROR "Install the toolchain's libgflags-dev")
endif()
cs_install()
cs_export(INCLUDE_DIRS ${GFLAGS_INCLUDE_DIR} LIBRARIES ${GFLAGS_LIBRARY})
''')
