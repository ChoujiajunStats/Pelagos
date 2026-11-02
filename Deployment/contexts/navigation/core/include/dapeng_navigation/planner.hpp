#pragma once
#include <functional>
#include <string>
#include <vector>
#include "dapeng_navigation/planning_map.hpp"
namespace dapeng_navigation {
struct PlannerConfig {
  std::string algorithm="multi_res_theta_star";
  float refinement_epsilon=.01f;
  int initialization_min_height=0;
  double max_result_latency_ms=2000.;
  size_t max_collision_cells=2000000;
  void validate(int tree_height) const;
};
struct PlanResult {
  uint64_t map_version=0;
  std::string status;
  std::vector<wavemap::Point3D> path;
  double planning_ms=0,collision_ms=0,length_m=0;
  bool collision_free=false;
};
PlanResult plan(const PlanningSnapshot& snapshot,const wavemap::Point3D& start,
                const wavemap::Point3D& goal,const PlannerConfig& config);
bool pathCollisionFree(const PlanningSnapshot& snapshot,
                       const std::vector<wavemap::Point3D>& path,
                       size_t max_cells=2000000);
// Serialize version changes and path publication. Late results from old goals
// or old maps can never publish after a newer invalidation.
class ResultGate {
 public:
  void advanceMap(uint64_t version,const std::function<void()>& invalidate={});
  uint64_t newGoal(const std::function<void()>& invalidate={});
  bool commit(uint64_t version,uint64_t goal,const std::function<void()>& publish);
  uint64_t goal() const;
 private:
  mutable std::mutex mutex_;
  uint64_t version_=0,goal_=0;
};
}
