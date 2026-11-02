#pragma once
#include <array>
#include <vector>
#include "dapeng_navigation/map_manager.hpp"

namespace dapeng_navigation {
struct VisualizationConfig {
  double local_radius_m=1.5, voxel_size_m=.1;
  double height_min_m=-1.,height_max_m=1.,local_alpha=.95,blue_alpha=.35;
  bool height_relative_to_robot=true,reverse_colors=false;
  size_t max_source_leaves=150000,max_local_voxels=60000,max_blue_points=100000,max_candidates=500000;
  double read_budget_ms=20.;
  void validate(float resolution) const;
};
struct OccupiedGeometry {
  wavemap::Point3D minimum;
  float width;
};
struct VisualizationSource {
  uint64_t version=0;
  float resolution=.1;
  std::vector<OccupiedGeometry> occupied;
  size_t visited_leaves=0;
  bool truncated=false;
  double capture_ms=0;
};
struct ColoredCube {
  wavemap::Point3D center;
  float width;
  std::array<float,4> color;
};
struct VisualizationFrame {
  uint64_t version=0;
  std::vector<ColoredCube> local;
  std::vector<wavemap::Point3D> blue;
  bool truncated=false;
  double build_ms=0;
};
// Call while holding MapManager's read lock. Only immutable occupied-leaf
// geometry is extracted. No occupancy tree, free cells or unknown cells copied.
VisualizationSource captureVisualization(const wavemap::MapBase& map,uint64_t version,
                                         const VisualizationConfig& config,float threshold);
// No native map access; subdivision of a confirmed occupied leaf does not
// infer extra occupancy outside that leaf. Rendered sizes never exceed it.
VisualizationFrame renderVisualization(const VisualizationSource& source,
                                      const wavemap::Point3D& robot,const VisualizationConfig& config);
std::array<float,4> heightColor(float height,const VisualizationConfig& config);
}  // namespace dapeng_navigation
