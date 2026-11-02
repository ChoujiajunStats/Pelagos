#pragma once
#include <array>
#include <map>
#include <set>
#include <vector>
#include "dapeng_navigation/visualization.hpp"
namespace dapeng_navigation {
using MeshKey=std::array<int,3>;
struct MeshConfig {
  double resolution_m=.1, radius_m=6.,read_budget_ms=8.;
  size_t max_blocks_per_update=1,max_source_leaves=80000,max_triangles=60000,max_blocks=128;
  void validate(float native_resolution,int height) const;
};
struct MeshBlockSource {
  MeshKey key;
  uint64_t fingerprint=0;
  int native_side=0;
  float native_resolution=0;
  std::vector<OccupiedGeometry> occupied;
};
struct MeshCapture {
  uint64_t version=0;
  std::set<MeshKey> retained;
  std::vector<MeshBlockSource> changed;
  size_t pending=0;
  bool budget_limited=false;
  double read_ms=0;
};
struct SurfaceBlock {
  MeshKey key;
  uint64_t fingerprint=0;
  std::vector<wavemap::Point3D> vertices; // triples, outward winding, no per-vertex colors
  double build_ms=0;
};
// Metadata and final surface geometry only. No persistent voxel or occupancy copy.
class SurfaceTracker {
 public:
  MeshCapture capture(const wavemap::MapBase& map,uint64_t version,const wavemap::Point3D& center,
                      const MeshConfig& config,float threshold);
  void acknowledge(const MeshBlockSource& block);
  void reset(){seen_.clear();order_.clear();sequence_=0;}
 private:
  std::map<MeshKey,uint64_t> seen_,order_;
  uint64_t sequence_=0;
};
SurfaceBlock greedySurface(const MeshBlockSource& source,const MeshConfig& config);
} // namespace dapeng_navigation
