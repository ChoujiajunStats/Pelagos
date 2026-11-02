#pragma once
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <wavemap/core/map/hashed_wavelet_octree.h>
#include <wavemap/core/utils/query/classified_map.h>
#include "dapeng_navigation/map_manager.hpp"

namespace dapeng_navigation {
using Block = wavemap::HashedWaveletOctreeBlock;
using BlockCopies = std::unordered_map<wavemap::Index3D, std::shared_ptr<const Block>, wavemap::Index3DHash>;
using BlockIndices = std::unordered_set<wavemap::Index3D,wavemap::Index3DHash>;
struct BlockDelta {
  uint64_t version = 0;
  wavemap::HashedWaveletOctreeConfig config;
  BlockCopies changed;
  BlockIndices retained;
  double capture_ms = 0;
  size_t copied_bytes = 0;
};
struct SnapshotRegion { wavemap::Point3D center,half_extent; };
// The caller holds the native-map owner's lock during capture. Only changed
// blocks are copied; this is a bounded planning mirror, never a second live map.
class DeltaTracker {
 public:
  DeltaTracker(size_t max_blocks=128,size_t max_bytes=64*1024*1024)
      : max_blocks_(max_blocks),max_bytes_(max_bytes) {}
  BlockDelta capture(const wavemap::MapBase& map,uint64_t version,bool force=false,
                     const std::optional<SnapshotRegion>& region=std::nullopt);
 private:
  std::unordered_map<wavemap::Index3D,wavemap::Timestamp,wavemap::Index3DHash> stamps_;
  const size_t max_blocks_,max_bytes_;
};
BlockDelta mergeDeltas(BlockDelta older, const BlockDelta& newer);

struct PlanningMapConfig {
  float occupancy_threshold_log_odds = 0.f;
  float robot_radius_m = .35f;
  wavemap::Point3D half_extent_m{3.f,3.f,2.f};
  size_t max_roi_cells = 500000;
  size_t max_mirror_bytes = 64*1024*1024;
  void validate(float resolution) const;
};
struct PlanningSnapshot {
  uint64_t version = 0;
  wavemap::ClassifiedMap::ConstPtr map;
  wavemap::HashedWaveletOctree::ConstPtr occupancy;
  BlockCopies source_blocks;
  wavemap::Index3D roi_minimum,roi_maximum;
  float occupancy_threshold_log_odds=0;
  wavemap::Point3D center;
  float robot_radius_m = 0, inflated_radius_m = 0;
  Clock::time_point created;
  double sdf_ms=0, classification_ms=0, roi_ms=0;
  size_t roi_memory_bytes=0, sdf_memory_bytes=0, classified_nodes=0;
  // ClassifiedMap's const methods modify its query cache. Every consumer must
  // hold this mutex, including Wavestar and path collision validation.
  mutable std::mutex queries;
};
class PlanningMapBuilder {
 public:
  explicit PlanningMapBuilder(PlanningMapConfig config):config_(std::move(config)) {}
  void apply(const BlockDelta& delta);
  std::shared_ptr<const PlanningSnapshot> generate(const wavemap::Point3D& center);
 private:
  PlanningMapConfig config_;
  BlockCopies blocks_;
  wavemap::HashedWaveletOctreeConfig map_config_;
  uint64_t version_=0;
};
enum class CellState { Free, Occupied, Unknown };
CellState occupancyAt(const PlanningSnapshot& snapshot,const wavemap::Point3D& point);
float sourceOccupancy(const PlanningSnapshot& snapshot,const wavemap::Index3D& index);
}  // namespace dapeng_navigation
