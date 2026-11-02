#pragma once

#include <atomic>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>

#include <wavemap/core/map/map_base.h>
#include <wavemap/pipeline/map_operations/prune_map_operation.h>
#include <wavemap/pipeline/map_operations/threshold_map_operation.h>

namespace dapeng_navigation {
using Clock = std::chrono::steady_clock;
double milliseconds(Clock::duration duration);

struct ManagementConfig {
  double threshold_period_s = 2.0;
  double prune_period_s = 10.0;
  double crop_period_s = 1.0;
  double crop_radius_m = 6.0;
  void validate() const;
};

struct MaintenanceResult {
  uint64_t version = 0;
  size_t memory_before_bytes = 0, memory_after_bytes = 0;
  size_t removed_blocks = 0;
  bool thresholded = false, pruned = false, cropped = false;
  double threshold_ms = 0, prune_ms = 0, crop_ms = 0, total_ms = 0;
};

// ROS 2 port of the official ROS 1 crop operation. The TF lookup is deliberately
// injected by the ROS adapter; block-AABB distance / erase semantics are kept.
// Source: ethz-asl/wavemap@db4f2fffdc1da1bec042d51bfff58029c0c9c827,
// interfaces/ros1/wavemap_ros/src/map_operations/crop_map_operation.cc.
class CropMapOperation final : public wavemap::MapOperationBase {
 public:
  CropMapOperation(wavemap::MapBase::Ptr map, double period_s, double radius_m);
  void setCenter(const wavemap::Point3D& center);
  void run(bool force_run) override;
  bool shouldRun(Clock::time_point now) const;
  size_t removedBlocks() const { return removed_blocks_; }
 private:
  const double period_s_, radius_m_;
  std::optional<wavemap::Point3D> center_;
  Clock::time_point last_run_{};
  size_t removed_blocks_ = 0;
};

// All native-map reads and writes go through this owner. Maintenance does not
// clone the map. tryMaintain never waits for an in-progress fusion transaction.
class MapManager {
 public:
  MapManager(wavemap::MapBase::Ptr map, ManagementConfig config);
  using Access = std::function<void(wavemap::MapBase&, uint64_t)>;
  using Read = std::function<void(const wavemap::MapBase&, uint64_t)>;
  uint64_t update(const Access& update);
  void read(const Read& reader) const;
  bool tryRead(const Read& reader) const;
  void setOnCommit(std::function<void(uint64_t)> callback);
  std::optional<MaintenanceResult> tryMaintain(
      const std::optional<wavemap::Point3D>& center, bool force = false);
  uint64_t version() const { return version_.load(); }
  const wavemap::MapBase* identity() const { return map_.get(); }
 private:
  const wavemap::MapBase::Ptr map_;
  const ManagementConfig config_;
  mutable std::mutex mutex_;
  std::atomic<uint64_t> version_{0};
  std::function<void(uint64_t)> on_commit_;
  wavemap::ThresholdMapOperation threshold_;
  wavemap::PruneMapOperation prune_;
  CropMapOperation crop_;
};
}  // namespace dapeng_navigation
