#include "dapeng_navigation/map_manager.hpp"

#include <cmath>
#include <stdexcept>
#include <wavemap/core/indexing/index_conversions.h>
#include <wavemap/core/map/hashed_chunked_wavelet_octree.h>
#include <wavemap/core/map/hashed_wavelet_octree.h>

namespace dapeng_navigation {
double milliseconds(Clock::duration duration) {
  return std::chrono::duration<double, std::milli>(duration).count();
}
void ManagementConfig::validate() const {
  for (double value : {threshold_period_s, prune_period_s, crop_period_s, crop_radius_m}) {
    if (!std::isfinite(value) || value <= 0) {
      throw std::invalid_argument("map management periods and crop radius must be finite and positive");
    }
  }
}
CropMapOperation::CropMapOperation(wavemap::MapBase::Ptr map, double period_s,
                                 double radius_m)
    : MapOperationBase(map ? std::move(map) : throw std::invalid_argument("null crop map")), period_s_(period_s), radius_m_(radius_m) {
  ManagementConfig{period_s, period_s, period_s, radius_m}.validate();
  if (!dynamic_cast<wavemap::HashedWaveletOctree*>(occupancy_map_.get()) &&
      !dynamic_cast<wavemap::HashedChunkedWaveletOctree*>(occupancy_map_.get())) {
    throw std::invalid_argument("crop requires a hashed wavelet octree map");
  }
}
void CropMapOperation::setCenter(const wavemap::Point3D& center) {
  if (!center.allFinite()) throw std::invalid_argument("nonfinite crop center");
  center_ = center;
}
bool CropMapOperation::shouldRun(Clock::time_point now) const {
  return std::chrono::duration<double>(now - last_run_).count() >= period_s_;
}
void CropMapOperation::run(bool force_run) {
  removed_blocks_ = 0;
  const auto now = Clock::now();
  if (!center_ || (!force_run && !shouldRun(now))) return;
  last_run_ = now;
  const auto height = occupancy_map_->getTreeHeight();
  const auto width = occupancy_map_->getMinCellWidth();
  const auto outside = [&](const wavemap::Index3D& index, const auto&) {
    const auto aabb = wavemap::convert::nodeIndexToAABB(
        wavemap::OctreeIndex{height, index}, width);
    const bool remove = radius_m_ < aabb.minDistanceTo(*center_);
    removed_blocks_ += remove;
    return remove;
  };
  if (auto* map = dynamic_cast<wavemap::HashedWaveletOctree*>(occupancy_map_.get())) {
    map->eraseBlockIf(outside);
  } else {
    dynamic_cast<wavemap::HashedChunkedWaveletOctree&>(*occupancy_map_).eraseBlockIf(outside);
  }
}
namespace {
ManagementConfig validated(ManagementConfig config) { config.validate(); return config; }
wavemap::MapBase::Ptr checkedMap(wavemap::MapBase::Ptr map) {
  if (!map) throw std::invalid_argument("null native map");
  return map;
}
wavemap::ThresholdMapOperationConfig thresholdConfig(double period) {
  wavemap::ThresholdMapOperationConfig config; config.once_every = period; return config;
}
wavemap::PruneMapOperationConfig pruneConfig(double period) {
  wavemap::PruneMapOperationConfig config; config.once_every = period; return config;
}
}
MapManager::MapManager(wavemap::MapBase::Ptr map, ManagementConfig config)
    : map_(checkedMap(std::move(map))), config_(validated(config)),
      threshold_(thresholdConfig(config_.threshold_period_s), map_),
      prune_(pruneConfig(config_.prune_period_s), map_),
      crop_(map_, config_.crop_period_s, config_.crop_radius_m) {
  if (!map_) throw std::invalid_argument("null native map");
}
uint64_t MapManager::update(const Access& update) {
  std::lock_guard<std::mutex> lock(mutex_);
  update(*map_, version_.load());
  const auto version=++version_;
  if(on_commit_)on_commit_(version);
  return version;
}
void MapManager::read(const Read& reader) const {
  std::lock_guard<std::mutex> lock(mutex_);
  reader(*map_, version_.load());
}
bool MapManager::tryRead(const Read& reader) const {
  std::unique_lock<std::mutex> lock(mutex_,std::try_to_lock);
  if(!lock.owns_lock())return false;
  reader(*map_,version_.load());return true;
}
void MapManager::setOnCommit(std::function<void(uint64_t)> callback) {
  std::lock_guard<std::mutex> lock(mutex_);on_commit_=std::move(callback);
}
std::optional<MaintenanceResult> MapManager::tryMaintain(
    const std::optional<wavemap::Point3D>& center, bool force) {
  if (center && !center->allFinite()) throw std::invalid_argument("nonfinite maintenance pose");
  std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
  if (!lock.owns_lock()) return std::nullopt;
  const auto start = Clock::now();
  MaintenanceResult result;
  result.memory_before_bytes = map_->getMemoryUsage();
  const auto now = wavemap::Time::now();
  auto tick = Clock::now();
  if (force || threshold_.shouldRun(now)) {
    threshold_.run(true); result.thresholded = true;
  }
  result.threshold_ms = milliseconds(Clock::now() - tick); tick = Clock::now();
  if (force || prune_.shouldRun(now)) {
    prune_.run(true); result.pruned = true;
  }
  result.prune_ms = milliseconds(Clock::now() - tick); tick = Clock::now();
  if (center && (force || crop_.shouldRun(Clock::now()))) {
    crop_.setCenter(*center); crop_.run(true); result.cropped = true;
    result.removed_blocks = crop_.removedBlocks();
  }
  result.crop_ms = milliseconds(Clock::now() - tick);
  if (result.thresholded || result.pruned || result.removed_blocks) {
    const auto version=++version_;if(on_commit_)on_commit_(version);
  }
  result.version = version_.load();
  result.memory_after_bytes = map_->getMemoryUsage();
  result.total_ms = milliseconds(Clock::now() - start);
  return result;
}
}  // namespace dapeng_navigation
