#include "dapeng_navigation/planning_map.hpp"
#include <cmath>
#include <stdexcept>
#include <wavemap/core/indexing/index_conversions.h>
#include <wavemap/core/map/hashed_chunked_wavelet_octree.h>
#include <wavemap/core/utils/sdf/quasi_euclidean_sdf_generator.h>

namespace dapeng_navigation {
namespace {
template<class SourceNode> void copyNode(const SourceNode& source,Block::OctreeType::NodeRefType target) {
  target.data()=source.data();
  for(int child=0;child<8;++child) {
    const auto* pointer=source.getChild(child);
    if(pointer) copyNode(*pointer,target.getOrAllocateChild(child));
  }
}
// Chunked nodes return NodePtr proxies rather than raw pointers.
template<class SourceBlock> std::shared_ptr<Block> cloneBlock(const SourceBlock& source,
                                                           const wavemap::MapBase& map) {
  auto target=std::make_shared<Block>(map.getTreeHeight(),map.getMinLogOdds(),map.getMaxLogOdds());
  target->getRootScale()=source.getRootScale();
  copyNode(source.getRootNode(),target->getRootNode());
  return target;
}
template<class SourceNode> void copyProxyNode(const SourceNode& source,Block::OctreeType::NodeRefType target) {
  target.data()=source.data();
  for(int child=0;child<8;++child) {
    auto pointer=source.getChild(child);
    if(pointer) copyProxyNode(*pointer,target.getOrAllocateChild(child));
  }
}
std::shared_ptr<Block> cloneBlock(const wavemap::HashedChunkedWaveletOctreeBlock& source,
                                const wavemap::MapBase& map) {
  auto target=std::make_shared<Block>(map.getTreeHeight(),map.getMinLogOdds(),map.getMaxLogOdds());
  target->getRootScale()=source.getRootScale();
  copyProxyNode(source.getRootNode(),target->getRootNode());
  return target;
}
}
BlockDelta DeltaTracker::capture(const wavemap::MapBase& map,uint64_t version,bool force,
                                 const std::optional<SnapshotRegion>& region) {
  if(region && (!region->center.allFinite()||!region->half_extent.allFinite()||
                (region->half_extent.array()<=0).any()))throw std::invalid_argument("invalid snapshot ROI");
  const auto start=Clock::now();
  BlockDelta delta;delta.version=version;
  delta.config.min_cell_width=map.getMinCellWidth();
  delta.config.min_log_odds=map.getMinLogOdds();delta.config.max_log_odds=map.getMaxLogOdds();
  delta.config.tree_height=map.getTreeHeight();
  auto stamps=stamps_; // Tiny block-index metadata; never an occupancy map copy.
  const auto visit=[&](const auto& source){
    source.forEachBlock([&](const wavemap::Index3D& index,const auto& block){
      if(region) {
        const float width=map.getMinCellWidth()*float(1<<map.getTreeHeight());
        const wavemap::Point3D low=index.cast<float>()*width,high=low+wavemap::Point3D::Constant(width);
        const auto extent=region->half_extent.array()+map.getMinCellWidth();
        if((high.array()<(region->center.array()-extent)).any()||
           (region->center.array()+extent<low.array()).any())return;
      }
      if(delta.retained.size()>=max_blocks_) throw std::runtime_error("planning snapshot exceeds block budget");
      delta.retained.insert(index);
      auto it=stamps_.find(index);
      if(force || it==stamps_.end() || it->second!=block.getLastUpdatedStamp()) {
        auto copy=cloneBlock(block,map);
        delta.copied_bytes+=copy->getMemoryUsage();
        if(delta.copied_bytes>max_bytes_) throw std::runtime_error("planning snapshot exceeds memory budget");
        delta.changed.emplace(index,std::move(copy));
      }
      stamps[index]=block.getLastUpdatedStamp();
    });
  };
  if(auto source=dynamic_cast<const wavemap::HashedWaveletOctree*>(&map))visit(*source);
  else if(auto source=dynamic_cast<const wavemap::HashedChunkedWaveletOctree*>(&map))visit(*source);
  else throw std::invalid_argument("planning snapshot requires a hashed wavelet octree");
  for(auto it=stamps.begin();it!=stamps.end();) {
    if(!delta.retained.count(it->first))it=stamps.erase(it);else ++it;
  }
  stamps_=std::move(stamps);delta.capture_ms=milliseconds(Clock::now()-start);
  return delta;
}
BlockDelta mergeDeltas(BlockDelta older,const BlockDelta& newer) {
  if(newer.version<older.version)throw std::invalid_argument("backward map delta version");
  if(older.config.tree_height!=newer.config.tree_height || older.config.min_cell_width!=newer.config.min_cell_width)
    throw std::invalid_argument("cannot merge different map geometries");
  for(const auto& [index,block]:newer.changed)older.changed[index]=block;
  for(auto it=older.changed.begin();it!=older.changed.end();) {
    if(!newer.retained.count(it->first))it=older.changed.erase(it);else ++it;
  }
  older.retained=newer.retained;older.version=newer.version;older.config=newer.config;
  older.capture_ms+=newer.capture_ms;older.copied_bytes+=newer.copied_bytes;
  return older;
}
void PlanningMapConfig::validate(float resolution) const {
  if(!std::isfinite(resolution)||resolution<=0 || !std::isfinite(occupancy_threshold_log_odds)
     || !std::isfinite(robot_radius_m)||robot_radius_m<=0 || !half_extent_m.allFinite()
     || (half_extent_m.array()<=robot_radius_m+resolution).any() || max_roi_cells==0 || max_mirror_bytes==0)
    throw std::invalid_argument("invalid planning map geometry or occupancy configuration");
  const double cells=(2.*half_extent_m.cast<double>().array()/resolution+2.).prod();
  if(cells>max_roi_cells)throw std::invalid_argument("planning ROI exceeds voxel budget");
}
void PlanningMapBuilder::apply(const BlockDelta& delta) {
  config_.validate(delta.config.min_cell_width);
  if(delta.version<version_)throw std::invalid_argument("backward planning map version");
  if(version_ && (map_config_.tree_height!=delta.config.tree_height ||
                 map_config_.min_cell_width!=delta.config.min_cell_width))
    throw std::invalid_argument("planning geometry changed during a run");
  auto next=blocks_;
  for(auto it=next.begin();it!=next.end();) {
    if(!delta.retained.count(it->first))it=next.erase(it);else ++it;
  }
  for(const auto& [index,block]:delta.changed)next[index]=block;
  size_t bytes=0;for(const auto& [index,block]:next)bytes+=block->getMemoryUsage();
  if(bytes>config_.max_mirror_bytes)throw std::runtime_error("planning mirror exceeds memory budget");
  blocks_=std::move(next);
  map_config_=delta.config;version_=delta.version;
}
namespace {
void copyClippedLeaf(const wavemap::OctreeIndex& index,float value,
                     const wavemap::Index3D& minimum,const wavemap::Index3D& maximum,
                     wavemap::HashedWaveletOctree& target) {
  const auto low=wavemap::convert::nodeIndexToMinCornerIndex(index);
  const auto high=wavemap::convert::nodeIndexToMaxCornerIndex(index);
  if((high.array()<minimum.array()).any() || (low.array()>maximum.array()).any())return;
  if((minimum.array()<=low.array()).all() && (high.array()<=maximum.array()).all()) {
    const auto block_index=target.indexToBlockIndex(index);
    auto& block=target.getOrAllocateBlock(block_index);
    if(index.height==target.getTreeHeight())block.getRootScale()=value;
    else {
      const auto local=target.indexToCellIndex(index);
      block.addToCellValue(local,value-block.getCellValue(local));
    }
    return;
  }
  for(int child=0;child<8;++child)copyClippedLeaf(index.computeChildIndex(child),value,minimum,maximum,target);
}
}
std::shared_ptr<const PlanningSnapshot> PlanningMapBuilder::generate(const wavemap::Point3D& center) {
  if(!center.allFinite()||!version_)throw std::invalid_argument("planning requires a finite center and committed map version");
  const auto start=Clock::now();config_.validate(map_config_.min_cell_width);
  const auto minimum=((center-config_.half_extent_m)/map_config_.min_cell_width).array().floor().cast<int>().matrix().eval();
  const auto maximum=(((center+config_.half_extent_m)/map_config_.min_cell_width).array().ceil()-1).cast<int>().matrix().eval();
  auto roi_pointer=std::make_shared<wavemap::HashedWaveletOctree>(map_config_);
  auto& roi=*roi_pointer;
  for(const auto& [index,block]:blocks_) {
    block->forEachLeaf(index,[&](const auto& leaf,float value){
      if(wavemap::OccupancyClassifier::isObserved(value))copyClippedLeaf(leaf,value,minimum,maximum,roi);
    });
  }
  roi.threshold();roi.prune();
  auto result=std::make_shared<PlanningSnapshot>();result->version=version_;result->center=center;
  result->robot_radius_m=config_.robot_radius_m;
  result->roi_memory_bytes=roi.getMemoryUsage();result->roi_ms=milliseconds(Clock::now()-start);
  // Compensate the documented quasi-Euclidean overestimate and the half voxel
  // diagonal, so a center-to-center planner does not cut through obstacle cells.
  result->inflated_radius_m=config_.robot_radius_m*
      (1.f+wavemap::QuasiEuclideanSDFGenerator::kMaxRelativeOverEstimate)+
      .5f*std::sqrt(3.f)*map_config_.min_cell_width;
  auto tick=Clock::now();
  wavemap::QuasiEuclideanSDFGenerator generator(result->inflated_radius_m+map_config_.min_cell_width,
                                               config_.occupancy_threshold_log_odds);
  const auto sdf=generator.generate(roi);result->sdf_ms=milliseconds(Clock::now()-tick);
  result->sdf_memory_bytes=sdf.getMemoryUsage();tick=Clock::now();
  auto classified=std::make_shared<wavemap::ClassifiedMap>(roi,
      wavemap::OccupancyClassifier{config_.occupancy_threshold_log_odds},sdf,result->inflated_radius_m);
  classified->forEachBlock([&](const auto&,const auto& block){result->classified_nodes+=block.size();});
  result->map=std::move(classified);result->classification_ms=milliseconds(Clock::now()-tick);
  result->occupancy=std::move(roi_pointer);
  result->source_blocks=blocks_;result->roi_minimum=minimum;result->roi_maximum=maximum;
  result->occupancy_threshold_log_odds=config_.occupancy_threshold_log_odds;
  result->created=Clock::now();return result;
}
CellState occupancyAt(const PlanningSnapshot& snapshot,const wavemap::Point3D& point) {
  if(!snapshot.map||!point.allFinite())return CellState::Unknown;
  std::lock_guard<std::mutex> lock(snapshot.queries);
  const auto index=wavemap::convert::pointToNearestIndex(point,1.f/snapshot.map->getMinCellWidth());
  if(snapshot.map->isFully(index,wavemap::Occupancy::kFree))return CellState::Free;
  if(snapshot.map->isFully(index,wavemap::Occupancy::kOccupied))return CellState::Occupied;
  return CellState::Unknown;
}
float sourceOccupancy(const PlanningSnapshot& snapshot,const wavemap::Index3D& index) {
  if(!snapshot.map)return 0.f;
  if((index.array()<snapshot.roi_minimum.array()).any() || (snapshot.roi_maximum.array()<index.array()).any())return 0.f;
  const int height=snapshot.map->getTreeHeight();
  const auto block_index=wavemap::convert::indexToBlockIndex(index,height);
  const auto it=snapshot.source_blocks.find(block_index);
  if(it==snapshot.source_blocks.end())return 0.f;
  return it->second->getCellValue(wavemap::OctreeIndex{0,index-block_index*(1<<height)});
}
}  // namespace dapeng_navigation
