#pragma once
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <vector>
#include <octomap/OcTree.h>
#include <wavemap/core/map/map_base.h>
#include <wavemap/core/utils/query/occupancy_classifier.h>

namespace underwater_wavemap_ros {
struct ProjectionStats {
  size_t observed_leaves=0,free_cells=0,occupied_cells=0,mixed_unknown_cells=0;
};
struct Projection {
  std::unique_ptr<octomap::OcTree> tree;
  ProjectionStats stats;
};
struct OccupancyLeaf {wavemap::OctreeIndex index;float value;};
struct OccupancySnapshot {double min_cell_width;std::vector<OccupancyLeaf> leaves;};
inline int64_t floorDiv(int64_t n,int64_t d) {
  const auto q=n/d,r=n%d;return q-(r<0);
}

// Conservative format projection, not a second measurement integrator.
// At target resolution: any occupied subvolume wins; free requires full
// observed-free coverage. Unknown cells are never allocated in OctoMap.
template <typename LeafVisitor>
inline Projection projectOctomapLeaves(double min_cell_width,double resolution,LeafVisitor visit) {
  const double ratio=resolution/min_cell_width;
  if(!std::isfinite(ratio)||ratio<1.-1.e-4||ratio>1024.+1.e-4)throw std::runtime_error("Invalid OctoMap/Wavemap resolution ratio");
  const auto factor=static_cast<int64_t>(std::llround(ratio));
  if(factor<1||factor>1024||std::abs(ratio-factor)>1.e-4||
      (factor&(factor-1)))throw std::runtime_error("OctoMap resolution must be a power-of-two multiple of Wavemap cells");
  struct Cell {
    uint64_t free_volume=0;
    bool occupied=false;
    float free_max=-std::numeric_limits<float>::infinity();
    float occupied_max=-std::numeric_limits<float>::infinity();
  };
  std::map<std::array<int64_t,3>,Cell> cells;
  Projection result;result.tree=std::make_unique<octomap::OcTree>(resolution);
  const uint64_t full_volume=factor*factor*factor;
  const wavemap::OccupancyClassifier classifier; // official threshold 0 log odds
  visit([&](const wavemap::OctreeIndex& index,float value) {
    if(!std::isfinite(value))throw std::runtime_error("Nonfinite Wavemap occupancy");
    if(!wavemap::OccupancyClassifier::isObserved(value))return;
    if(index.height<0||index.height>30)throw std::runtime_error("Invalid Wavemap leaf height");
    ++result.stats.observed_leaves;
    const int64_t width=int64_t(1)<<index.height;
    std::array<int64_t,3> low,high,first,last;
    for(int k=0;k<3;++k) {
      low[k]=int64_t(index.position[k])*width;high[k]=low[k]+width;
      first[k]=floorDiv(low[k],factor);last[k]=floorDiv(high[k]-1,factor);
      if(first[k]<-32768||last[k]>32767)throw std::runtime_error("Map exceeds native OctoMap coordinate range");
    }
    const bool occupied=classifier.classify(value)==wavemap::Occupancy::kOccupied;
    for(int64_t x=first[0];x<=last[0];++x)
      for(int64_t y=first[1];y<=last[1];++y)
        for(int64_t z=first[2];z<=last[2];++z) {
          const std::array<int64_t,3> key{x,y,z};uint64_t volume=1;
          for(int k=0;k<3;++k)
            volume*=std::min(high[k],(key[k]+1)*factor)-std::max(low[k],key[k]*factor);
          auto& cell=cells[key];
          if(occupied) {cell.occupied=true;cell.occupied_max=std::max(cell.occupied_max,value);}
          else {cell.free_volume+=volume;cell.free_max=std::max(cell.free_max,value);}
          if(cell.free_volume>full_volume)throw std::runtime_error("Overlapping Wavemap leaf coverage");
        }
  });
  for(const auto& [key,cell]:cells) {
    if(!cell.occupied&&cell.free_volume!=full_volume) {++result.stats.mixed_unknown_cells;continue;}
    const octomap::point3d center((key[0]+.5)*resolution,(key[1]+.5)*resolution,(key[2]+.5)*resolution);
    result.tree->setNodeValue(center,cell.occupied?cell.occupied_max:cell.free_max,true);
    if(cell.occupied)++result.stats.occupied_cells;else ++result.stats.free_cells;
  }
  result.tree->updateInnerOccupancy();result.tree->prune();return result;
}
inline Projection projectOctomap(const wavemap::MapBase& source,double resolution) {
  return projectOctomapLeaves(source.getMinCellWidth(),resolution,
    [&](const auto& visitor){source.forEachLeaf(visitor);});
}
inline Projection projectOctomap(const OccupancySnapshot& source,double resolution) {
  return projectOctomapLeaves(source.min_cell_width,resolution,[&](const auto& visitor){
    for(const auto& leaf:source.leaves)visitor(leaf.index,leaf.value);
  });
}
}
