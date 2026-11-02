#pragma once
#include <memory>
#include <stdexcept>
#include <wavemap/core/map/hashed_chunked_wavelet_octree.h>
#include <wavemap/core/map/hashed_wavelet_octree.h>

namespace underwater_wavemap_ros {
// Wavemap's official serializer writes chunked maps in HashedWaveletOctree
// format. Copy the saved Haar coefficients directly; do not re-integrate or
// assign reconstructed leaves (which could perturb the occupancy evidence).
inline wavemap::HashedChunkedWaveletOctree::Ptr restoreChunked(const wavemap::MapBase& source) {
  auto* saved=dynamic_cast<const wavemap::HashedWaveletOctree*>(&source);
  if(!saved)throw std::runtime_error("Expected Wavemap's serialized hashed wavelet format");
  wavemap::HashedChunkedWaveletOctreeConfig config;
  config.min_cell_width=source.getMinCellWidth();config.tree_height=source.getTreeHeight();
  config.min_log_odds=source.getMinLogOdds();config.max_log_odds=source.getMaxLogOdds();
  auto result=std::make_shared<wavemap::HashedChunkedWaveletOctree>(config);
  const auto copy=[](auto&& self,const auto& from,auto to)->void {
    to.data()=from.data();
    for(int i=0;i<8;++i)if(auto* child=from.getChild(i))self(self,*child,to.getOrAllocateChild(i));
  };
  saved->forEachBlock([&](const auto& index,const auto& block) {
    auto& restored=result->getOrAllocateBlock(index);restored.getRootScale()=block.getRootScale();
    copy(copy,block.getRootNode(),restored.getRootNode());
  });
  return result;
}
}
