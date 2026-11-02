#include <wavemap/core/map/hashed_wavelet_octree.h>
#include <wavemap/core/map/hashed_chunked_wavelet_octree.h>
#include <wavemap/io/file_conversions.h>
#include <iostream>

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  wavemap::MapBase::Ptr map;
  if (std::string(argv[1]) == "chunked") {
    wavemap::HashedChunkedWaveletOctree::Config c;
    c.min_cell_width=.1f; c.tree_height=6;
    map=std::make_shared<wavemap::HashedChunkedWaveletOctree>(c);
  } else {
    wavemap::HashedWaveletOctree::Config c;
    c.min_cell_width=.1f; c.tree_height=6;
    map=std::make_shared<wavemap::HashedWaveletOctree>(c);
  }
  map->setCellValue({-3,-2,-1},-2.f);
  map->setCellValue({3,2,1},3.f);
  map->setCellValue({5,4,3},.2f);
  map->prune();
  return wavemap::io::mapToFile(*map,argv[2]) ? 0 : 1;
}
