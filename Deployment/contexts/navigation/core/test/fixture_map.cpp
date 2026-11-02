#include <iostream>
#include <wavemap/core/map/hashed_wavelet_octree.h>
#include <wavemap/io/file_conversions.h>
int main(int argc,char** argv) {
  google::InitGoogleLogging(argv[0]);if(argc!=2)return 2;
  wavemap::HashedWaveletOctreeConfig config;config.min_cell_width=.2f;config.tree_height=4;
  wavemap::HashedWaveletOctree map(config);
  for(int x=-20;x<=20;++x)for(int y=-15;y<=15;++y)for(int z=-10;z<=10;++z)
    map.addToCellValue(wavemap::Index3D{x,y,z},-2.f);
  map.threshold();map.prune();
  if(!wavemap::io::mapToFile(map,argv[1]))return 1;
  std::cout<<"Synthetic observed free room saved, resolution .2 m, tree height 4\n";
}
