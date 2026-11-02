#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <wavemap/core/map/hashed_wavelet_octree.h>
#include <wavemap/core/map/hashed_chunked_wavelet_octree.h>
#include "dapeng_navigation/visualization.hpp"
using namespace dapeng_navigation;
void require(bool ok,const char* reason){if(!ok)throw std::runtime_error(reason);}
template<class Map> void check() {
  typename Map::Config c;c.min_cell_width=.1;c.tree_height=3;
  auto map=std::make_shared<Map>(c);VisualizationConfig config;config.local_radius_m=.5;
  auto empty=captureVisualization(*map,0,config,0);
  require(empty.occupied.empty(),"empty native map must produce no filled space");
  map->setCellValue(wavemap::Index3D{1,1,1},2.f);
  map->setCellValue(wavemap::Index3D{12,1,1},2.f);
  map->setCellValue(wavemap::Index3D{2,1,1},-2.f);
  map->threshold();map->prune();
  const auto memory=map->getMemoryUsage();auto src=captureVisualization(*map,3,config,0);
  auto frame=renderVisualization(src,wavemap::Point3D::Zero(),config);
  require(frame.version==3 && frame.local.size()==1 && frame.blue.size()==1,"only occupied cells split into local and blue");
  require(map->getMemoryUsage()==memory,"visualization must not change native occupancy map");
  auto moved=renderVisualization(src,{1.2f,0,0},config);
  require(moved.local.size()==1&&moved.blue.size()==1&&moved.local.front().center.x()>1,"local window follows robot and old local becomes blue");
  ManagementConfig management;management.crop_radius_m=.2;
  MapManager manager(map,management);manager.tryMaintain(wavemap::Point3D::Zero(),true);
  auto cropped=captureVisualization(*map,4,config,0);auto after=renderVisualization(cropped,wavemap::Point3D::Zero(),config);
  require(after.blue.empty(),"cropped observations must disappear from active blue cloud");
  config.max_source_leaves=1;auto bounded=captureVisualization(*map,5,config,0);
  require(bounded.truncated,"bounded extraction must report truncation");
}
int main(int argc,char** argv) {
  google::InitGoogleLogging(argv[0]);
  try {
    check<wavemap::HashedWaveletOctree>();check<wavemap::HashedChunkedWaveletOctree>();
    VisualizationConfig c;
    for(int i=0;i<6;++i){auto rgb=heightColor(float(-1.+i*.4),c);require(std::abs(*std::max_element(rgb.begin(),rgb.begin()+3)-1)<1e-5,"rainbow must remain saturated");}
    VisualizationSource coarse;coarse.occupied.push_back({{0,0,0},.4f});
    auto dense=renderVisualization(coarse,{0,0,0},c);require(dense.local.size()==64,"coarse occupied leaves subdivide into regular cubes");
    c.max_local_voxels=2;auto limited=renderVisualization(coarse,{0,0,0},c);require(limited.local.size()==2&&limited.truncated,"render budget bounded and reported");
    c.voxel_size_m=.15;bool rejected=false;try{c.validate(.1);}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"unaligned display sizes must be rejected");
    std::cout<<"PASS occupied-only native extraction, real empty start, rolling local/blue split, crop removal, colors and budgets\n";
  }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
