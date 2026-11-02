#include <iostream>
#include <stdexcept>
#include <thread>
#include <wavemap/core/map/hashed_chunked_wavelet_octree.h>
#include "dapeng_navigation/planning_map.hpp"
using namespace dapeng_navigation;
void require(bool value,const char* why){if(!value)throw std::runtime_error(why);}
template<class Map> void check() {
  typename Map::Config config;config.min_cell_width=.2f;config.tree_height=4;
  auto source=std::make_shared<Map>(config);MapManager manager(source,{});
  manager.update([](auto& map,auto){
    for(int x=-9;x<=9;++x)for(int y=-9;y<=9;++y)for(int z=-5;z<=5;++z)
      map.addToCellValue(wavemap::Index3D{x,y,z},-1.f);
    map.addToCellValue(wavemap::Index3D{5,0,0},3.f);
    map.addToCellValue(wavemap::Index3D{200,0,0},2.f);
  });
  DeltaTracker tracker;BlockDelta first;
  manager.read([&](const auto& map,auto version){first=tracker.capture(map,version);});
  PlanningMapConfig cfg;cfg.robot_radius_m=.15f;cfg.half_extent_m={2.f,2.f,1.2f};
  DeltaTracker bounded;BlockDelta local;
  manager.read([&](const auto& map,auto version){local=bounded.capture(map,version,true,
      SnapshotRegion{wavemap::Point3D::Zero(),cfg.half_extent_m});});
  require(local.retained.size()<first.retained.size(),"planning snapshots must not copy the whole native map");
  PlanningMapBuilder builder(cfg);builder.apply(first);auto old=builder.generate(wavemap::Point3D::Zero());
  require(old->version==1,"snapshot must carry the native map version");
  require(occupancyAt(*old,{0.1f,0.1f,0.1f})==CellState::Free,"observed free cell");
  require(occupancyAt(*old,{1.1f,0.1f,0.1f})==CellState::Occupied,"obstacle classification");
  require(occupancyAt(*old,{.9f,.1f,.1f})==CellState::Occupied,"robot-radius inflation");
  require(occupancyAt(*old,{4.1f,.1f,.1f})==CellState::Unknown,"outside planning ROI must remain unknown");
  manager.update([](auto& map,auto){map.addToCellValue(wavemap::Index3D{0,0,0},3.f);});
  BlockDelta second;manager.read([&](const auto& map,auto v){second=tracker.capture(map,v);});
  require(second.changed.size()==1,"unchanged blocks must not be copied");
  builder.apply(second);auto current=builder.generate(wavemap::Point3D::Zero());
  require(occupancyAt(*current,{.1f,.1f,.1f})==CellState::Occupied,"new version must include new obstacle");
  require(occupancyAt(*old,{.1f,.1f,.1f})==CellState::Free,"old snapshot must be immutable");
  std::thread a([&]{for(int i=0;i<10000;++i)require(occupancyAt(*old,{.1f,.1f,.1f})==CellState::Free,"concurrent free query");});
  std::thread b([&]{for(int i=0;i<10000;++i)require(occupancyAt(*old,{1.1f,.1f,.1f})==CellState::Occupied,"concurrent obstacle query");});
  a.join();b.join();
  auto merged=mergeDeltas(first,second);PlanningMapBuilder lagged(cfg);lagged.apply(merged);
  require(occupancyAt(*lagged.generate(wavemap::Point3D::Zero()),{.1f,.1f,.1f})==CellState::Occupied,"coalesced updates must preserve dirty blocks");
  manager.tryMaintain(wavemap::Point3D{20,0,0},true);
  BlockDelta cropped;manager.read([&](const auto& map,auto v){cropped=tracker.capture(map,v,true);});
  builder.apply(cropped);auto empty=builder.generate(wavemap::Point3D{20,0,0});
  require(occupancyAt(*empty,{.1f,.1f,.1f})==CellState::Unknown,"cropped source must disappear from planning map");
  bool rejected=false;try{PlanningMapConfig bad;bad.max_roi_cells=1;bad.validate(.1f);}catch(const std::invalid_argument&){rejected=true;}
  require(rejected,"oversized ROI must fail before allocation");
  std::cout << "planning_map PASS: snapshot_ms=" << first.capture_ms << ", sdf_ms=" << old->sdf_ms
            << ", classified_ms=" << old->classification_ms << ", roi_bytes=" << old->roi_memory_bytes << '\n';
}
int main(int argc,char** argv) {
  google::InitGoogleLogging(argv[0]);
  try{check<wavemap::HashedWaveletOctree>();check<wavemap::HashedChunkedWaveletOctree>();}
  catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
