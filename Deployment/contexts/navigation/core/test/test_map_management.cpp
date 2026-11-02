#include <future>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <thread>
#include <wavemap/core/map/hashed_chunked_wavelet_octree.h>
#include <wavemap/core/map/hashed_wavelet_octree.h>
#include <wavemap/core/utils/query/occupancy_classifier.h>
#include "dapeng_navigation/map_manager.hpp"

void require(bool condition, const char* message) {
  if (!condition) throw std::runtime_error(message);
}
template<class Map> void check() {
  typename Map::Config config; config.min_cell_width = .1f; config.tree_height = 3;
  auto native = std::make_shared<Map>(config);
  dapeng_navigation::ManagementConfig management; management.crop_radius_m = 1.;
  dapeng_navigation::MapManager manager(native, management);
  require(manager.identity() == native.get(), "manager must own the original map");
  manager.update([](auto& map, auto) {
    map.setCellValue(wavemap::Index3D{1,1,1},100.f);
    map.setCellValue(wavemap::Index3D{12,1,1},-100.f);
    map.setCellValue(wavemap::Index3D{200,1,1},4.f);
  });
  const auto memory = native->getMemoryUsage();
  auto result = manager.tryMaintain(wavemap::Point3D::Zero(),true);
  require(result.has_value() && result->removed_blocks == 1,"only far blocks must be removed");
  require(native->getCellValue(wavemap::Index3D{1,1,1}) <= config.max_log_odds + 1e-4f,"threshold occupied value");
  require(native->getCellValue(wavemap::Index3D{12,1,1}) >= config.min_log_odds - 1e-4f,"keep intersecting block, clamp free value");
  require(wavemap::OccupancyClassifier::isUnobserved(native->getCellValue(wavemap::Index3D{200,1,1})),"removed cells must become unknown");
  require(native->getMemoryUsage() < memory,"crop must free native memory");
  auto version = manager.version();
  manager.tryMaintain(wavemap::Point3D::Zero());
  require(manager.version() == version,"no version change for a throttled no-op");
  manager.tryMaintain(wavemap::Point3D{20.f,0.f,0.f},true);
  require(wavemap::OccupancyClassifier::isUnobserved(native->getCellValue(wavemap::Index3D{1,1,1})),"rolling crop must forget old space");
  manager.update([](auto& map,auto){map.setCellValue(wavemap::Index3D{200,1,1},-1.f);});
  require(native->getCellValue(wavemap::Index3D{200,1,1}) < 0,"revisited space can be observed again");
  std::promise<void> entered,release; auto released=release.get_future();
  std::thread fusion([&]{manager.update([&](auto&,auto){entered.set_value();released.wait();});});
  entered.get_future().wait();
  const auto now=dapeng_navigation::Clock::now();
  const auto deferred=manager.tryMaintain(wavemap::Point3D::Zero(),true);
  const auto elapsed=dapeng_navigation::milliseconds(dapeng_navigation::Clock::now()-now);
  release.set_value();fusion.join();
  require(!deferred && elapsed < 50,"maintenance must defer without waiting on fusion");
  bool rejected=false;
  try{manager.tryMaintain(wavemap::Point3D{std::numeric_limits<float>::quiet_NaN(),0,0});}
  catch(const std::invalid_argument&){rejected=true;}
  require(rejected,"invalid pose must be rejected");
  std::cout << "map_operations_and_rolling_map PASS: memory " << memory << " -> " << result->memory_after_bytes
            << ", maintenance_ms " << result->total_ms << ", version " << manager.version() << '\n';
}
int main(int argc,char** argv) {
  google::InitGoogleLogging(argv[0]);
  try {
    check<wavemap::HashedWaveletOctree>();
    check<wavemap::HashedChunkedWaveletOctree>();
    dapeng_navigation::ManagementConfig invalid; invalid.crop_radius_m=-1;
    bool rejected=false;try{invalid.validate();}catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"invalid YAML/config values must be rejected before official CHECKs");
    rejected=false;
    try{dapeng_navigation::MapManager null_manager(nullptr,{});}
    catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"null maps must fail with an exception before official CHECKs");
    rejected=false;
    try{dapeng_navigation::CropMapOperation null_crop(nullptr,1.,1.);}
    catch(const std::invalid_argument&){rejected=true;}
    require(rejected,"standalone crop must reject a null map without aborting");
  } catch(const std::exception& e){std::cerr << e.what() << '\n';return 1;}
  return 0;
}
