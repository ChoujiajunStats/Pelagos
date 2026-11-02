#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <yaml-cpp/yaml.h>
#include <wavemap/core/map/hashed_chunked_wavelet_octree.h>
#include <wavemap/io/file_conversions.h>
#include "dapeng_navigation/planner.hpp"
using namespace dapeng_navigation;
int main(int argc,char** argv) {
  google::InitGoogleLogging(argv[0]);
  if(argc!=4 && argc!=7 && argc!=10){std::cerr<<"offline_check map.wvmp navigation.yaml output_dir [start_x start_y start_z [goal_x goal_y goal_z]]\n";return 2;}
  try {
    const auto yaml=YAML::LoadFile(argv[2])["/**"]["ros__parameters"];
    ManagementConfig management;
    auto p=yaml["management"];
    management.threshold_period_s=p["threshold_period_s"].as<double>();
    management.prune_period_s=p["prune_period_s"].as<double>();
    management.crop_period_s=p["crop_period_s"].as<double>();
    management.crop_radius_m=p["crop_radius_m"].as<double>();management.validate();
    PlanningMapConfig configuration;p=yaml["planning_map"];
    configuration.robot_radius_m=p["robot_radius_m"].as<float>();
    configuration.occupancy_threshold_log_odds=p["occupancy_threshold_log_odds"].as<float>();
    const auto extent=p["half_extent_m"].as<std::vector<float>>();
    if(extent.size()!=3)throw std::invalid_argument("half_extent_m must have three components");
    configuration.half_extent_m={extent[0],extent[1],extent[2]};
    configuration.max_roi_cells=p["max_roi_cells"].as<size_t>();
    configuration.max_mirror_bytes=p["max_mirror_bytes"].as<size_t>();
    wavemap::MapBase::Ptr loaded;
    if(!wavemap::io::fileToMap(argv[1],loaded)||!loaded||loaded->empty())throw std::runtime_error("map load failed or map empty");
    wavemap::Point3D center=wavemap::Point3D::Zero();
    if(argc>=7)center={std::stof(argv[4]),std::stof(argv[5]),std::stof(argv[6])};
    else {
      bool found=false;
      loaded->forEachLeaf([&](const auto& index,float value){
        if(!found && value<-.01f){center=wavemap::convert::nodeIndexToCenterPoint(index,loaded->getMinCellWidth());found=true;}
      });
    }
    MapManager manager(loaded,management);manager.update([](auto&,auto){});
    const size_t memory_before=loaded->getMemoryUsage();
    const auto maintenance=manager.tryMaintain(center,true);
    if(!maintenance)throw std::runtime_error("unexpected maintenance deferral");
    DeltaTracker tracker(p["max_snapshot_blocks"].as<size_t>(),configuration.max_mirror_bytes);
    BlockDelta delta;manager.read([&](const auto& map,auto v){
      delta=tracker.capture(map,v,true,SnapshotRegion{center,configuration.half_extent_m});
    });
    PlanningMapBuilder builder(configuration);builder.apply(delta);auto snapshot=builder.generate(center);
    size_t compared=0;float error=0;
    manager.read([&](const auto& source,auto){source.forEachLeaf([&](const auto& index,float value){
      if(compared>=10000)return;
      auto point=wavemap::convert::nodeIndexToCenterPoint(index,source.getMinCellWidth());
      if(((point-center).array().abs() < (configuration.half_extent_m.array()-source.getMinCellWidth())).all()) {
        const auto cell=wavemap::convert::pointToNearestIndex(point,1.f/source.getMinCellWidth());
        const float exact=sourceOccupancy(*snapshot,cell);
        if(std::abs(source.getCellValue(cell)-exact)>.00001f)
          throw std::runtime_error("native block snapshot occupancy mismatch");
        error=std::max(error,std::abs(value-snapshot->occupancy->getCellValue(cell)));++compared;
      }
    });});
    if(!compared || error>.002f)throw std::runtime_error("native and planning ROI occupancy validation failed");
    std::vector<wavemap::Point3D> free_points;
    snapshot->map->forEachLeafMatching(wavemap::Occupancy::kFree,[&](const auto& index,auto){
      if(free_points.size()<100)free_points.push_back(wavemap::convert::nodeIndexToCenterPoint(index,snapshot->map->getMinCellWidth()));
    });
    PlanResult result;result.status="no_valid_free_pair";
    PlannerConfig planner_config;
    if(yaml["planner"]["algorithm"])planner_config.algorithm=yaml["planner"]["algorithm"].as<std::string>();
    wavemap::Point3D goal=wavemap::Point3D::Zero();
    if(argc==10) {
      goal={std::stof(argv[7]),std::stof(argv[8]),std::stof(argv[9])};
      result=plan(*snapshot,center,goal,planner_config);
    } else {
      for(size_t i=1;i<free_points.size();++i) {
        if((free_points[i]-free_points.front()).norm()<.5f)continue;
        result=plan(*snapshot,free_points.front(),free_points[i],planner_config);
        if(result.status=="success")break;
      }
    }
    const std::filesystem::path output(argv[3]);std::filesystem::create_directories(output);
    const auto saved=output/"maintained.wvmp";
    if(!wavemap::io::mapToFile(*loaded,saved.string()))throw std::runtime_error("maintained map save failed");
    wavemap::MapBase::Ptr reloaded;
    if(!wavemap::io::fileToMap(saved.string(),reloaded)||!reloaded)throw std::runtime_error("maintained map reload failed");
    YAML::Node report;report["map_type"]=dynamic_cast<wavemap::HashedChunkedWaveletOctree*>(loaded.get())?"hashed_chunked_wavelet_octree":"hashed_wavelet_octree";
    report["version"]=manager.version();report["robot_radius_m"]=configuration.robot_radius_m;
    for(int i=0;i<3;++i)report["center"].push_back(center[i]);
    report["memory_before_bytes"]=memory_before;report["memory_after_bytes"]=loaded->getMemoryUsage();
    report["maintenance_ms"]=maintenance->total_ms;report["removed_blocks"]=maintenance->removed_blocks;
    report["capture_ms"]=delta.capture_ms;report["copied_block_bytes"]=delta.copied_bytes;
    report["roi_ms"]=snapshot->roi_ms;report["sdf_ms"]=snapshot->sdf_ms;
    report["classification_ms"]=snapshot->classification_ms;report["roi_memory_bytes"]=snapshot->roi_memory_bytes;
    report["sdf_memory_bytes"]=snapshot->sdf_memory_bytes;report["classified_nodes"]=snapshot->classified_nodes;
    report["occupancy_samples_compared"]=compared;report["maximum_occupancy_error"]=error;
    report["planning_status"]=result.status;report["planning_ms"]=result.planning_ms;
    report["collision_ms"]=result.collision_ms;report["collision_free"]=result.collision_free;
    report["path_points"]=result.path.size();report["map_reloaded"]=true;
    report["algorithm"]=planner_config.algorithm;report["exact_input_endpoints"]=argc==10;
    if(argc==10) {
      for(int i=0;i<3;++i)report["goal"].push_back(goal[i]);
      const float inverse=1.f/loaded->getMinCellWidth();
      report["start_native_log_odds"]=loaded->getCellValue(wavemap::convert::pointToNearestIndex(center,inverse));
      report["goal_native_log_odds"]=loaded->getCellValue(wavemap::convert::pointToNearestIndex(goal,inverse));
      report["occupancy_threshold_log_odds"]=configuration.occupancy_threshold_log_odds;
      report["start_collision_free"]=pathCollisionFree(*snapshot,{center});
      report["goal_collision_free"]=pathCollisionFree(*snapshot,{goal});
    }
    std::ofstream(output/"report.yaml")<<report<<'\n';
    std::ofstream path(output/"path.csv");path<<"x,y,z\n";
    for(const auto& point:result.path)path<<point.x()<<','<<point.y()<<','<<point.z()<<'\n';
    std::cout<<report<<'\n';
  }catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}
}
