#include <iostream>
#include <stdexcept>
#include "dapeng_navigation/planner.hpp"
using namespace dapeng_navigation;
void require(bool v,const char* reason){if(!v)throw std::runtime_error(reason);}
int main(int argc,char** argv) {
  google::InitGoogleLogging(argv[0]);
  try {
    wavemap::HashedWaveletOctreeConfig mc;mc.min_cell_width=.2f;mc.tree_height=4;
    auto native=std::make_shared<wavemap::HashedWaveletOctree>(mc);MapManager manager(native,{});
    manager.update([](auto& map,auto){
      for(int x=-12;x<=12;++x)for(int y=-8;y<=8;++y)for(int z=-6;z<=6;++z)
        map.addToCellValue(wavemap::Index3D{x,y,z},-1.f);
      for(int y=-8;y<=8;++y)for(int z=-6;z<=0;++z)
        map.addToCellValue(wavemap::Index3D{0,y,z},3.f);
    });
    DeltaTracker tracker;PlanningMapConfig map_cfg;map_cfg.robot_radius_m=.1f;
    PlanningMapBuilder builder(map_cfg);BlockDelta delta;
    manager.read([&](const auto& map,auto v){delta=tracker.capture(map,v);});
    builder.apply(delta);auto snapshot=builder.generate(wavemap::Point3D::Zero());
    const wavemap::Point3D start{-1.5f,-.5f,-.5f},goal{1.5f,-.5f,-.5f};
    for(const auto& algorithm:{"multi_res_theta_star","theta_star","a_star"}) {
      PlannerConfig cfg;cfg.algorithm=algorithm;
      const auto result=plan(*snapshot,start,goal,cfg);
      std::cout << algorithm << ": " << result.status << ", points=" << result.path.size()
                << ", planning_ms=" << result.planning_ms << ", collision_ms=" << result.collision_ms << '\n';
      require(result.status=="success" && result.collision_free,"3D wall-overflight path must succeed");
      require((result.path.front()-start).norm()<1e-5 && (result.path.back()-goal).norm()<1e-5,"include exact endpoints");
      bool climbed=false;for(const auto& p:result.path)climbed|=p.z()>.3f;
      require(climbed,"planner must use the third dimension");
      require(pathCollisionFree(*snapshot,result.path),"independent continuous collision recheck");
      require(!pathCollisionFree(*snapshot,{start,goal}),"straight path through wall must fail");
      require(plan(*snapshot,start,{5.f,0,0},cfg).path.empty(),"unknown goal must fail without a path");
    }
    manager.update([](auto& map,auto){
      for(int y=-8;y<=8;++y)for(int z=1;z<=6;++z)
        map.addToCellValue(wavemap::Index3D{0,y,z},3.f);
    });
    manager.read([&](const auto& map,auto v){delta=tracker.capture(map,v);});builder.apply(delta);
    auto blocked=builder.generate(wavemap::Point3D::Zero());
    for(const auto& algorithm:{"multi_res_theta_star","theta_star","a_star"}) {
      PlannerConfig cfg;cfg.algorithm=algorithm;auto result=plan(*blocked,start,goal,cfg);
      require(result.status=="unreachable" && result.path.empty(),"closed wall must fail explicitly");
    }
    require(!pathCollisionFree(*snapshot,{{-1.5f,1.75f,.1f},{-1.1f,1.75f,.1f}}),"body footprint into unknown must fail");
    ResultGate gate;int published=0;gate.advanceMap(1);auto old_goal=gate.newGoal();
    gate.advanceMap(2);require(!gate.commit(1,old_goal,[&]{++published;}),"old-map result must not publish");
    auto current_goal=gate.newGoal();require(!gate.commit(2,old_goal,[&]{++published;}),"old-goal result must not publish");
    require(gate.commit(2,current_goal,[&]{++published;}) && published==1,"current result may publish");
    PlanningSnapshot missing;
    require(plan(missing,start,goal,{}).status=="missing_planning_map","missing map must fail explicitly");
    require(!pathCollisionFree(missing,{start,goal}),"missing map must never pass collision checking");
  } catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}
}
