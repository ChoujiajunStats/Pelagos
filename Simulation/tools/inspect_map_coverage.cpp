// Read-only comparison of saved Wavemap projections at source/planner resolution.
// No planner, sensor integration or motion interfaces are used.
#include <octomap/OcTree.h>
#include <array>
#include <cmath>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

int main(int argc,char** argv) {
  if(argc!=7) {std::cerr<<"Usage: inspect_map_coverage COARSE.ot FINE.ot X Y Z YAW\n";return 2;}
  try {
    std::unique_ptr<octomap::AbstractOcTree> coarse_map(octomap::AbstractOcTree::read(argv[1]));
    std::unique_ptr<octomap::AbstractOcTree> fine_map(octomap::AbstractOcTree::read(argv[2]));
    auto* coarse_tree=dynamic_cast<octomap::OcTree*>(coarse_map.get());
    auto* fine_tree=dynamic_cast<octomap::OcTree*>(fine_map.get());
    if(!coarse_tree||!fine_tree)throw std::runtime_error("Failed to load full OcTree files");
    const auto& coarse=*coarse_tree;const auto& fine=*fine_tree;
    using P=octomap::point3d;
    P body(std::stod(argv[3]),std::stod(argv[4]),std::stod(argv[5]));
    double yaw=std::stod(argv[6]);
    if(std::abs(coarse.getResolution()-.4)>1.e-6||std::abs(fine.getResolution()-.1)>1.e-6)
      throw std::runtime_error("Expected unchanged planner 0.4 m and source 0.1 m grids");
    auto status=[](const octomap::OcTree& tree,const P& p) {
      auto* node=tree.search(p);return !node?"unknown":tree.isNodeOccupied(node)?"occupied":"free";
    };
    auto point=[](const P& p){std::cout<<'['<<p.x()<<','<<p.y()<<','<<p.z()<<']';};
    auto describe=[&](const P& p) {
      const auto center=coarse.keyToCoord(coarse.coordToKey(p));
      std::array<int,3> counts{};
      for(int x=0;x<4;++x)for(int y=0;y<4;++y)for(int z=0;z<4;++z) {
        auto* node=fine.search(center+P((x-1.5)*.1,(y-1.5)*.1,(z-1.5)*.1));
        ++counts[!node?2:fine.isNodeOccupied(node)?1:0];
      }
      std::cout<<"{\"point\":";point(p);std::cout<<",\"coarse_status\":\""<<status(coarse,p)
        <<"\",\"fine_point_status\":\""<<status(fine,p)<<"\",\"coarse_center\":";point(center);
      std::cout<<",\"fine_cell_counts\":{\"free\":"<<counts[0]<<",\"occupied\":"<<counts[1]
        <<",\"unknown\":"<<counts[2]<<"}}";
    };
    std::vector<P> offsets;
    for(int x=0;x<3;++x)for(int y=0;y<3;++y)for(int z=0;z<2;++z)
      offsets.emplace_back(-.25+x*.25,-.25+y*.25,-.15+z*.3);
    std::cout<<"{\"execution_authorized\":false,\"body\":";point(body);
    std::cout<<",\"footprint\":[";
    for(size_t i=0;i<offsets.size();++i) {if(i)std::cout<<',';describe(body+offsets[i]);}
    auto low=coarse.coordToKey(body+P(-.25,-.25,-.15)),high=coarse.coordToKey(body+P(.25,.25,.15));
    auto in_self=[&](const octomap::OcTreeKey& key) {
      for(int k=0;k<3;++k)if(key[k]<low[k]||high[k]<key[k])return false;
      return true;
    };
    std::cout<<"],\"first_nonfree_forward_cells_outside_initial_self\":[";
    bool comma=false;
    for(const auto& offset:offsets) {
      const auto a=body+offset,b=a+P(1.5*std::cos(yaw),1.5*std::sin(yaw),0.);
      octomap::KeyRay ray;coarse.computeRayKeys(a,b,ray);
      std::vector<octomap::OcTreeKey> keys(ray.begin(),ray.end());keys.push_back(coarse.coordToKey(b));
      for(const auto& key:keys) {
        if(in_self(key))continue;
        const auto center=coarse.keyToCoord(key);auto* node=coarse.search(key);
        if(!node||coarse.isNodeOccupied(node)) {
          if(comma)std::cout<<',';comma=true;describe(center);break;
        }
      }
    }
    std::cout<<"]}\n";return 0;
  }catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
