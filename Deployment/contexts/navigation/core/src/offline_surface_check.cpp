#include <iostream>
#include <wavemap/io/file_conversions.h>
#include "dapeng_navigation/surface_mesh.hpp"
int main(int argc,char** argv){google::InitGoogleLogging(argv[0]);if(argc!=5){std::cerr<<"usage: offline_surface_check map.wvmp x y z\n";return 2;}
 try{wavemap::MapBase::Ptr map;if(!wavemap::io::fileToMap(argv[1],map)||!map)throw std::runtime_error("map load failed");
  dapeng_navigation::MeshConfig c;c.resolution_m=map->getMinCellWidth();c.max_blocks_per_update=8;c.read_budget_ms=50;c.max_source_leaves=1000000;
  dapeng_navigation::SurfaceTracker tracker;wavemap::Point3D center{float(std::stod(argv[2])),float(std::stod(argv[3])),float(std::stod(argv[4]))};
  size_t blocks=0,triangles=0;double read_ms=0,build_ms=0;bool completed=false;
  for(int round=0;round<128;++round){const auto source=tracker.capture(*map,1,center,c,0);read_ms+=source.read_ms;
   for(const auto& b:source.changed){const auto mesh=dapeng_navigation::greedySurface(b,c);triangles+=mesh.vertices.size()/3;build_ms+=mesh.build_ms;++blocks;tracker.acknowledge(b);}
   if(source.pending==0){completed=true;break;}}
  std::cout<<"{\"completed\":"<<(completed?"true":"false")<<",\"blocks\":"<<blocks<<",\"triangles\":"<<triangles<<",\"read_ms\":"<<read_ms<<",\"build_ms\":"<<build_ms<<"}\n";
  return completed?0:1;
 }catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
