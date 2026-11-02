#include <cmath>
#include <iostream>
#include <stdexcept>
#include <wavemap/core/map/hashed_wavelet_octree.h>
#include <wavemap/core/map/hashed_chunked_wavelet_octree.h>
#include "dapeng_navigation/surface_mesh.hpp"
using namespace dapeng_navigation;
void require(bool ok,const char* why){if(!ok)throw std::runtime_error(why);}
double area(const SurfaceBlock& b){double a=0;for(size_t i=0;i<b.vertices.size();i+=3)a+=(b.vertices[i+1]-b.vertices[i]).cross(b.vertices[i+2]-b.vertices[i]).norm()*.5;return a;}
template<class Map> void test(){
  typename Map::Config mc;mc.min_cell_width=.1f;mc.tree_height=3;Map m(mc);MeshConfig c;c.max_blocks_per_update=8;c.read_budget_ms=50;
  SurfaceTracker tracker;auto e=tracker.capture(m,0,wavemap::Point3D::Zero(),c,0);require(e.changed.empty()&&e.retained.empty(),"empty map must have no surface");
  // Six adjacent occupied cells form one rectangular solid, including a native block seam.
  for(int x=5;x<=10;++x)m.setCellValue(wavemap::Index3D{x,1,1},2);
  m.setCellValue(wavemap::Index3D{6,2,1},-2); // FREE and the surrounding UNKNOWN remain empty.
  auto a=tracker.capture(m,1,wavemap::Point3D::Zero(),c,0);require(a.changed.size()==2,"two native blocks changed");
  double total=0;for(const auto& b:a.changed){auto mesh=greedySurface(b,c);total+=area(mesh);tracker.acknowledge(b);
    require(mesh.vertices.size()/3==10,"greedy merge and seam culling: five rectangles per block");
    for(size_t i=0;i<mesh.vertices.size();i+=3){const auto p=(mesh.vertices[i]+mesh.vertices[i+1]+mesh.vertices[i+2])/3;
      const auto normal=(mesh.vertices[i+1]-mesh.vertices[i]).cross(mesh.vertices[i+2]-mesh.vertices[i]);
      require(normal.dot(p-wavemap::Point3D{.8f,.15f,.15f})>0,"triangle winding must point outward");}}
  require(std::abs(total-.26)<1e-5,"surface must match exact rectangular solid area, no internal faces");
  require(tracker.capture(m,2,wavemap::Point3D::Zero(),c,0).changed.empty(),"unchanged map sends no rebuilt geometry");
  // Dirty neighbor must invalidate the previously hidden seam.
  m.eraseBlock(wavemap::Index3D{1,0,0});auto cropped=tracker.capture(m,3,wavemap::Point3D::Zero(),c,0);
  require(cropped.retained.size()==1&&cropped.changed.size()==1,"crop deletes block and rebuilds neighbor boundary");
  require(greedySurface(cropped.changed[0],c).vertices.size()==36,"exposed block after crop has six rectangles");
  MeshConfig coarse=c;coarse.resolution_m=.2;
  require(greedySurface(cropped.changed[0],coarse).vertices.empty(),"coarse display must never enlarge thin occupied geometry into unknown");
  SurfaceTracker limited;MeshConfig budget=c;budget.max_source_leaves=1;
  auto partial=limited.capture(m,4,wavemap::Point3D::Zero(),budget,0);
  require(partial.budget_limited&&partial.changed.empty(),"partial block must not create false surface");
}
int main(int,char** argv){google::InitGoogleLogging(argv[0]);try{test<wavemap::HashedWaveletOctree>();test<wavemap::HashedChunkedWaveletOctree>();
  MeshConfig c;bool rejected=false;try{c.resolution_m=.15;c.validate(.1,6);}catch(const std::invalid_argument&){rejected=true;}require(rejected,"unaligned mesh rejected");
  std::cout<<"PASS HWO/HCWO exact occupied surface, greedy area, winding, seams, incremental stamps, crop, unknown and budgets\n";
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
