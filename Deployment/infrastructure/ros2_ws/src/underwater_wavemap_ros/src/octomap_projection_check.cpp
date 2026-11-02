#include <iostream>
#include <stdexcept>
#include <sstream>
#include <wavemap/io/file_conversions.h>
#include <wavemap/io/stream_conversions.h>
#include <wavemap/core/map/hashed_chunked_wavelet_octree.h>
#include "underwater_wavemap_ros/octomap_projection.hpp"
#include "underwater_wavemap_ros/chunked_restore.hpp"

void require(bool value,const char* text) {if(!value)throw std::runtime_error(text);}
int main(int argc,char** argv) {
  try {
    wavemap::HashedChunkedWaveletOctreeConfig config;config.min_cell_width=.1f;config.tree_height=6;
    wavemap::HashedChunkedWaveletOctree source(config);
    // Build exact fixtures with the additive API used by Wavemap integration.
    // This checkout's setCellValue subtracts the parent scale coefficient.
    auto assign=[&](const wavemap::Index3D& index,float value) {
      source.addToCellValue(index,value-source.getCellValue(index));
      require(std::abs(source.getCellValue(index)-value)<1.e-5f,"Fixture assignment must reconstruct requested occupancy");
    };
    auto fill=[&](int begin,float value) {
      for(int x=begin;x<begin+4;++x)for(int y=begin;y<begin+4;++y)for(int z=begin;z<begin+4;++z)
        assign(wavemap::Index3D(x,y,z),value);
    };
    fill(0,-.8f);
    auto full=underwater_wavemap_ros::projectOctomap(source,.4);
    auto native=underwater_wavemap_ros::projectOctomap(source,.1);
    require(native.tree->search(.05,.05,.05),"Native float32 Wavemap resolution must accept decimal 0.1 m");
    underwater_wavemap_ros::OccupancySnapshot snapshot;snapshot.min_cell_width=source.getMinCellWidth();
    source.forEachLeaf([&](const auto& index,float value){snapshot.leaves.push_back({index,value});});
    auto copied=underwater_wavemap_ros::projectOctomap(snapshot,.4);
    require(copied.stats.free_cells==full.stats.free_cells&&copied.stats.occupied_cells==full.stats.occupied_cells,
      "Immutable worker snapshot must preserve source projection");
    auto* cell=full.tree->search(.2,.2,.2);
    require(cell&&!full.tree->isNodeOccupied(cell),"Full free coverage must be free");
    require(!full.tree->search(.6,.2,.2),"Unobserved neighbor must remain unknown");
    assign(wavemap::Index3D(3,3,3),0.f);
    auto partial=underwater_wavemap_ros::projectOctomap(source,.4);
    require(!partial.tree->search(.2,.2,.2),"63/64 free plus unknown must not become free");
    assign(wavemap::Index3D(0,0,0),1.2f);
    auto obstacle=underwater_wavemap_ros::projectOctomap(source,.4);
    cell=obstacle.tree->search(.2,.2,.2);
    require(cell&&obstacle.tree->isNodeOccupied(cell),"Thin occupied subcell must dominate free and unknown");
    fill(-4,-.7f);source.prune();
    auto negative=underwater_wavemap_ros::projectOctomap(source,.4);
    cell=negative.tree->search(-.2,-.2,-.2);
    require(cell&&!negative.tree->isNodeOccupied(cell),"Negative grid coordinates must use floor division");
    require(wavemap::OccupancyClassifier::isUnobserved(source.getCellValue(wavemap::Index3D(3,3,3))),"Projection must not change source unknown state");
    wavemap::MapBase::Ptr saved;
    if(argc==2) {
      require(wavemap::io::fileToMap(argv[1],saved),"Real map reload must succeed");
    } else {
      std::stringstream stream;require(wavemap::io::mapToStream(source,stream),"Fixture save failed");
      require(wavemap::io::streamToMap(stream,saved),"Fixture reload failed");
    }
    auto restored=underwater_wavemap_ros::restoreChunked(*saved);
    auto before=underwater_wavemap_ros::projectOctomap(*saved,.1);
    auto after=underwater_wavemap_ros::projectOctomap(*restored,.1);
    std::ostringstream a,b;before.tree->writeData(a);after.tree->writeData(b);
    require(a.str()==b.str(),"Restoring chunked fusion storage must preserve every projected probability and unknown cell");
    std::cout<<"PASS: free/unknown/occupied, negative coordinates, native float resolution, immutable snapshot, coefficient restore roundtrip\n";
    return 0;
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
