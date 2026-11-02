#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <wavemap/io/file_conversions.h>
#include "underwater_wavemap_ros/octomap_projection.hpp"

int main(int argc,char** argv) {
  if(argc!=4) {std::cerr<<"Usage: wavemap_to_octomap INPUT.wvmp OUTPUT.ot RESOLUTION\n";return 2;}
  try {
    wavemap::MapBase::Ptr source;
    if(!wavemap::io::fileToMap(argv[1],source)||!source)throw std::runtime_error("Wavemap load failed");
    const auto started=std::chrono::steady_clock::now();
    auto output=underwater_wavemap_ros::projectOctomap(*source,std::stod(argv[3]));
    if(!output.tree->write(argv[2]))throw std::runtime_error("Full OctoMap write failed");
    const auto& s=output.stats;
    std::cout<<"{\"observed_source_leaves\":"<<s.observed_leaves<<",\"free_cells\":"<<s.free_cells
      <<",\"occupied_cells\":"<<s.occupied_cells<<",\"mixed_unknown_cells\":"<<s.mixed_unknown_cells
      <<",\"conversion_ms\":"<<std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count()<<"}\n";
    return 0;
  } catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
