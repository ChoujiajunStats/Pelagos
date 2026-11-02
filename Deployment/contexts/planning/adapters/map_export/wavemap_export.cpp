// Read-only bounded rasterization of an official .wvmp snapshot. No map fusion.
#include <wavemap/core/map/map_base.h>
#include <wavemap/io/file_conversions.h>
#include <wavemap/core/utils/query/occupancy_classifier.h>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <vector>

int main(int argc, char** argv) {
  try {
    if (argc != 6 && argc != 7) throw std::runtime_error("usage: wavemap_export map.wvmp output.f32 x y z [official]");
    const bool official = argc == 7 && std::string(argv[6]) == "official";
    if (argc==7 && !official) throw std::runtime_error("unknown occupancy policy");
    const auto start = std::chrono::steady_clock::now();
    wavemap::MapBase::Ptr map;
    if (!wavemap::io::fileToMap(argv[1], map) || !map || map->empty())
      throw std::runtime_error("empty or unreadable Wavemap snapshot");
    const double resolution = map->getMinCellWidth();
    if (!std::isfinite(resolution) || resolution < .02 || resolution > .5)
      throw std::runtime_error("unsupported native resolution");
    // Aligned native cells: querying their centers cannot miss a finer obstacle.
    const double extent[3] = {8., 8., 3.};
    int low[3], shape[3];
    size_t count = 1;
    for (int a = 0; a < 3; ++a) {
      const double center = std::stod(argv[3+a]);
      if (!std::isfinite(center) || std::abs(center) > 1000)
        throw std::runtime_error("invalid ROI center");
      low[a] = int(std::floor((center - extent[a]/2) / resolution));
      shape[a] = int(std::ceil(extent[a] / resolution));
      count *= shape[a];
    }
    if (count > 1000000) throw std::runtime_error("ROI exceeds one million cells");
    const float free_threshold = std::log(.3f/.7f);
    const float occupied_threshold = std::log(.7f/.3f);
    const wavemap::OccupancyClassifier classifier;
    std::vector<float> cells;
    cells.reserve(count);
    size_t free = 0, occupied = 0;
    for (int x = 0; x < shape[0]; ++x)
      for (int y = 0; y < shape[1]; ++y)
        for (int z = 0; z < shape[2]; ++z) {
          const float value = map->getCellValue(wavemap::Index3D(low[0]+x, low[1]+y, low[2]+z));
          float output = std::numeric_limits<float>::quiet_NaN();
          if (std::isfinite(value) && (official ? classifier.is(value,wavemap::Occupancy::kFree) : value < free_threshold)) { output = 1.f; ++free; }
          else if (std::isfinite(value) && (official ? classifier.is(value,wavemap::Occupancy::kOccupied) : value > occupied_threshold)) { output = 0.f; ++occupied; }
          cells.push_back(output);
        }
    std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
    output.write(reinterpret_cast<const char*>(cells.data()), cells.size()*sizeof(float));
    output.close();
    if (!output) throw std::runtime_error("ROI write failed");
    const double ms = std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    std::cout << std::setprecision(12) << "{\"origin\":[" << low[0]*resolution << ','
              << low[1]*resolution << ',' << low[2]*resolution << "],\"shape\":["
              << shape[0] << ',' << shape[1] << ',' << shape[2] << "],\"resolution\":"
              << resolution << ",\"free_cells\":" << free << ",\"occupied_cells\":" << occupied
              << ",\"unknown_cells\":" << count-free-occupied << ",\"export_ms\":" << ms
              << ",\"native_memory_bytes\":" << map->getMemoryUsage() << "}\n";
  } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
