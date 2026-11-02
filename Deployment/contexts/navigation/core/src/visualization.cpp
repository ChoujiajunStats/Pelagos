#include "dapeng_navigation/visualization.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <wavemap/core/indexing/index_conversions.h>
#include <wavemap/core/utils/query/occupancy_classifier.h>

namespace dapeng_navigation {
void VisualizationConfig::validate(float resolution) const {
  if(!std::isfinite(resolution)||resolution<=0 || !std::isfinite(local_radius_m)||local_radius_m<=0 ||
     !std::isfinite(voxel_size_m)||voxel_size_m<resolution*(1.-1e-5) ||
     !std::isfinite(height_min_m)||!std::isfinite(height_max_m)||height_min_m>=height_max_m ||
     !std::isfinite(local_alpha)||local_alpha<=0||local_alpha>1 ||
     !std::isfinite(blue_alpha)||blue_alpha<=0||blue_alpha>1 ||
     !max_source_leaves||!max_local_voxels||!max_blue_points||!max_candidates ||
     !std::isfinite(read_budget_ms)||read_budget_ms<=0)
    throw std::invalid_argument("invalid incremental visualization configuration");
  const double exponent=std::log2(voxel_size_m/resolution);
  if(std::abs(exponent-std::round(exponent))>1e-4 || exponent>16)
    throw std::invalid_argument("display voxel size must equal native resolution times a power of two");
}
std::array<float,4> heightColor(float height,const VisualizationConfig& config) {
  static constexpr float stops[6][3]={{1,0,0},{1,1,0},{0,1,0},{0,1,1},{0,0,1},{1,0,1}};
  double t=std::clamp((height-config.height_min_m)/(config.height_max_m-config.height_min_m),0.,1.);
  if(config.reverse_colors)t=1.-t;
  const double f=t*5.;const int a=std::min(4,int(f));const float u=float(f-a);
  return {(1-u)*stops[a][0]+u*stops[a+1][0],(1-u)*stops[a][1]+u*stops[a+1][1],
          (1-u)*stops[a][2]+u*stops[a+1][2],float(config.local_alpha)};
}
VisualizationSource captureVisualization(const wavemap::MapBase& map,uint64_t version,
                                         const VisualizationConfig& config,float threshold) {
  config.validate(map.getMinCellWidth());
  if(!std::isfinite(threshold))throw std::invalid_argument("nonfinite occupancy threshold");
  const auto start=Clock::now();VisualizationSource output;
  output.version=version;output.resolution=map.getMinCellWidth();
  const wavemap::OccupancyClassifier classifier(threshold);
  struct BudgetReached {};
  try {
    map.forEachLeaf([&](const auto& index,float value){
      if(++output.visited_leaves>config.max_source_leaves ||
         ((output.visited_leaves%256)==0 && milliseconds(Clock::now()-start)>config.read_budget_ms))throw BudgetReached{};
      if(!classifier.is(value,wavemap::Occupancy::kOccupied))return;
      const wavemap::Point3D minimum=wavemap::convert::nodeIndexToMinCornerIndex(index).template cast<float>()*output.resolution;
      output.occupied.push_back({minimum,float(1<<index.height)*output.resolution});
    });
  }catch(const BudgetReached&){output.truncated=true;}
  output.capture_ms=milliseconds(Clock::now()-start);return output;
}
VisualizationFrame renderVisualization(const VisualizationSource& source,const wavemap::Point3D& robot,
                                      const VisualizationConfig& config) {
  config.validate(source.resolution);
  if(!robot.allFinite())throw std::invalid_argument("nonfinite visualization center");
  const auto start=Clock::now();VisualizationFrame result;
  result.version=source.version;result.truncated=source.truncated;
  size_t candidates=0;bool exhausted=false;
  for(const auto& leaf:source.occupied) {
    const float size=std::min(float(config.voxel_size_m),leaf.width);
    const int count=std::max(1,int(std::lround(leaf.width/size)));
    for(int x=0;x<count&&!exhausted;++x)for(int y=0;y<count&&!exhausted;++y)for(int z=0;z<count;++z) {
      if(++candidates>config.max_candidates){result.truncated=true;exhausted=true;break;}
      const wavemap::Point3D p=leaf.minimum+wavemap::Point3D{(x+.5f)*size,(y+.5f)*size,(z+.5f)*size};
      if((p-robot).squaredNorm()<=config.local_radius_m*config.local_radius_m) {
        if(result.local.size()>=config.max_local_voxels){result.truncated=true;continue;}
        const float height=p.z()-(config.height_relative_to_robot?robot.z():0.f);
        result.local.push_back({p,size,heightColor(height,config)});
      } else {
        if(result.blue.size()>=config.max_blue_points){result.truncated=true;continue;}
        result.blue.push_back(p);
      }
    }
    if(exhausted)break;
  }
  result.build_ms=milliseconds(Clock::now()-start);return result;
}
}  // namespace dapeng_navigation
