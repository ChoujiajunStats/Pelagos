#include "dapeng_navigation/planner.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include "a_star.h"
#include "theta_star.h"
#include "multi_res_theta_star.h"

namespace dapeng_navigation {
void PlannerConfig::validate(int height) const {
  if(algorithm!="a_star" && algorithm!="theta_star" && algorithm!="multi_res_theta_star")
    throw std::invalid_argument("unknown Wavestar algorithm");
  if(!std::isfinite(refinement_epsilon)||refinement_epsilon<=0 ||
     initialization_min_height<0 || initialization_min_height>height ||
     !std::isfinite(max_result_latency_ms)||max_result_latency_ms<=0 || max_collision_cells==0)
    throw std::invalid_argument("invalid planner parameters");
}
namespace {
bool validPoint(const wavemap::Point3D& point) {
  return point.allFinite() && (point.array().abs()<100000.f).all();
}
double segmentBoxDistanceSquared(const wavemap::Point3D& from,const wavemap::Point3D& to,
                                 const wavemap::Point3D& minimum,const wavemap::Point3D& maximum) {
  const Eigen::Vector3d origin=from.cast<double>(),direction=(to-from).cast<double>();
  std::vector<double> cuts{0.,1.};
  for(int axis=0;axis<3;++axis) {
    if(std::abs(direction[axis])>1e-12) {
      for(double boundary:{double(minimum[axis]),double(maximum[axis])}) {
        const double t=(boundary-origin[axis])/direction[axis];
        if(0<t && t<1)cuts.push_back(t);
      }
    }
  }
  std::sort(cuts.begin(),cuts.end());double best=std::numeric_limits<double>::infinity();
  const auto distance=[&](double t){
    const Eigen::Vector3d p=origin+t*direction;
    return (p-p.cwiseMax(minimum.cast<double>()).cwiseMin(maximum.cast<double>())).squaredNorm();
  };
  for(size_t i=1;i<cuts.size();++i) {
    const double left=cuts[i-1],right=cuts[i],mid=(left+right)*.5;
    const Eigen::Vector3d point=origin+mid*direction;double a=0,b=0;
    for(int axis=0;axis<3;++axis) {
      if(point[axis]<minimum[axis] || maximum[axis]<point[axis]) {
        const double boundary=point[axis]<minimum[axis]?minimum[axis]:maximum[axis];
        a+=direction[axis]*direction[axis];b+=direction[axis]*(origin[axis]-boundary);
      }
    }
    const double t=a>0?std::clamp(-b/a,left,right):left;
    best=std::min({best,distance(left),distance(right),distance(t)});
  }
  return best;
}
bool collisionFreeUnlocked(const PlanningSnapshot& snapshot,
                           const std::vector<wavemap::Point3D>& path,size_t budget) {
  if(path.empty())return false;
  const float resolution=snapshot.map->getMinCellWidth(),radius=snapshot.robot_radius_m;
  for(const auto& p:path)if(!validPoint(p))return false;
  // Independently test the swept sphere against ORIGINAL occupancy voxel AABBs,
  // including unknown space and corners. The bounded ROI is already built for
  // SDF generation and retained by pointer, with no second map copy here.
  const wavemap::OccupancyClassifier classifier(snapshot.occupancy_threshold_log_odds);
  size_t checked=0;
  for(size_t segment=0;segment<std::max(size_t{1},path.size()-1);++segment) {
    const auto& a=path[segment];const auto& b=path[path.size()==1?0:segment+1];
    const wavemap::Index3D minimum=((a.cwiseMin(b).array()-radius)/resolution).floor().cast<int>();
    const wavemap::Index3D maximum=((a.cwiseMax(b).array()+radius)/resolution).floor().cast<int>();
    const double cells=(maximum.cast<double>()-minimum.cast<double>()+wavemap::Point3D::Ones().cast<double>()).prod();
    if(cells>static_cast<double>(budget-std::min(budget,checked)))return false;
    for(int x=minimum.x();x<=maximum.x();++x)for(int y=minimum.y();y<=maximum.y();++y)
      for(int z=minimum.z();z<=maximum.z();++z) {
        ++checked;const wavemap::Index3D index{x,y,z};
        if(classifier.is(sourceOccupancy(snapshot,index),wavemap::Occupancy::kFree))continue;
        const wavemap::Point3D low=index.cast<float>().array()*resolution;
        const wavemap::Point3D high=(index.cast<float>().array()+1.f)*resolution;
        if(segmentBoxDistanceSquared(a,b,low,high)<=double(radius)*radius+1e-10)return false;
      }
  }
  return true;
}
}
bool pathCollisionFree(const PlanningSnapshot& snapshot,const std::vector<wavemap::Point3D>& path,size_t max_cells) {
  if(!snapshot.map||!snapshot.occupancy)return false;
  std::lock_guard<std::mutex> lock(snapshot.queries);
  return collisionFreeUnlocked(snapshot,path,max_cells);
}
PlanResult plan(const PlanningSnapshot& snapshot,const wavemap::Point3D& start,
                const wavemap::Point3D& goal,const PlannerConfig& config) {
  PlanResult result;result.map_version=snapshot.version;
  if(!snapshot.map||!snapshot.occupancy){result.status="missing_planning_map";return result;}
  config.validate(snapshot.map->getTreeHeight());
  if(!validPoint(start)||!validPoint(goal)){result.status="invalid_position";return result;}
  const auto begin=Clock::now();std::lock_guard<std::mutex> lock(snapshot.queries);
  const float inverse=1.f/snapshot.map->getMinCellWidth();
  for(const auto& endpoint:{std::make_pair("start",start),std::make_pair("goal",goal)}) {
    auto index=wavemap::convert::pointToNearestIndex(endpoint.second,inverse);
    if(!snapshot.map->isFully(index,wavemap::Occupancy::kFree)) {
      result.status=std::string(endpoint.first)+"_not_free";return result;
    }
  }
  wavemap::PlannerBase::Ptr planner;
  if(config.algorithm=="a_star")planner=std::make_shared<wavemap::AStar>(wavemap::AStarConfig{},snapshot.map);
  else if(config.algorithm=="theta_star")planner=std::make_shared<wavemap::ThetaStar>(snapshot.map);
  else planner=std::make_shared<wavemap::MultiResThetaStar>(
      wavemap::MultiResThetaStarConfig{config.refinement_epsilon,config.initialization_min_height},snapshot.map);
  auto points=planner->searchPath(start,goal);result.planning_ms=milliseconds(Clock::now()-begin);
  if(points.empty()){result.status="unreachable";return result;}
  if(result.planning_ms>config.max_result_latency_ms){result.status="result_latency_exceeded";return result;}
  std::vector<wavemap::Point3D> candidate{start};
  for(const auto& p:points)if((candidate.back()-p).norm()>1e-5f)candidate.push_back(p);
  if((candidate.back()-goal).norm()>1e-5f)candidate.push_back(goal);
  const auto collision_start=Clock::now();
  result.collision_free=collisionFreeUnlocked(snapshot,candidate,config.max_collision_cells);
  result.collision_ms=milliseconds(Clock::now()-collision_start);
  if(!result.collision_free){result.status="path_collision_or_unknown";return result;}
  for(size_t i=1;i<candidate.size();++i)result.length_m+=(candidate[i]-candidate[i-1]).norm();
  result.path=std::move(candidate);result.status="success";return result;
}
void ResultGate::advanceMap(uint64_t version,const std::function<void()>& invalidate) {
  std::lock_guard<std::mutex> lock(mutex_);
  if(version<version_)throw std::invalid_argument("map version moved backward");
  if(version_!=version){version_=version;if(invalidate)invalidate();}
}
uint64_t ResultGate::newGoal(const std::function<void()>& invalidate) {
  std::lock_guard<std::mutex> lock(mutex_);++goal_;if(invalidate)invalidate();return goal_;
}
bool ResultGate::commit(uint64_t version,uint64_t goal,const std::function<void()>& publish) {
  std::lock_guard<std::mutex> lock(mutex_);
  if(version!=version_ || goal!=goal_)return false;
  publish();return true;
}
uint64_t ResultGate::goal() const{std::lock_guard<std::mutex> lock(mutex_);return goal_;}
}
