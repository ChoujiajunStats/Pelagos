#pragma once
#include <Eigen/Geometry>
#include <octomap/OcTree.h>
#include <cmath>
#include <stdexcept>

namespace pelagos {
enum class CellState {Free,Occupied,Unknown};
inline bool segmentIntersectsBox(const Eigen::Vector3d& a,const Eigen::Vector3d& b,
    const Eigen::Vector3d& center,const Eigen::Vector3d& half) {
  const Eigen::Vector3d delta=b-a;double enter=0.,leave=1.;
  for(int i=0;i<3;++i) {
    if(std::abs(delta[i])<1.e-12) {if(std::abs(a[i]-center[i])>half[i])return false;}
    else {double low=(center[i]-half[i]-a[i])/delta[i],high=(center[i]+half[i]-a[i])/delta[i];
      if(low>high)std::swap(low,high);enter=std::max(enter,low);leave=std::min(leave,high);}
  }
  return enter<=leave;
}

// Geometry-only fixed exception mask. It owns no occupancy observations and
// never inserts nodes into the planner's map. A key is eligible only inside
// the initial swept vehicle corridor AND outside the initial calibrated view,
// or where it already intersects the initial vehicle footprint.
class StartupBlindMask {
  octomap::OcTree grid_;
  octomap::KeySet allowed_;
  size_t self_cells_=0;
  Eigen::Vector3d low_,high_;
public:
  StartupBlindMask(double resolution,const Eigen::Vector3d& body,
      const Eigen::Vector3d& camera,const Eigen::Matrix3d& world_camera,
      const Eigen::Vector3d& direction,const Eigen::Vector3d& box,
      double corridor_length,double min_depth,double horizontal_deg,double vertical_deg):grid_(resolution) {
    if(!body.allFinite()||!camera.allFinite()||!world_camera.allFinite()||!direction.allFinite()||
       !box.allFinite()||resolution<=0||min_depth<=0||corridor_length<=0||
       horizontal_deg<=0||horizontal_deg>=180||vertical_deg<=0||vertical_deg>=180)
      throw std::runtime_error("Invalid calibrated startup geometry");
    const auto end=(body+corridor_length*direction).eval();
    low_=(body.cwiseMin(end)-box/2.).eval();high_=(body.cwiseMax(end)+box/2.).eval();
    auto p=[](const Eigen::Vector3d& v){return octomap::point3d(v.x(),v.y(),v.z());};
    octomap::OcTreeKey lo,hi,self_lo,self_hi;
    if(!grid_.coordToKeyChecked(p(low_),lo)||!grid_.coordToKeyChecked(p(high_),hi)||
       !grid_.coordToKeyChecked(p(body-box/2.),self_lo)||!grid_.coordToKeyChecked(p(body+box/2.),self_hi))
      throw std::runtime_error("Startup geometry exceeds native OctoMap key range");
    const double pi=std::acos(-1.),th=std::tan(horizontal_deg*pi/360.),tv=std::tan(vertical_deg*pi/360.);
    const Eigen::Vector3d delta=end-body,half=box/2.+Eigen::Vector3d::Constant(resolution/2.);
    for(unsigned x=lo[0];x<=hi[0];++x)for(unsigned y=lo[1];y<=hi[1];++y)for(unsigned z=lo[2];z<=hi[2];++z) {
      const octomap::OcTreeKey key(x,y,z);const auto center=grid_.keyToCoord(key);
      const Eigen::Vector3d c(center.x(),center.y(),center.z());
      bool self=true;for(int i=0;i<3;++i)self=self&&self_lo[i]<=key[i]&&key[i]<=self_hi[i];
      if(self) {allowed_.insert(key);++self_cells_;continue;}
      // Exact slab test: does this voxel intersect the swept, axis-aligned
      // source footprint? An outer AABB alone would admit diagonal side cells.
      double enter=0.,leave=1.;bool intersects=true;
      for(int i=0;i<3;++i) {
        if(std::abs(delta[i])<1.e-12) {if(std::abs(body[i]-c[i])>half[i])intersects=false;}
        else {double a=(c[i]-half[i]-body[i])/delta[i],b=(c[i]+half[i]-body[i])/delta[i];
          if(a>b)std::swap(a,b);enter=std::max(enter,a);leave=std::min(leave,b);}
      }
      if(!intersects||enter>leave)continue;
      const Eigen::Vector3d optical=world_camera.transpose()*(c-camera);
      const bool blind=optical.z()<min_depth || std::abs(optical.x())>th*optical.z() ||
          std::abs(optical.y())>tv*optical.z();
      if(blind)allowed_.insert(key);
    }
  }
  size_t cells() const {return allowed_.size();}
  size_t selfCells() const {return self_cells_;}
  double resolution() const {return grid_.getResolution();}
  bool contains(const octomap::point3d& point) const {return allowed_.count(grid_.coordToKey(point));}
  template <typename Query>
  bool traversable(const octomap::point3d& a,const octomap::point3d& b,Query query) const {
    octomap::OcTreeKey ka,kb;
    if(!grid_.coordToKeyChecked(a,ka)||!grid_.coordToKeyChecked(b,kb))return false;
    const auto check=[&](const octomap::OcTreeKey& key) {
      const auto state=query(grid_.keyToCoord(key));
      return state==CellState::Free || (state==CellState::Unknown&&allowed_.count(key));
    };
    if(!check(ka)||!check(kb))return false;
    octomap::KeyRay ray;if(!grid_.computeRayKeys(a,b,ray))return ka==kb;
    for(const auto& key:ray)if(!check(key))return false;
    return true;
  }
};
}
