#pragma once
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace pelagos {
struct ForwardObstacleContact {
  bool found=false;
  double front_clearance_m=std::numeric_limits<double>::infinity();
  Eigen::Vector3d center=Eigen::Vector3d::Zero();
  double size_m=0.;
};

// Ray of body centers against a voxel expanded by the source body half extents.
// Entry is the translation until first physical-footprint contact, already a
// front/bumper clearance. Do not subtract the body half length a second time.
inline bool forwardVoxelClearance(const Eigen::Vector3d& body,
    const Eigen::Vector3d& direction,const Eigen::Vector3d& body_box,
    const Eigen::Vector3d& voxel_center,double voxel_size,double* clearance) {
  if(!body.allFinite()||!direction.allFinite()||direction.squaredNorm()==0.||
     !body_box.allFinite()||(body_box.array()<=0.).any()||
     !voxel_center.allFinite()||!std::isfinite(voxel_size)||voxel_size<=0.)return false;
  const Eigen::Vector3d ray=direction.normalized();
  const Eigen::Vector3d half=body_box/2.+Eigen::Vector3d::Constant(voxel_size/2.);
  double enter=0.,leave=std::numeric_limits<double>::infinity();
  for(int i=0;i<3;++i) {
    const double low=voxel_center[i]-half[i],high=voxel_center[i]+half[i];
    if(std::abs(ray[i])<1.e-12) {
      if(body[i]<low||high<body[i])return false;
    } else {
      double a=(low-body[i])/ray[i],b=(high-body[i])/ray[i];
      if(a>b)std::swap(a,b);
      enter=std::max(enter,a);leave=std::min(leave,b);
      if(enter>leave)return false;
    }
  }
  if(clearance)*clearance=enter;
  return true;
}

inline ForwardObstacleContact nearestForwardObstacle(const Eigen::Vector3d& body,
    const Eigen::Vector3d& direction,const Eigen::Vector3d& body_box,
    const std::vector<std::pair<Eigen::Vector3d,double>>& occupied) {
  ForwardObstacleContact result;
  for(const auto& cell:occupied) {
    double clearance;
    if(forwardVoxelClearance(body,direction,body_box,cell.first,cell.second,&clearance)&&
       clearance<result.front_clearance_m) {
      result.found=true;result.front_clearance_m=clearance;
      result.center=cell.first;result.size_m=cell.second;
    }
  }
  return result;
}
}
