#pragma once
#include <Eigen/Core>
#include <cmath>
#include <vector>
#include "forward_obstacle.hpp"

namespace pelagos {
struct SourceCruiseDirection {
  bool valid=false;
  const char* reason="no_cached_source_edge";
  Eigen::Vector3d direction=Eigen::Vector3d::Zero();
};

struct CruiseDirectionClearance {
  bool valid=false;
  bool direction_valid=false;
  Eigen::Vector3d direction=Eigen::Vector3d::Zero();
  ForwardObstacleContact contact;
};

// Test the horizontal direction that will actually be executed, at the actual
// current body altitude, rather than the source branch's possible 3D slope.
// No known hit means only that: unknown space is not asserted to be free.
inline CruiseDirectionClearance cruiseDirectionClearance(
    const Eigen::Vector3d& body,const Eigen::Vector3d& displacement,
    const Eigen::Vector3d& body_box,
    const std::vector<std::pair<Eigen::Vector3d,double>>& occupied,double minimum_clearance) {
  CruiseDirectionClearance result;
  const double length=std::hypot(displacement.x(),displacement.y());
  if(!body.allFinite()||!displacement.allFinite()||!body_box.allFinite()||
      (body_box.array()<=0.).any()||!std::isfinite(length)||length==0.||
      !std::isfinite(minimum_clearance)||minimum_clearance<=0.)return result;
  result.direction={displacement.x()/length,displacement.y()/length,0.};
  result.direction_valid=true;
  result.contact=nearestForwardObstacle(body,result.direction,body_box,occupied);
  result.valid=!result.contact.found||result.contact.front_clearance_m>minimum_clearance;
  return result;
}

// Geometry must be the actual last accepted source edge, transported without
// changing XYZ. ROS doubles round-trip exactly through the existing JSON link.
// View yaw is deliberately not part of this heading-only task constraint.
inline SourceCruiseDirection sourceEdgeCruiseDirection(
    const std::vector<Eigen::Vector3d>& source,
    const std::vector<Eigen::Vector3d>& requested) {
  SourceCruiseDirection result;
  if(source.size()<2)return result;
  result.reason="source_edge_geometry_mismatch";
  if(requested.size()!=source.size())return result;
  for(size_t i=0;i<source.size();++i)
    if(!source[i].allFinite()||!requested[i].allFinite()||
       !(source[i].array()==requested[i].array()).all())return result;
  const Eigen::Vector3d displacement=source.back()-source.front();
  const double length=std::hypot(displacement.x(),displacement.y());
  result.reason="source_edge_has_no_horizontal_direction";
  if(!displacement.allFinite()||!std::isfinite(length)||length==0.)return result;
  result.direction={displacement.x()/length,displacement.y()/length,0.};
  result.valid=true;result.reason="cruise_direction_set_from_source_edge";
  return result;
}
}
