// Transport and independent execution checks around the original NBV algorithm.
#include <ros/ros.h>
#include <nbvplanner/nbvp.h>
#include <nbvplanner/nbvp.hpp>
#include <pelagos_nbv/Step.h>
#include <cmath>
#include <cstdlib>
#include <stdexcept>
#include <boost/make_shared.hpp>
#include <boost/property_tree/ptree.hpp>
#include <boost/property_tree/json_parser.hpp>
#include <sstream>
#include <algorithm>
#include <memory>
#include <map>
#include <tuple>
#include <octomap_msgs/conversions.h>
#include "startup_blind_mask.hpp"
#include "forward_obstacle.hpp"
#include "cruise_direction.hpp"

class Integration {
  ros::NodeHandle nh_, private_{"~"};
  nbvInspection::nbvPlanner<Eigen::Vector4d> planner_{nh_, private_};
  ros::ServiceServer service_;
  uint64_t revision_ = 0;
  Eigen::Vector3d box_;
  double pcl_throttle_=0.;
  ros::Time last_cloud_;
  bool map_frozen_=false;
  std::string map_source_;
  uint64_t source_revision_=0,map_imports_=0,depth_insertions_=0;
  bool forward_start_active_=false;
  Eigen::Vector3d forward_start_origin_;
  double extension_=0.,overshoot_=0.;
  double obstacle_plan_distance_=3.;
  std::string startup_unknown_policy_;
  double startup_min_depth_=0.;
  std::unique_ptr<pelagos::StartupBlindMask> startup_blind_mask_;
  bool startup_origin_fixed_=false,startup_limit_segment_=false;
  double startup_max_distance_=0.;
  Eigen::Vector3d startup_direction_;
  bool forward_task_=false,forward_reference_set_=false;
  bool allow_self_unknown_=false;
  Eigen::Vector3d forward_reference_=Eigen::Vector3d::Zero();
  Eigen::Vector3d forward_reference_origin_=Eigen::Vector3d::Zero();
  std::vector<Eigen::Vector3d> last_source_edge_geometry_;
  uint64_t last_source_edge_map_revision_=0;
  bool setForwardReference(const geometry_msgs::Pose& pose) {
    const auto& q=pose.orientation;
    const double norm=q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w;
    if(!std::isfinite(norm)||std::abs(norm-1.)>1.e-3)return false;
    if(forward_reference_set_)return true;
    const double yaw=std::atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z));
    forward_reference_=Eigen::Vector3d(std::cos(yaw),std::sin(yaw),0.);
    forward_reference_origin_=point(pose);forward_reference_set_=true;return true;
  }
  using Json = boost::property_tree::ptree;
  Json vector(const Eigen::Vector3d& value) {
    Json result;
    for (int i=0;i<3;++i) { Json item;item.put("",value[i]);result.push_back({"",item}); }
    return result;
  }
  std::string status(int value) {
    return value==0 ? "free" : value==1 ? "occupied" : "unknown";
  }
  Json inspect(const geometry_msgs::Pose& body) {
    auto map=planner_.pelagosMap();auto root=point(body);
    Json report;Eigen::Vector3d low=Eigen::Vector3d::Zero(),high=Eigen::Vector3d::Zero();
    std::vector<std::pair<Eigen::Vector3d,double>> free,occupied;
    map->getAllFreeBoxes(&free);map->getAllOccupiedBoxes(&occupied);
    if(!free.empty()||!occupied.empty())map->getMapBounds(&low,&high);
    report.put("map_source",map_source_);report.put("source_revision",source_revision_);
    report.put("self_unknown_collision_exception",allow_self_unknown_);
    report.put("map_imports",map_imports_);report.put("native_depth_insertions",depth_insertions_);
    report.add_child("body_position",vector(root));
    report.add_child("map_min",vector(low));report.add_child("map_max",vector(high));
    report.add_child("map_size",vector(high-low));
    report.add_child("robot_box",vector(box_));report.put("resolution_m",map->getResolution());
    report.put("root_point",status(map->getCellStatusPoint(root)));
    report.put("root_box",status(map->getCellStatusBoundingBox(root,box_)));
    // Query the same footprint sample locations used by the upstream swept
    // box check. A center/leaf-bounds query alone can miss unknown corners.
    Json footprint;int unknown_corners=0,occupied_corners=0;
    Eigen::Vector3d step;
    for(int i=0;i<3;++i)step[i]=box_[i]/std::ceil((box_[i]+.001)/map->getResolution());
    for(double x=-box_.x()/2;x<=box_.x()/2;x+=step.x())
      for(double y=-box_.y()/2;y<=box_.y()/2;y+=step.y())
        for(double z=-box_.z()/2;z<=box_.z()/2;z+=step.z()) {
          Eigen::Vector3d sample=root+Eigen::Vector3d(x,y,z);
          auto value=map->getCellStatusPoint(sample);Json entry;
          entry.add_child("point",vector(sample));entry.put("status",status(value));
          footprint.push_back({"",entry});
          if(value==volumetric_mapping::WorldBase::kUnknown)++unknown_corners;
          if(value==volumetric_mapping::WorldBase::kOccupied)++occupied_corners;
        }
    report.add_child("root_footprint_samples",footprint);
    report.put("root_unknown_samples",unknown_corners);report.put("root_occupied_samples",occupied_corners);
    report.put("extension_m",extension_);report.put("overshoot_m",overshoot_);
    report.put("free_leaves",free.size());report.put("occupied_leaves",occupied.size());
    report.put("map_frozen",map_frozen_);
    std::sort(free.begin(),free.end(),[&](const auto& a,const auto& b){
      return (a.first-root).squaredNorm()<(b.first-root).squaredNorm();});
    for (const auto& cell:free) {
      if (map->getCellStatusBoundingBox(cell.first,box_)==volumetric_mapping::WorldBase::kFree) {
        report.add_child("nearest_known_free_box_center",vector(cell.first));
        report.put("nearest_known_free_box_distance_m",(cell.first-root).norm());break;
      }
    }
    Json directions;int free_directions=0;
    for(int x=-1;x<=1;++x)for(int y=-1;y<=1;++y)for(int z=-1;z<=1;++z) {
      if(x==0&&y==0&&z==0)continue;
      Eigen::Vector3d d=Eigen::Vector3d(x,y,z).normalized();
      auto result=map->getLineStatusBoundingBox(root,root+(extension_+overshoot_)*d,box_);
      Json entry;entry.add_child("direction",vector(d));entry.put("status",status(result));
      directions.push_back({"",entry});if(result==volumetric_mapping::WorldBase::kFree)++free_directions;
    }
    report.add_child("native_extension_checks",directions);report.put("free_extension_directions",free_directions);
    return report;
  }
  Eigen::Vector3d point(const geometry_msgs::Pose& p) {
    return {p.position.x, p.position.y, p.position.z};
  }
  bool freePath(const geometry_msgs::Pose& body, const std::vector<geometry_msgs::Pose>& path) {
    if (path.empty()) return false;
    Eigen::Vector3d previous = point(body);
    if (!previous.allFinite()) return false;
    for (const auto& pose : path) {
      auto next = point(pose);
      const auto& q = pose.orientation;
      double norm = q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w;
      if (!next.allFinite() || !std::isfinite(norm) || std::abs(norm-1.)>1.e-3 ||
          planner_.pelagosMap()->getLineStatusBoundingBox(previous, next, box_) !=
          volumetric_mapping::WorldBase::kFree) return false;
      // Upstream computeRayKeys excludes the endpoint. Check its complete
      // footprint explicitly, including endpoints lying in the same voxel.
      auto map=planner_.pelagosMap();Eigen::Vector3d step;
      for(int i=0;i<3;++i)step[i]=box_[i]/std::ceil((box_[i]+.001)/map->getResolution());
      for(double x=-box_.x()/2;x<=box_.x()/2;x+=step.x())for(double y=-box_.y()/2;y<=box_.y()/2;y+=step.y())
        for(double z=-box_.z()/2;z<=box_.z()/2;z+=step.z()) {
          const Eigen::Vector3d sample=next+Eigen::Vector3d(x,y,z);const auto state=map->getCellStatusPoint(sample);
          if(state==volumetric_mapping::WorldBase::kOccupied)return false;
          if(state==volumetric_mapping::WorldBase::kUnknown) {
            octomap::OcTreeKey key;map->coordToKey(sample,&key);if(!map->pelagosIsSelfUnknownKey(key))return false;
          }
        }
      previous = next;
    }
    return true;
  }
  bool strictFreePath(const geometry_msgs::Pose& body,const std::vector<geometry_msgs::Pose>& path) {
    auto map=planner_.pelagosMap();map->pelagosSelfUnknownEnabled_=false;
    const bool valid=freePath(body,path);map->pelagosSetSelfUnknown(point(body),box_,allow_self_unknown_);return valid;
  }
  bool occupiedOnlyFree(const geometry_msgs::Pose& body,const std::vector<geometry_msgs::Pose>& path) {
    if(path.empty())return false;
    std::vector<std::pair<Eigen::Vector3d,double>> occupied;
    planner_.pelagosMap()->getAllOccupiedBoxes(&occupied);
    auto previous=point(body);if(!previous.allFinite())return false;
    for(const auto& pose:path) {
      auto next=point(pose);if(!next.allFinite())return false;
      for(const auto& cell:occupied) {
        const Eigen::Vector3d half=box_/2.+Eigen::Vector3d::Constant(cell.second/2.);
        if(pelagos::segmentIntersectsBox(previous,next,cell.first,half))return false;
      }
      previous=next;
    }
    return true;
  }
  Json forwardObstacleStatus(const geometry_msgs::Pose& body) {
    std::vector<std::pair<Eigen::Vector3d,double>> occupied;
    planner_.pelagosMap()->getAllOccupiedBoxes(&occupied);
    const auto contact=pelagos::nearestForwardObstacle(point(body),forward_reference_,box_,occupied);
    Json report;report.put("forward_obstacle_found",contact.found);
    report.put("obstacle_observation",contact.found?"known_occupied":"none_in_current_map");
    report.put("known_occupied_boxes_tested",occupied.size());
    report.put("map_revision",revision_);report.put("source_revision",source_revision_);
    report.put("map_source_stamp_ns",source_revision_);
    report.add_child("forward_direction",vector(forward_reference_));
    report.add_child("forward_origin",vector(forward_reference_origin_));
    report.add_child("body_position",vector(point(body)));
    report.add_child("corridor_body_box",vector(box_));
    report.put("clearance_semantics","translation_until_source_body_first_touches_known_occupied_voxel_at_current_altitude");
    report.put("no_obstacle_semantics","no_known_occupied_voxel_in_this_forward_corridor_in_installed_map;unknown_is_not_observed_free");
    if(contact.found) {
      report.put("nearest_obstacle_front_clearance_m",contact.front_clearance_m);
      report.add_child("nearest_obstacle_center",vector(contact.center));
      report.put("nearest_obstacle_size_m",contact.size_m);
    }
    return report;
  }
  Json startupStatus(const geometry_msgs::Pose& body) {
    Json report;auto map=planner_.pelagosMap();auto root=point(body);Eigen::Vector3d step;
    for(int i=0;i<3;++i)step[i]=box_[i]/std::ceil((box_[i]+.001)/map->getResolution());
    int unknown=0,occupied=0;
    for(double x=-box_.x()/2;x<=box_.x()/2;x+=step.x())for(double y=-box_.y()/2;y<=box_.y()/2;y+=step.y())
      for(double z=-box_.z()/2;z<=box_.z()/2;z+=step.z()) {
        auto value=map->getCellStatusPoint(root+Eigen::Vector3d(x,y,z));
        unknown+=value==volumetric_mapping::WorldBase::kUnknown;occupied+=value==volumetric_mapping::WorldBase::kOccupied;
      }
    report.put("startup_body_known",unknown==0&&occupied==0);report.put("root_unknown_samples",unknown);
    report.put("root_occupied_samples",occupied);report.put("startup_limit_segment",startup_limit_segment_);
    report.put("startup_progress_m",startup_origin_fixed_?(root-forward_start_origin_).dot(startup_direction_):0.);
    report.put("startup_max_distance_m",startup_max_distance_);
    int exits=0;
    for(int x=-1;x<=1;++x)for(int y=-1;y<=1;++y)for(int z=-1;z<=1;++z) {
      if(x==0&&y==0&&z==0)continue;Eigen::Vector3d d=Eigen::Vector3d(x,y,z).normalized();
      if(forward_task_&&d.dot(forward_reference_)<=0.)continue;
      if(map->getLineStatusBoundingBox(root,root+(extension_+overshoot_)*d,box_)==volumetric_mapping::WorldBase::kFree)++exits;
    }
    report.put("startup_can_plan_forward",occupied==0&&exits>0);report.put("observed_forward_exit_directions",exits);
    return report;
  }
  bool forwardStartFree(const geometry_msgs::Pose& body,
                        const std::vector<geometry_msgs::Pose>& path) {
    if(!forward_start_active_ || path.empty())return false;
    auto map=planner_.pelagosMap();
    if(startup_unknown_policy_=="forward_until_observed")return occupiedOnlyFree(body,path);
    if(startup_unknown_policy_=="calibrated_initial_blind") {
      if(!startup_blind_mask_||std::abs(map->getResolution()-startup_blind_mask_->resolution())>1.e-6)return false;
      auto query=[&](const octomap::point3d& p) {
        auto value=map->getCellStatusPoint(Eigen::Vector3d(p.x(),p.y(),p.z()));
        return value==volumetric_mapping::WorldBase::kFree?pelagos::CellState::Free:
          value==volumetric_mapping::WorldBase::kOccupied?pelagos::CellState::Occupied:pelagos::CellState::Unknown;
      };
      Eigen::Vector3d step;
      for(int i=0;i<3;++i)step[i]=box_[i]/std::ceil((box_[i]+.001)/map->getResolution());
      auto previous=point(body);
      for(const auto& pose:path) {
        const auto next=point(pose);if(!next.allFinite())return false;
        for(double x=-box_.x()/2;x<=box_.x()/2;x+=step.x())
          for(double y=-box_.y()/2;y<=box_.y()/2;y+=step.y())
            for(double z=-box_.z()/2;z<=box_.z()/2;z+=step.z()) {
              const Eigen::Vector3d offset(x,y,z),a=previous+offset,b=next+offset;
              if(!startup_blind_mask_->traversable(octomap::point3d(a.x(),a.y(),a.z()),
                 octomap::point3d(b.x(),b.y(),b.z()),query))return false;
            }
        previous=next;
      }
      return true;
    }
    Eigen::Vector3d low=forward_start_origin_-box_/2.;
    Eigen::Vector3d high=forward_start_origin_+box_/2.;
    // OctoMap collision evidence is voxel-valued. Freeze the set of native
    // cells already intersected by the initial footprint, including partially
    // intersected boundary cells. This is not a tunable clearance expansion.
    octomap::OcTreeKey low_key,high_key;
    map->coordToKey(low,&low_key);map->coordToKey(high,&high_key);
    map->keyToCoord(low_key,&low);map->keyToCoord(high_key,&high);
    low.array()-=map->getResolution()/2.;high.array()+=map->getResolution()/2.;
    // The exception is the fixed set of initially intersected self cells only.
    // Never exempt an occupied map cell, or move the exception with the ROV.
    std::vector<std::pair<Eigen::Vector3d,double>> occupied;
    map->getAllOccupiedBoxes(&occupied);
    for(const auto& cell:occupied) {
      if(((cell.first.array()+cell.second/2.)>low.array()).all() &&
         ((cell.first.array()-cell.second/2.)<high.array()).all())return false;
    }
    auto strict=[&](const Eigen::Vector3d& a,const Eigen::Vector3d& b) {
      return map->getLineStatus(a,b)==volumetric_mapping::WorldBase::kFree &&
             map->getCellStatusPoint(b)==volumetric_mapping::WorldBase::kFree;
    };
    auto line=[&](const Eigen::Vector3d& a,const Eigen::Vector3d& b) {
      const Eigen::Vector3d delta=b-a;double enter=0.,leave=1.;
      for(int i=0;i<3;++i) {
        if(std::abs(delta[i])<1.e-12) {
          if(a[i]<low[i]-1.e-9 || a[i]>high[i]+1.e-9)return strict(a,b);
        } else {
          double u=(low[i]-a[i])/delta[i],v=(high[i]-a[i])/delta[i];
          if(u>v)std::swap(u,v);enter=std::max(enter,u);leave=std::min(leave,v);
        }
      }
      if(enter>leave)return strict(a,b);
      const double epsilon=.001/std::max(delta.norm(),.001); // upstream spatial epsilon
      if(enter>epsilon && !strict(a,a+std::max(0.,enter-epsilon)*delta))return false;
      if(leave<1.-epsilon && !strict(a+std::min(1.,leave+epsilon)*delta,b))return false;
      return true;
    };
    Eigen::Vector3d step;
    for(int i=0;i<3;++i)step[i]=box_[i]/std::ceil((box_[i]+.001)/map->getResolution());
    auto previous=point(body);
    for(const auto& pose:path) {
      const auto next=point(pose);
      if(!next.allFinite())return false;
      for(double x=-box_.x()/2;x<=box_.x()/2;x+=step.x())
        for(double y=-box_.y()/2;y<=box_.y()/2;y+=step.y())
          for(double z=-box_.z()/2;z<=box_.z()/2;z+=step.z()) {
            const Eigen::Vector3d offset(x,y,z);
            if(!line(previous+offset,next+offset))return false;
          }
      previous=next;
    }
    return true;
  }
  bool step(pelagos_nbv::Step::Request& req, pelagos_nbv::Step::Response& res) {
    res.map_source=map_source_;res.source_revision=source_revision_;
    res.depth_fusion_enabled=map_source_=="native_depth";
    if (req.header.frame_id != "rov01/odom" || !point(req.body).allFinite()) {
      res.reason = "invalid_frame_or_pose"; return true;
    }
    // /clock arrives independently of planner requests, including while an
    // original plannerCallback is expanding its tree.
    nav_msgs::Odometry state; state.header = req.header; state.child_frame_id = "rov01/base_link";
    state.pose.pose = req.body;
    planner_.odomCallback(state);
    auto map = planner_.pelagosMap();
    map->pelagosSetSelfUnknown(point(req.body),box_,allow_self_unknown_);
    if (req.operation == "observe") {
      if(map_source_=="wavemap") {
        res.valid=source_revision_>0;res.reason="native_depth_fusion_disabled_wavemap_owner";
        res.map_revision=revision_;return true;
      }
      if(map_frozen_) {
        res.valid=true;res.reason="frozen_observed_snapshot";res.map_revision=revision_;return true;
      }
      if(!last_cloud_.isZero() && (req.header.stamp-last_cloud_).toSec()<pcl_throttle_) {
        res.valid=true;res.reason="upstream_pcl_throttle";res.map_revision=revision_;return true;
      }
      const auto& q = req.camera.orientation;
      Eigen::Quaterniond rotation(q.w,q.x,q.y,q.z);
      if (!rotation.coeffs().allFinite() || std::abs(rotation.norm()-1.)>1.e-3 ||
          !point(req.camera).allFinite() || req.cloud.width == 0) {
        res.reason = "invalid_observation"; return true;
      }
      volumetric_mapping::Transformation transform(rotation,point(req.camera));
      map->insertPointcloud(transform, boost::make_shared<sensor_msgs::PointCloud2>(req.cloud));
      ++depth_insertions_;
      last_cloud_=req.header.stamp;
      ++revision_; res.valid = true; res.reason = "inserted";
    } else if (req.operation=="replace_map") {
      if(map_source_!="wavemap" || req.map.header.frame_id!="rov01/odom" || req.map.binary ||
          req.map.id!="OcTree" || !std::isfinite(req.map.resolution) ||
          std::abs(req.map.resolution-map->getResolution())>1.e-6 || !req.source_revision) {
        res.reason="invalid_wavemap_octomap_contract";return true;
      }
      if(req.source_revision<=source_revision_) {
        res.valid=true;res.reason="wavemap_snapshot_already_installed";res.map_revision=revision_;return true;
      }
      if(req.map.data.empty())map->resetMap();
      else map->setOctomapFromMsg(req.map); // source API: replaces the complete tree, no ray integration
      source_revision_=req.source_revision;++revision_;++map_imports_;map_frozen_=false;
      res.valid=true;res.reason="wavemap_snapshot_installed";res.source_revision=source_revision_;
    } else if (req.operation == "load_snapshot") {
      const char* run=std::getenv("PELAGOS_RUN");
      res.valid=run&&map->loadOctomapFromFile(std::string(run)+"/nbv-frozen.bt");
      if(res.valid) { map_frozen_=true;++revision_; }
      res.reason=res.valid?"observed_snapshot_loaded_and_frozen":"snapshot_load_failed";
    } else if (req.operation == "unfreeze") {
      map_frozen_=false;res.valid=true;res.reason="live_observations_resumed";
    } else if (req.operation == "inspect") {
      res.valid=true;res.reason="map_and_root_inspected";
    } else if (req.operation == "reset_execution") {
      forward_reference_set_=false;
      if(!setForwardReference(req.body)) {res.reason="invalid_initial_heading";return true;}
      last_source_edge_geometry_.clear();last_source_edge_map_revision_=0;
      static_cast<nbvInspection::RrtTree*>(planner_.tree_)->pelagosResetExecution();
      forward_start_active_=false;
      startup_origin_fixed_=false;startup_limit_segment_=false;
      res.valid=true;res.reason="execution_root_reset_map_retained";
    } else if(req.operation=="set_cruise_direction") {
      std::vector<Eigen::Vector3d> requested;
      for(const auto& pose:req.path)requested.push_back(point(pose));
      const auto direction=pelagos::sourceEdgeCruiseDirection(last_source_edge_geometry_,requested);
      const auto& q=req.body.orientation;
      const double norm=q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w;
      const bool body_valid=std::isfinite(norm)&&std::abs(norm-1.)<=1.e-3;
      // Recheck the actual source edge against the currently installed map.
      // The native unknown/occupied rule is retained here. Continuous cruise
      // then uses the existing occupied-only sweep and 3m obstacle trigger.
      const bool collision_valid=direction.valid&&body_valid&&freePath(req.body,req.path);
      std::vector<std::pair<Eigen::Vector3d,double>> occupied;
      if(collision_valid)map->getAllOccupiedBoxes(&occupied);
      const auto clearance=pelagos::cruiseDirectionClearance(point(req.body),direction.direction,box_,occupied,obstacle_plan_distance_);
      res.valid=collision_valid&&clearance.valid;
      res.reason=!direction.valid?direction.reason:!body_valid?"invalid_cruise_origin_pose":
        !collision_valid?"source_edge_blocked_or_unknown_current_map":
        !clearance.valid?"source_turn_direction_no_longer_cruisable":direction.reason;
      if(res.valid) {
        forward_reference_=direction.direction;forward_reference_origin_=point(req.body);forward_reference_set_=true;
        static_cast<nbvInspection::RrtTree*>(planner_.tree_)->pelagosResetExecution();
        forward_start_active_=false;startup_origin_fixed_=false;startup_limit_segment_=false;
      }
      Json diagnostic;
      diagnostic.put("direction_source","last_valid_native_nbv_first_edge_geometry");
      diagnostic.put("source_edge_direction_valid",direction.valid);
      diagnostic.put("source_edge_map_revision",last_source_edge_map_revision_);
      diagnostic.put("source_edge_points",last_source_edge_geometry_.size());
      diagnostic.put("current_map_collision_valid",collision_valid);
      diagnostic.put("current_horizontal_clearance_valid",collision_valid&&clearance.valid);
      diagnostic.put("turn_clearance_minimum_m",obstacle_plan_distance_);
      diagnostic.put("forward_obstacle_found",clearance.contact.found);
      if(clearance.contact.found)diagnostic.put("nearest_obstacle_front_clearance_m",clearance.contact.front_clearance_m);
      diagnostic.put("no_obstacle_semantics","no_known_occupied_hit;unknown_is_not_observed_free");
      diagnostic.put("source_view_yaw_executed",false);
      diagnostic.put("gain_execution_limit","source_gain_preserved_but_sampled_view_yaw_not_executed_by_heading_only_policy");
      diagnostic.add_child("forward_direction",vector(forward_reference_));
      diagnostic.add_child("forward_origin",vector(forward_reference_origin_));
      std::ostringstream encoded;boost::property_tree::write_json(encoded,diagnostic,false);res.diagnostics_json=encoded.str();
    } else if(req.operation=="forward_obstacle_status" || req.operation=="plan_forward_cruise" ||
              req.operation=="validate_forward_cruise") {
      if(!setForwardReference(req.body)) {res.reason="invalid_initial_heading";return true;}
      Json diagnostic=forwardObstacleStatus(req.body);
      if(req.operation=="forward_obstacle_status") {
        res.valid=true;res.reason=diagnostic.get<bool>("forward_obstacle_found")?
          "known_forward_obstacle_measured":"no_known_forward_obstacle_in_map";
      } else if(req.operation=="plan_forward_cruise") {
        diagnostic.put("path_kind","forward_cruise_not_rrt");
        const double progress=(point(req.body)-forward_reference_origin_).dot(forward_reference_);
        geometry_msgs::Pose end=req.body;
        const Eigen::Vector3d goal=forward_reference_origin_+(progress+extension_)*forward_reference_;
        end.position.x=goal.x();end.position.y=goal.y();end.position.z=goal.z();
        geometry_msgs::Pose check=end;
        check.position.x+=overshoot_*forward_reference_.x();check.position.y+=overshoot_*forward_reference_.y();
        // Preserve the original body->overshoot chord and check the complete
        // body->goal leg. Tracking error can make those two chords noncollinear.
        // In the configured common rov01/odom frame, source samplePath positions
        // are affine points on body->goal. Its full sweep therefore covers all
        // sampled subsegments without rescanning every occupied voxel ~50 times.
        res.valid=occupiedOnlyFree(req.body,{check})&&occupiedOnlyFree(req.body,{end});
        if(res.valid) {
          const auto& q=req.body.orientation;
          const double yaw=std::atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z));
          Eigen::Vector4d start;start<<point(req.body),yaw;
          Eigen::Vector4d target=start;target.head<3>()=goal;
          target[3]=std::atan2(forward_reference_.y(),forward_reference_.x());
          res.path=static_cast<nbvInspection::RrtTree*>(planner_.tree_)->samplePath(start,target,req.header.frame_id);
          res.valid=!res.path.empty();
        }
        res.reason=res.valid?"forward_cruise_no_known_obstacle":"forward_cruise_known_occupied_or_invalid";
        diagnostic.add_child("cruise_goal",vector(goal));diagnostic.put("cruise_progress_m",progress);
        if(!res.valid)res.path.clear();
      } else {
        res.valid=!req.path.empty()&&occupiedOnlyFree(req.body,req.path);
        if(res.valid) {
          auto check=req.path.back();
          check.position.x+=overshoot_*forward_reference_.x();check.position.y+=overshoot_*forward_reference_.y();
          res.valid=occupiedOnlyFree(req.body,{check});
        }
        res.reason=res.valid?"forward_cruise_corridor_valid":"forward_cruise_corridor_known_occupied_or_invalid";
      }
      diagnostic.put("collision_policy","unknown_allowed_known_occupied_rejected_swept_source_body_plus_source_overshoot");
      diagnostic.put("extension_m",extension_);diagnostic.put("overshoot_m",overshoot_);
      diagnostic.put("cruise_corridor_valid",req.operation=="forward_obstacle_status"?false:res.valid);
      diagnostic.put("native_path_points",res.path.size());
      std::ostringstream encoded;boost::property_tree::write_json(encoded,diagnostic,false);res.diagnostics_json=encoded.str();
    } else if (req.operation == "plan_forward_start") {
      const auto& q=req.body.orientation;
      const double yaw=std::atan2(2*(q.w*q.z+q.x*q.y),1-2*(q.y*q.y+q.z*q.z));
      Eigen::Vector3d forward(std::cos(yaw),std::sin(yaw),0.);
      if(startup_unknown_policy_=="forward_until_observed") {
        if(!startup_origin_fixed_) {forward_start_origin_=point(req.body);startup_direction_=forward;startup_origin_fixed_=true;}
        forward=startup_direction_;
      } else forward_start_origin_=point(req.body);
      forward_start_active_=true;
      if(startup_unknown_policy_=="calibrated_initial_blind") {
        const auto& c=req.camera.orientation;Eigen::Quaterniond rotation(c.w,c.x,c.y,c.z);
        const auto& params=planner_.pelagosParams();
        if(!rotation.coeffs().allFinite()||std::abs(rotation.norm()-1.)>1.e-3||!point(req.camera).allFinite()||
           params.camHorizontal_.empty()||params.camVertical_.empty()) {
          forward_start_active_=false;res.reason="invalid_startup_camera_geometry";return true;
        }
        startup_blind_mask_=std::make_unique<pelagos::StartupBlindMask>(map->getResolution(),
          point(req.body),point(req.camera),rotation.toRotationMatrix(),forward,box_,extension_+overshoot_,
          startup_min_depth_,params.camHorizontal_.front(),params.camVertical_.front());
      }
      geometry_msgs::Pose end=req.body;
      end.position.x+=extension_*forward.x();end.position.y+=extension_*forward.y();
      if(startup_unknown_policy_=="forward_until_observed") {
        // User requested one continuous ~5 m advance, not source-edge stop
        // points each metre. RRT extension remains unchanged after handoff.
        const double target=startup_max_distance_;
        const Eigen::Vector3d goal=forward_start_origin_+target*forward;
        end.position.x=goal.x();end.position.y=goal.y();end.position.z=goal.z();
        startup_limit_segment_=target>=startup_max_distance_;
      }
      geometry_msgs::Pose check=end;
      check.position.x+=overshoot_*forward.x();check.position.y+=overshoot_*forward.y();
      res.valid=forwardStartFree(req.body,{check});
      if(res.valid) {
        Eigen::Vector4d start;start<<point(req.body),yaw;
        Eigen::Vector4d goal=start;goal.head<3>()=point(end);
        res.path=static_cast<nbvInspection::RrtTree*>(planner_.tree_)->samplePath(start,goal,req.header.frame_id);
        res.valid=!res.path.empty()&&forwardStartFree(req.body,res.path);
      }
      res.reason=res.valid?(startup_unknown_policy_=="forward_until_observed"?
        "startup_forward_no_known_obstacle":"forward_start_front_corridor_free"):"forward_start_front_blocked_or_unknown";
      Json diagnostic=inspect(req.body);
      diagnostic.put("path_kind","forward_start_not_rrt");
      diagnostic.put("collision_policy",startup_unknown_policy_=="forward_until_observed"?
        "startup_unknown_allowed_known_occupied_rejected_until_body_observed_or_5m_limit":startup_unknown_policy_=="calibrated_initial_blind"?
        "unknown_allowed_only_in_fixed_initial_self_or_calibrated_camera_blind_cells":
        "unknown_allowed_only_in_fixed_initial_self_intersecting_cells");
      if(startup_unknown_policy_!="forward_until_observed")diagnostic.put(startup_unknown_policy_=="calibrated_initial_blind"?
        "outside_initial_blind_cells_known_free":"outside_initial_self_cells_known_free",res.valid);
      else diagnostic.put("startup_corridor_has_no_known_occupied",res.valid);
      if(startup_blind_mask_) {
        diagnostic.put("startup_exception_cells",startup_blind_mask_->cells());
        diagnostic.put("startup_self_cells",startup_blind_mask_->selfCells());
        diagnostic.put("startup_min_depth_m",startup_min_depth_);
      }
      diagnostic.put("path_known_free",res.valid&&strictFreePath(req.body,res.path));
      if(startup_unknown_policy_=="forward_until_observed") {
        const auto state=startupStatus(req.body);for(const auto& entry:state)diagnostic.put_child(entry.first,entry.second);
      }
      diagnostic.put("native_path_points",res.path.size());
      diagnostic.put("start_distance_m",startup_unknown_policy_=="forward_until_observed"?
        (point(end)-point(req.body)).norm():extension_);diagnostic.put("overshoot_m",overshoot_);
      std::ostringstream encoded;boost::property_tree::write_json(encoded,diagnostic,false);
      res.diagnostics_json=encoded.str();
      if(!res.valid) {res.path.clear();forward_start_active_=false;}
    } else if (req.operation == "validate_forward_start") {
      res.valid=forwardStartFree(req.body,req.path);
      if(res.valid&&startup_unknown_policy_=="forward_until_observed"&&!req.path.empty()) {
        auto check=req.path.back();check.position.x+=overshoot_*startup_direction_.x();check.position.y+=overshoot_*startup_direction_.y();
        res.valid=occupiedOnlyFree(req.body,{check});
      }
      res.reason=res.valid?"forward_start_corridor_valid":"forward_start_corridor_invalid";
      if(startup_unknown_policy_=="forward_until_observed") {
        std::ostringstream encoded;boost::property_tree::write_json(encoded,startupStatus(req.body),false);res.diagnostics_json=encoded.str();
      }
    } else if (req.operation == "plan" || req.operation == "plan_map_window" || req.operation=="plan_turn") {
      last_source_edge_geometry_.clear();last_source_edge_map_revision_=0;
      setForwardReference(req.body);
      auto* source_tree=static_cast<nbvInspection::RrtTree*>(planner_.tree_);
      source_tree->pelagosSetForwardDirection(forward_task_?forward_reference_:Eigen::Vector3d::Zero());
      const bool turn_only=req.operation=="plan_turn";
      const auto previous_turn_filter=source_tree->pelagosTurnDirectionAllowed_;
      std::vector<std::pair<Eigen::Vector3d,double>> turn_occupied;
      std::map<std::tuple<double,double,double>,pelagos::CruiseDirectionClearance> turn_checks;
      source_tree->pelagosTurnDirectionAllowed_={};
      if(turn_only) {
        // getBestEdge advances exact_root to its unexecuted endpoint. A new
        // heading-only request must initialize the real current-body root.
        source_tree->pelagosResetExecution();map->getAllOccupiedBoxes(&turn_occupied);
        source_tree->pelagosTurnDirectionAllowed_=[&](const Eigen::Vector3d& displacement) {
          const auto key=std::make_tuple(displacement.x(),displacement.y(),displacement.z());
          auto found=turn_checks.find(key);
          if(found==turn_checks.end())found=turn_checks.emplace(key,
              pelagos::cruiseDirectionClearance(point(req.body),displacement,box_,turn_occupied,obstacle_plan_distance_)).first;
          return found->second.valid;
        };
      }
      auto source_params=planner_.pelagosParams();auto selected_params=source_params;
      if(req.operation=="plan_map_window") {
        // Exact values explicitly approved for the 2026-10-30 nonexecuting
        // contrast. Do not silently expand them if a later map grows.
        selected_params.minX_=0.;selected_params.minY_=-3.2;selected_params.minZ_=-2.8;
        selected_params.maxX_=5.2;selected_params.maxY_=3.2;selected_params.maxZ_=2.8;
        planner_.tree_->setParams(selected_params);
      }
      nbvplanner::nbvp_srv::Request request; nbvplanner::nbvp_srv::Response response;
      request.header = req.header;
      try { planner_.plannerCallback(request, response); }
      catch(...) { planner_.tree_->setParams(source_params);source_tree->pelagosTurnDirectionAllowed_=previous_turn_filter;throw; }
      planner_.tree_->setParams(source_params);
      source_tree->pelagosTurnDirectionAllowed_=previous_turn_filter;
      res.path = response.path;
      const bool path_collision_valid=freePath(req.body, res.path);
      res.valid = path_collision_valid;
      const bool forward_path_check_ran=res.valid&&forward_task_;
      if(res.valid&&forward_task_) {
        const Eigen::Vector3d progress=point(res.path.back())-point(req.body);
        res.valid=progress.dot(forward_reference_)>0.;
      }
      // Recheck the actual extracted first edge as well: source samplePath can
      // have an early fallback outside normal extraction. Never approve a turn
      // solely because the inspected tree contains another valid direction.
      pelagos::CruiseDirectionClearance selected_turn;
      if(turn_only&&res.valid) {
        selected_turn=pelagos::cruiseDirectionClearance(point(req.body),point(res.path.back())-point(res.path.front()),
            box_,turn_occupied,obstacle_plan_distance_);
        res.valid=selected_turn.valid;
      }
      res.reason = res.valid ? "planned" : turn_only?"no_cruisable_turn_direction":
        forward_task_?"no_forward_known_free_edge":"no_valid_known_free_edge";
      Json diagnostic=inspect(req.body);
      double native_path_length=0.;
      for(size_t i=1;i<response.path.size();++i)
        native_path_length+=(point(response.path[i])-point(response.path[i-1])).norm();
      diagnostic.put("native_path_points",res.path.size());
      diagnostic.put("native_path_length_m",native_path_length);
      diagnostic.put("execution_path_points",res.valid?res.path.size():0);
      diagnostic.put("source_callback_termination",planner_.pelagosTermination_);
      diagnostic.put("source_tree_initialized_this_callback",planner_.pelagosTreeInitialized_);
      diagnostic.put("source_forward_selection_ran",planner_.pelagosForwardSelectionRan_);
      diagnostic.put("source_ros_ok_after_callback",ros::ok());
      diagnostic.put("accepted_tree_nodes",planner_.tree_->getCounter());
      diagnostic.put("positive_gain_found",planner_.tree_->gainFound());
      diagnostic.put("path_known_free",path_collision_valid&&strictFreePath(req.body,res.path));
      diagnostic.put("path_collision_valid",path_collision_valid);
      diagnostic.put("path_execution_valid",res.valid);
      diagnostic.put("source_path_validation",response.path.empty()?"empty_source_path":
        !path_collision_valid?"collision_rejected":!res.valid?"forward_constraint_rejected":"accepted");
      diagnostic.put("forward_path_validation",!forward_task_?"not_required":
        !forward_path_check_ran?"not_evaluated":res.valid?"passed":"failed");
      diagnostic.put("forward_task_constraint",forward_task_);
      diagnostic.put("forward_view_constraint",false);
      diagnostic.put("forward_task_constraint_policy","positive_first_edge_translation_preserve_source_yaw_and_gain");
      diagnostic.put("turn_direction_filter_active",turn_only);
      if(turn_only) {
        diagnostic.put("turn_clearance_minimum_m",obstacle_plan_distance_);
        diagnostic.put("turn_selector_eligible_nodes",source_tree->pelagosForwardEligible_);
        diagnostic.put("turn_direction_source","actual_native_rrt_first_edge_then_source_getBestEdge");
        diagnostic.put("turn_execution_root","latest_estimated_body_not_unexecuted_source_endpoint");
        diagnostic.put("turn_no_obstacle_semantics","no_known_occupied_hit;unknown_is_not_observed_free");
        Json checked;
        for(const auto& item:turn_checks) {
          Json entry;const auto& value=item.second;
          entry.add_child("source_first_edge_displacement",vector(Eigen::Vector3d(std::get<0>(item.first),std::get<1>(item.first),std::get<2>(item.first))));
          entry.add_child("horizontal_execution_direction",vector(value.direction));
          entry.put("cruisable",value.valid);entry.put("forward_obstacle_found",value.contact.found);
          if(value.contact.found)entry.put("nearest_obstacle_front_clearance_m",value.contact.front_clearance_m);
          checked.push_back({"",entry});
        }
        diagnostic.add_child("turn_checked_source_first_edges",checked);
        diagnostic.put("selected_turn_cruisable",res.valid&&selected_turn.valid);
        diagnostic.put("selected_turn_forward_obstacle_found",selected_turn.contact.found);
        if(selected_turn.contact.found)diagnostic.put("selected_turn_front_clearance_m",selected_turn.contact.front_clearance_m);
      }
      diagnostic.add_child("forward_task_direction",vector(forward_reference_));
      // Recount from the final source tree without setting bestNode_/bestGain_.
      // An early upstream fallback can return before the selector ever runs.
      int actual_forward_eligible=0;
      const auto* root=source_tree->pelagosRoot();
      pelagos::selectForwardNode(root,forward_reference_,selected_params.zero_gain_,&actual_forward_eligible);
      diagnostic.put("forward_eligible_tree_nodes",actual_forward_eligible);
      diagnostic.put("source_selector_eligible_tree_nodes",source_tree->pelagosForwardEligible_);
      diagnostic.put("forward_eligibility_scope",planner_.pelagosTreeInitialized_?
        "current_callback_tree":"retained_tree_not_initialized_this_callback");
      // Read-only export of the actual source RRT; no reconstructed branches.
      Json nodes;
      std::vector<std::pair<const nbvInspection::Node<Eigen::Vector4d>*,int>> pending;
      if(root)pending.push_back({root,-1});
      for(size_t i=0;i<pending.size();++i) {
        const auto* node=pending[i].first;Json entry;
        entry.add_child("position",vector(node->state_.head<3>()));
        entry.put("yaw",node->state_[3]);entry.put("gain",node->gain_);
        entry.put("parent",pending[i].second);nodes.push_back({"",entry});
        for(const auto* child:node->children_)pending.push_back({child,static_cast<int>(i)});
      }
      diagnostic.add_child("tree_nodes",nodes);
      diagnostic.put("sampling_profile",req.operation=="plan_map_window"?"observed_map_window":"approved_full_porth");
      diagnostic.add_child("sampling_min",vector(Eigen::Vector3d(selected_params.minX_,selected_params.minY_,selected_params.minZ_)));
      diagnostic.add_child("sampling_max",vector(Eigen::Vector3d(selected_params.maxX_,selected_params.maxY_,selected_params.maxZ_)));
      std::ostringstream encoded;boost::property_tree::write_json(encoded,diagnostic,false);
      res.diagnostics_json=encoded.str();
      if (!res.valid) res.path.clear();
      else {
        for(const auto& pose:res.path)last_source_edge_geometry_.push_back(point(pose));
        last_source_edge_map_revision_=revision_;
      }
    } else if (req.operation == "validate") {
      res.valid = freePath(req.body, req.path);
      res.reason = res.valid ? "known_free" : "blocked_or_unknown";
    } else { res.reason = "unknown_operation"; return true; }
    res.map_revision = revision_;
    if(req.operation=="inspect") {
      std::ostringstream encoded;boost::property_tree::write_json(encoded,inspect(req.body),false);
      res.diagnostics_json=encoded.str();
      const char* run=std::getenv("PELAGOS_RUN");
      if(run) {
        // The upstream writeBinary() mutates probabilities/pruning. Inspect a
        // copy so diagnostics cannot change the live Wavemap query snapshot.
        octomap_msgs::Octomap snapshot;map->getOctomapFullMsg(&snapshot);
        std::unique_ptr<octomap::AbstractOcTree> copy(octomap_msgs::fullMsgToMap(snapshot));
        auto* tree=dynamic_cast<octomap::OcTree*>(copy.get());
        if(tree) {tree->write(std::string(run)+"/nbv-map.ot");tree->writeBinaryConst(std::string(run)+"/nbv-map.bt");}
      }
    }
    if (req.include_map && (req.operation == "plan" || req.operation == "plan_map_window" || req.operation=="plan_turn" ||
        req.operation == "plan_forward_start" || req.operation == "plan_forward_cruise")) {
      std::vector<std::pair<Eigen::Vector3d,double>> occupied;
      map->getAllOccupiedBoxes(&occupied);
      for (const auto& cell : occupied) {
        geometry_msgs::Point p; p.x=cell.first.x(); p.y=cell.first.y(); p.z=cell.first.z();
        res.occupied.push_back(p); res.sizes.push_back(cell.second);
      }
    }
    return true;
  }
public:
  Integration() {
    private_.param<std::string>("pelagos_map_source",map_source_,"wavemap");
    if(map_source_!="wavemap"&&map_source_!="native_depth")throw std::runtime_error("Invalid map source");
    if(map_source_=="wavemap")planner_.pelagosDisableMapInputs();
    for (int i=0;i<3;++i) {
      const std::string axis(1,"xyz"[i]);
      if(!private_.getParam("system/bbx/"+axis,box_[i]))throw std::runtime_error("Missing upstream robot box");
    }
    private_.getParam("pcl_throttle",pcl_throttle_);
    private_.getParam("nbvp/tree/extension_range",extension_);
    private_.getParam("system/bbx/overshoot",overshoot_);
    private_.param("pelagos_obstacle_plan_distance_m",obstacle_plan_distance_,3.);
    if(!std::isfinite(obstacle_plan_distance_)||obstacle_plan_distance_<=0.)throw std::runtime_error("Invalid user obstacle planning distance");
    private_.param("pelagos_forward_task",forward_task_,false);
    private_.param("pelagos_allow_self_unknown",allow_self_unknown_,false);
    private_.param<std::string>("pelagos_start_unknown_policy",startup_unknown_policy_,"initial_self_only");
    private_.param("pelagos_start_min_depth_m",startup_min_depth_,.3);
    private_.param("pelagos_start_max_distance_m",startup_max_distance_,extension_);
    if(!std::isfinite(startup_max_distance_)||startup_max_distance_<=0.)throw std::runtime_error("Invalid user startup distance");
    if(startup_unknown_policy_!="initial_self_only"&&startup_unknown_policy_!="calibrated_initial_blind"&&
       startup_unknown_policy_!="forward_until_observed"&&startup_unknown_policy_!="obstacle_triggered")
      throw std::runtime_error("Unsupported startup unknown policy");
    service_ = nh_.advertiseService("pelagos_step", &Integration::step, this);
  }
};
int main(int argc, char** argv) {
  ros::init(argc,argv,"pelagos_nbv");
  Integration integration; ros::spin();
}
