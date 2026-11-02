#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <deque>
#include <fstream>
#include <filesystem>
#include <future>
#include <cstdlib>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <mutex>
#include <map>
#include <sstream>
#include <string>
#include <stdexcept>
#include <utility>
#include <vector>

#include <Eigen/Geometry>

#include <geometry_msgs/msg/point.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <octomap_msgs/conversions.h>
#include <octomap_msgs/msg/octomap.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/msg/point_field.hpp>
#include <std_msgs/msg/u_int64.hpp>
#include <std_msgs/msg/string.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <visualization_msgs/msg/marker_array.hpp>

#include <wavemap/core/common.h>
#include <wavemap/core/data_structure/image.h>
#include <wavemap/core/indexing/index_conversions.h>
#include <wavemap/core/integrator/measurement_model/continuous_ray.h>
#include <wavemap/core/integrator/projective/coarse_to_fine/hashed_chunked_wavelet_integrator.h>
#include <wavemap/core/integrator/projection_model/pinhole_camera_projector.h>
#include <wavemap/core/map/hashed_chunked_wavelet_octree.h>
#include <wavemap/io/file_conversions.h>
#include "underwater_wavemap_ros/octomap_projection.hpp"
#include "underwater_wavemap_ros/chunked_restore.hpp"
#include "underwater_wavemap_ros/depth_measurements.hpp"

namespace {

struct StampedPose {
  rclcpp::Time stamp;
  geometry_msgs::msg::Pose pose;
};

using underwater_wavemap_ros::ValidDepthContinuousRay;

class WavemapNode final : public rclcpp::Node {
 public:
  WavemapNode()
      : Node("underwater_wavemap") {
    width_ = declare_parameter<int>("width", 960);
    height_ = declare_parameter<int>("height", 540);
    fx_ = declare_parameter<double>("fx", 0.0);
    fy_ = declare_parameter<double>("fy", 0.0);
    cx_ = declare_parameter<double>("cx", 0.0);
    cy_ = declare_parameter<double>("cy", 0.0);
    min_range_ = declare_parameter<double>("min_range_m", 0.30);
    max_range_ = declare_parameter<double>("max_range_m", 12.0);
    max_pose_gap_ = declare_parameter<double>("max_pose_gap_s", 0.10);
    min_cell_width_ = declare_parameter<double>("min_cell_width_m", 0.10);
    tree_height_ = declare_parameter<int>("tree_height", 6);
    publish_every_n_frames_ = std::max(1, static_cast<int>(
        declare_parameter<int64_t>("publish_every_n_frames", 1)));
    depth_topic_ = declare_parameter<std::string>("depth_topic", "/stereo/depth");
    odom_topic_ = declare_parameter<std::string>("odom_topic", "/odometry/estimated");
    world_frame_ = declare_parameter<std::string>("world_frame", "world");
    map_output_path_ = declare_parameter<std::string>("map_output_path", "");
    const auto map_input_path=declare_parameter<std::string>("same_session_map_input", "");
    const auto octomap_topic=declare_parameter<std::string>("octomap_topic", "");
    octomap_resolution_=declare_parameter<double>("octomap_resolution_m", .4);
    camera_frame_ = declare_parameter<std::string>("camera_frame", "cam0_rect_optical");
    status_path_ = declare_parameter<std::string>("status_path", "");
    const auto localization_topic=declare_parameter<std::string>("localization_status_topic", "");
    pixel_stride_ = declare_parameter<int>("pixel_stride", 1);
    integration_rate_ = declare_parameter<double>("integration_rate_hz", 0.0);
    publish_rate_ = declare_parameter<double>("publish_rate_hz", 0.0);
    pose_wait_ = declare_parameter<double>("pose_wait_timeout_s", 0.3);
    prune_period_ = declare_parameter<double>("prune_period_s", 0.0);
    occupied_probability_ = declare_parameter<double>("occupied_probability", 0.70);
    const double sigma = declare_parameter<double>("range_sigma_m", 0.03);
    const double free_weight = declare_parameter<double>("scaling_free", 0.20);
    const double occupied_weight = declare_parameter<double>("scaling_occupied", 0.40);
    const int threads = declare_parameter<int>("integration_threads", 4);
    edge_absolute_ = declare_parameter<double>("depth_edge_absolute_m", 0.0);
    edge_relative_ = declare_parameter<double>("depth_edge_relative", 0.0);

    if (width_ <= 0 || height_ <= 0 || fx_ <= 0.0 || fy_ <= 0.0 || max_range_ <= min_range_ ||
        pixel_stride_<1 || width_%pixel_stride_ || height_%pixel_stride_ ||
        !std::isfinite(sigma) || sigma<=0 || threads<1 || threads>64 ||
        integration_rate_<0 || publish_rate_<0 || pose_wait_<=0 || max_pose_gap_<=0 ||
        occupied_probability_<=.5 || occupied_probability_>=1.) {
      throw std::invalid_argument("invalid camera or range parameters");
    }

    wavemap::HashedChunkedWaveletOctree::Config map_config;
    map_config.min_cell_width = static_cast<float>(min_cell_width_);
    map_config.min_log_odds = -2.0f;
    map_config.max_log_odds = 4.0f;
    map_config.tree_height = tree_height_;
    map_ = std::make_shared<wavemap::HashedChunkedWaveletOctree>(map_config);
    if(!map_input_path.empty()) {
      // A process reload may reuse this live estimator's map. Never allow a
      // prior run's map to be relabelled into a new estimator coordinate frame.
      const char* run=std::getenv("PELAGOS_RUN");
      if(!run || std::filesystem::weakly_canonical(std::filesystem::path(map_input_path).parent_path())!=
          std::filesystem::weakly_canonical(run))throw std::runtime_error("Map restore must belong to the active estimator run");
      wavemap::MapBase::Ptr restored;
      if(!wavemap::io::fileToMap(map_input_path,restored)||!restored ||
         std::abs(restored->getMinCellWidth()-map_->getMinCellWidth())>1.e-6 ||
         restored->getTreeHeight()!=map_->getTreeHeight())throw std::runtime_error("Invalid same-session Wavemap restore");
      auto typed=std::dynamic_pointer_cast<wavemap::HashedChunkedWaveletOctree>(restored);
      if(!typed)typed=underwater_wavemap_ros::restoreChunked(*restored);
      map_=std::move(typed);
      RCLCPP_INFO(get_logger(),"Restored current estimator session map: %s",map_input_path.c_str());
    }

    projector_ = std::make_shared<wavemap::PinholeCameraProjector>(
        wavemap::PinholeCameraProjectorConfig{
            static_cast<float>(fx_/pixel_stride_), static_cast<float>(fy_/pixel_stride_), static_cast<float>(cx_/pixel_stride_),
            static_cast<float>(cy_/pixel_stride_), height_/pixel_stride_, width_/pixel_stride_});
    range_image_ = std::make_shared<wavemap::PosedImage<>>(width_/pixel_stride_, height_/pixel_stride_);
    beam_offsets_ = std::make_shared<wavemap::Image<wavemap::Vector2D>>(width_/pixel_stride_, height_/pixel_stride_);
    measurement_model_ = std::make_shared<ValidDepthContinuousRay>(
        wavemap::ContinuousRayConfig{static_cast<float>(sigma),static_cast<float>(free_weight),
                                    static_cast<float>(occupied_weight)}, projector_, range_image_);
    integrator_ = std::make_unique<wavemap::HashedChunkedWaveletIntegrator>(
        wavemap::ProjectiveIntegratorConfig{static_cast<float>(min_range_),
                                            static_cast<float>(max_range_)},
        projector_, range_image_, beam_offsets_, measurement_model_, map_,
        std::make_shared<wavemap::ThreadPool>(threads));

    occupied_pub_ = create_publisher<sensor_msgs::msg::PointCloud2>(
        "/mapping/occupied", rclcpp::QoS(1).transient_local());
    occupied_cells_pub_ = create_publisher<visualization_msgs::msg::MarkerArray>(
        "/mapping/occupied_cells", rclcpp::QoS(1).transient_local());
    path_pub_ = create_publisher<nav_msgs::msg::Path>(
        "/trajectory/estimated", rclcpp::QoS(10));
    integrated_pub_ = create_publisher<std_msgs::msg::UInt64>(
        "/mapping/integrated_frames", rclcpp::QoS(10).transient_local());
    status_pub_ = create_publisher<std_msgs::msg::String>("/mapping/status",rclcpp::QoS(1).transient_local());
    if(!octomap_topic.empty())octomap_pub_=create_publisher<octomap_msgs::msg::Octomap>(
        octomap_topic,rclcpp::QoS(1).transient_local());
    depth_sub_ = create_subscription<sensor_msgs::msg::Image>(
        depth_topic_, rclcpp::SensorDataQoS(),
        std::bind(&WavemapNode::depthCallback, this, std::placeholders::_1));
    odom_sub_ = create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_, rclcpp::SensorDataQoS(),
        std::bind(&WavemapNode::odomCallback, this, std::placeholders::_1));
    if(!localization_topic.empty())localization_sub_=create_subscription<std_msgs::msg::String>(
        localization_topic,1,[this](const std_msgs::msg::String& message) {
          if(message.data.find("okvis_native_tracking_lost_restart_required")!=std::string::npos) {
            localization_lost_=true;pending_depth_.clear();
            RCLCPP_ERROR(get_logger(),"Native OKVIS declared Lost; map fusion frozen until session restart");
          }
        });
    save_service_ = create_service<std_srvs::srv::Trigger>(
        "save_map", std::bind(&WavemapNode::saveMapService, this,
                              std::placeholders::_1, std::placeholders::_2));
    process_timer_=create_wall_timer(std::chrono::milliseconds(20),[this](){processPending();pollProjection();});
    if (publish_rate_>0) {
      publish_timer_=create_wall_timer(std::chrono::duration<double>(1./publish_rate_),[this](){
        if(integrated_frames_>published_frames_)publishMap(last_header_);
        publishStatus();
      });
    }

    RCLCPP_INFO(get_logger(), "Wavemap ROS2 node ready: depth=%s odom=%s size=%dx%d",
                depth_topic_.c_str(), odom_topic_.c_str(), width_, height_);
  }

  ~WavemapNode() override {
    if (!map_output_path_.empty()) {
      saveMap(map_output_path_);
    }
  }

 private:
  struct ProjectionJob {
    underwater_wavemap_ros::OccupancySnapshot snapshot;
    std_msgs::msg::Header header;
    double resolution;
  };
  struct ProjectionResult {
    octomap_msgs::msg::Octomap message;
    underwater_wavemap_ros::ProjectionStats stats;
    double milliseconds;
  };
  void pollProjection() {
    if(projection_future_.valid() && projection_future_.wait_for(std::chrono::seconds(0))==std::future_status::ready) {
      try {
        auto result=projection_future_.get();octomap_bytes_=result.message.data.size();
        projected_stamp_ns_=rclcpp::Time(result.message.header.stamp).nanoseconds();
        projection_stats_=result.stats;projection_ms_=result.milliseconds;
        octomap_pub_->publish(result.message);
      } catch(const std::exception& error) {
        RCLCPP_ERROR(get_logger(),"Wavemap OctoMap projection failed: %s",error.what());
      }
    }
    if(!projection_future_.valid() && pending_projection_) {
      auto job=std::move(pending_projection_);
      projection_future_=std::async(std::launch::async,[job=std::move(job)]() {
        const auto start=std::chrono::steady_clock::now();
        auto projection=underwater_wavemap_ros::projectOctomap(job->snapshot,job->resolution);
        ProjectionResult result;result.stats=projection.stats;
        if(!octomap_msgs::fullMapToMsg(*projection.tree,result.message))throw std::runtime_error("Full OctoMap serialization failed");
        result.message.header=job->header;
        result.milliseconds=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
        return result;
      });
    }
  }
  void odomCallback(const nav_msgs::msg::Odometry::SharedPtr msg) {
    if(localization_lost_)return;
    if(msg->header.frame_id!=world_frame_ || msg->child_frame_id!=camera_frame_)return;
    const auto& p=msg->pose.pose.position;const auto& q=msg->pose.pose.orientation;
    if(!std::isfinite(p.x)||!std::isfinite(p.y)||!std::isfinite(p.z)||
       !std::isfinite(q.x)||!std::isfinite(q.y)||!std::isfinite(q.z)||!std::isfinite(q.w)||
       std::abs(q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w-1.)>1.e-3)return;
    const rclcpp::Time stamp(msg->header.stamp);
    if (!odom_buffer_.empty() && stamp < odom_buffer_.back().stamp) {
      // A map may not mix two estimator epochs. A normal run restarts all
      // components together; an in-session rewind needs an explicit restart.
      estimator_rewound_=true;pending_depth_.clear();return;
    }
    if (!odom_buffer_.empty() && stamp == odom_buffer_.back().stamp) {
      return;
    }
    odom_buffer_.push_back(StampedPose{stamp, msg->pose.pose});
    while (odom_buffer_.size() > 512) {
      odom_buffer_.pop_front();
    }

    nav_msgs::msg::Path path;
    path.header.stamp = msg->header.stamp;
    path.header.frame_id = world_frame_;
    path.poses.reserve(odom_buffer_.size());
    for (const auto& sample : odom_buffer_) {
      geometry_msgs::msg::PoseStamped pose;
      pose.header.stamp = sample.stamp;
      pose.header.frame_id = world_frame_;
      pose.pose = sample.pose;
      path.poses.push_back(std::move(pose));
    }
    path_pub_->publish(path);
    processPending();
  }

  bool interpolatePose(const rclcpp::Time& stamp, geometry_msgs::msg::Pose* pose) {
    if (odom_buffer_.empty()) {
      return false;
    }
    for(auto it=odom_buffer_.begin();it!=odom_buffer_.end();++it) {
      if(it->stamp==stamp){*pose=it->pose;return true;}
      auto next=std::next(it);
      if(next==odom_buffer_.end())break;
      if(it->stamp<stamp && stamp<next->stamp) {
        const double gap=(next->stamp-it->stamp).seconds();
        if(gap>max_pose_gap_)return false;
        const double alpha=(stamp-it->stamp).seconds()/gap;
        const auto& a=it->pose;const auto& b=next->pose;
        Eigen::Quaterniond qa(a.orientation.w,a.orientation.x,a.orientation.y,a.orientation.z);
        Eigen::Quaterniond qb(b.orientation.w,b.orientation.x,b.orientation.y,b.orientation.z);
        const auto q=qa.normalized().slerp(alpha,qb.normalized());
        pose->position.x=a.position.x+alpha*(b.position.x-a.position.x);
        pose->position.y=a.position.y+alpha*(b.position.y-a.position.y);
        pose->position.z=a.position.z+alpha*(b.position.z-a.position.z);
        pose->orientation.x=q.x();pose->orientation.y=q.y();pose->orientation.z=q.z();pose->orientation.w=q.w();
        return true;
      }
    }
    return false;
  }

  bool decodeDepth(const sensor_msgs::msg::Image& message, Eigen::MatrixXf* depth) {
    if (static_cast<int>(message.width) != width_ || static_cast<int>(message.height) != height_) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "Ignoring depth with dimensions %ux%u, expected %dx%d",
                           message.width, message.height, width_, height_);
      return false;
    }
    const bool float_depth = message.encoding == "32FC1";
    const bool uint_depth = message.encoding == "16UC1";
    if (!float_depth && !uint_depth) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "Ignoring unsupported depth encoding %s", message.encoding.c_str());
      return false;
    }
    if(message.is_bigendian)return false;
    const size_t element_bytes = float_depth ? sizeof(float) : sizeof(uint16_t);
    if (message.step < static_cast<uint32_t>(width_ * element_bytes) ||
        message.data.size() < static_cast<size_t>(message.step) * height_) {
      RCLCPP_WARN(get_logger(), "Ignoring truncated depth image");
      return false;
    }
    const int rows=height_/pixel_stride_,columns=width_/pixel_stride_;
    depth->resize(rows,columns);
    for (int row = 0; row < rows; ++row) {
      const auto* source = message.data.data() + static_cast<size_t>(row*pixel_stride_) * message.step;
      for (int column = 0; column < columns; ++column) {
        float value = 0.f;
        if (float_depth) {
          std::memcpy(&value, source + column*pixel_stride_ * sizeof(float), sizeof(float));
        } else {
          uint16_t millimetres = 0;
          std::memcpy(&millimetres, source + column*pixel_stride_ * sizeof(uint16_t), sizeof(uint16_t));
          value = static_cast<float>(millimetres) * 0.001f;
        }
        (*depth)(row, column) = underwater_wavemap_ros::sanitizeDepthMeasurement(value,min_range_);
      }
    }
    underwater_wavemap_ros::filterDepthEdges(depth,edge_absolute_,edge_relative_);
    valid_depth_fraction_=static_cast<double>((depth->array()>0).count())/depth->size();
    return true;
  }

  void depthCallback(const sensor_msgs::msg::Image::SharedPtr message) {
    if(localization_lost_)return;
    if(estimator_rewound_ || message->header.frame_id!=camera_frame_){++skipped_frames_;return;}
    const rclcpp::Time stamp(message->header.stamp);
    if(last_received_depth_ns_>=stamp.nanoseconds()){++skipped_frames_;return;}
    last_received_depth_ns_=stamp.nanoseconds();
    if(pending_depth_.size()>=2){pending_depth_.pop_front();++skipped_frames_;}
    pending_depth_.push_back({message,std::chrono::steady_clock::now()});
    processPending();
  }

  void processPending() {
    if(localization_lost_)return;
    if(estimator_rewound_ || pending_depth_.empty())return;
    auto pending=pending_depth_.front();
    const rclcpp::Time stamp(pending.message->header.stamp);
    if(last_integrated_ns_>0 && (stamp.nanoseconds()<=last_integrated_ns_ ||
       (integration_rate_>0 && (stamp.nanoseconds()-last_integrated_ns_)*1.e-9<1./integration_rate_))) {
      pending_depth_.pop_front();++throttled_frames_;return;
    }
    geometry_msgs::msg::Pose pose_message;
    if (!interpolatePose(stamp, &pose_message)) {
      const double wait=std::chrono::duration<double>(std::chrono::steady_clock::now()-pending.received).count();
      if(wait<pose_wait_)return;
      pending_depth_.pop_front();skipped_frames_++;
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "Skipping depth frame: no odometry within %.3f s", max_pose_gap_);
      return;
    }
    pending_depth_.pop_front();
    integrateDepth(pending.message,pose_message);
  }

  void integrateDepth(const sensor_msgs::msg::Image::SharedPtr& message,
                      const geometry_msgs::msg::Pose& pose_message) {
    Eigen::MatrixXf depth;
    if (!decodeDepth(*message, &depth)) {
      skipped_frames_++;
      return;
    }
    const auto& q = pose_message.orientation;
    const Eigen::Quaternionf rotation(static_cast<float>(q.w), static_cast<float>(q.x),
                                       static_cast<float>(q.y), static_cast<float>(q.z));
    if (rotation.norm() < 1e-6f) {
      RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                           "Skipping depth frame with zero-norm pose quaternion");
      skipped_frames_++;
      return;
    }
    const Eigen::Vector3f translation(static_cast<float>(pose_message.position.x),
                                      static_cast<float>(pose_message.position.y),
                                      static_cast<float>(pose_message.position.z));
    const wavemap::Transformation3D pose(rotation.normalized(), translation);
    const auto started=std::chrono::steady_clock::now();
    range_image_->setPose(pose);
    range_image_->getData() = depth.transpose();
    integrator_->integrate(*range_image_);
    integration_ms_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
    last_integrated_ns_=rclcpp::Time(message->header.stamp).nanoseconds();
    last_header_=message->header;
    integrated_frames_++;
    std_msgs::msg::UInt64 integrated_message;
    integrated_message.data = integrated_frames_;
    integrated_pub_->publish(integrated_message);
    if (publish_rate_==0 && integrated_frames_ % publish_every_n_frames_ == 0) {
      publishMap(message->header);
      publishStatus();
    }
  }

  void publishMap(const std_msgs::msg::Header& source_header) {
    const auto now=std::chrono::steady_clock::now();
    if(prune_period_==0 || std::chrono::duration<double>(now-last_prune_).count()>=prune_period_) {
      map_->prune();last_prune_=now;
    }
    const float occupied_threshold = std::log(occupied_probability_/(1.-occupied_probability_));
    std::vector<wavemap::Point3D> points;
    std::vector<float> values;
    std::map<int,std::vector<geometry_msgs::msg::Point>> levels;
    map_->forEachLeaf([&](const wavemap::OctreeIndex& index, wavemap::FloatingPoint value) {
      if (value > occupied_threshold) {
        points.push_back(wavemap::convert::nodeIndexToCenterPoint(index, map_->getMinCellWidth()));
        values.push_back(value);
        geometry_msgs::msg::Point p;
        p.x=points.back().x();p.y=points.back().y();p.z=points.back().z();
        levels[index.height].push_back(p);
      }
    });

    sensor_msgs::msg::PointCloud2 cloud;
    cloud.header = source_header;
    cloud.header.frame_id = world_frame_;
    cloud.height = 1;
    cloud.width = static_cast<uint32_t>(points.size());
    cloud.is_dense = true;
    cloud.is_bigendian = false;
    cloud.point_step = 16;
    cloud.row_step = cloud.point_step * cloud.width;
    const auto make_field = [](const std::string& name, uint32_t offset) {
      sensor_msgs::msg::PointField field;
      field.name = name;
      field.offset = offset;
      field.datatype = sensor_msgs::msg::PointField::FLOAT32;
      field.count = 1;
      return field;
    };
    cloud.fields = {make_field("x", 0), make_field("y", 4), make_field("z", 8),
                    make_field("occupancy_log_odds", 12)};
    cloud.data.resize(static_cast<size_t>(cloud.row_step) * cloud.height);
    for (size_t i = 0; i < points.size(); ++i) {
      float values_to_write[4] = {points[i].x(), points[i].y(), points[i].z(), values[i]};
      std::memcpy(cloud.data.data() + i * cloud.point_step, values_to_write, sizeof(values_to_write));
    }
    occupied_pub_->publish(cloud);

    visualization_msgs::msg::MarkerArray markers;
    visualization_msgs::msg::Marker clear;
    clear.header=cloud.header;clear.action=visualization_msgs::msg::Marker::DELETEALL;
    markers.markers.push_back(clear);
    for(auto& level:levels) {
      visualization_msgs::msg::Marker marker;
      marker.header=cloud.header;marker.ns="wavemap_occupied_cells";marker.id=level.first;
      marker.type=visualization_msgs::msg::Marker::CUBE_LIST;
      marker.action=visualization_msgs::msg::Marker::ADD;marker.pose.orientation.w=1.;
      const auto width=wavemap::convert::heightToCellWidth(map_->getMinCellWidth(),level.first);
      marker.scale.x=marker.scale.y=marker.scale.z=width;
      marker.color.r=1.f;marker.color.g=.55f;marker.color.b=.08f;marker.color.a=.9f;
      marker.points=std::move(level.second);markers.markers.push_back(std::move(marker));
    }
    occupied_cells_pub_->publish(markers);
    if(octomap_pub_) {
      const auto started=std::chrono::steady_clock::now();
      auto job=std::make_unique<ProjectionJob>();job->header=cloud.header;job->resolution=octomap_resolution_;
      job->snapshot.min_cell_width=map_->getMinCellWidth();
      map_->forEachLeaf([&](const wavemap::OctreeIndex& index,float value) {
        if(wavemap::OccupancyClassifier::isObserved(value))job->snapshot.leaves.push_back({index,value});
      });
      // Only the executor reads the mutable Wavemap. The worker owns immutable
      // leaves. One running + one newest pending snapshot bounds memory/latency.
      pending_projection_=std::move(job);
      projection_snapshot_ms_=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count();
      pollProjection();
    }
    published_frames_=integrated_frames_;occupied_leaves_=points.size();
    RCLCPP_INFO_THROTTLE(get_logger(), *get_clock(), 5000,
                         "Published %zu occupied cells after %zu integrated frames (%zu skipped)",
                         points.size(), integrated_frames_, skipped_frames_);
  }

  void publishStatus() {
    const auto now=std::chrono::steady_clock::now();
    const auto elapsed=std::chrono::duration<double>(now-last_status_wall_).count();
    const double hz=elapsed>0 ? (integrated_frames_-last_status_frames_)/elapsed : 0.;
    last_status_frames_=integrated_frames_;last_status_wall_=now;
    std::ostringstream stream;
    stream<<std::setprecision(9)<<"{\"backend\":\"wavemap\",\"world_frame\":"<<std::quoted(world_frame_)
      <<",\"integrated_frames\":"<<integrated_frames_<<",\"skipped_frames\":"<<skipped_frames_
      <<",\"throttled_frames\":"<<throttled_frames_<<",\"occupied_leaves\":"<<occupied_leaves_
      <<",\"memory_bytes\":"<<map_->getMemoryUsage()<<",\"integration_ms\":"<<integration_ms_
      <<",\"integration_hz\":"<<hz
      <<",\"valid_depth_fraction\":"<<valid_depth_fraction_<<",\"source_stamp_ns\":"<<last_integrated_ns_
      <<",\"estimator_rewound\":"<<(estimator_rewound_?"true":"false")
      <<",\"localization_lost\":"<<(localization_lost_?"true":"false")
      <<",\"octomap_projection_enabled\":"<<(octomap_pub_?"true":"false")
      <<",\"octomap_resolution_m\":"<<octomap_resolution_
      <<",\"octomap_free_cells\":"<<projection_stats_.free_cells
      <<",\"octomap_occupied_cells\":"<<projection_stats_.occupied_cells
      <<",\"octomap_mixed_unknown_cells\":"<<projection_stats_.mixed_unknown_cells
      <<",\"octomap_bytes\":"<<octomap_bytes_<<",\"octomap_projection_ms\":"<<projection_ms_
      <<",\"octomap_projection_async\":true,\"octomap_snapshot_ms\":"<<projection_snapshot_ms_
      <<",\"octomap_projected_stamp_ns\":"<<projected_stamp_ns_
      <<",\"position_crop\":false,\"min_cell_width_m\":"<<min_cell_width_
      <<",\"pose_matching\":\"capture-time interpolation, no extrapolation\"}";
    std_msgs::msg::String message;message.data=stream.str();status_pub_->publish(message);
    if(!status_path_.empty()){std::ofstream file(status_path_);file<<message.data<<'\n';}
  }

  bool saveMap(const std::string& path) {
    if (path.empty()) {
      return false;
    }
    map_->prune();
    const bool saved = wavemap::io::mapToFile(*map_, path);
    wavemap::MapBase::Ptr restored_map;
    const bool reloaded = saved && wavemap::io::fileToMap(path, restored_map) && restored_map &&
                          !restored_map->empty() &&
                          restored_map->getMinCellWidth() == map_->getMinCellWidth();
    if (reloaded) {
      RCLCPP_INFO(get_logger(), "Saved and reloaded Wavemap to %s (%zu nodes)",
                  path.c_str(), restored_map->size());
    } else {
      RCLCPP_ERROR(get_logger(), "Failed Wavemap save/reload validation for %s", path.c_str());
    }
    return reloaded;
  }

  void saveMapService(const std::shared_ptr<std_srvs::srv::Trigger::Request>,
                     std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
    if (map_output_path_.empty()) {
      response->success = false;
      response->message = "map_output_path parameter is empty";
      return;
    }
    response->success = saveMap(map_output_path_);
    response->message = response->success ? map_output_path_ : "map save failed";
  }

  int width_;
  int height_;
  int tree_height_;
  int publish_every_n_frames_;
  double fx_;
  double fy_;
  double cx_;
  double cy_;
  double min_range_;
  double max_range_;
  double max_pose_gap_;
  double min_cell_width_;
  std::string depth_topic_;
  std::string odom_topic_;
  std::string world_frame_;
  std::string map_output_path_;
  double octomap_resolution_=0.,projection_ms_=0.;
  size_t octomap_bytes_=0;
  underwater_wavemap_ros::ProjectionStats projection_stats_;
  std::future<ProjectionResult> projection_future_;
  std::unique_ptr<ProjectionJob> pending_projection_;
  double projection_snapshot_ms_=0.;
  int64_t projected_stamp_ns_=0;
  rclcpp::Publisher<octomap_msgs::msg::Octomap>::SharedPtr octomap_pub_;
  bool localization_lost_=false;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr localization_sub_;
  std::string camera_frame_,status_path_;
  int pixel_stride_;
  double integration_rate_,publish_rate_,pose_wait_,prune_period_,occupied_probability_;
  double edge_absolute_,edge_relative_,integration_ms_=0.,valid_depth_fraction_=0.;
  int64_t last_received_depth_ns_=0,last_integrated_ns_=0;
  bool estimator_rewound_=false;
  struct PendingDepth {
    sensor_msgs::msg::Image::SharedPtr message;
    std::chrono::steady_clock::time_point received;
  };
  std::deque<PendingDepth> pending_depth_;
  std::chrono::steady_clock::time_point last_prune_;
  std::chrono::steady_clock::time_point last_status_wall_=std::chrono::steady_clock::now();
  std_msgs::msg::Header last_header_;
  rclcpp::TimerBase::SharedPtr process_timer_,publish_timer_;
  std::deque<StampedPose> odom_buffer_;
  size_t integrated_frames_ = 0;
  size_t skipped_frames_ = 0;
  size_t throttled_frames_=0,published_frames_=0,occupied_leaves_=0,last_status_frames_=0;

  wavemap::HashedChunkedWaveletOctree::Ptr map_;
  std::shared_ptr<wavemap::PinholeCameraProjector> projector_;
  std::shared_ptr<wavemap::PosedImage<>> range_image_;
  std::shared_ptr<wavemap::Image<wavemap::Vector2D>> beam_offsets_;
  std::shared_ptr<wavemap::ContinuousRay> measurement_model_;
  std::unique_ptr<wavemap::HashedChunkedWaveletIntegrator> integrator_;

  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr depth_sub_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr odom_sub_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr occupied_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr occupied_cells_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_;
  rclcpp::Publisher<std_msgs::msg::UInt64>::SharedPtr integrated_pub_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr status_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_service_;
};

}  // namespace

int main(int argc, char** argv) {
  rclcpp::init(argc, argv);
  try {
    rclcpp::spin(std::make_shared<WavemapNode>());
  } catch (const std::exception& error) {
    RCLCPP_FATAL(rclcpp::get_logger("underwater_wavemap"), "%s", error.what());
    rclcpp::shutdown();
    return 1;
  }
  rclcpp::shutdown();
  return 0;
}
