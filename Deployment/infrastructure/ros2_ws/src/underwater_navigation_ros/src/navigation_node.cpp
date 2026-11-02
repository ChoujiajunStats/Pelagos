#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstring>
#include <deque>
#include <limits>
#include <optional>
#include <thread>
#include <Eigen/Geometry>
#include <rclcpp/rclcpp.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/u_int64.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <tf2_ros/buffer.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_ros/transform_broadcaster.h>
#include <tf2_ros/static_transform_broadcaster.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <yaml-cpp/yaml.h>
#include <wavemap/core/integrator/measurement_model/continuous_ray.h>
#include <wavemap/core/integrator/projective/coarse_to_fine/hashed_wavelet_integrator.h>
#include <wavemap/core/integrator/projective/coarse_to_fine/hashed_chunked_wavelet_integrator.h>
#include <wavemap/core/integrator/projection_model/pinhole_camera_projector.h>
#include <wavemap/io/file_conversions.h>
#include "dapeng_navigation/planner.hpp"
#include "dapeng_navigation/recorder.hpp"
#include "native_visualizer.hpp"

namespace dn=dapeng_navigation;
using Diagnostic=diagnostic_msgs::msg::DiagnosticStatus;
using Image=sensor_msgs::msg::Image;
using Odom=nav_msgs::msg::Odometry;
using PoseStamped=geometry_msgs::msg::PoseStamped;
using Cloud=sensor_msgs::msg::PointCloud2;
using Version=std_msgs::msg::UInt64;

class NavigationNode final:public rclcpp::Node {
 public:
  NavigationNode():Node("wavemap_navigation"),tf_buffer_(get_clock()),tf_listener_(tf_buffer_),
                   tf_broadcaster_(*this),static_tf_(*this) {
    source_world_=declare_parameter<std::string>("frames.source_world","world");
    source_body_=declare_parameter<std::string>("frames.source_body","body");
    odom_frame_=declare_parameter<std::string>("frames.odom","odom");
    base_frame_=declare_parameter<std::string>("frames.base","base_link");
    camera_frame_=declare_parameter<std::string>("frames.camera","cam0_rect_optical");
    const auto translation=declare_parameter<std::vector<double>>("frames.body_to_base_translation_m",{0.,0.,0.});
    const auto rotation=declare_parameter<std::vector<double>>("frames.body_to_base_quaternion_xyzw",{0.,0.,0.,1.});
    if(translation.size()!=3||rotation.size()!=4)throw std::invalid_argument("invalid body-to-base extrinsic size");
    body_to_base_translation_={translation[0],translation[1],translation[2]};
    body_to_base_rotation_={rotation[3],rotation[0],rotation[1],rotation[2]};
    if(!body_to_base_translation_.allFinite()||!body_to_base_rotation_.coeffs().allFinite()||
       body_to_base_rotation_.norm()<1e-6)throw std::invalid_argument("nonfinite body-to-base extrinsic");
    body_to_base_rotation_.normalize();
    for(const auto& frame:{source_world_,source_body_,odom_frame_,base_frame_,camera_frame_})
      if(frame.empty()||frame.front()=='/'||frame.find(' ')!=std::string::npos)throw std::invalid_argument("invalid TF frame");
    width_=declare_parameter<int>("mapping.width",960);height_=declare_parameter<int>("mapping.height",600);
    const float fx=declare_parameter<double>("mapping.fx",655.41835425112);
    const float fy=declare_parameter<double>("mapping.fy",655.41835425112);
    const float cx=declare_parameter<double>("mapping.cx",481.3433380126953);
    const float cy=declare_parameter<double>("mapping.cy",299.0089359283447);
    min_range_=declare_parameter<double>("mapping.min_range_m",.3);
    max_range_=declare_parameter<double>("mapping.max_range_m",12.);
    max_gap_=declare_parameter<double>("mapping.max_pose_gap_s",.1);
    pose_wait_=declare_parameter<double>("mapping.pose_wait_timeout_s",.5);
    max_pose_age_=declare_parameter<double>("planner.max_pose_age_s",3.);
    max_snapshot_age_=declare_parameter<double>("planner.max_snapshot_age_s",3.);
    max_depth_age_=declare_parameter<double>("planner.max_depth_age_s",3.);
    map_output_=declare_parameter<std::string>("mapping.map_output_path","");
    const auto input=declare_parameter<std::string>("mapping.map_input_path","");
    const auto calibration=declare_parameter<std::string>("mapping.calibration_path","");
    auto metrics_path=declare_parameter<std::string>("recording.metrics_path","");
    const bool recording=declare_parameter<bool>("recording.enabled",true);
    if(metrics_path.empty()&&!map_output_.empty()) {
      const auto slash=map_output_.find_last_of('/');
      metrics_path=(slash==std::string::npos?std::string{}:map_output_.substr(0,slash+1))+"navigation-metrics.jsonl";
    }
    recorder_=std::make_unique<dn::Recorder>(recording?metrics_path:std::string{});
    if(width_<=0||height_<=0||!std::isfinite(fx)||!std::isfinite(fy)||fx<=0||fy<=0||
       !std::isfinite(cx)||!std::isfinite(cy)||!std::isfinite(min_range_)||min_range_<=0||
       !std::isfinite(max_range_)||max_range_<=min_range_)
      throw std::invalid_argument("invalid camera/range configuration");
    for(double value:{max_gap_,pose_wait_,max_pose_age_,max_snapshot_age_,max_depth_age_})
      if(!std::isfinite(value)||value<=0)throw std::invalid_argument("invalid timing configuration");
    if(!calibration.empty()) {
      const auto c=YAML::LoadFile(calibration);
      if(c["width"].as<int>()!=width_||c["height"].as<int>()!=height_||
         std::abs(c["P_left"][0][0].as<double>()-fx)>.001 ||
         std::abs(c["P_left"][1][1].as<double>()-fy)>.001 ||
         std::abs(c["P_left"][0][2].as<double>()-cx)>.001 ||
         std::abs(c["P_left"][1][2].as<double>()-cy)>.001 ||
         c["camera_frame"].as<std::string>()!=camera_frame_)
        throw std::invalid_argument("navigation camera geometry differs from frozen LAS2 calibration");
    }
    wavemap::HashedWaveletOctreeConfig map_config;
    map_config.min_cell_width=declare_parameter<double>("mapping.min_cell_width_m",.1);
    map_config.tree_height=declare_parameter<int>("mapping.tree_height",6);
    if(!std::isfinite(float(map_config.min_cell_width))||map_config.min_cell_width<=0||
       map_config.tree_height<3||map_config.tree_height>9)throw std::invalid_argument("invalid map geometry");
    const auto type=declare_parameter<std::string>("mapping.type","hashed_wavelet_octree");
    if(type!="hashed_wavelet_octree"&&type!="hashed_chunked_wavelet_octree")throw std::invalid_argument("unsupported map type");
    if(!input.empty()) {
      if(!wavemap::io::fileToMap(input,map_)||!map_||map_->empty())throw std::runtime_error("failed to load initial native map");
      if(std::abs(map_->getMinCellWidth()-map_config.min_cell_width)>1e-6||map_->getTreeHeight()!=map_config.tree_height)
        throw std::invalid_argument("saved map geometry does not match configuration");
    } else if(type=="hashed_wavelet_octree")map_=std::make_shared<wavemap::HashedWaveletOctree>(map_config);
    else {
      wavemap::HashedChunkedWaveletOctreeConfig chunk;
      chunk.min_cell_width=map_config.min_cell_width;chunk.tree_height=map_config.tree_height;
      map_=std::make_shared<wavemap::HashedChunkedWaveletOctree>(chunk);
    }
    dn::ManagementConfig management;
    management.threshold_period_s=declare_parameter<double>("management.threshold_period_s",2.);
    management.prune_period_s=declare_parameter<double>("management.prune_period_s",10.);
    management.crop_period_s=declare_parameter<double>("management.crop_period_s",1.);
    management.crop_radius_m=declare_parameter<double>("management.crop_radius_m",6.);
    manager_=std::make_unique<dn::MapManager>(map_,management);
    dn::PlanningMapConfig pm;
    pm.robot_radius_m=declare_parameter<double>("planning_map.robot_radius_m",.35);
    pm.occupancy_threshold_log_odds=declare_parameter<double>("planning_map.occupancy_threshold_log_odds",0.);
    const auto extent=declare_parameter<std::vector<double>>("planning_map.half_extent_m",{3.,3.,2.});
    if(extent.size()!=3)throw std::invalid_argument("planning half extent must have three values");
    pm.half_extent_m={float(extent[0]),float(extent[1]),float(extent[2])};
    pm.max_roi_cells=positiveBudget("planning_map.max_roi_cells",500000);
    pm.max_mirror_bytes=positiveBudget("planning_map.max_mirror_bytes",64*1024*1024);
    pm.validate(map_->getMinCellWidth());threshold_=pm.occupancy_threshold_log_odds;
    planning_half_extent_=pm.half_extent_m;
    const auto max_blocks=positiveBudget("planning_map.max_snapshot_blocks",128);
    tracker_=std::make_unique<dn::DeltaTracker>(max_blocks,pm.max_mirror_bytes);
    builder_=std::make_unique<dn::PlanningMapBuilder>(pm);
    snapshot_period_=declare_parameter<double>("planning_map.update_period_s",.5);
    if(!std::isfinite(snapshot_period_)||snapshot_period_<=0)throw std::invalid_argument("invalid planning update period");
    planner_config_.algorithm=declare_parameter<std::string>("planner.algorithm","multi_res_theta_star");
    planner_config_.refinement_epsilon=declare_parameter<double>("planner.refinement_epsilon",.01);
    planner_config_.initialization_min_height=declare_parameter<int>("planner.initialization_min_height",0);
    planner_config_.max_result_latency_ms=declare_parameter<double>("planner.max_result_latency_ms",2000.);
    planner_config_.max_collision_cells=positiveBudget("planner.max_collision_cells",2000000);
    planner_config_.validate(map_->getTreeHeight());
    projector_=std::make_shared<wavemap::PinholeCameraProjector>(wavemap::PinholeCameraProjectorConfig{fx,fy,cx,cy,height_,width_});
    ranges_=std::make_shared<wavemap::PosedImage<>>(width_,height_);
    offsets_=std::make_shared<wavemap::Image<wavemap::Vector2D>>(width_,height_);
    model_=std::make_shared<wavemap::ContinuousRay>(wavemap::ContinuousRayConfig{.03f,.20f,.40f},projector_,ranges_);
    if(auto map=std::dynamic_pointer_cast<wavemap::HashedWaveletOctree>(map_))
      integrator_=std::make_shared<wavemap::HashedWaveletIntegrator>(wavemap::ProjectiveIntegratorConfig{min_range_,max_range_},projector_,ranges_,offsets_,model_,map);
    else if(auto map=std::dynamic_pointer_cast<wavemap::HashedChunkedWaveletOctree>(map_))
      integrator_=std::make_shared<wavemap::HashedChunkedWaveletIntegrator>(wavemap::ProjectiveIntegratorConfig{min_range_,max_range_},projector_,ranges_,offsets_,model_,map);
    else throw std::invalid_argument("unsupported loaded map type");
    const auto durable=rclcpp::QoS(1).transient_local();
    version_pub_=create_publisher<Version>("/mapping/map_version",durable);
    planning_version_pub_=create_publisher<Version>("/mapping/planning_map_version",durable);
    frames_pub_=create_publisher<Version>("/mapping/integrated_frames",durable);
    map_pub_=create_publisher<Cloud>("/mapping/occupied",durable);
    path_pub_=create_publisher<nav_msgs::msg::Path>("/planning/path",durable);
    visual_path_pub_=create_publisher<nav_msgs::msg::Path>("/visualization/planned_path",durable);
    goal_pub_=create_publisher<PoseStamped>("/planning/goal_visualization",durable);
    status_pub_=create_publisher<Diagnostic>("/planning/status",durable);
    metrics_pub_=create_publisher<Diagnostic>("/mapping/metrics",durable);
    visualizer_=std::make_unique<NativeVisualizer>(*this,*manager_,map_->getMinCellWidth(),threshold_,map_->getTreeHeight(),odom_frame_,input.empty(),
      [this](const Diagnostic& message){
        std::vector<std::pair<std::string,std::string>> values;
        for(const auto& item:message.values)values.push_back({item.key,item.value});
        recorder_->write(now().nanoseconds(),"/mapping/visualization_metrics",message.message,values);
      });
    manager_->setOnCommit([this](uint64_t version){
      gate_.advanceMap(version,[&]{publishEmpty();
        if(has_goal_&&inputs_healthy_)status("map_updated",Diagnostic::WARN,{{"map_version",std::to_string(version)}});
      });
      recorder_->write(now().nanoseconds(),"/mapping/map_version","committed",{{"map_version",std::to_string(version)}});
      Version message;message.data=version;version_pub_->publish(message);
    });
    group_=create_callback_group(rclcpp::CallbackGroupType::Reentrant);
    rclcpp::SubscriptionOptions options;options.callback_group=group_;
    body_sub_=create_subscription<Odom>(declare_parameter<std::string>("inputs.body_odometry","/okvis/okvis_odometry"),rclcpp::SensorDataQoS(),
        [this](Odom::ConstSharedPtr message){bodyCallback(*message);},options);
    camera_sub_=create_subscription<Odom>(declare_parameter<std::string>("inputs.camera_odometry","/odometry/estimated"),rclcpp::SensorDataQoS(),
        [this](Odom::ConstSharedPtr message){cameraCallback(*message);},options);
    depth_sub_=create_subscription<Image>(declare_parameter<std::string>("inputs.depth","/stereo/depth"),rclcpp::SensorDataQoS().keep_last(2),
        [this](Image::ConstSharedPtr message){depthCallback(std::move(message));},options);
    goal_sub_=create_subscription<PoseStamped>("/planning/goal",rclcpp::QoS(10),
        [this](PoseStamped::ConstSharedPtr message){goalCallback(*message);});
    save_service_=create_service<std_srvs::srv::Trigger>("/save_map",[this](
        const std::shared_ptr<std_srvs::srv::Trigger::Request>,std::shared_ptr<std_srvs::srv::Trigger::Response> response){
      response->success=saveMap();response->message=response->success?map_output_:"map save failed or output path unset";
    });
    if(source_world_!=odom_frame_) {
      geometry_msgs::msg::TransformStamped transform;transform.header.frame_id=source_world_;
      transform.child_frame_id=odom_frame_;transform.header.stamp=now();transform.transform.rotation.w=1.;static_tf_.sendTransform(transform);
    }
    watchdog_=create_wall_timer(std::chrono::milliseconds(100),[this]{watchdog();});
    parameter_guard_=add_on_set_parameters_callback([](const std::vector<rclcpp::Parameter>& parameters){
      rcl_interfaces::msg::SetParametersResult result;result.successful=true;
      for(const auto& parameter:parameters)if(parameter.get_name()!="use_sim_time" && parameter.get_name().rfind("visualization.",0)!=0) {
        result.successful=false;result.reason="Navigation configuration is frozen at startup; use YAML/launch overrides and restart.";break;
      }
      return result;
    });
    status("idle_no_goal",Diagnostic::OK);
    if(!input.empty())manager_->update([](auto&,auto){});
    try {
      fusion_thread_=std::thread([this]{fusionLoop();});
      maintenance_thread_=std::thread([this]{maintenanceLoop();});
      builder_thread_=std::thread([this]{builderLoop();});
      planner_thread_=std::thread([this]{plannerLoop();});
    }catch(...){stopWorkers();throw;}
    RCLCPP_INFO(get_logger(),"Native navigation ready: %s -> %s, collision radius %.3f m, Wavestar %s",
                odom_frame_.c_str(),base_frame_.c_str(),pm.robot_radius_m,planner_config_.algorithm.c_str());
  }
  ~NavigationNode() override {
    visualizer_.reset();
    stopWorkers();
    recorder_.reset();
    if(manager_&&!map_output_.empty())saveMap();
  }
 private:
  void stopWorkers() {
    stopping_=true;depth_cv_.notify_all();builder_cv_.notify_all();planner_cv_.notify_all();
    for(auto* thread:{&fusion_thread_,&maintenance_thread_,&builder_thread_,&planner_thread_})if(thread->joinable())thread->join();
  }
  void requestPlanLocked(){++request_;request_time_=dn::Clock::now();}
  size_t positiveBudget(const std::string& name,int64_t fallback) {
    auto value=declare_parameter<int64_t>(name,fallback);if(value<=0)throw std::invalid_argument("invalid budget "+name);return value;
  }
  static wavemap::Point3D position(const geometry_msgs::msg::Point& p){return {float(p.x),float(p.y),float(p.z)};}
  bool validPose(const geometry_msgs::msg::Pose& pose) const {
    const auto& q=pose.orientation;const double norm=q.x*q.x+q.y*q.y+q.z*q.z+q.w*q.w;
    return position(pose.position).allFinite()&&(position(pose.position).array().abs()<100000.f).all()
        &&std::isfinite(norm)&&norm>.8&&norm<1.2;
  }
  bool validOdom(const Odom& message,const std::string& child) const {
    if(message.header.frame_id!=source_world_||message.child_frame_id!=child||!validPose(message.pose.pose))return false;
    try{const auto t=rclcpp::Time(message.header.stamp);const double age=(now()-t).seconds();return t.nanoseconds()>0&&age>=-.5&&age<=max_pose_age_;}
    catch(const std::exception&){return false;}
  }
  void bodyCallback(const Odom& message) {
    if(!validOdom(message,source_body_)){++rejected_inputs_;return;}
    const auto stamp=rclcpp::Time(message.header.stamp).nanoseconds();
    auto previous=last_body_stamp_.load();
    while(stamp>previous&&!last_body_stamp_.compare_exchange_weak(previous,stamp)){}
    if(stamp<=previous)return;
    geometry_msgs::msg::TransformStamped transform;transform.header=message.header;
    transform.header.frame_id=odom_frame_;transform.child_frame_id=base_frame_;
    const auto& p=message.pose.pose.position;const auto& q=message.pose.pose.orientation;
    const Eigen::Quaterniond rotation=Eigen::Quaterniond(q.w,q.x,q.y,q.z).normalized();
    const Eigen::Vector3d base=Eigen::Vector3d(p.x,p.y,p.z)+rotation*body_to_base_translation_;
    const auto orientation=rotation*body_to_base_rotation_;
    transform.transform.translation.x=base.x();transform.transform.translation.y=base.y();transform.transform.translation.z=base.z();
    transform.transform.rotation.w=orientation.w();transform.transform.rotation.x=orientation.x();
    transform.transform.rotation.y=orientation.y();transform.transform.rotation.z=orientation.z();tf_broadcaster_.sendTransform(transform);
    PoseStamped robot_pose;robot_pose.header=transform.header;
    robot_pose.pose.position.x=base.x();robot_pose.pose.position.y=base.y();robot_pose.pose.position.z=base.z();
    robot_pose.pose.orientation=transform.transform.rotation;
    visualizer_->observePose(robot_pose);
  }
  void cameraCallback(const Odom& message) {
    if(!validOdom(message,camera_frame_)){++rejected_inputs_;return;}
    std::lock_guard<std::mutex> lock(camera_mutex_);
    const rclcpp::Time stamp(message.header.stamp);
    if(!cameras_.empty()&&stamp<=cameras_.back().stamp)return;
    cameras_.push_back({stamp,message.pose.pose});while(cameras_.size()>512)cameras_.pop_front();depth_cv_.notify_all();
  }
  void depthCallback(Image::ConstSharedPtr message) {
    if(message->encoding!="32FC1"||int(message->width)!=width_||int(message->height)!=height_||
       message->header.frame_id!=camera_frame_||message->step<message->width*4||
       message->data.size()<size_t(message->step)*message->height){++rejected_inputs_;return;}
    try {
      const auto stamp=rclcpp::Time(message->header.stamp);const double age=(now()-stamp).seconds();
      if(stamp.nanoseconds()<=0||age<-.5||age>max_depth_age_){++rejected_inputs_;return;}
    }catch(const std::exception&){++rejected_inputs_;return;}
    std::lock_guard<std::mutex> lock(depth_mutex_);
    if(pending_depth_)++superseded_depth_;
    pending_depth_=std::move(message);depth_cv_.notify_all();
  }
  std::optional<geometry_msgs::msg::Pose> nearestCamera(const rclcpp::Time& stamp) {
    std::lock_guard<std::mutex> lock(camera_mutex_);double gap=max_gap_;
    std::optional<geometry_msgs::msg::Pose> result;
    for(const auto& entry:cameras_) {
      const double error=std::abs((entry.stamp-stamp).seconds());
      if(error<=gap){gap=error;result=entry.pose;}
    }
    return result;
  }
  std::optional<wavemap::Point3D> robotPoint() {
    try {
      const auto t=tf_buffer_.lookupTransform(odom_frame_,base_frame_,tf2::TimePointZero);
      const double age=(now()-rclcpp::Time(t.header.stamp)).seconds();
      if(age<-.5||age>max_pose_age_)return std::nullopt;
      wavemap::Point3D point{float(t.transform.translation.x),float(t.transform.translation.y),float(t.transform.translation.z)};
      if(point.allFinite())return point;
    }catch(const tf2::TransformException&){}
    return std::nullopt;
  }
  bool depthFresh() const {
    const auto stamp=last_depth_stamp_.load();if(stamp<=0)return false;
    const double age=(now()-rclcpp::Time(stamp,get_clock()->get_clock_type())).seconds();
    return -.5<=age&&age<=max_depth_age_;
  }
  void publishEmpty(){nav_msgs::msg::Path path;path.header.frame_id=odom_frame_;path.header.stamp=now();path_pub_->publish(path);visual_path_pub_->publish(path);}
  void status(const std::string& text,int level=Diagnostic::WARN,
              const std::vector<std::pair<std::string,std::string>>& values={}) {
    Diagnostic message;message.name="wavestar";message.hardware_id="dapeng";message.level=level;message.message=text;
    for(const auto& item:values){diagnostic_msgs::msg::KeyValue v;v.key=item.first;v.value=item.second;message.values.push_back(v);}
    diagnostic_msgs::msg::KeyValue v;v.key="stamp_ns";v.value=std::to_string(now().nanoseconds());message.values.push_back(v);
    recorder_->write(now().nanoseconds(),"/planning/status",text,values);
    status_pub_->publish(message);
  }
  void fail(const std::string& text){publishEmpty();status(text,Diagnostic::ERROR);}
  bool saveMap() {
    if(map_output_.empty())return false;
    bool success=false;manager_->read([&](const auto& map,auto){success=wavemap::io::mapToFile(map,map_output_);});return success;
  }
  void fusionLoop() {
    try {
      while(!stopping_) {
        Image::ConstSharedPtr image;
        {std::unique_lock<std::mutex> lock(depth_mutex_);depth_cv_.wait(lock,[&]{return stopping_||pending_depth_;});
         if(stopping_)break;
         image=std::move(pending_depth_);}
        const auto deadline=dn::Clock::now()+std::chrono::duration<double>(pose_wait_);
        auto pose=nearestCamera(rclcpp::Time(image->header.stamp));
        while(!pose&&!stopping_&&dn::Clock::now()<deadline) {
          std::unique_lock<std::mutex> lock(depth_mutex_);depth_cv_.wait_for(lock,std::chrono::milliseconds(10));lock.unlock();
          pose=nearestCamera(rclcpp::Time(image->header.stamp));
        }
        if(stopping_)break;
        if(!pose){++unmatched_depth_;continue;}
        const auto stamp=rclcpp::Time(image->header.stamp).nanoseconds();
        if(stamp<=last_depth_stamp_.load()){++rejected_inputs_;continue;}
        Eigen::MatrixXf depth(width_,height_);size_t valid_pixels=0;
        for(int row=0;row<height_;++row)for(int col=0;col<width_;++col) {
          uint32_t bits;std::memcpy(&bits,image->data.data()+size_t(row)*image->step+col*4,4);
          if(image->is_bigendian)bits=__builtin_bswap32(bits);
          float value;std::memcpy(&value,&bits,4);
          const bool valid=std::isfinite(value)&&min_range_<=value&&value<=max_range_;
          depth(col,row)=valid?value:std::numeric_limits<float>::quiet_NaN();valid_pixels+=valid;
        }
        if(!valid_pixels){++rejected_inputs_;continue;}
        const auto& q=pose->orientation;const Eigen::Quaternionf rotation(float(q.w),float(q.x),float(q.y),float(q.z));
        ranges_->setPose(wavemap::Transformation3D(rotation.normalized(),position(pose->position)));ranges_->getData()=std::move(depth);
        const auto start=dn::Clock::now();manager_->update([&](auto&,auto){integrator_->integrate(*ranges_);});
        last_fusion_ms_=dn::milliseconds(dn::Clock::now()-start);last_depth_stamp_=stamp;
        Version message;message.data=++integrated_frames_;frames_pub_->publish(message);
      }
    }catch(const std::exception& error){fail(std::string("fusion_error: ")+error.what());rclcpp::shutdown();}
  }
  Cloud cloudFromMap(const wavemap::MapBase& map) {
    std::vector<std::array<float,4>> points;
    map.forEachLeaf([&](const auto& index,float value){
      if(wavemap::OccupancyClassifier::isObserved(value)&&value>=threshold_) {
        if(points.size()>=100000)return;
        const auto p=wavemap::convert::nodeIndexToCenterPoint(index,map.getMinCellWidth());points.push_back({p.x(),p.y(),p.z(),value});
      }
    });
    Cloud cloud;cloud.header.frame_id=odom_frame_;cloud.header.stamp=now();cloud.height=1;cloud.width=points.size();
    cloud.is_dense=true;cloud.point_step=16;cloud.row_step=cloud.width*16;
    for(int i=0;i<4;++i){sensor_msgs::msg::PointField field;field.name=std::array<std::string,4>{"x","y","z","occupancy_log_odds"}[i];
      field.offset=i*4;field.datatype=sensor_msgs::msg::PointField::FLOAT32;field.count=1;cloud.fields.push_back(field);}
    cloud.data.resize(points.size()*16);if(!points.empty())std::memcpy(cloud.data.data(),points.data(),cloud.data.size());return cloud;
  }
  void metrics(const std::vector<std::pair<std::string,double>>& values) {
    Diagnostic message;message.name="wavemap_navigation_metrics";message.hardware_id="dapeng";message.level=Diagnostic::OK;message.message="measurement";
    std::vector<std::pair<std::string,std::string>> recorded;
    for(const auto& item:values){diagnostic_msgs::msg::KeyValue v;v.key=item.first;v.value=std::to_string(item.second);message.values.push_back(v);}
    for(const auto& item:message.values)recorded.push_back({item.key,item.value});
    recorded.push_back({"log_dropped",std::to_string(recorder_->dropped())});
    recorded.push_back({"log_failed",recorder_->failed()?"true":"false"});
    if(recorder_->failed()){message.level=Diagnostic::WARN;message.message="metrics_log_write_failed";}
    recorder_->write(now().nanoseconds(),"/mapping/metrics",message.message,recorded);
    metrics_pub_->publish(message);
  }
  void maintenanceLoop() {
    uint64_t captured=0;bool force=true;std::optional<wavemap::Point3D> previous_center;
    auto last=dn::Clock::now()-std::chrono::seconds(10);
    try {
      while(!stopping_) {
        const auto center=robotPoint();const auto result=manager_->tryMaintain(center);
        if(center&&previous_center&&(*center-*previous_center).norm()>.15f&&manager_->version()==captured)
          manager_->update([](auto&,auto){}); // Commit a moved rolling planning window.
        if(result) {
          force|=result->thresholded||result->pruned;
          if(result->thresholded||result->pruned||result->cropped)
            metrics({{"map_version",double(result->version)},{"native_memory_bytes",double(result->memory_after_bytes)},
                     {"maintenance_ms",result->total_ms},{"threshold_ms",result->threshold_ms},{"prune_ms",result->prune_ms},
                     {"crop_ms",result->crop_ms},{"removed_blocks",double(result->removed_blocks)}});
        }
        force|=force_capture_.exchange(false);
        if(center&&(manager_->version()!=captured||force)&&std::chrono::duration<double>(dn::Clock::now()-last).count()>=snapshot_period_) {
          const auto read_start=dn::Clock::now();dn::BlockDelta delta;Cloud cloud;bool available=false;
          try {
            available=manager_->tryRead([&](const auto& map,auto version){
              delta=tracker_->capture(map,version,force,dn::SnapshotRegion{*center,planning_half_extent_});cloud=cloudFromMap(map);
            });
          }catch(const std::exception& error) {
            planning_healthy_=false;force=true;last=dn::Clock::now();
            fail(std::string("planning_snapshot_error: ")+error.what());
          }
          if(available) {
            captured=delta.version;force=false;last=dn::Clock::now();previous_center=*center;
            metrics({{"capture_ms",delta.capture_ms},{"map_read_hold_ms",dn::milliseconds(dn::Clock::now()-read_start)},
                     {"snapshot_block_bytes",double(delta.copied_bytes)},
                     {"fusion_ms",last_fusion_ms_.load()},{"integrated_frames",double(integrated_frames_.load())},
                     {"depth_superseded",double(superseded_depth_.load())},{"depth_unmatched",double(unmatched_depth_.load())},
                     {"rejected_inputs",double(rejected_inputs_.load())}});
            {std::lock_guard<std::mutex> lock(builder_mutex_);
             if(pending_delta_)pending_delta_=dn::mergeDeltas(std::move(*pending_delta_),delta);else pending_delta_=std::move(delta);
             pending_center_=*center;}
            builder_cv_.notify_one();map_pub_->publish(cloud);
          }
        }
        std::unique_lock<std::mutex> lock(depth_mutex_);depth_cv_.wait_for(lock,std::chrono::milliseconds(100),[&]{return stopping_.load();});
      }
    }catch(const std::exception& error){fail(std::string("map_management_error: ")+error.what());rclcpp::shutdown();}
  }
  void builderLoop() {
    while(!stopping_) {
      dn::BlockDelta delta;wavemap::Point3D center;
      {std::unique_lock<std::mutex> lock(builder_mutex_);builder_cv_.wait(lock,[&]{return stopping_||pending_delta_.has_value();});
       if(stopping_)break;
       delta=std::move(*pending_delta_);pending_delta_.reset();center=pending_center_;}
      try {
        builder_->apply(delta);auto snapshot=builder_->generate(center);
        metrics({{"planning_map_version",double(snapshot->version)},{"roi_ms",snapshot->roi_ms},{"sdf_ms",snapshot->sdf_ms},
                 {"classification_ms",snapshot->classification_ms},{"roi_memory_bytes",double(snapshot->roi_memory_bytes)},
                 {"sdf_memory_bytes",double(snapshot->sdf_memory_bytes)},{"classified_nodes",double(snapshot->classified_nodes)}});
        Version message;message.data=snapshot->version;planning_version_pub_->publish(message);
        {std::lock_guard<std::mutex> lock(planner_mutex_);snapshot_=std::move(snapshot);planning_healthy_=true;requestPlanLocked();}
        planner_cv_.notify_one();
      }catch(const std::exception& error){planning_healthy_=false;force_capture_=true;fail(std::string("planning_map_error: ")+error.what());}
    }
  }
  void goalCallback(const PoseStamped& source) {
    has_goal_=false;
    const auto id=gate_.newGoal([&]{publishEmpty();});
    try {
      if(source.header.frame_id.empty()||!validPose(source.pose))throw std::invalid_argument("invalid goal pose/frame");
      if(rclcpp::Time(source.header.stamp).nanoseconds()!=0) {
        const double age=(now()-rclcpp::Time(source.header.stamp)).seconds();
        if(age<-.5||age>30.)throw std::invalid_argument("goal timestamp is stale or in the future");
      }
      PoseStamped target=source;
      if(source.header.frame_id!=odom_frame_) {
        const auto transform=tf_buffer_.lookupTransform(odom_frame_,source.header.frame_id,rclcpp::Time(source.header.stamp));
        tf2::doTransform(source,target,transform);
      }
      {std::lock_guard<std::mutex> lock(planner_mutex_);goal_=Goal{position(target.pose.position),id,dn::Clock::now()};has_goal_=true;requestPlanLocked();}
      target.header.frame_id=odom_frame_;goal_pub_->publish(target);
      status("goal_accepted",Diagnostic::OK,{{"goal_id",std::to_string(id)},
        {"x",std::to_string(target.pose.position.x)},{"y",std::to_string(target.pose.position.y)},{"z",std::to_string(target.pose.position.z)}});
      planner_cv_.notify_one();
    }catch(const std::exception& error){
      {std::lock_guard<std::mutex> lock(planner_mutex_);goal_.reset();requestPlanLocked();}
      fail(std::string("invalid_goal: ")+error.what());
    }
  }
  void plannerLoop() {
    uint64_t processed=0;
    while(!stopping_) {
      std::shared_ptr<const dn::PlanningSnapshot> snapshot;Goal goal;dn::Clock::time_point request_time;
      {std::unique_lock<std::mutex> lock(planner_mutex_);planner_cv_.wait(lock,[&]{return stopping_||(request_!=processed&&snapshot_&&goal_);});
       if(stopping_)break;
       processed=request_;snapshot=snapshot_;goal=*goal_;request_time=request_time_;}
      try {
        const auto start=robotPoint();
        if(!start||!depthFresh()||!planning_healthy_||std::chrono::duration<double>(dn::Clock::now()-snapshot->created).count()>max_snapshot_age_) {
          fail("stale_pose_or_planning_map");continue;
        }
        const auto result=dn::plan(*snapshot,*start,goal.position,planner_config_);
        const bool committed=gate_.commit(result.map_version,goal.id,[&]{
          const auto current=robotPoint();
          if(!current||!depthFresh()||!planning_healthy_||
             std::chrono::duration<double>(dn::Clock::now()-snapshot->created).count()>max_snapshot_age_) {
            publishEmpty();status("stale_inputs_at_commit",Diagnostic::ERROR);return;
          }
          if((*current-*start).norm()>.15f) {
            publishEmpty();status("start_moved_replanning",Diagnostic::WARN);
            {std::lock_guard<std::mutex> lock(planner_mutex_);requestPlanLocked();}planner_cv_.notify_one();return;
          }
          nav_msgs::msg::Path path;path.header.frame_id=odom_frame_;path.header.stamp=now();
          for(const auto& point:result.path){PoseStamped pose;pose.header=path.header;pose.pose.orientation.w=1.;
            pose.pose.position.x=point.x();pose.pose.position.y=point.y();pose.pose.position.z=point.z();path.poses.push_back(pose);}
          path_pub_->publish(path);visual_path_pub_->publish(path);
          status(result.status,result.status=="success"?Diagnostic::OK:Diagnostic::ERROR,
            {{"map_version",std::to_string(result.map_version)},{"goal_id",std::to_string(goal.id)},
             {"algorithm",planner_config_.algorithm},{"planning_ms",std::to_string(result.planning_ms)},
             {"planning_latency_ms",std::to_string(dn::milliseconds(dn::Clock::now()-request_time))},
             {"goal_age_ms",std::to_string(dn::milliseconds(dn::Clock::now()-goal.received))},
             {"collision_ms",std::to_string(result.collision_ms)},{"collision_free",result.collision_free?"true":"false"},
             {"path_points",std::to_string(result.path.size())},{"length_m",std::to_string(result.length_m)}});
        });
        if(!committed)++discarded_results_;
      }catch(const std::exception& error){fail(std::string("planner_error: ")+error.what());}
    }
  }
  void watchdog() {
    bool snapshot_fresh=false;
    {std::lock_guard<std::mutex> lock(planner_mutex_);snapshot_fresh=snapshot_&&
      std::chrono::duration<double>(dn::Clock::now()-snapshot_->created).count()<=max_snapshot_age_;}
    const bool fresh=robotPoint().has_value()&&depthFresh()&&planning_healthy_&&snapshot_fresh;
    inputs_healthy_=fresh;
    if(!fresh&&was_fresh_){publishEmpty();status("stale_pose_depth_or_planning_map",Diagnostic::ERROR);}
    if(fresh&&!was_fresh_){std::lock_guard<std::mutex> lock(planner_mutex_);requestPlanLocked();planner_cv_.notify_one();}
    was_fresh_=fresh;
  }
  struct CameraPose{rclcpp::Time stamp;geometry_msgs::msg::Pose pose;};
  struct Goal{wavemap::Point3D position;uint64_t id=0;dn::Clock::time_point received;};
  std::string source_world_,source_body_,odom_frame_,base_frame_,camera_frame_,map_output_;
  int width_,height_;float min_range_,max_range_,threshold_;
  double max_gap_,pose_wait_,max_pose_age_,max_snapshot_age_,max_depth_age_,snapshot_period_;
  Eigen::Vector3d body_to_base_translation_;Eigen::Quaterniond body_to_base_rotation_;
  wavemap::Point3D planning_half_extent_;
  tf2_ros::Buffer tf_buffer_;tf2_ros::TransformListener tf_listener_;
  tf2_ros::TransformBroadcaster tf_broadcaster_;tf2_ros::StaticTransformBroadcaster static_tf_;
  wavemap::MapBase::Ptr map_;std::unique_ptr<dn::MapManager> manager_;
  std::unique_ptr<dn::DeltaTracker> tracker_;std::unique_ptr<dn::PlanningMapBuilder> builder_;
  std::unique_ptr<dn::Recorder> recorder_;
  std::unique_ptr<NativeVisualizer> visualizer_;
  wavemap::PinholeCameraProjector::Ptr projector_;wavemap::PosedImage<>::Ptr ranges_;
  wavemap::Image<wavemap::Vector2D>::Ptr offsets_;std::shared_ptr<wavemap::ContinuousRay> model_;
  wavemap::ProjectiveIntegrator::Ptr integrator_;dn::PlannerConfig planner_config_;dn::ResultGate gate_;
  std::atomic<bool> stopping_{false};std::atomic<uint64_t> integrated_frames_{0},superseded_depth_{0},unmatched_depth_{0},rejected_inputs_{0};
  std::atomic<bool> force_capture_{false},planning_healthy_{false};
  std::atomic<bool> has_goal_{false},inputs_healthy_{false};
  std::atomic<uint64_t> discarded_results_{0};
  std::atomic<int64_t> last_body_stamp_{0},last_depth_stamp_{0};std::atomic<double> last_fusion_ms_{0};
  std::mutex depth_mutex_,camera_mutex_,builder_mutex_,planner_mutex_;
  std::condition_variable depth_cv_,builder_cv_,planner_cv_;
  Image::ConstSharedPtr pending_depth_;std::deque<CameraPose> cameras_;
  std::optional<dn::BlockDelta> pending_delta_;wavemap::Point3D pending_center_;
  std::shared_ptr<const dn::PlanningSnapshot> snapshot_;std::optional<Goal> goal_;uint64_t request_=0;bool was_fresh_=false;
  dn::Clock::time_point request_time_;
  std::thread fusion_thread_,maintenance_thread_,builder_thread_,planner_thread_;
  rclcpp::CallbackGroup::SharedPtr group_;
  rclcpp::Subscription<Odom>::SharedPtr body_sub_,camera_sub_;rclcpp::Subscription<Image>::SharedPtr depth_sub_;
  rclcpp::Subscription<PoseStamped>::SharedPtr goal_sub_;
  rclcpp::Publisher<Version>::SharedPtr version_pub_,planning_version_pub_,frames_pub_;
  rclcpp::Publisher<Cloud>::SharedPtr map_pub_;rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_,visual_path_pub_;
  rclcpp::Publisher<PoseStamped>::SharedPtr goal_pub_;rclcpp::Publisher<Diagnostic>::SharedPtr status_pub_,metrics_pub_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr save_service_;rclcpp::TimerBase::SharedPtr watchdog_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_guard_;
};
int main(int argc,char** argv) {
  rclcpp::init(argc,argv);google::InitGoogleLogging(argv[0]);
  try {
    auto node=std::make_shared<NavigationNode>();rclcpp::executors::MultiThreadedExecutor executor(rclcpp::ExecutorOptions(),4);
    executor.add_node(node);executor.spin();executor.remove_node(node);node.reset();
  }catch(const std::exception& error){RCLCPP_ERROR(rclcpp::get_logger("navigation"),"%s",error.what());rclcpp::shutdown();return 1;}
  rclcpp::shutdown();return 0;
}
