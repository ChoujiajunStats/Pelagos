#pragma once
#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <map>
#include <mutex>
#include <thread>
#include <rclcpp/rclcpp.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <diagnostic_msgs/msg/diagnostic_status.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <nav_msgs/msg/path.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include "dapeng_navigation/surface_mesh.hpp"

class NativeVisualizer {
 public:
  using Pose=geometry_msgs::msg::PoseStamped;
  using Diagnostic=diagnostic_msgs::msg::DiagnosticStatus;
  NativeVisualizer(rclcpp::Node& node,dapeng_navigation::MapManager& manager,float resolution,
                   float threshold,int tree_height,std::string frame,bool empty_start,std::function<void(const Diagnostic&)> record);
  ~NativeVisualizer();
  void observePose(const Pose& pose);
 private:
  struct Settings {
    dapeng_navigation::VisualizationConfig render;
    dapeng_navigation::MeshConfig mesh;
    std::string mode="surface";
    bool lighting=true,robot_model=true;
    std::string robot_resource="package://underwater_navigation_ros/meshes/bluerov2_lowpoly.obj";
    double alpha=1.,highlight_s=3.,keyframe_s=10.;
    std::array<double,3> surface_color{.32,.43,.53},new_color{.12,.82,.86};
    bool enabled=false,require_empty_start=true;
    double update_hz=1.,pose_max_age_s=1.,trajectory_min_distance_m=.02,trajectory_min_period_s=.2,
      trajectory_max_interval_s=1.,trajectory_jump_m=.75;
    size_t trajectory_max_points=5000;
    std::string colormap="rainbow";
  };
  Settings settings(const std::vector<rclcpp::Parameter>& changes={}) const;
  void loop();
  void publishPoints(const dapeng_navigation::VisualizationFrame& data,const Settings& config);
  void publishSurface(const dapeng_navigation::MeshCapture& data,const Settings& config);
  visualization_msgs::msg::Marker surfaceMarker(const dapeng_navigation::MeshKey& key,const Settings& config,int shade) const;
  void clear();
  void diagnostic(const std::string& text,int level,const std::vector<std::pair<std::string,std::string>>& values);
  rclcpp::Node& node_;
  dapeng_navigation::MapManager& manager_;
  float resolution_,threshold_;
  int tree_height_;
  dapeng_navigation::SurfaceTracker surface_tracker_;
  struct CachedSurface {dapeng_navigation::SurfaceBlock mesh;dapeng_navigation::Clock::time_point changed;int id=0,shade=-1;};
  std::map<dapeng_navigation::MeshKey,CachedSurface> surfaces_;
  int next_surface_id_=0;
  size_t previous_subscribers_=0;
  dapeng_navigation::Clock::time_point keyframe_time_{};
  std::string previous_mode_,previous_style_;
  double previous_mesh_resolution_=0;
  size_t triangles_=0;
  double mesh_build_ms_=0;
  std::string frame_;
  bool empty_start_;
  std::function<void(const Diagnostic&)> record_;
  std::vector<std::string> names_;
  std::atomic<bool> stop_{false};
  std::mutex wait_mutex_,pose_mutex_;
  std::condition_variable cv_;
  Pose latest_pose_;
  bool have_pose_=false;
  std::deque<Pose> trajectory_;
  size_t trajectory_resets_=0;
  std::thread thread_;
  rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr cloud_pub_;
  rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr voxels_pub_,surface_pub_,robot_pub_;
  rclcpp::Publisher<nav_msgs::msg::Path>::SharedPtr path_pub_,visual_path_pub_;
  rclcpp::Publisher<Pose>::SharedPtr pose_pub_;
  rclcpp::Publisher<Diagnostic>::SharedPtr metrics_pub_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr parameter_callback_;
};
