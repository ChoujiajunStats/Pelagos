#include "native_visualizer.hpp"
#include <cmath>
#include <cstring>
#include <stdexcept>
namespace dn=dapeng_navigation;

NativeVisualizer::NativeVisualizer(rclcpp::Node& node,dn::MapManager& manager,float resolution,float threshold,int tree_height,
    std::string frame,bool empty_start,std::function<void(const Diagnostic&)> record)
  :node_(node),manager_(manager),resolution_(resolution),threshold_(threshold),tree_height_(tree_height),frame_(std::move(frame)),
   empty_start_(empty_start),record_(std::move(record)) {
  const auto boolean=[&](const std::string& name,bool value){names_.push_back("visualization."+name);node_.declare_parameter<bool>(names_.back(),value);};
  const auto number=[&](const std::string& name,double value){names_.push_back("visualization."+name);node_.declare_parameter<double>(names_.back(),value);};
  const auto count=[&](const std::string& name,int64_t value){names_.push_back("visualization."+name);node_.declare_parameter<int64_t>(names_.back(),value);};
  boolean("enabled",false);boolean("require_empty_start",true);boolean("height_relative_to_robot",true);
  number("update_hz",2.);number("local_radius_m",1.5);number("voxel_size_m",.1);
  number("height_min_m",-1.);number("height_max_m",1.);number("local_alpha",.95);number("blue_alpha",.35);
  number("read_budget_ms",20.);number("pose_max_age_s",1.);
  count("max_source_leaves",150000);count("max_local_voxels",60000);count("max_blue_points",100000);count("max_candidates",500000);
  number("trajectory_min_distance_m",.02);number("trajectory_min_period_s",.2);number("trajectory_max_interval_s",1.);
  number("trajectory_jump_m",.75);count("trajectory_max_points",5000);
  names_.push_back("visualization.colormap");node_.declare_parameter<std::string>(names_.back(),"rainbow");
  names_.push_back("visualization.mode");node_.declare_parameter<std::string>(names_.back(),"surface");
  boolean("robot_model",true);
  names_.push_back("visualization.robot_resource");node_.declare_parameter<std::string>(names_.back(),"package://underwater_navigation_ros/meshes/bluerov2_lowpoly.obj");
  boolean("lighting",true);number("surface_alpha",1.);number("highlight_seconds",3.);number("keyframe_seconds",10.);
  number("mesh_resolution_m",.1);number("update_radius_m",6.);number("mesh_read_budget_ms",8.);
  count("max_mesh_blocks_per_update",1);count("max_mesh_triangles",60000);count("max_mesh_blocks",128);
  for(const auto& key:{"surface_color","new_color"}){names_.push_back(std::string("visualization.")+key);
    node_.declare_parameter<std::vector<double>>(names_.back(),key==std::string("surface_color")?std::vector<double>{.32,.43,.53}:std::vector<double>{.12,.82,.86});}
  settings();
  const auto qos=rclcpp::QoS(1).transient_local();
  cloud_pub_=node_.create_publisher<sensor_msgs::msg::PointCloud2>("/mapping/explored_cloud",qos);
  voxels_pub_=node_.create_publisher<visualization_msgs::msg::MarkerArray>("/mapping/local_voxels",qos);
  robot_pub_=node_.create_publisher<visualization_msgs::msg::MarkerArray>("/visualization/robot_model",qos);
  surface_pub_=node_.create_publisher<visualization_msgs::msg::MarkerArray>("/visualization/surface_mesh",qos);
  visual_path_pub_=node_.create_publisher<nav_msgs::msg::Path>("/visualization/trajectory",qos);
  path_pub_=node_.create_publisher<nav_msgs::msg::Path>("/vio/trajectory",qos);
  pose_pub_=node_.create_publisher<Pose>("/vio/robot_pose",qos);
  metrics_pub_=node_.create_publisher<Diagnostic>("/mapping/visualization_metrics",qos);
  parameter_callback_=node_.add_on_set_parameters_callback([this](const std::vector<rclcpp::Parameter>& parameters){
    rcl_interfaces::msg::SetParametersResult result;result.successful=true;
    try{settings(parameters);}catch(const std::exception& e){result.successful=false;result.reason=e.what();}
    return result; // Pure validation. Worker reads parameters only after commit.
  });
  clear();
  thread_=std::thread([this]{loop();});
}
NativeVisualizer::~NativeVisualizer() {
  stop_=true;cv_.notify_all();if(thread_.joinable())thread_.join();
}
NativeVisualizer::Settings NativeVisualizer::settings(const std::vector<rclcpp::Parameter>& changes) const {
  std::map<std::string,rclcpp::Parameter> p;
  for(const auto& value:node_.get_parameters(names_))p.emplace(value.get_name().substr(14),value);
  for(const auto& value:changes) {
    if(value.get_name().rfind("visualization.",0)!=0)continue;
    const auto key=value.get_name().substr(14);
    if(!p.count(key))throw std::invalid_argument("unknown visualization parameter");
    p[key]=value;
  }
  const auto n=[&](const char* k){return p.at(k).as_double();};
  const auto b=[&](const char* k){return p.at(k).as_bool();};
  const auto positive=[&](const char* k){const auto v=p.at(k).as_int();if(v<=0||v>2000000)throw std::invalid_argument("invalid visualization count");return size_t(v);};
  Settings c;c.enabled=b("enabled");c.require_empty_start=b("require_empty_start");
  if(c.enabled&&c.require_empty_start&&!empty_start_)throw std::invalid_argument("online visualization requires an empty map_input; saved maps are offline-only");
  c.update_hz=n("update_hz");c.pose_max_age_s=n("pose_max_age_s");
  c.render.local_radius_m=n("local_radius_m");c.render.voxel_size_m=n("voxel_size_m");
  c.render.height_min_m=n("height_min_m");c.render.height_max_m=n("height_max_m");
  c.render.height_relative_to_robot=b("height_relative_to_robot");c.render.local_alpha=n("local_alpha");c.render.blue_alpha=n("blue_alpha");
  c.render.read_budget_ms=n("read_budget_ms");c.render.max_source_leaves=positive("max_source_leaves");
  c.render.max_local_voxels=positive("max_local_voxels");c.render.max_blue_points=positive("max_blue_points");c.render.max_candidates=positive("max_candidates");
  c.colormap=p.at("colormap").as_string();
  if(c.colormap!="rainbow"&&c.colormap!="reversed_rainbow")throw std::invalid_argument("colormap must be rainbow or reversed_rainbow");
  c.render.reverse_colors=c.colormap=="reversed_rainbow";
  c.trajectory_max_points=positive("trajectory_max_points");c.trajectory_min_distance_m=n("trajectory_min_distance_m");
  c.trajectory_min_period_s=n("trajectory_min_period_s");c.trajectory_max_interval_s=n("trajectory_max_interval_s");c.trajectory_jump_m=n("trajectory_jump_m");
  c.mode=p.at("mode").as_string();if(c.mode!="surface"&&c.mode!="points")throw std::invalid_argument("visualization mode must be surface or points");
  c.robot_model=b("robot_model");c.robot_resource=p.at("robot_resource").as_string();
  c.lighting=b("lighting");c.alpha=n("surface_alpha");c.highlight_s=n("highlight_seconds");c.keyframe_s=n("keyframe_seconds");
  c.mesh.resolution_m=n("mesh_resolution_m");c.mesh.radius_m=n("update_radius_m");c.mesh.read_budget_ms=n("mesh_read_budget_ms");
  c.mesh.max_blocks_per_update=positive("max_mesh_blocks_per_update");c.mesh.max_triangles=positive("max_mesh_triangles");
  c.mesh.max_blocks=positive("max_mesh_blocks");c.mesh.max_source_leaves=c.render.max_source_leaves;
  for(const auto& key:{"surface_color","new_color"}){const auto color=p.at(key).as_double_array();
    if(color.size()!=3)throw std::invalid_argument("surface colors require three RGB values");
    for(double v:color)if(!std::isfinite(v)||v<0||v>1)throw std::invalid_argument("surface RGB outside [0,1]");
    auto& target=key==std::string("surface_color")?c.surface_color:c.new_color;std::copy(color.begin(),color.end(),target.begin());}
  if(!std::isfinite(c.alpha)||c.alpha<=0||c.alpha>1||!std::isfinite(c.highlight_s)||c.highlight_s<=0||!std::isfinite(c.keyframe_s)||c.keyframe_s<1)
    throw std::invalid_argument("invalid surface alpha or highlight/keyframe duration");
  c.mesh.validate(resolution_,tree_height_);
  c.render.validate(resolution_);
  for(double v:{c.update_hz,c.pose_max_age_s,c.trajectory_min_distance_m,c.trajectory_min_period_s,c.trajectory_max_interval_s,c.trajectory_jump_m})
    if(!std::isfinite(v)||v<=0)throw std::invalid_argument("visualization timing and trajectory distances must be positive");
  if(c.update_hz>5||c.render.read_budget_ms>50||c.trajectory_max_points>20000||c.trajectory_min_period_s>c.trajectory_max_interval_s)
    throw std::invalid_argument("visualization frequency, read budget or trajectory budget exceeds limits");
  return c;
}
void NativeVisualizer::observePose(const Pose& pose) {
  std::lock_guard<std::mutex> lock(pose_mutex_);
  if(have_pose_&&rclcpp::Time(pose.header.stamp)<=rclcpp::Time(latest_pose_.header.stamp))return;
  latest_pose_=pose;have_pose_=true; // Constant work on the odometry callback.
}
void NativeVisualizer::clear() {
  visualization_msgs::msg::MarkerArray markers;visualization_msgs::msg::Marker remove;
  remove.header.frame_id=frame_;remove.header.stamp=node_.now();remove.action=visualization_msgs::msg::Marker::DELETEALL;
  markers.markers.push_back(remove);voxels_pub_->publish(markers);surface_pub_->publish(markers);
  surface_tracker_.reset();surfaces_.clear();triangles_=0;
  sensor_msgs::msg::PointCloud2 cloud;cloud.header=remove.header;cloud.height=1;cloud.point_step=16;cloud.is_dense=true;
  for(int i=0;i<4;++i){sensor_msgs::msg::PointField field;field.name=std::array<std::string,4>{"x","y","z","rgba"}[i];field.offset=i*4;
    field.datatype=i==3?sensor_msgs::msg::PointField::UINT32:sensor_msgs::msg::PointField::FLOAT32;field.count=1;cloud.fields.push_back(field);}
  cloud_pub_->publish(cloud);
}
void NativeVisualizer::publishPoints(const dn::VisualizationFrame& data,const Settings& config) {
  using Marker=visualization_msgs::msg::Marker;
  visualization_msgs::msg::MarkerArray markers;Marker remove;
  remove.header.frame_id=frame_;remove.header.stamp=node_.now();remove.action=Marker::DELETEALL;markers.markers.push_back(remove);
  voxels_pub_->publish(markers); // Retired CUBE_LIST topic stays empty.
  sensor_msgs::msg::PointCloud2 cloud;cloud.header=remove.header;cloud.height=1;cloud.width=data.blue.size();
  cloud.is_dense=true;cloud.point_step=16;cloud.row_step=cloud.width*16;
  for(int i=0;i<4;++i){sensor_msgs::msg::PointField field;field.name=std::array<std::string,4>{"x","y","z","rgba"}[i];field.offset=i*4;
    field.datatype=i==3?sensor_msgs::msg::PointField::UINT32:sensor_msgs::msg::PointField::FLOAT32;field.count=1;cloud.fields.push_back(field);}
  cloud.data.resize(cloud.row_step);
  const uint32_t rgba=(uint32_t(std::lround(config.render.blue_alpha*255))<<24)|(45u<<16)|(135u<<8)|255u;
  for(size_t i=0;i<data.blue.size();++i){for(int axis=0;axis<3;++axis){float v=data.blue[i][axis];std::memcpy(cloud.data.data()+16*i+4*axis,&v,4);}std::memcpy(cloud.data.data()+16*i+12,&rgba,4);}
  cloud_pub_->publish(cloud);
}
void NativeVisualizer::diagnostic(const std::string& text,int level,const std::vector<std::pair<std::string,std::string>>& values) {
  Diagnostic d;d.name="native_incremental_visualization";d.hardware_id="dapeng";d.level=level;d.message=text;
  for(const auto& [key,value]:values){diagnostic_msgs::msg::KeyValue item;item.key=key;item.value=value;d.values.push_back(item);}
  metrics_pub_->publish(d);if(record_)record_(d);
}
visualization_msgs::msg::Marker NativeVisualizer::surfaceMarker(const dn::MeshKey& key,const Settings& config,int shade) const {
  using Marker=visualization_msgs::msg::Marker;const auto& item=surfaces_.at(key);Marker m;
  m.header.frame_id=frame_;m.header.stamp=node_.now();m.ns="native_occupancy_surface";m.id=item.id;
  m.type=Marker::TRIANGLE_LIST;m.action=Marker::ADD;m.pose.orientation.w=1.;m.scale.x=m.scale.y=m.scale.z=1.;
  const double t=shade/3.;m.color.r=config.surface_color[0]*(1-t)+config.new_color[0]*t;
  m.color.g=config.surface_color[1]*(1-t)+config.new_color[1]*t;m.color.b=config.surface_color[2]*(1-t)+config.new_color[2]*t;m.color.a=config.alpha;
  m.points.reserve(item.mesh.vertices.size());
  for(const auto& p:item.mesh.vertices){geometry_msgs::msg::Point point;point.x=p.x();point.y=p.y();point.z=p.z();m.points.push_back(point);}
  // Humble TRIANGLE_LIST computes normals from winding; vertex colors disable lighting.
  if(!config.lighting){auto color=m.color;color.a=1.;m.colors.assign(m.points.size(),color);}
  return m;
}
void NativeVisualizer::publishSurface(const dn::MeshCapture& data,const Settings& config) {
  using Marker=visualization_msgs::msg::Marker;visualization_msgs::msg::MarkerArray message;
  const auto began=dn::Clock::now();mesh_build_ms_=0;
  const auto remove=[&](int id){Marker m;m.header.frame_id=frame_;m.header.stamp=node_.now();m.ns="native_occupancy_surface";m.id=id;m.action=Marker::DELETE;message.markers.push_back(m);};
  for(auto it=surfaces_.begin();it!=surfaces_.end();) {
    if(!data.retained.count(it->first)){remove(it->second.id);triangles_-=it->second.mesh.vertices.size()/3;it=surfaces_.erase(it);}else ++it;
  }
  for(const auto& block:data.changed){
    try {
      auto mesh=dn::greedySurface(block,config.mesh);mesh_build_ms_+=mesh.build_ms;
      auto it=surfaces_.find(block.key);const size_t old=it==surfaces_.end()?0:it->second.mesh.vertices.size()/3;
      if(mesh.vertices.empty()){
        if(it!=surfaces_.end()){remove(it->second.id);triangles_-=old;surfaces_.erase(it);}
        surface_tracker_.acknowledge(block);continue;
      }
      if(triangles_-old+mesh.vertices.size()/3>config.mesh.max_triangles)throw std::runtime_error("total surface triangle budget exceeded");
      // A refreshed occupancy stamp with identical geometry must not flash cyan again.
      bool same=it!=surfaces_.end()&&it->second.mesh.vertices.size()==mesh.vertices.size();
      if(same)for(size_t i=0;i<mesh.vertices.size();++i)if(it->second.mesh.vertices[i]!=mesh.vertices[i]){same=false;break;}
      if(it==surfaces_.end()){CachedSurface item;item.id=next_surface_id_++;it=surfaces_.emplace(block.key,std::move(item)).first;}
      triangles_=triangles_-old+mesh.vertices.size()/3;
      if(!same){it->second.changed=began;it->second.shade=-1;}
      it->second.mesh=std::move(mesh);surface_tracker_.acknowledge(block);
    }catch(const std::exception& e){
      // Do not retain stale geometry when a changed block exceeds a budget.
      auto it=surfaces_.find(block.key);if(it!=surfaces_.end()){remove(it->second.id);triangles_-=it->second.mesh.vertices.size()/3;surfaces_.erase(it);}
      RCLCPP_WARN_THROTTLE(node_.get_logger(),*node_.get_clock(),5000,"Surface block omitted: %s",e.what());
    }
  }
  const auto subscribers=surface_pub_->get_subscription_count();
  const bool keyframe=subscribers!=previous_subscribers_||dn::milliseconds(began-keyframe_time_)>config.keyframe_s*1000;
  previous_subscribers_=subscribers;
  if(keyframe){message.markers.clear();Marker m;m.action=Marker::DELETEALL;m.header.frame_id=frame_;m.header.stamp=node_.now();message.markers.push_back(m);keyframe_time_=began;}
  for(auto& [key,item]:surfaces_){
    const double age=std::chrono::duration<double>(began-item.changed).count();
    const int shade=std::max(0,std::min(3,int(std::ceil(3.*(1.-age/config.highlight_s)))));
    if(keyframe||item.shade!=shade){message.markers.push_back(surfaceMarker(key,config,shade));item.shade=shade;}
  }
  if(!message.markers.empty())surface_pub_->publish(message);
}
void NativeVisualizer::loop() {
  while(!stop_&&rclcpp::ok()) {
    double period=1.;
    try {
      const auto config=settings();period=1./config.update_hz;const auto began=dn::Clock::now();
      Pose pose;bool available=false;
      {std::lock_guard<std::mutex> lock(pose_mutex_);available=have_pose_;if(available)pose=latest_pose_;}
      const double age=available?(node_.now()-rclcpp::Time(pose.header.stamp)).seconds():1e9;
      if(!available||age<-.5||age>config.pose_max_age_s) {
        clear();diagnostic("waiting_for_fresh_robot_pose",Diagnostic::WARN,{});
      } else {
        if(!trajectory_.empty()) {
          const auto& a=trajectory_.back().pose.position;const auto& b=pose.pose.position;
          const double distance=std::hypot(std::hypot(b.x-a.x,b.y-a.y),b.z-a.z);
          const double dt=(rclcpp::Time(pose.header.stamp)-rclcpp::Time(trajectory_.back().header.stamp)).seconds();
          if(distance>config.trajectory_jump_m){trajectory_.clear();++trajectory_resets_;}
          else if(dt>=config.trajectory_min_period_s&&(distance>=config.trajectory_min_distance_m||dt>=config.trajectory_max_interval_s))trajectory_.push_back(pose);
        }
        if(trajectory_.empty())trajectory_.push_back(pose);
        while(trajectory_.size()>config.trajectory_max_points)trajectory_.pop_front();
        nav_msgs::msg::Path path;path.header=pose.header;path.poses.assign(trajectory_.begin(),trajectory_.end());
        path_pub_->publish(path);visual_path_pub_->publish(path);pose_pub_->publish(pose);
        visualization_msgs::msg::MarkerArray vehicle;visualization_msgs::msg::Marker model;
        model.header=pose.header;model.ns="nominal_bluerov2";model.id=0;model.type=visualization_msgs::msg::Marker::MESH_RESOURCE;
        model.action=config.robot_model?visualization_msgs::msg::Marker::ADD:visualization_msgs::msg::Marker::DELETE;
        model.pose=pose.pose;model.scale.x=model.scale.y=model.scale.z=1.;model.mesh_resource=config.robot_resource;
        model.mesh_use_embedded_materials=false;model.color.r=.84;model.color.g=.89;model.color.b=.92;model.color.a=1.;
        vehicle.markers.push_back(model);robot_pub_->publish(vehicle);
        if(!config.enabled){clear();diagnostic("disabled; pose_and_trajectory_active",Diagnostic::WARN,{});}
        else {
          if(previous_mode_!=config.mode||previous_mesh_resolution_!=config.mesh.resolution_m){clear();previous_mode_=config.mode;previous_mesh_resolution_=config.mesh.resolution_m;}
          std::string style=std::to_string(config.alpha)+std::to_string(config.lighting);
          for(double x:config.surface_color)style+=std::to_string(x);
          for(double x:config.new_color)style+=std::to_string(x);
          if(style!=previous_style_){keyframe_time_={};previous_style_=style;}
          const auto& p=pose.pose.position;const wavemap::Point3D center{float(p.x),float(p.y),float(p.z)};
          bool captured=false;dn::MeshCapture mesh;dn::VisualizationSource source;
          captured=manager_.tryRead([&](const auto& map,auto version){
            if(config.mode=="surface")mesh=surface_tracker_.capture(map,version,center,config.mesh,threshold_);
            else source=dn::captureVisualization(map,version,config.render,threshold_);
          });
          if(captured){
            if(config.mode=="surface"){
              publishSurface(mesh,config);
              diagnostic(mesh.budget_limited?"surface_update_budget_limited":"native_incremental_surface",mesh.budget_limited?Diagnostic::WARN:Diagnostic::OK,
                {{"map_version",std::to_string(mesh.version)},{"triangles",std::to_string(triangles_)},{"mesh_blocks",std::to_string(surfaces_.size())},
                 {"changed_blocks",std::to_string(mesh.changed.size())},{"pending_blocks",std::to_string(mesh.pending)},
                 {"native_read_hold_ms",std::to_string(mesh.read_ms)},{"build_ms",std::to_string(mesh_build_ms_)},
                 {"total_ms",std::to_string(dn::milliseconds(dn::Clock::now()-began))},{"trajectory_resets",std::to_string(trajectory_resets_)},
                 {"history_policy","active_native_map_only; cropped_blocks_deleted"}});
            } else {
              auto c=config.render;c.local_radius_m=1e-6; // Low cost points only, never dense cubes.
              auto rendered=dn::renderVisualization(source,center,c);publishPoints(rendered,config);
              diagnostic("low_cost_observed_points",Diagnostic::OK,{{"map_version",std::to_string(source.version)},{"blue_points",std::to_string(rendered.blue.size())}});
            }
          }else diagnostic("native_map_busy; surface_update_deferred",Diagnostic::WARN,{});
        }
      }
    }catch(const std::exception& e){clear();diagnostic(std::string("visualization_error: ")+e.what(),Diagnostic::ERROR,{});}
    std::unique_lock<std::mutex> lock(wait_mutex_);cv_.wait_for(lock,std::chrono::duration<double>(period),[&]{return stop_.load();});
  }
}
