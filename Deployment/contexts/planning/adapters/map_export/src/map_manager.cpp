#include "map_manager/map_manager.h"
#include <cmath>
#include <set>

namespace {
dapeng::Point point(const Eigen::Vector3d& p) { return {p.x(), p.y(), p.z()}; }
Eigen::Vector3d eigen(const dapeng::Point& p) { return {p[0], p[1], p[2]}; }
}

MapManager::MapManager(ros::NodeHandle& nh, ros::NodeHandle& private_nh)
    : nh_(nh), nh_private_(private_nh) {
  nh_private_.param("map_timeout", map_timeout_, 5.0);
  grid_subscriber_ = nh_.subscribe("nvblox_esdf", 1, &MapManager::gridCallback, this);
}

void MapManager::gridCallback(const dapeng_planning_msgs::EsdfGrid::ConstPtr& message) {
  if (message->header.frame_id != "world" || message->header.stamp.isZero()) return;
  try {
    grid_.assign({message->origin.x, message->origin.y, message->origin.z}, message->voxel_size,
                 {message->size_x, message->size_y, message->size_z}, message->distance);
    last_map_stamp_ = message->header.stamp;
    last_map_sequence_ = message->sequence;
  } catch (const std::exception& error) {
    grid_.clear();
    ROS_ERROR_STREAM("Rejected nvblox ESDF: " << error.what());
  }
}

double MapManager::getResolution() { return grid_.resolution(); }
bool MapManager::getStatus() {
  return !grid_.empty() && !last_map_stamp_.isZero() &&
         std::abs((ros::Time::now() - last_map_stamp_).toSec()) <= map_timeout_;
}
VoxelStatus MapManager::getVoxelStatus(const Eigen::Vector3d& p) {
  return getStatus() ? static_cast<VoxelStatus>(grid_.status(point(p))) : kUnknown;
}
float MapManager::getVoxelDistance(const Eigen::Vector3d& p) {
  return getStatus() ? grid_.distance(point(p)) : std::numeric_limits<float>::quiet_NaN();
}
double MapManager::getPointDistance(const Eigen::Vector3d& p) {
  const float d = getVoxelDistance(p);
  return std::isfinite(d) ? d : -1.0;
}
Eigen::Vector3d MapManager::getPointGradient(const Eigen::Vector3d& p) {
  Eigen::Vector3d gradient = Eigen::Vector3d::Zero();
  for (int axis = 0; axis < 3; ++axis) {
    Eigen::Vector3d offset = Eigen::Vector3d::Zero(); offset[axis] = getResolution();
    const float a = getVoxelDistance(p + offset), b = getVoxelDistance(p - offset);
    if (std::isfinite(a) && std::isfinite(b)) gradient[axis] = (a - b) / (2 * getResolution());
  }
  return gradient;
}
VoxelStatus MapManager::getBoxStatus(const Eigen::Vector3d& center,
                                   const Eigen::Vector3d& size, bool stop) {
  return getStatus() ? static_cast<VoxelStatus>(grid_.box(point(center), point(size), stop)) : kUnknown;
}
VoxelStatus MapManager::getPathStatus(const Eigen::Vector3d& start, const Eigen::Vector3d& end,
                                    const Eigen::Vector3d& size, bool stop) {
  return getStatus() ? static_cast<VoxelStatus>(grid_.path(point(start), point(end), point(size), stop)) : kUnknown;
}
bool MapManager::augmentFreeBox(const Eigen::Vector3d& position, const Eigen::Vector3d& size) {
  if (!getStatus()) return false;
  grid_.augmentRobotBox(point(position), point(size));
  return true;
}
VoxelStatus MapManager::getRayStatus(const Eigen::Vector3d& start, const Eigen::Vector3d& end,
                                   bool stop) {
  Eigen::Vector3d voxel; double distance;
  return getRayStatus(start, end, stop, voxel, distance);
}
VoxelStatus MapManager::getRayStatus(const Eigen::Vector3d& start, const Eigen::Vector3d& end,
                                   bool stop, Eigen::Vector3d& voxel, double& distance) {
  const double length = (end - start).norm();
  if (!getStatus() || !std::isfinite(length) || length > 100) return kUnknown;
  const int steps = std::max(1, int(std::ceil(length / (getResolution() * .5))));
  for (int i = 0; i <= steps; ++i) {
    voxel = start + (end - start) * double(i) / steps;
    const VoxelStatus status = getVoxelStatus(voxel);
    distance = getPointDistance(voxel);
    if (status == kOccupied || (stop && status == kUnknown)) return status;
  }
  return kFree;
}

void MapManager::getScanStatus(Eigen::Vector3d& position,
    std::vector<Eigen::Vector3d>& endpoints, std::tuple<int,int,int>& gain,
    std::vector<std::pair<Eigen::Vector3d,VoxelStatus>>& log, SensorParamsBase&) {
  std::set<std::array<int,3>> seen;
  int unknown = 0, free = 0, occupied = 0;
  for (const auto& end : endpoints) {
    const double length = (end - position).norm();
    if (!std::isfinite(length) || length > 100) continue;
    const int steps = std::max(1, int(std::ceil(length / getResolution())));
    for (int i = 0; i <= steps; ++i) {
      const Eigen::Vector3d p = position + (end - position) * double(i) / steps;
      const std::array<int,3> key{int(std::floor(p.x()/getResolution())),
        int(std::floor(p.y()/getResolution())), int(std::floor(p.z()/getResolution()))};
      const VoxelStatus s = getVoxelStatus(p);
      if (seen.insert(key).second) {
        log.emplace_back(p,s);
        unknown += s == kUnknown; free += s == kFree; occupied += s == kOccupied;
      }
      if (s == kOccupied) break;
    }
  }
  gain = {unknown, free, occupied};
}
void MapManager::getCameraScanStatus(Eigen::Vector3d& position,
    std::vector<Eigen::Vector3d>& endpoints, std::tuple<int,int,int>& gain,
    std::vector<std::pair<Eigen::Vector3d,VoxelStatus>>& log, SensorParamsBase& sensor) {
  getScanStatus(position, endpoints, gain, log, sensor);
}
void MapManager::getCameraScanStatus(StateVec& state, SensorParamsBase& sensor,
                                    std::vector<VoxelLog>& logs) {
  std::vector<Eigen::Vector3d> endpoints;
  sensor.getFrustumEndpoints(state, endpoints);
  std::tuple<int,int,int> gain;
  std::vector<std::pair<Eigen::Vector3d,VoxelStatus>> scan;
  Eigen::Vector3d position = state.head<3>();
  getScanStatus(position, endpoints, gain, scan, sensor);
  for (const auto& entry : scan) if (entry.second == kOccupied) {
    const auto& p = entry.first;
    size_t h = std::hash<int>()(int(std::floor(p.x()/getResolution())));
    h ^= std::hash<int>()(int(std::floor(p.y()/getResolution()))) << 1;
    h ^= std::hash<int>()(int(std::floor(p.z()/getResolution()))) << 2;
    logs.emplace_back(p,h);
  }
}
void MapManager::annotateCameraVoxels(Eigen::Vector3d&, std::vector<Eigen::Vector3d>&) {}
void MapManager::augmentFreeFrustum() {}

void MapManager::extractLocalMap(const Eigen::Vector3d& center, const Eigen::Vector3d& size,
    std::vector<Eigen::Vector3d>& occupied, std::vector<Eigen::Vector3d>& free) {
  grid_.forEach([&](const dapeng::Point& p, dapeng::Status s) {
    const Eigen::Vector3d v = eigen(p);
    if (((v-center).cwiseAbs().array() > size.array()/2).any()) return;
    if (s == dapeng::Occupied) occupied.push_back(v);
    if (s == dapeng::Free) free.push_back(v);
  });
}
void MapManager::extractLocalMapAlongAxis(const Eigen::Vector3d& center,
    const Eigen::Vector3d& axis, const Eigen::Vector3d& size,
    std::vector<Eigen::Vector3d>& occupied, std::vector<Eigen::Vector3d>& free) {
  const Eigen::Vector3d x = axis.norm() > 1e-6 ? axis.normalized() : Eigen::Vector3d::UnitX();
  const Eigen::Vector3d helper = std::abs(x.z()) < .9 ? Eigen::Vector3d::UnitZ() : Eigen::Vector3d::UnitY();
  const Eigen::Vector3d y = helper.cross(x).normalized(), z = x.cross(y);
  grid_.forEach([&](const dapeng::Point& p, dapeng::Status s) {
    const Eigen::Vector3d v = eigen(p), d = v-center;
    if (std::abs(d.dot(x)) > size.x()/2 || std::abs(d.dot(y)) > size.y()/2 || std::abs(d.dot(z)) > size.z()/2) return;
    if (s == dapeng::Occupied) occupied.push_back(v);
    if (s == dapeng::Free) free.push_back(v);
  });
}
void MapManager::getFreeSpacePointCloud(std::vector<Eigen::Vector3d> endpoints, StateVec state,
                                       pcl::PointCloud<pcl::PointXYZ>::Ptr cloud) {
  const Eigen::Vector3d start = state.head<3>();
  for (const auto& end : endpoints) {
    const double length = (end-start).norm();
    if (!std::isfinite(length) || length > 100) continue;
    const int steps = std::max(1,int(std::ceil(length/getResolution())));
    for (int i = 0; i <= steps; ++i) {
      const Eigen::Vector3d p = start + (end-start)*double(i)/steps;
      const auto s = getVoxelStatus(p);
      if (s == kOccupied) break;
      if (s == kFree) cloud->push_back(pcl::PointXYZ(p.x(),p.y(),p.z()));
    }
  }
}
void MapManager::getLocalPointcloud(const Eigen::Vector3d& center, const double& range,
    const double& yaw, pcl::PointCloud<pcl::PointXYZI>& cloud, bool include_unknown) {
  const Eigen::Vector2d limits(-std::numeric_limits<double>::infinity(),std::numeric_limits<double>::infinity());
  getLocalPointcloud(center,range,yaw,limits,cloud,include_unknown);
}
void MapManager::getLocalPointcloud(const Eigen::Vector3d& center, const double& range,
    const double&, const Eigen::Vector2d& limits, pcl::PointCloud<pcl::PointXYZI>& cloud,
    bool include_unknown) {
  grid_.forEach([&](const dapeng::Point& p,dapeng::Status s) {
    const Eigen::Vector3d v=eigen(p);
    if ((v-center).norm()>range || v.z()<limits[0] || v.z()>limits[1] || (!include_unknown && s==dapeng::Unknown)) return;
    pcl::PointXYZI q; q.x=v.x();q.y=v.y();q.z=v.z();q.intensity=static_cast<int>(s);cloud.push_back(q);
  });
}
void MapManager::resetMap() { grid_.clear(); }
void MapManager::setRaycastingParams(bool, double) {}
void MapManager::setRobotRadius(double radius) { robot_radius_=radius; }
void MapManager::setBoxCheckMethod(int) {}
void MapManager::setLineCheckMethod(int) {}
