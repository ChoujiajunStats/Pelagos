#include <glog/logging.h>
#include "gbplanner/gbplanner.h"
#include "dapeng_planning_msgs/Plan.h"

class GoalPlanner {
 public:
  GoalPlanner() : nh_(), private_("~"), map_(nh_,private_), planner_(nh_,private_,&map_) {
    std::vector<double> size;
    private_.getParam("RobotParams/size", size);
    if (size.size() != 3) throw std::runtime_error("RobotParams/size is required");
    robot_size_ = Eigen::Vector3d(size[0],size[1],size[2]);
    service_ = nh_.advertiseService("omniplanner/plan", &GoalPlanner::plan, this);
  }
 private:
  bool plan(dapeng_planning_msgs::Plan::Request& request, dapeng_planning_msgs::Plan::Response& response) {
    if (request.header.frame_id != "world" || !map_.getStatus() ||
        request.map_sequence != map_.lastSequence()) return false;
    response.map_sequence = map_.lastSequence();
    const auto& root = request.root_pose.position;
    const auto& goal = request.goal_pose.position;
    if (map_.getBoxStatus({root.x,root.y,root.z},robot_size_,true) != kFree ||
        map_.getBoxStatus({goal.x,goal.y,goal.z},robot_size_,true) != kFree) return true;
    planner_msgs::planner_srv::Request upstream_request;
    upstream_request.header = request.header;
    upstream_request.root_pose = request.root_pose;
    upstream_request.bound_mode = upstream_request.kExactBound;
    planner_.rrg_->setLocalNavGoal({goal.x,goal.y,goal.z});
    planner_.setPlannerSrvReq(upstream_request);
    planner_.getLocalNavigationPath();
    planner_msgs::planner_srv::Response upstream_response;
    planner_.getPlannerSrvRes(upstream_response);
    response.path = upstream_response.path;
    response.status = upstream_response.status;
    for (size_t i=1; i<response.path.size(); ++i) {
      const auto& a=response.path[i-1].position;
      const auto& b=response.path[i].position;
      if (map_.getPathStatus({a.x,a.y,a.z},{b.x,b.y,b.z},robot_size_,true)!=kFree) {
        response.path.clear();
        break;
      }
    }
    return true;
  }
  ros::NodeHandle nh_, private_;
  MapManager map_;
  Gbplanner planner_;
  ros::ServiceServer service_;
  Eigen::Vector3d robot_size_;
};

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  ros::init(argc,argv,"dapeng_omniplanner");
  GoalPlanner planner;
  ros::spin();
}
