#include <iostream>
#include <limits>
#include <stdexcept>
#include "cruise_direction.hpp"
#include "forward_obstacle.hpp"

void require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
int main() {
  try {
    std::vector<Eigen::Vector3d> edge;
    for(int i=0;i<50;++i)edge.push_back({2.+.14*i/49.,3.-.97*i/49.,-.4+.02*i/49.});
    auto requested=edge;
    const auto first=pelagos::sourceEdgeCruiseDirection(edge,requested);
    require(first.valid,"Actual source edge geometry must set a direction");
    require(std::abs(first.direction.norm()-1.)<1.e-12&&first.direction.z()==0.,"Cruise direction must be horizontal and normalized");
    require(first.direction.y()<-.98&&first.direction.x()>0.,"Direction must follow source translation rather than sampled view yaw");
    const auto repeated=pelagos::sourceEdgeCruiseDirection(edge,requested);
    require(repeated.valid&&(repeated.direction-first.direction).norm()==0.,"Same cached edge retry must preserve direction");
    requested[20].x()+=.001;
    require(!pelagos::sourceEdgeCruiseDirection(edge,requested).valid,"Changed intermediate geometry must not be accepted as source edge");
    requested={edge.front(),edge.back()};
    require(!pelagos::sourceEdgeCruiseDirection(edge,requested).valid,"Truncated request cannot stand in for complete cached source edge");
    require(!pelagos::sourceEdgeCruiseDirection({},edge).valid,"No cached native edge must reject direction request");
    require(!pelagos::sourceEdgeCruiseDirection({edge.front()},{edge.front()}).valid,"Single pose has no source direction");
    std::vector<Eigen::Vector3d> vertical{{0,0,0},{0,0,1}};
    require(!pelagos::sourceEdgeCruiseDirection(vertical,vertical).valid,"Vertical source edge cannot establish forward heading");
    std::vector<Eigen::Vector3d> invalid{{0,0,0},{std::numeric_limits<double>::quiet_NaN(),1,0}};
    require(!pelagos::sourceEdgeCruiseDirection(invalid,invalid).valid,"Nonfinite geometry must reject without new direction");
    std::vector<Eigen::Vector3d> turned{{0,0,0},{0,-1,0}};
    const auto turn=pelagos::sourceEdgeCruiseDirection(turned,turned);
    require(turn.valid&&(turn.direction-Eigen::Vector3d(0,-1,0)).norm()==0.,"Source -Y leg must become current forward");
    const Eigen::Vector3d origin(20,10,-3),box(.5,.5,.3);
    const std::vector<std::pair<Eigen::Vector3d,double>> occupied{{origin+Eigen::Vector3d(0,-3.3,0),.1}};
    const auto contact=pelagos::nearestForwardObstacle(origin,turn.direction,box,occupied);
    require(contact.found&&std::abs(contact.front_clearance_m-3.)<1.e-12,"Existing clearance must use the new direction and actual cruise origin");
    require(!pelagos::nearestForwardObstacle(origin,{1,0,0},box,occupied).found,"Old forward direction would miss the turned-corridor obstacle");
    std::cout<<"PASS: cached source-edge proof, unchanged geometry, horizontal direction, idempotent retry, invalid rejection, rotated occupied clearance\n";
    return 0;
  }catch(const std::exception& error) {std::cerr<<error.what()<<'\n';return 1;}
}
