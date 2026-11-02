#include <iostream>
#include <stdexcept>
#include "startup_blind_mask.hpp"
void require(bool value,const char* message) {if(!value)throw std::runtime_error(message);}
int main() {
  try {
    Eigen::Matrix3d camera;camera<<0,0,1,-1,0,0,0,-1,0;
    pelagos::StartupBlindMask mask(.1,{0,0,0},{.16,-.0725,-.15},camera,{1,0,0},{.5,.5,.3},1.5,.3,75.,59.840444);
    using P=octomap::point3d;using S=pelagos::CellState;
    require(mask.contains(P(0,0,0)),"Initial self cells must retain their exception");
    require(mask.contains(P(.35,0,0)),"Calibrated near blind cells must be eligible");
    require(!mask.contains(P(1.05,0,0)),"Visible corridor unknown must not be exempt");
    require(!mask.contains(P(.35,2,0)),"Cells outside initial swept footprint must not be exempt");
    octomap::OcTree grid(.1);auto near=grid.coordToKey(P(.35,0,0)),far=grid.coordToKey(P(1.05,0,0));
    const auto query=[&](S near_state,S far_state) {
      return [&,near_state,far_state](const P& p) {
        auto key=grid.coordToKey(p);return key==near?near_state:key==far?far_state:S::Free;
      };
    };
    require(mask.traversable(P(0,0,0),P(1.5,0,0),query(S::Unknown,S::Free)),"Near blind unknown + observed corridor must pass");
    require(!mask.traversable(P(0,0,0),P(1.5,0,0),query(S::Occupied,S::Free)),"Occupied inside blind region must reject");
    require(!mask.traversable(P(0,0,0),P(1.5,0,0),query(S::Unknown,S::Unknown)),"Visible unknown must reject");
    require(!mask.traversable(P(1.05,0,0),P(2,0,0),query(S::Free,S::Unknown)),"Exception must not follow a moving vehicle");
    require(pelagos::segmentIntersectsBox({0,0,0},{1.5,0,0},{1,0,0},{.3,.3,.2}),"Forward occupied wall must block the swept body");
    require(!pelagos::segmentIntersectsBox({0,0,0},{1.5,0,0},{1,2,0},{.3,.3,.2}),"Off-corridor occupied wall must not block");
    require(pelagos::segmentIntersectsBox({0,0,0},{0,0,0},{0,0,0},{.3,.3,.2}),"Occupied initial footprint must block");
    require(!pelagos::segmentIntersectsBox({0,0,0},{0,0,0},{1,0,0},{.3,.3,.2}),"Stationary clear footprint must pass");
    std::cout<<"PASS: initial self, calibrated blind, occupied rejection, visible unknown rejection, fixed spatial mask\n";
    return 0;
  }catch(const std::exception& e) {std::cerr<<e.what()<<'\n';return 1;}
}
