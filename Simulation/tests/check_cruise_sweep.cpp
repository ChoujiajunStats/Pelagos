#include <iostream>
#include <stdexcept>
#include "startup_blind_mask.hpp"

void require(bool condition,const char* message){if(!condition)throw std::runtime_error(message);}
int main(){try{
  using V=Eigen::Vector3d;
  // Same source dimensions as cruise: body .5/.5/.3, occupied voxel .1.
  const V half(.3,.3,.2),goal(1,0,0),overshoot(1.5,0,0);
  const V starts[]={V(0,.4,0),V(0,0,.257)};
  const V obstacles[]={V(1.05,-.29,0),V(1.05,0,-.19)};
  for(int case_index=0;case_index<2;++case_index){
    const V body=starts[case_index],obstacle=obstacles[case_index];
    require(!pelagos::segmentIntersectsBox(body,overshoot,obstacle,half),
      "Original overshoot chord alone must demonstrate the off-line collision gap");
    require(pelagos::segmentIntersectsBox(body,goal,obstacle,half),
      "Full actual-goal leg must reject the lateral/altitude tracking-error obstacle");
    bool sampled_collision=false;
    for(int i=0;i<50;++i){
      const V a=body+(double(i)/50.)*(goal-body);
      const V b=body+(double(i+1)/50.)*(goal-body);
      if(pelagos::segmentIntersectsBox(a,b,obstacle,half))sampled_collision=true;
    }
    require(sampled_collision,"Fixture must also reject through the previous sampled path");
  }
  // Retaining the original overshoot chord preserves its previous rejections.
  require(pelagos::segmentIntersectsBox(V::Zero(),overshoot,V(1.5,0,0),half),
    "Original overshoot-only obstacle must remain rejected");
  require(!pelagos::segmentIntersectsBox(V::Zero(),goal,V(1.5,0,0),half),
    "Overshoot fixture must be outside the commanded goal sweep");
  require(!pelagos::segmentIntersectsBox(starts[0],goal,V(1,2,2),half)&&
          !pelagos::segmentIntersectsBox(starts[0],overshoot,V(1,2,2),half),
    "Both complete sweeps must preserve a clear corridor");
  std::cout<<"PASS: full goal sweep covers source path, tracking-error collision gaps rejected, original overshoot coverage retained\n";
  return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
