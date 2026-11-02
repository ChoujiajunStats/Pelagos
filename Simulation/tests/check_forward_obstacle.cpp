#include <cmath>
#include <iostream>
#include <stdexcept>
#include "forward_obstacle.hpp"

void require(bool value,const char* message){if(!value)throw std::runtime_error(message);}
void near(double actual,double expected,const char* message){require(std::abs(actual-expected)<1.e-10,message);}
int main(){try{
  using V=Eigen::Vector3d;
  const V body=V::Zero(),box(.5,.5,.3),direction(1,0,0);
  double distance=-1.;
  require(pelagos::forwardVoxelClearance(body,direction,box,{3.3,0,0},.1,&distance),"Front obstacle must intersect swept body");
  near(distance,3.,"Clearance must already be from body front, including voxel half width");
  require(!pelagos::forwardVoxelClearance(body,direction,box,{-1,0,0},.1,&distance),"Obstacle wholly behind vehicle must be excluded");
  require(!pelagos::forwardVoxelClearance(body,direction,box,{3.3,.301,0},.1,&distance),"Obstacle outside lateral footprint plus voxel half width must be excluded");
  require(!pelagos::forwardVoxelClearance(body,direction,box,{3.3,0,.201},.1,&distance),"Obstacle outside same-altitude vertical footprint must be excluded");
  require(pelagos::forwardVoxelClearance(body,direction,box,{.1,0,0},.1,&distance),"Known occupied overlapping current body must be included");
  near(distance,0.,"Already overlapping occupied cell has zero clearance");
  require(pelagos::forwardVoxelClearance(body,{1,1,0},box,{4,4,0},.1,&distance),"Diagonal direction must use exact slab intersection");
  near(distance,(4.-.3)*std::sqrt(2.),"Diagonal clearance must account for both lateral axes");
  require(pelagos::forwardVoxelClearance(body,{2,0,0},box,{1.2,0,0},.1,&distance),"Nonunit direction must normalize without changing metres");
  near(distance,.9,"A known obstacle inside the existing 1m extension must not disappear");
  std::vector<std::pair<V,double>> cells={{{5,0,0},.2},{{2,0,0},.1},{{-1,0,0},.1},{{1,2,0},.1}};
  auto nearest=pelagos::nearestForwardObstacle(body,direction,box,cells);
  require(nearest.found,"Nearest forward occupied cell must be reported");near(nearest.front_clearance_m,1.7,"Nearest occupied contact must win");
  require(nearest.center==V(2,0,0),"Reported voxel must identify the measured occupied leaf");
  require(!pelagos::nearestForwardObstacle(body,direction,box,{}).found,"No occupied evidence must remain absent, not an observed-free distance");
  std::cout<<"PASS: front clearance, behind/lateral/vertical exclusion, overlap, diagonal, normalization, 1m obstacle, nearest leaf and absent evidence\n";return 0;
}catch(const std::exception& error){std::cerr<<error.what()<<'\n';return 1;}}
