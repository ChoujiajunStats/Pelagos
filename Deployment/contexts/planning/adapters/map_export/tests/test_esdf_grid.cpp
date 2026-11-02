#include "esdf_grid.hpp"
#include <cassert>
#include <iostream>

int main() {
  dapeng::EsdfGrid g;
  assert(g.status({0,0,0})==dapeng::Unknown);
  std::vector<float> values(1000,1.0F);
  g.assign({-.5,-.5,-.5},.1,{10,10,10},values);
  assert(g.status({0,0,0})==dapeng::Free);
  assert(g.status({-.5001,0,0})==dapeng::Unknown);
  assert(g.status({1e99,0,0})==dapeng::Unknown);
  assert(g.status({std::numeric_limits<double>::quiet_NaN(),0,0})==dapeng::Unknown);
  assert(g.path({-.3,0,0},{.3,0,0},{.1,.1,.1},true)==dapeng::Free);
  for(int y=0;y<10;++y)for(int z=0;z<10;++z)values[(5*10+y)*10+z]=-1000;
  g.assign({-.5,-.5,-.5},.1,{10,10,10},values);
  assert(g.path({-.3,0,0},{.3,0,0},{.1,.1,.1},true)==dapeng::Unknown);
  assert(g.box({-.06,0,0},{.2,.2,.2},true)==dapeng::Unknown);
  g.augmentRobotBox({.05,0,0},{.2,.4,.4});
  assert(g.status({.05,0,0})==dapeng::Free);
  values[(5*10+5)*10+5]=0;
  g.assign({-.5,-.5,-.5},.1,{10,10,10},values);
  g.augmentRobotBox({.05,.05,.05},{.2,.2,.2});
  assert(g.status({.05,.05,.05})==dapeng::Occupied);
  assert(g.path({-.3,.05,.05},{.3,.05,.05},{.1,.1,.1},true)!=dapeng::Free);
  bool rejected=false;
  try{g.assign({0,0,0},.1,{10,10,10},{1});}catch(const std::invalid_argument&){rejected=true;}
  assert(rejected);
  g.clear();
  assert(g.status({0,0,0})==dapeng::Unknown);
  std::cout<<"ESDF indexing, unknown-space, collision-volume, segment and input validation checks passed\n";
}
