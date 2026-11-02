// Read-only diagnosis using the same fixed startup mask as the live backend.
#include <iostream>
#include <map>
#include <memory>
#include <tuple>
#include "startup_blind_mask.hpp"
int main(int argc,char** argv) {
  if(argc!=19)return 2; // full .ot, body pose 7, camera pose 7, min depth, HFOV, VFOV
  std::unique_ptr<octomap::AbstractOcTree> base(octomap::AbstractOcTree::read(argv[1]));
  auto* tree=dynamic_cast<octomap::OcTree*>(base.get());if(!tree)return 3;
  Eigen::Vector3d body(std::stod(argv[2]),std::stod(argv[3]),std::stod(argv[4]));
  Eigen::Quaterniond bq(std::stod(argv[8]),std::stod(argv[5]),std::stod(argv[6]),std::stod(argv[7]));
  Eigen::Vector3d camera(std::stod(argv[9]),std::stod(argv[10]),std::stod(argv[11]));
  Eigen::Quaterniond cq(std::stod(argv[15]),std::stod(argv[12]),std::stod(argv[13]),std::stod(argv[14]));
  double near=std::stod(argv[16]),h=std::stod(argv[17]),v=std::stod(argv[18]),pi=std::acos(-1.);
  double yaw=std::atan2(2*(bq.w()*bq.z()+bq.x()*bq.y()),1-2*(bq.y()*bq.y()+bq.z()*bq.z()));
  Eigen::Vector3d direction(std::cos(yaw),std::sin(yaw),0),box(.5,.5,.3);
  Eigen::Matrix3d rotation=cq.toRotationMatrix();
  pelagos::StartupBlindMask mask(tree->getResolution(),body,camera,rotation,direction,box,1.5,near,h,v);
  struct Failure {Eigen::Vector3d center,optical;bool occupied,partial_blind;};
  std::map<std::tuple<int,int,int>,Failure> failures;
  auto query=[&](const octomap::point3d& p) {
    auto* n=tree->search(p);auto state=!n?pelagos::CellState::Unknown:tree->isNodeOccupied(n)?pelagos::CellState::Occupied:pelagos::CellState::Free;
    if(state==pelagos::CellState::Occupied||(state==pelagos::CellState::Unknown&&!mask.contains(p))) {
      Eigen::Vector3d center(p.x(),p.y(),p.z()),optical=rotation.transpose()*(center-camera);
      bool partial_blind=false;
      for(int x=-1;x<=1;x+=2)for(int y=-1;y<=1;y+=2)for(int z=-1;z<=1;z+=2) {
        Eigen::Vector3d c=rotation.transpose()*(center+tree->getResolution()/2.*Eigen::Vector3d(x,y,z)-camera);
        partial_blind=partial_blind||c.z()<near||std::abs(c.x())>std::tan(h*pi/360.)*c.z()||std::abs(c.y())>std::tan(v*pi/360.)*c.z();
      }
      auto key=tree->coordToKey(p);failures[{key[0],key[1],key[2]}]={center,optical,state==pelagos::CellState::Occupied,partial_blind};
    }
    return state;
  };
  Eigen::Vector3d step;for(int i=0;i<3;++i)step[i]=box[i]/std::ceil((box[i]+.001)/tree->getResolution());
  int passed=0,total=0;
  for(double x=-box.x()/2.;x<=box.x()/2.;x+=step.x())for(double y=-box.y()/2.;y<=box.y()/2.;y+=step.y())
    for(double z=-box.z()/2.;z<=box.z()/2.;z+=step.z()) {
      Eigen::Vector3d a=body+Eigen::Vector3d(x,y,z),b=a+1.5*direction;++total;
      passed+=mask.traversable(octomap::point3d(a.x(),a.y(),a.z()),octomap::point3d(b.x(),b.y(),b.z()),query);
    }
  std::cout<<"{\"passed_rays\":"<<passed<<",\"total_rays\":"<<total<<",\"failures\":[";
  bool comma=false;for(const auto& entry:failures) {if(comma)std::cout<<',';comma=true;const auto& f=entry.second;
    std::cout<<"{\"center\":["<<f.center.x()<<','<<f.center.y()<<','<<f.center.z()<<"],\"optical\":["<<f.optical.x()<<','<<f.optical.y()<<','<<f.optical.z()
      <<"],\"occupied\":"<<(f.occupied?"true":"false")<<",\"intersects_blind_volume\":"<<(f.partial_blind?"true":"false")<<'}';}
  std::cout<<"]}\n";
}
