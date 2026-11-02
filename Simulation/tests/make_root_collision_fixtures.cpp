#include <octomap/OcTree.h>
#include <string>
int main(int argc,char** argv) {
  if(argc!=2)return 2;
  octomap::OcTree map(.1);
  // Source body volume is unknown; the observed exit corridor starts just
  // outside it. These are isolated test maps, never loaded into the simulator.
  for(int x=3;x<23;++x)for(int y=-8;y<8;++y)for(int z=-6;z<6;++z)
    map.setNodeValue(octomap::point3d((x+.5)*.1,(y+.5)*.1,(z+.5)*.1),-.8,true);
  map.updateInnerOccupancy();map.write(std::string(argv[1])+"/free-exit.ot");
  map.deleteNode(octomap::point3d(1.25,.05,.05));
  map.write(std::string(argv[1])+"/unknown-endpoint.ot");
  map.setNodeValue(octomap::point3d(1.25,.05,.05),1.2,true);map.updateInnerOccupancy();
  map.write(std::string(argv[1])+"/occupied-endpoint.ot");
  map.setNodeValue(octomap::point3d(1.25,.05,.05),-.8,true);map.updateInnerOccupancy();
  map.deleteNode(octomap::point3d(.85,.05,.05));
  map.write(std::string(argv[1])+"/unknown-exit.ot");
  map.setNodeValue(octomap::point3d(.85,.05,.05),1.2,true);map.updateInnerOccupancy();
  map.write(std::string(argv[1])+"/occupied-exit.ot");
  map.setNodeValue(octomap::point3d(.85,.05,.05),-.8,true);
  map.setNodeValue(octomap::point3d(.05,.05,.05),1.2,true);map.updateInnerOccupancy();
  return map.write(std::string(argv[1])+"/occupied-self.ot")?0:1;
}
