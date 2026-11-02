#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <yaml-cpp/yaml.h>
#include "dapeng_navigation/recorder.hpp"
int main() {
  const auto path=std::filesystem::temp_directory_path()/("dapeng-navigation-recorder-"+
      std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())+".jsonl");
  std::filesystem::remove(path);uint64_t dropped=0;
  {
    dapeng_navigation::Recorder recorder(path.string(),4);
    for(int i=0;i<1000;++i)recorder.write(i,"/mapping/metrics","test\n\t\"\\中文",{{"map_version",std::to_string(i)}});
    dropped=recorder.dropped();
  }
  std::ifstream input(path);std::string line;size_t count=0;
  while(std::getline(input,line)) {
    auto record=YAML::Load(line);
    if(record["message"].as<std::string>()!="test\n\t\"\\中文")throw std::runtime_error("invalid JSON escaping");
    ++count;
  }
  std::filesystem::remove(path);
  if(count+dropped!=1000)throw std::runtime_error("recorder lost unaccounted metadata");
  {
    dapeng_navigation::Recorder recorder("/dev/full");recorder.write(1,"t","m",{});
    const auto end=std::chrono::steady_clock::now()+std::chrono::seconds(2);
    while(!recorder.failed()&&std::chrono::steady_clock::now()<end)std::this_thread::yield();
    if(!recorder.failed())throw std::runtime_error("disk failure was not reported");
  }
  std::cout<<"recorder PASS: written="<<count<<", bounded queue drops="<<dropped<<'\n';
}
