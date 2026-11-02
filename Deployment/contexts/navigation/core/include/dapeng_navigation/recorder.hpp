#pragma once
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <fstream>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>
namespace dapeng_navigation {
// Small bounded metadata queue. Disk IO never occurs on a map/ROS callback.
class Recorder {
 public:
  explicit Recorder(const std::string& path,size_t capacity=256);
  ~Recorder();
  void write(int64_t stamp_ns,const std::string& topic,const std::string& message,
             const std::vector<std::pair<std::string,std::string>>& values);
  bool enabled() const{return enabled_;}
  bool failed() const{return failed_.load();}
  uint64_t dropped() const{return dropped_.load();}
 private:
  void loop();
  std::ofstream stream_;
  const bool enabled_;
  const size_t capacity_;
  std::mutex mutex_;std::condition_variable condition_;
  std::deque<std::string> pending_;
  std::thread worker_;
  bool stopping_=false;
  std::atomic<bool> failed_{false};std::atomic<uint64_t> dropped_{0};
};
}
