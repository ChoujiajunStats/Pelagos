#include "dapeng_navigation/recorder.hpp"
#include <sstream>
#include <stdexcept>
namespace dapeng_navigation {
namespace {
std::string quoted(const std::string& value) {
  static constexpr char hex[]="0123456789abcdef";
  std::string out="\"";
  for(unsigned char c:value) {
    if(c=='"'||c=='\\'){out+='\\';out+=char(c);}
    else if(c<32){out+="\\u00";out+=hex[c>>4];out+=hex[c&15];}
    else out+=char(c);
  }
  out+='"';return out;
}
}
Recorder::Recorder(const std::string& path,size_t capacity):enabled_(!path.empty()),capacity_(capacity) {
  if(!capacity_)throw std::invalid_argument("zero recorder capacity");
  if(path.empty())return;
  stream_.open(path,std::ios::app);
  if(!stream_)throw std::runtime_error("cannot open navigation metrics output: "+path);
  worker_=std::thread([this]{loop();});
}
Recorder::~Recorder(){
  {std::lock_guard<std::mutex> lock(mutex_);stopping_=true;}
  condition_.notify_all();if(worker_.joinable())worker_.join();
}
void Recorder::write(int64_t stamp,const std::string& topic,const std::string& message,
                     const std::vector<std::pair<std::string,std::string>>& values) {
  if(!enabled()||failed_)return;
  std::ostringstream text;text<<"{\"stamp_ns\":"<<stamp<<",\"topic\":"<<quoted(topic)
      <<",\"message\":"<<quoted(message)<<",\"values\":{";
  bool first=true;for(const auto& [key,value]:values){if(!first)text<<',';first=false;text<<quoted(key)<<':'<<quoted(value);}
  text<<"}}\n";
  {std::lock_guard<std::mutex> lock(mutex_);
   if(pending_.size()>=capacity_){pending_.pop_front();++dropped_;}
   pending_.push_back(text.str());}
  condition_.notify_one();
}
void Recorder::loop() {
  while(true) {
    std::deque<std::string> batch;
    {std::unique_lock<std::mutex> lock(mutex_);condition_.wait(lock,[&]{return stopping_||!pending_.empty();});
     if(pending_.empty()&&stopping_)break;
     batch.swap(pending_);}
    for(const auto& line:batch)stream_<<line;
    stream_.flush();
    if(!stream_){failed_=true;return;}
  }
}
}
