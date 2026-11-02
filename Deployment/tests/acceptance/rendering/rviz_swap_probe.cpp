// Optional diagnostic interposer: per-drawable GLX swaps, not configured FPS.
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <mutex>
#include <map>
#include <algorithm>
namespace {
using Clock=std::chrono::steady_clock;
struct Counter {Clock::time_point begin,last;unsigned count=0;double max_gap=0;};
std::mutex lock;
FILE* output=nullptr;
std::map<unsigned long,Counter> counters;
void observe(void* display,unsigned long drawable){
  std::lock_guard<std::mutex> guard(lock);const auto now=Clock::now();
  if(!output){const char* path=std::getenv("DAPENG_RVIZ_SWAP_LOG");if(!path)return;output=std::fopen(path,"a");if(!output)return;}
  auto it=counters.find(drawable);
  if(it==counters.end()){Counter c;c.begin=c.last=now;it=counters.emplace(drawable,c).first;}
  auto& c=it->second;c.max_gap=std::max(c.max_gap,std::chrono::duration<double>(now-c.last).count());c.last=now;++c.count;
  const double elapsed=std::chrono::duration<double>(now-c.begin).count();
  if(elapsed>=1.){
    unsigned width=0,height=0;
    using Query=void(*)(void*,unsigned long,int,unsigned*);
    static auto query=reinterpret_cast<Query>(dlsym(RTLD_NEXT,"glXQueryDrawable"));
    if(query){query(display,drawable,0x801D,&width);query(display,drawable,0x801E,&height);}
    std::fprintf(output,"{\"drawable\":%lu,\"width\":%u,\"height\":%u,\"duration_s\":%.6f,\"swaps\":%u,\"fps\":%.3f,\"max_gap_ms\":%.3f}\n",drawable,width,height,elapsed,c.count,c.count/elapsed,c.max_gap*1000.);
    std::fflush(output);c.begin=now;c.count=0;c.max_gap=0;
  }
}
}
extern "C" void glXSwapBuffers(void* display,unsigned long drawable){
  using Function=void(*)(void*,unsigned long);static auto original=reinterpret_cast<Function>(dlsym(RTLD_NEXT,"glXSwapBuffers"));
  if(original)original(display,drawable);observe(display,drawable);
}
