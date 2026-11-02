#include "pelagos_rviz/mission_flow.hpp"
#include <cassert>
#include <algorithm>
#include <iostream>
using namespace pelagos_rviz;
constexpr int64_t now=100000000000LL;
MissionStatus status(std::string mode="manual",int64_t stamp=now) {
  MissionStatus s;s.run="session";s.mode=mode;s.guard="DISARMED";
  s.protocol=1;s.healthy=true;s.wall_ns=stamp;return s;
}
int main() {
  {MissionFlow f("test");f.update(status(),now);
   auto select=f.activate(MissionAction::Explore,now+1);assert(select && select->action=="select_autonomy");
   assert(!f.reply(select->id,"other",true,"accepted",now+2));
   assert(!f.reply(select->id,"session",true,"accepted",now+3));
   auto switching=status("autonomy",now+4);switching.pending_mode="autonomy";
   assert(!f.update(switching,now+5));
   auto start=f.update(status("autonomy",now+6),now+7);
   assert(start && start->action=="start_exploration" && start->issued_ns==now+7);
   assert(!f.update(status("autonomy",now+8),now+9));}
  {MissionFlow f("test");f.update(status(),now);
   auto select=f.activate(MissionAction::Explore,now+1);
   auto pause=f.activate(MissionAction::Pause,now+2);assert(pause && pause->action=="pause");
   assert(!f.reply(select->id,"session",true,"accepted",now+3));
   assert(!f.update(status("autonomy",now+4),now+5));}
  {MissionFlow f("test");f.update(status(),now);
   auto select=f.activate(MissionAction::Explore,now+1);
   auto manual=f.activate(MissionAction::Manual,now+2);assert(manual && manual->action=="select_manual");
   assert(!f.reply(select->id,"session",true,"accepted",now+3));
   assert(!f.update(status("autonomy",now+4),now+5));}
  {MissionFlow f("test");f.update(status(),now);
   auto preview=f.activate(MissionAction::Preview,now+1);assert(preview && preview->action=="plan_preview");}
  {MissionFlow f("test");auto s=status();s.enabled=true;f.update(s,now);
   assert(!f.activate(MissionAction::Preview,now+1));}
  {MissionFlow f("test");f.update(status("autonomy"),now);
   auto start=f.activate(MissionAction::Explore,now+1);assert(start && start->action=="start_exploration");}
  {MissionFlow f("test");f.update(status(),now);
   assert(!f.activate(MissionAction::Explore,now+1000000001LL));
   auto pause=f.activate(MissionAction::Pause,now+1000000002LL);assert(pause && pause->action=="pause");}
  {MissionFlow f("test");f.update(status(),now);
   auto select=f.activate(MissionAction::Explore,now+1);
   f.reply(select->id,"session",true,"accepted",now+2);
   assert(!f.update(status("autonomy",now),now+3)); // queued old state cannot confirm ownership
   f.tick(now+5000000002LL);
   assert(!f.update(status("autonomy",now+5000000003LL),now+5000000004LL));}
  {MissionFlow f("test");f.update(status(),now);
   auto select=f.activate(MissionAction::Explore,now+1);
   auto next=status("autonomy",now+2);next.run="new-session";
   assert(!f.update(next,now+3));assert(!f.reply(select->id,"session",true,"accepted",now+4));}
  {MissionFlow f("test");auto s=status();s.planning_only=true;f.update(s,now);
   assert(!f.activate(MissionAction::Explore,now+1));
   assert(!f.activate(MissionAction::Manual,now+2));
   auto preview=f.activate(MissionAction::Preview,now+3);assert(preview && preview->action=="plan_preview");}
  for(bool accepted:{false,true}) {
    MissionFlow f("test");f.update(status(),now);
    auto request=f.activate(MissionAction::Preview,now+1);
    f.reply(request->id,"session",accepted,"已接收任务操作／等待估计状态",now+2);
    assert(std::all_of(f.message().begin(),f.message().end(),[](unsigned char c){return c<128;}));
    assert(f.message().find(accepted?"Request accepted":"Request rejected")==0);
  }
  {MissionFlow f("test");f.update(status("autonomy"),now);
   auto request=f.activate(MissionAction::Explore,now+1);
   f.reply(request->id,"session",true,"accepted",now+2);
   auto fault=status("autonomy",now+3);fault.guard="FAULT";fault.guard_reason="state_clock_mismatch";
   f.update(fault,now+4);const auto text=f.feedback(now+5);
   assert(f.selected(MissionAction::Auto,now+5)); // ownership remains selected, not moving
   assert(text.find("Guard: FAULT (state/clock mismatch)")!=std::string::npos);
   assert(text.find("Task: Stopped")!=std::string::npos);
   assert(text.find("Request accepted")==std::string::npos);}
  {MissionFlow f("test");auto fault=status();fault.guard="FAULT";fault.guard_reason="等待估计状态";
   f.update(fault,now);const auto text=f.feedback(now+1);
   assert(text.find("Guard: FAULT (safety fault)")!=std::string::npos);
   assert(std::all_of(text.begin(),text.end(),[](unsigned char c){return c<128;}));}
  {MissionFlow f("test");auto running=status("autonomy");running.guard="ARMED";running.enabled=true;
   f.update(running,now);assert(f.feedback(now+1).find("Task: Running")!=std::string::npos);
   f.update(status("autonomy",now+2),now+3);
   assert(f.feedback(now+4).find("Guard: DISARMED | Task: Paused")!=std::string::npos);
   assert(f.feedback(now+1000000004LL).find("Control: Offline | Guard: UNKNOWN")!=std::string::npos);}
  std::cout<<"PASS: 15 mission-flow/feedback scenarios, including fault precedence and English-only localized input\n";
}
