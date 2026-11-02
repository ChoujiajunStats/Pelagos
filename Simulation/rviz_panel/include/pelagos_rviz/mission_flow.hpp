#pragma once
#include <cstdint>
#include <optional>
#include <string>
#include <utility>

namespace pelagos_rviz {
enum class MissionAction {Explore, Preview, Manual, Auto, Pause};
struct MissionStatus {
  std::string run, mode, pending_mode, guard, guard_reason;
  int protocol=0;
  int64_t wall_ns=0;
  bool healthy=false, enabled=false, planning_only=false;
};
struct MissionRequest {
  std::string id, run, action;
  int64_t issued_ns;
};

// Protocol-only state machine. No authority, navigation or actuator publisher.
class MissionFlow {
public:
  explicit MissionFlow(std::string prefix) : prefix_(std::move(prefix)) {}
  bool connected(int64_t now) const {
    return status_.protocol==1 && !status_.run.empty() && received_ns_>0 &&
      now>=received_ns_ && now-received_ns_<1000000000LL &&
      now>=status_.wall_ns && now-status_.wall_ns<1000000000LL;
  }
  const MissionStatus & status() const {return status_;}
  const std::string & message() const {return message_;}
  bool busy() const {return !pending_id_.empty() || waiting_owner_;}
  bool selected(MissionAction action,int64_t now) const {
    return connected(now) && !status_.planning_only && status_.pending_mode.empty() &&
      ((action==MissionAction::Manual && status_.mode=="manual") ||
       (action==MissionAction::Auto && status_.mode=="autonomy"));
  }
  std::string feedback(int64_t now) const {
    if(!connected(now))return "Control: Offline | Guard: UNKNOWN | Task: Unknown";
    const std::string owner=status_.planning_only?"Planning only":
      !status_.pending_mode.empty()?(status_.pending_mode=="manual"?"Switching to Manual":
        status_.pending_mode=="autonomy"?"Switching to Auto":"Switching control"):
      status_.mode=="manual"?"Manual":status_.mode=="autonomy"?"Auto":"Unknown";
    std::string guard=status_.guard=="ARMED"?"ARMED":status_.guard=="DISARMED"?"DISARMED":
      status_.guard=="FAULT"?"FAULT":"UNKNOWN";
    if(status_.guard=="FAULT") {
      const auto & reason=status_.guard_reason;
      const std::string detail=reason=="state_clock_mismatch"?"state/clock mismatch":
        reason.rfind("estimator_LOST",0)==0?"localization lost":
        reason=="original_request_expired" || reason=="terminal:original_request_expired"?
          "request expired":"safety fault";
      // Faults take precedence over earlier request acknowledgements. Never
      // expose an arbitrary, possibly localized raw reason in the toolbar.
      return "Control: "+owner+" | Guard: FAULT ("+detail+") | Task: Stopped";
    }
    const std::string task=status_.planning_only?"Planning only":!status_.enabled?"Paused":
      status_.guard=="ARMED"?"Running":"Waiting";
    return "Control: "+owner+" | Guard: "+guard+" | Task: "+task+" | "+message_;
  }

  std::optional<MissionRequest> update(MissionStatus status,int64_t now) {
    if(status.run!=status_.run)cancel();
    status_=std::move(status);received_ns_=now;
    return advance(now);
  }
  std::optional<MissionRequest> reply(const std::string & id,const std::string & run,
                                     bool accepted,const std::string & reason,int64_t now) {
    if(id!=pending_id_ || run!=status_.run || pending_id_.empty())return {};
    const auto action=pending_action_;
    pending_id_.clear();pending_action_.clear();
    // Coordinator diagnostics can be localized. Keep visible toolbar feedback
    // English; the ROS client logs the original reason separately.
    (void)reason;
    message_=(accepted?"Request accepted: ":"Request rejected: ")+action;
    if(!accepted) {cancel();return {};}
    if(action=="select_autonomy" && explore_after_owner_)waiting_owner_=true;
    return advance(now);
  }
  void tick(int64_t now) {
    if(busy() && now-flow_started_ns_>5000000000LL) {
      cancel();message_="No confirmation. Check the current state before retrying.";
    }
  }
  std::optional<MissionRequest> activate(MissionAction action,int64_t now) {
    tick(now);
    // Revocation overrides a queued Explore; late responses cannot restart it.
    if(action==MissionAction::Pause) {
      cancel();
      if(status_.run.empty()) {message_="Waiting for a simulation session.";return {};}
      return send("pause",now);
    }
    if(!connected(now)) {message_="Coordinator offline. No request sent.";return {};}
    if(action==MissionAction::Manual) {
      cancel();
      if(status_.planning_only) {message_="Planning-only session: motion is unavailable.";return {};}
      return send("select_manual",now);
    }
    if(busy()) {message_="Waiting for the previous request. Pause remains available.";return {};}
    if(action==MissionAction::Auto) {
      if(status_.planning_only) {message_="Planning-only session: motion is unavailable.";return {};}
      return send("select_autonomy",now);
    }
    if(action==MissionAction::Preview) {
      if(!status_.healthy || status_.enabled ||
         (!status_.planning_only && status_.guard!="DISARMED")) {
        message_="Preview requires a healthy, paused session.";return {};
      }
      return send("plan_preview",now);
    }
    if(status_.planning_only || !status_.healthy || status_.enabled || status_.guard=="FAULT") {
      message_="Explore requires a healthy, idle autonomous session.";return {};
    }
    if(status_.mode=="autonomy" && status_.pending_mode.empty())return send("start_exploration",now);
    explore_after_owner_=true;owner_requested_ns_=now;
    return send("select_autonomy",now);
  }
private:
  void cancel() {
    pending_id_.clear();pending_action_.clear();waiting_owner_=false;explore_after_owner_=false;
  }
  std::optional<MissionRequest> send(const std::string & action,int64_t now) {
    if(!busy())flow_started_ns_=now;
    pending_id_=prefix_+"-"+std::to_string(++sequence_);pending_action_=action;
    message_="Waiting for confirmation: "+action;
    return MissionRequest{pending_id_,status_.run,action,now};
  }
  std::optional<MissionRequest> advance(int64_t now) {
    tick(now);
    if(!waiting_owner_ || !pending_id_.empty() || !connected(now))return {};
    if(status_.guard=="FAULT" || status_.planning_only || !status_.healthy) {
      cancel();message_="Explore cancelled: session is not ready.";return {};
    }
    // A response only acknowledges the selection request. The bridge's actual
    // ownership status must also confirm completion before starting once.
    if(status_.wall_ns<=owner_requested_ns_ || status_.mode!="autonomy" ||
       !status_.pending_mode.empty())return {};
    waiting_owner_=false;explore_after_owner_=false;
    if(status_.enabled) {message_="Exploration is already running.";return {};}
    return send("start_exploration",now);
  }
  MissionStatus status_;
  std::string prefix_,pending_id_,pending_action_,message_="Waiting for the mission coordinator.";
  int64_t received_ns_=0,flow_started_ns_=0,owner_requested_ns_=0;
  uint64_t sequence_=0;
  bool waiting_owner_=false,explore_after_owner_=false;
};
}
