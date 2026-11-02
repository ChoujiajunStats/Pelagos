#include "pelagos_rviz/mission_tools.hpp"
#include <rviz_common/display_context.hpp>
#include <rviz_common/tool_manager.hpp>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_msgs/msg/string.hpp>
#include <QApplication>
#include <QJsonDocument>
#include <QJsonObject>
#include <QPointer>
#include <QStyle>
#include <QToolBar>
#include <QTimer>
#include <QUuid>
#include <chrono>
#include <functional>
#include <vector>

namespace pelagos_rviz {
namespace {
int64_t wallNow() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::steady_clock::now().time_since_epoch()).count();
}
}
class MissionClient : public QObject {
public:
  static std::shared_ptr<MissionClient> shared() {
    static std::weak_ptr<MissionClient> instance;
    auto client=instance.lock();
    if(!client) {client=std::make_shared<MissionClient>();instance=client;}
    return client;
  }
  MissionClient() : flow(QUuid::createUuid().toString(QUuid::Id128).toStdString()) {
    node_=std::make_shared<rclcpp::Node>("pelagos_rviz_tools_"+
      QUuid::createUuid().toString(QUuid::Id128).toStdString());
    requests_=node_->create_publisher<std_msgs::msg::String>("/pelagos/mission/request",10);
    status_=node_->create_subscription<std_msgs::msg::String>("/pelagos/planner/status",1,
      [this](std_msgs::msg::String::ConstSharedPtr msg) {
        const auto document=QJsonDocument::fromJson(QByteArray::fromStdString(msg->data));
        if(!document.isObject())return;
        const auto value=document.object();
        MissionStatus status;
        status.run=value["run"].toString().toStdString();
        status.mode=value["control_mode"].toString().toStdString();
        status.pending_mode=value["control_mode_pending"].toString().toStdString();
        status.guard=value["guard"].toString().toStdString();
        status.guard_reason=value["guard_reason"].toString().toStdString();
        status.protocol=value["mission_protocol"].toInt();
        status.wall_ns=static_cast<int64_t>(value["wall_ns"].toDouble());
        status.healthy=value["healthy"].toBool();status.enabled=value["enabled"].toBool();
        status.planning_only=value["planning_only"].toBool();
        publish(flow.update(std::move(status),wallNow()));
      });
    response_=node_->create_subscription<std_msgs::msg::String>("/pelagos/mission/response",10,
      [this](std_msgs::msg::String::ConstSharedPtr msg) {
        const auto value=QJsonDocument::fromJson(QByteArray::fromStdString(msg->data)).object();
        RCLCPP_INFO(node_->get_logger(),"Mission response %s: %s",
          value["request_id"].toString().toStdString().c_str(),
          value["reason"].toString().toStdString().c_str());
        publish(flow.reply(value["request_id"].toString().toStdString(),
          value["run_id"].toString().toStdString(),value["accepted"].toBool(),
          value["reason"].toString().toStdString(),wallNow()));
      });
    QObject::connect(&timer_,&QTimer::timeout,this,[this]() {
      const auto context=node_->get_node_base_interface()->get_context();
      if(!rclcpp::ok(context)) {timer_.stop();return;}
      try {rclcpp::spin_some(node_);}
      catch(const rclcpp::exceptions::RCLError & error) {
        // Shutdown can invalidate the context between the check and spin_some.
        if(rclcpp::ok(context))RCLCPP_ERROR(node_->get_logger(),"Mission toolbar: %s",error.what());
        timer_.stop();return;
      }
      flow.tick(wallNow());
      for(const auto & observer:observers)observer();
    });
    timer_.start(50);
  }
  ~MissionClient() override {timer_.stop();}
  void activate(MissionAction action) {publish(flow.activate(action,wallNow()));}
  MissionFlow flow;
  std::vector<std::function<void()>> observers;
private:
  void publish(const std::optional<MissionRequest> & request) {
    if(!request || !rclcpp::ok(node_->get_node_base_interface()->get_context()))return;
    const auto & r=*request;
    QJsonObject value{{"request_id",QString::fromStdString(r.id)},
      {"run_id",QString::fromStdString(r.run)},{"epoch",1},
      {"issued_steady_ns",static_cast<double>(r.issued_ns)},
      {"action",QString::fromStdString(r.action)}};
    std_msgs::msg::String message;
    message.data=QJsonDocument(value).toJson(QJsonDocument::Compact).toStdString();
    requests_->publish(message);
  }
  rclcpp::Node::SharedPtr node_;
  rclcpp::Publisher<std_msgs::msg::String>::SharedPtr requests_;
  rclcpp::Subscription<std_msgs::msg::String>::SharedPtr status_,response_;
  QTimer timer_;
};

MissionTool::MissionTool(MissionAction action,const char * name,int icon)
 : action_(action),label_(name),icon_(icon) {}
MissionTool::~MissionTool()=default;
void MissionTool::onInitialize() {
  setName(label_);
  setIcon(QApplication::style()->standardIcon(static_cast<QStyle::StandardPixmap>(icon_)));
  client_=MissionClient::shared();
  QPointer<MissionTool> self(this);
  client_->observers.emplace_back([self](){if(self)self->refresh();});
  // ToolManager initializes the Tool before emitting toolAdded. The frame's
  // QAction does not exist during onInitialize; refreshing it there crashes.
  QObject::connect(context_->getToolManager(),&rviz_common::ToolManager::toolAdded,
    this,[this](rviz_common::Tool * added) {
      if(added==this)QTimer::singleShot(0,this,[this]() {
        toolbar_ready_=true;refresh();
      });
    });
}
void MissionTool::activate() {
  if(client_ && toolbar_ready_) {client_->activate(action_);refresh();}
  // These are one-shot toolbar actions; restore RViz's normal mouse tool.
  QTimer::singleShot(0,this,[this]() {
    auto * manager=context_->getToolManager();
    if(manager->getCurrentTool()==this && manager->getDefaultTool()!=this)
      manager->setCurrentTool(manager->getDefaultTool());
  });
}
void MissionTool::refresh() {
  if(!client_ || !toolbar_ready_)return;
  const auto & flow=client_->flow;const auto now=wallNow();
  const QString name=label_+(flow.selected(action_,now)?" (selected)":"");
  const QString feedback=QString::fromStdString(flow.feedback(now));
  QString help;
  switch(action_) {
    case MissionAction::Explore:help="Select Auto, wait for confirmed ownership, then start exploration once.";break;
    case MissionAction::Preview:help="Compute and display a path in a paused session. No motion.";break;
    case MissionAction::Manual:help="Pause autonomy and select manual control. Does not arm the joystick.";break;
    case MissionAction::Auto:help="Select autonomous ownership. Does not start motion.";break;
    case MissionAction::Pause:help="Cancel a queued start and revoke motion authority.";break;
  }
  const QString description=help+"\n"+feedback;
  setDescription(description);
  if(!toolbar_action_) {
    for(auto * window:QApplication::topLevelWidgets())
      for(auto * toolbar:window->findChildren<QToolBar *>())
        for(auto * action:toolbar->actions())
          if(action->text()==label_)toolbar_action_=action;
  }
  if(toolbar_action_) {
    // Keep the Tool's stable name for RViz's Remove Tool menu. Only its native
    // QAction presentation shows ownership and current Guard/task feedback.
    toolbar_action_->setIconText(name);
    toolbar_action_->setToolTip(description);
    toolbar_action_->setStatusTip(feedback);
  }
  if(last_status_!=feedback) {setStatus(feedback);last_status_=feedback;}
}
ExploreTool::ExploreTool():MissionTool(MissionAction::Explore,"Explore",QStyle::SP_MediaPlay) {}
PreviewTool::PreviewTool():MissionTool(MissionAction::Preview,"Plan Preview",QStyle::SP_FileDialogDetailedView) {}
ManualTool::ManualTool():MissionTool(MissionAction::Manual,"Manual",QStyle::SP_ComputerIcon) {}
AutoTool::AutoTool():MissionTool(MissionAction::Auto,"Auto",QStyle::SP_ArrowForward) {}
PauseTool::PauseTool():MissionTool(MissionAction::Pause,"Pause",QStyle::SP_MediaPause) {}
}
PLUGINLIB_EXPORT_CLASS(pelagos_rviz::ExploreTool,rviz_common::Tool)
PLUGINLIB_EXPORT_CLASS(pelagos_rviz::PreviewTool,rviz_common::Tool)
PLUGINLIB_EXPORT_CLASS(pelagos_rviz::ManualTool,rviz_common::Tool)
PLUGINLIB_EXPORT_CLASS(pelagos_rviz::AutoTool,rviz_common::Tool)
PLUGINLIB_EXPORT_CLASS(pelagos_rviz::PauseTool,rviz_common::Tool)
