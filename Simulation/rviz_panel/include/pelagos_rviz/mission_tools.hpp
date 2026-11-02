#pragma once
#include <memory>
#include <QAction>
#include <QPointer>
#include <rviz_common/tool.hpp>
#include "pelagos_rviz/mission_flow.hpp"

namespace pelagos_rviz {
class MissionClient;
class MissionTool : public rviz_common::Tool {
public:
  MissionTool(MissionAction action,const char * name,int icon);
  ~MissionTool() override;
  void activate() override;
  void deactivate() override {}
protected:
  void onInitialize() override;
private:
  void refresh();
  MissionAction action_;
  QString label_;
  QString last_status_;
  int icon_;
  bool toolbar_ready_=false;
  QPointer<QAction> toolbar_action_;
  std::shared_ptr<MissionClient> client_;
};
class ExploreTool : public MissionTool {public: ExploreTool();};
class PreviewTool : public MissionTool {public: PreviewTool();};
class ManualTool : public MissionTool {public: ManualTool();};
class AutoTool : public MissionTool {public: AutoTool();};
class PauseTool : public MissionTool {public: PauseTool();};
}
