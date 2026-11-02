"""Verify the copied-source scheduling patch and its actual lifecycle code."""
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest


ROOT=Path(__file__).resolve().parents[2]
PACKAGE=ROOT/'Simulation/patches'
SOURCE=ROOT/'TP/Localization/okvis2'
MANIFEST=json.loads((PACKAGE/'okvis-realtime-ingress-sha256.json').read_text())


def stripped(text):
    pattern=r'//[^\n]*|/\*[\s\S]*?\*/|"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\''
    return re.sub(pattern,lambda m:' '*len(m.group()),text)


class OkvisRealtimePatchChecks(unittest.TestCase):
    def copy(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        target=Path(temporary.name)/'okvis'
        for name in MANIFEST['files']:
            path=target/name;path.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(SOURCE/name,path)
        return target

    def apply(self,target):
        return subprocess.run(['python3',str(PACKAGE/'apply-okvis-realtime-ingress.py'),str(target)],
                              capture_output=True,text=True)

    def patched(self):
        target=self.copy();result=self.apply(target)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        return target

    def test_exact_build_copy_applies_and_upstream_source_remains_unchanged(self):
        target=self.patched()
        for name,expected in MANIFEST['files'].items():
            self.assertEqual(hashlib.sha256((target/name).read_bytes()).hexdigest(),expected['after_sha256'])
            self.assertEqual(hashlib.sha256((SOURCE/name).read_bytes()).hexdigest(),expected['before_sha256'])

    def test_source_drift_rejects_before_any_file_is_patched(self):
        target=self.copy();name=next(iter(MANIFEST['files']))
        (target/name).write_text((target/name).read_text()+'\n// unrelated source drift\n')
        before={p:p.read_bytes() for p in target.rglob('*') if p.is_file()}
        result=self.apply(target)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('before SHA256 mismatch',result.stderr)
        self.assertEqual({p:p.read_bytes() for p in before},before)

    def test_original_rate_parameters_and_actual_imu_state_timestamps_are_retained(self):
        target=self.patched()
        node=(target/'okvis_ros2/src/okvis_node_subscriber.cpp').read_text()
        original=(SOURCE/'okvis_ros2/src/okvis_node_subscriber.cpp').read_text()
        rate=lambda text:[line.strip() for line in text.splitlines()
                          if 'imu_propagated_state_publishing_rate' in line or 'setOdometryPublishingRate' in line]
        self.assertEqual(rate(node),rate(original))
        publisher=(target/'okvis_ros2/src/Publisher.cpp').read_text()
        realtime=publisher[publisher.index('bool Publisher::realtimePredictAndPublish'):publisher.index('void Publisher::publishEstimatorUpdate')]
        self.assertIn('rclcpp::Time t(state.timestamp.sec, state.timestamp.nsec)',realtime)
        self.assertIn('odometryMsg.header.stamp = t;',realtime)
        self.assertIn('lastTime_ = state.timestamp;',realtime)
        self.assertNotIn('node_->now()',realtime)
        self.assertNotIn('MultiThreadedExecutor',node)
        self.assertNotIn('spin_some',node)
        self.assertEqual(set(MANIFEST['files']),{'okvis_ros2/src/Publisher.cpp',
            'okvis_ros2/include/okvis/ros2/Publisher.hpp','okvis_ros2/src/okvis_node_subscriber.cpp'})

    def test_every_trajectory_access_has_real_mutex_scope_and_ros_publishing_is_outside(self):
        target=self.patched();text=(target/'okvis_ros2/src/Publisher.cpp').read_text();mask=stripped(text)
        stack=[];blocks=[]
        for i,char in enumerate(mask):
            if char=='{':stack.append(i)
            elif char=='}':blocks.append((stack.pop(),i))
        guards=list(re.finditer(r'std::(?:unique_lock|lock_guard)<std::mutex>\s+lock\(trajectoryMutex_',mask))
        self.assertEqual(len(guards),4)
        scopes=[]
        for guard in guards:
            scope=min((b for b in blocks if b[0]<guard.start()<b[1]),key=lambda b:b[1]-b[0])
            scopes.append((guard.start(),scope[1]))
        accesses=list(re.finditer(r'trajectory_\.(?:addImuMeasurement|update|getState)\(',mask))
        self.assertEqual(len(accesses),4)
        for access in accesses:
            self.assertTrue(any(a<access.start()<b for a,b in scopes),text[access.start():access.start()+80])
        for publish in re.finditer(r'(?:->publish|\.publish|->sendTransform)\(',mask):
            self.assertFalse(any(a<publish.start()<b for a,b in scopes),'ROS publication holds trajectory mutex')
        self.assertIn('std::try_to_lock',text)
        self.assertIn('if(!lock.owns_lock())',text)
        self.assertNotIn('trajectoryLocked_',text)

    def test_extracted_real_ingress_lifecycle_keeps_callbacks_live_and_joins_on_errors(self):
        compiler=shutil.which('g++')
        if not compiler:self.skipTest('Run in project image with the retained C++ compiler')
        target=self.patched();node=(target/'okvis_ros2/src/okvis_node_subscriber.cpp').read_text()
        start=node.index('  rclcpp::executors::SingleThreadedExecutor ingressExecutor;')
        end=node.index('  if(ingressError) std::rethrow_exception(ingressError);',start)+len('  if(ingressError) std::rethrow_exception(ingressError);')
        actual=node[start:end]
        fixture=r'''
#include <algorithm>
#include <atomic>
#include <chrono>
#include <exception>
#include <iostream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
std::atomic_bool shtdown;
struct NullLog { template<class T> NullLog& operator<<(const T&){return *this;} };
#define LOG(level) NullLog()
namespace cv {using Mat=int;}
struct TestNode {std::atomic_bool alive{false},visual{false};std::atomic_int duringVisual{0};bool callbackError=false;};
namespace rclcpp {
bool ok(){return true;}
namespace executors {
class SingleThreadedExecutor {
  std::shared_ptr<TestNode> node_;std::atomic_bool cancelled_{false};
public:
  void add_node(std::shared_ptr<TestNode> n){node_=n;}
  void cancel(){cancelled_.store(true);}
  void spin(){
    node_->alive.store(true);
    try {
      int callbacks=0;
      while(!cancelled_.load()){
        if(node_->visual.load())++node_->duringVisual;
        if(node_->callbackError&&++callbacks>3)throw std::runtime_error("callback error");
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
      }
    }catch(...){node_->alive.store(false);throw;}
    node_->alive.store(false);
  }
};}}
struct Estimator {
 std::shared_ptr<TestNode> node;bool visualError;
 bool processFrame(){node->visual.store(true);std::this_thread::sleep_for(std::chrono::milliseconds(120));
   if(visualError)throw std::runtime_error("visual error");shtdown.store(true);return true;}
 void display(std::map<std::string,cv::Mat>&){}
};
struct Subscriber {std::shared_ptr<TestNode> node;bool& closed;
 void shutdown(){if(node->alive.load())throw std::runtime_error("destroying subscriber before join");closed=true;}
};
struct Publisher {void publishImages(const std::map<std::string,cv::Mat>&){};};
void run(bool visualError,bool callbackError){
 shtdown.store(false);auto node=std::make_shared<TestNode>();node->callbackError=callbackError;
 Estimator estimator{node,visualError};bool closed=false;Subscriber subscriber{node,closed};Publisher publisher;
 bool threw=false;
 try {
 ACTUAL_CODE
 }catch(const std::runtime_error&){threw=true;}
 if(!closed||node->alive.load())throw std::runtime_error("ingress was not joined and closed");
 if(threw!=(visualError||callbackError))throw std::runtime_error("exception propagation changed");
 if(!callbackError&&node->duringVisual.load()<2)throw std::runtime_error("visual work blocked callback ingress");
}
int main(){try{run(false,false);run(true,false);run(false,true);std::cout<<"PASS actual lifecycle normal/visual-error/callback-error, live ingress during visual stall\n";return 0;}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;}}
'''.replace('ACTUAL_CODE',actual)
        cpp=target/'lifecycle.cpp';cpp.write_text(fixture);binary=target/'lifecycle'
        subprocess.run([compiler,'-std=c++14','-O1','-pthread',str(cpp),'-o',str(binary)],check=True,capture_output=True,text=True)
        result=subprocess.run([str(binary)],capture_output=True,text=True,timeout=10)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertIn('PASS actual lifecycle',result.stdout)


if __name__=='__main__':unittest.main(verbosity=2)
