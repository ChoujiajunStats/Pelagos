#!/usr/bin/env python3
"""在隔离 ROS 域回放同一输入，导出真实地图阶段快照供可视化。"""
import argparse,json,os,signal,subprocess,time
from pathlib import Path
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile,DurabilityPolicy,ReliabilityPolicy,qos_profile_sensor_data
from sensor_msgs.msg import PointCloud2,Image
from nav_msgs.msg import Odometry
from nvblox_msgs.srv import FilePath


def main():
 p=argparse.ArgumentParser();p.add_argument('--bag',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--mapper',choices=['nvblox','wavemap'],required=True);a=p.parse_args()
 root=Path(__file__).resolve().parents[4];out=a.output.resolve();out.mkdir(parents=True,exist_ok=False)
 os.environ['ROS_DOMAIN_ID']='89';os.environ['DAPENG_USE_SIM_TIME']='true'
 rclpy.init();node=Node('map_replay_capture');children=[];streams=[];cloud=[None];poses=[];depth=[None];times=[None,0.];counts=[]
 def launch(name,cmd):
  f=(out/(name+'.log')).open('w');streams.append(f);q=subprocess.Popen(cmd,stdout=f,stderr=subprocess.STDOUT,start_new_session=True);children.append(q);return q
 def point_cb(m):
  dt=np.dtype({'names':['x','y','z'],'formats':['<f4']*3,'offsets':[next(f.offset for f in m.fields if f.name==k) for k in ['x','y','z']],'itemsize':m.point_step})
  v=np.frombuffer(bytes(m.data),dtype=dt);cloud[0]=np.column_stack([v[k] for k in ['x','y','z']]).copy()
 def depth_cb(m):
  stamp=m.header.stamp.sec+m.header.stamp.nanosec*1e-9
  if times[0] is None:times[0]=stamp
  times[1]=stamp-times[0]
  if m.encoding=='32FC1':depth[0]=np.ndarray((m.height,m.width),dtype='>f4' if m.is_bigendian else '<f4',buffer=bytes(m.data),strides=(m.step,4)).copy()
 def pose_cb(m):
  p=m.pose.pose.position;poses.append([p.x,p.y,p.z])
 node.create_subscription(PointCloud2,'/mapping/occupied',point_cb,QoSProfile(depth=1,durability=DurabilityPolicy.TRANSIENT_LOCAL,reliability=ReliabilityPolicy.RELIABLE))
 node.create_subscription(Image,'/stereo/depth',depth_cb,qos_profile_sensor_data)
 node.create_subscription(Odometry,'/odometry/estimated',pose_cb,qos_profile_sensor_data)
 cli=node.create_client(FilePath,'/nvblox_node/save_ply')
 def snapshot(idx):
  if a.mapper=='nvblox':
   if not cli.wait_for_service(timeout_sec=5):raise RuntimeError('save_ply unavailable')
   req=FilePath.Request();req.file_path=str(out/f'frame-{idx:02d}.ply');future=cli.call_async(req);rclpy.spin_until_future_complete(node,future,timeout_sec=25)
   if not future.done() or not future.result().success:raise RuntimeError('save_ply failed')
  elif cloud[0] is not None:np.save(out/f'frame-{idx:02d}.npy',cloud[0])
  if depth[0] is not None:np.save(out/f'depth-{idx:02d}.npy',depth[0])
  np.save(out/f'trajectory-{idx:02d}.npy',np.asarray(poses));counts.append({'index':idx,'source_elapsed_s':times[1],'point_count':None if cloud[0] is None else len(cloud[0])})
 try:
  if a.mapper=='nvblox':
   launch('pose_tf',['bash',str(root/'Deployment/contexts/mapping/scripts/start_nvblox_pose_tf.sh')]);mapper=launch('mapper',['bash',str(root/'Deployment/contexts/mapping/scripts/start_nvblox.sh')])
  else:mapper=launch('mapper',['bash',str(root/'Deployment/contexts/mapping/scripts/start_wavemap.sh'),str(out/'map.wvmp')])
  started=time.monotonic()
  while time.monotonic()-started<5:rclpy.spin_once(node,timeout_sec=.1)
  player=launch('player',['ros2','bag','play',str(a.bag.resolve()),'--delay','3','--clock','100'])
  idx=0;next_time=10.;deadline=time.monotonic()+150
  while player.poll() is None:
   rclpy.spin_once(node,timeout_sec=.05)
   if mapper.poll() is not None:raise RuntimeError('mapper stopped')
   if time.monotonic()>deadline:raise RuntimeError('replay timeout')
   if times[1]>=next_time and idx<5:
    snapshot(idx);idx+=1;next_time+=10
  if player.returncode:raise RuntimeError('bag failed')
  started=time.monotonic()
  while time.monotonic()-started<4:rclpy.spin_once(node,timeout_sec=.05)
  snapshot(idx)
  (out/'capture.json').write_text(json.dumps({'mapper':a.mapper,'bag':str(a.bag.resolve()),'snapshots':counts,'player_returncode':player.returncode},indent=2))
 finally:
  for q in reversed(children):
   if q.poll() is None:
    os.killpg(q.pid,signal.SIGINT)
    try:q.wait(timeout=30)
    except subprocess.TimeoutExpired:os.killpg(q.pid,signal.SIGKILL);q.wait()
  for f in streams:f.close()
  node.destroy_node();rclpy.shutdown()

if __name__=='__main__':main()
