#!/usr/bin/env python3
"""Operate the connected planner without introducing another ROS domain."""
import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
import yaml


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('action',choices=('planner','explore','pause','navigate','manual','autonomous','plan-preview','start-preview','explore-forward'))
    args=parser.parse_args();out=args.run_dir
    config=yaml.safe_load((out/'resolved_config.yaml').read_text())
    if config.get('planner_profile')!='ethz_nbvplanner':
        raise RuntimeError('当前为手动模式；用 ./Simulation/sim run --autonomy 启动 NBV + JEPA')
    if args.action=='planner':
        path=out/'planner-status.json'
        if not path.exists():raise RuntimeError('规划桥仍在启动，检查 logs/nbv_bridge.log')
        print(json.dumps(json.loads(path.read_text()),ensure_ascii=False,indent=2));return
    import rclpy
    from rclpy.node import Node
    from std_srvs.srv import SetBool
    os.environ['ROS_DOMAIN_ID']=str(config['ros_domain_id'])
    rclpy.init();node=Node('pelagos_exploration_cli')
    try:
        if args.action in ('navigate','manual','autonomous','plan-preview','start-preview','explore-forward'):
            from std_msgs.msg import String
            from mission_geometry import route_poses
            values={};action='select_manual' if args.action=='manual' else 'select_autonomy'
            if args.action=='plan-preview':action='plan_preview'
            if args.action=='start-preview':action='forward_start_preview'
            if args.action=='explore-forward':action='start_forward_exploration'
            if args.action=='navigate':
                data=json.load(sys.stdin)
                if not isinstance(data,dict):raise RuntimeError('路线文件必须是任务面板导出的 JSON 对象')
                if data.get('run_id')!=out.name or data.get('frame_id')!='rov01/odom' or data.get('epoch')!=1:
                    raise RuntimeError('路线不属于本次估计坐标系；请在当前任务面板重新保存')
                values=dict(frame_id='rov01/odom',poses=route_poses(data['poses']));action='start_route'
            identifier=uuid.uuid4().hex;responses=[]
            def response(message):
                value=json.loads(message.data)
                if value.get('request_id')==identifier and value.get('run_id')==out.name:responses.append(value)
            node.create_subscription(String,'/pelagos/mission/response',response,10)
            publisher=node.create_publisher(String,'/pelagos/mission/request',10)
            deadline=time.monotonic()+5
            while publisher.get_subscription_count()==0 and time.monotonic()<deadline:
                rclpy.spin_once(node,timeout_sec=.05)
            if publisher.get_subscription_count()==0:raise RuntimeError('任务协调器未连接，请重启更新后的自主仿真')
            publisher.publish(String(data=json.dumps(dict(run_id=out.name,epoch=1,request_id=identifier,
                issued_steady_ns=time.monotonic_ns(),action=action,**values))))
            deadline=time.monotonic()+5
            while not responses and time.monotonic()<deadline:rclpy.spin_once(node,timeout_sec=.05)
            if not responses:raise RuntimeError('任务协调器未应答')
            print(responses[0]['reason'])
            if not responses[0]['accepted']:raise SystemExit(1)
            if args.action in ('manual','autonomous'):
                desired='manual' if args.action=='manual' else 'autonomy'
                deadline=time.monotonic()+5
                while time.monotonic()<deadline:
                    state=json.loads((out/'planner-status.json').read_text())
                    if state.get('control_mode')==desired and not state.get('control_mode_pending'):
                        print('已切换为手动接管' if desired=='manual' else '已切换为自主任务；请显式开始或继续任务');return
                    rclpy.spin_once(node,timeout_sec=.05)
                raise RuntimeError('控制模式未确认切换完成，请查看面板状态')
            return
        client=node.create_client(SetBool,'/pelagos/exploration/enable')
        if not client.wait_for_service(timeout_sec=5):raise RuntimeError('NBV 控制服务尚不可用')
        future=client.call_async(SetBool.Request(data=args.action=='explore'))
        rclpy.spin_until_future_complete(node,future,timeout_sec=5)
        if not future.done():raise RuntimeError('NBV 控制服务未应答')
        response=future.result();print(response.message)
        if not response.success:raise SystemExit(1)
    finally:node.destroy_node();rclpy.try_shutdown()


if __name__=='__main__':main()
