#!/usr/bin/env python3
"""Run the existing simulator, Pelagos algorithms and RViz in one ROS domain."""
import argparse
from pathlib import Path
import os
import signal
import subprocess
import sys
import time
import yaml


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--scene', choices=('fixture', 'porth'), required=True)
    parser.add_argument('--ros-domain', type=int, required=True)
    args = parser.parse_args()
    autonomy = os.environ.get('PELAGOS_AUTONOMY', '0') == '1'
    planning_only=os.environ.get('PELAGOS_PLANNING_ONLY','0')=='1'
    out = args.run_dir.resolve()
    python = '/opt/research/bin/python3'
    root = Path('/pelagos/Simulation')
    mapper = Path('/opt/pelagos/mapping/underwater_wavemap_ros/lib/underwater_wavemap_ros/wavemap_node')
    if not mapper.is_file():
        print('当前镜像还没有 Wavemap；请执行 ./Simulation/sim build && ./Simulation/sim run',
              file=sys.stderr,flush=True)
        return 1
    subprocess.run([python, str(root / 'tools/prepare_scene.py'), '--run-dir', str(out),
                    '--scene', args.scene, '--ros-domain', str(args.ros_domain),
                    *(['--autonomy'] if autonomy else [])], check=True)
    ros = ['--ros-args', '-p', 'use_sim_time:=true']
    namespaced = ['--ros-args', '-r', '__ns:=/rov01', '-p', 'use_sim_time:=true']
    run_param = ['-p', f'run_dir:={out}']
    truth_remap = ['-p', 'state_frame_prefix:=rov01/truth',
                   '-r', 'state/odometry:=sim/ground_truth/odometry']

    def module(name):
        return [python, '-c', f'from {name} import main; main()']

    commands = [
        ('stonefish', ['/opt/uw_underlay/lib/stonefish_ros2/stonefish_simulator',
                       '/opt/uw_underlay/share/stonefish_bluerov2/data', str(out / 'scene.scn'),
                       '100.0', '960', '720', 'medium', *namespaced]),
        ('sensors', [*module('uw_perception.node'), *namespaced, *run_param]),
        ('truth', [*module('uw_simulations.truth_node'), *namespaced, '-p', 'frame_prefix:=rov01']),
        ('guard', [python, str(root / 'adapters/estimated_control.py'), 'guard',
                   *namespaced, *run_param] if autonomy else
                  [*module('uw_guard.node'), *namespaced, *run_param, *truth_remap]),
        ('controller', [python, str(root / 'adapters/estimated_control.py'), 'controller',
                        *namespaced, *run_param, '-p', 'controller_backend:=jepa_mppi',
                        '-p', f'jepa_settings_path:={out / "jepa-settings.yaml"}'] if autonomy else
                       [*module('uw_controller.node'), *namespaced, *run_param, *truth_remap]),
        ('actuators', [*module('uw_simulations.actuator_node'), *namespaced, *run_param]),
        ('sensor_bridge', [python, str(root / 'adapters/sensor_bridge.py'), *ros]),
        ('okvis2', ['/opt/pelagos/okvis2/lib/okvis/okvis_node_subscriber',
                    '--ros-args', '-r', '__ns:=/okvis', '-p', 'use_sim_time:=true',
                    '-p', f'config_filename:={out / "okvis2.yaml"}',
                    '-p', 'imu_propagated_state_publishing_rate:=30.0']),
        ('camera_pose', [python, '/pelagos/Deployment/contexts/localization/adapters/okvis_pose_adapter.py',
                         '--extrinsics', str(out / 'stereo-geometry.json'), *ros]),
        ('las2_s', [python, '/pelagos/Deployment/contexts/depth/adapters/las2_node.py',
                    '--checkpoint', '/pelagos/Deployment/artifacts/weights/LAS2_S.pth',
                    '--calibration', str(out / 'stereo-geometry.json'), '--variant', 's', *ros]),
        ('depth_comparison', [python, str(root / 'adapters/depth_comparison.py'), *ros]),
        ('wavemap', [str(mapper), '--ros-args', '-r', '__ns:=/pelagos/wavemap',
                     '-p', 'use_sim_time:=true', '--params-file',str(out / 'wavemap.yaml')]),
        ('vio_evidence', [python, str(root / 'adapters/vio_evidence.py'), *ros, *run_param]),
        ('pose_visualizer', [python, str(root / 'adapters/pose_visualizer.py'), *ros,
                             '-p', f'run_dir:={out}',
                             '-p', f'robot_profile_path:={out / "robot_profile.yaml"}',
                             *(['-p', 'cave_asset_path:=/porth'] if args.scene == 'porth' else [])]),
        ('rviz', ['env', 'LANG=C.UTF-8', 'LC_ALL=C.UTF-8', 'LANGUAGE=en',
                  'rviz2', '-d', str(root / 'config/pelagos.rviz'), *ros]),
    ]
    if autonomy:
        commands[-1:-1] = [
            ('body_state', [python, str(root / 'adapters/body_state.py'), *ros, *run_param]),
            ('navigation', [python, str(root/'adapters/estimated_navigation.py'), *namespaced, *run_param]),
            ('nbv_bridge', [python, str(root / 'adapters/nbv_bridge.py'), *ros, *run_param]),
            ('planner_visualizer', [python, str(root / 'adapters/planner_visualizer.py'), *ros]),
        ]
        if yaml.safe_load((out/'nbv.yaml').read_text()).get('pelagos_map_source','wavemap')=='wavemap':
            commands.insert(-1,('map_relay',[python,str(root/'adapters/wavemap_nbv_relay.py'),*ros,*run_param]))
    if planning_only:
        commands=[(name,command) for name,command in commands
                  if name not in ('guard','controller','actuators','navigation')]
    logs = out / 'logs'
    logs.mkdir(exist_ok=True)
    processes = []
    exited_viewers=set()
    stopping = False

    def stop(_signum, _frame):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)
    try:
        for name, command in commands:
            stream = (logs / f'{name}.log').open('w', buffering=1)
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            processes.append((name, process, stream))
        stack=('+ 原版 NBVPlanner（纯规划，不执行） ' if planning_only else '+ 原版 NBVPlanner + JEPA-MPPI ' if autonomy else '')
        print(f'Stonefish + OKVIS2 + LAS2-S {stack}已启动，日志：{out}/logs', flush=True)
        print('相机真值红色，估计绿色。' + ('用 ./Simulation/sim plan-preview 只生成轨迹。' if planning_only else 'Use the RViz toolbar: Explore / Plan Preview / Manual / Auto / Pause.' if autonomy else '手动模式，默认未授权运动。'), flush=True)
        while not stopping:
            for name, process, _ in processes:
                code = process.poll()
                if code is not None:
                    if name in exited_viewers:continue
                    print(f'{name} 已退出（{code}），详见 {logs / (name + ".log")}', file=sys.stderr, flush=True)
                    if name=='rviz' and code!=0:
                        # A viewer's GL resource failure must not kill the map,
                        # planner and their evidence. The operator may reopen it.
                        exited_viewers.add(name)
                        continue
                    return 0 if name == 'rviz' and code == 0 else 1
            time.sleep(0.25)
        return 0
    finally:
        for _, process, _ in reversed(processes):
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic() + 8
        for _, process, _ in reversed(processes):
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
        for _, _, stream in processes:
            stream.close()


if __name__ == '__main__':
    raise SystemExit(main())
