#!/usr/bin/env python3
"""板端同包顺序回放，保存后端日志、地图与 tegrastats；不启动相机或 VIO。"""
import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time


def stop(process):
    if process.poll() is None:
        os.killpg(process.pid, signal.SIGINT)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bag', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rate', type=float, default=1.)
    parser.add_argument('--mapper', choices=('nvblox', 'wavemap'), required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[4]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ, ROS_DOMAIN_ID='87', DAPENG_USE_SIM_TIME='true')
    children, streams = [], []

    def launch(name, command):
        stream = (output / (name + '.log')).open('w')
        streams.append(stream)
        process = subprocess.Popen(command, env=env, stdout=stream, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        children.append(process)
        return process

    record = dict(mapper=args.mapper, bag=str(args.bag.resolve()), rate=args.rate,
                  ros_domain_id=87, voxel_size_m=.1, started_at=time.time())
    try:
        launch('tegrastats', ['tegrastats', '--interval', '1000'])
        if args.mapper == 'nvblox':
            launch('pose_tf', ['bash', str(root/'Deployment/contexts/mapping/scripts/start_nvblox_pose_tf.sh')])
            mapper = launch('mapper', ['bash', str(root/'Deployment/contexts/mapping/scripts/start_nvblox.sh')])
        else:
            mapper = launch('mapper', ['bash', str(root/'Deployment/contexts/mapping/scripts/start_wavemap.sh'),
                                      str(output/'map.wvmp')])
        time.sleep(5)
        if mapper.poll() is not None:
            raise RuntimeError('地图节点启动失败')
        started = time.monotonic()
        player = launch('player', ['ros2', 'bag', 'play', str(args.bag.resolve()),
                                   '--rate', str(args.rate), '--delay', '3', '--clock', '100'])
        player.wait(timeout=300)
        record['playback_wall_s'] = time.monotonic() - started
        record['player_returncode'] = player.returncode
        if player.returncode:
            raise RuntimeError('数据包回放失败，见 player.log')
        time.sleep(8)
        if args.mapper == 'nvblox':
            for service, filename in [('save_map', 'map.nvblx'), ('save_timings', 'timings.txt'),
                                      ('save_rates', 'rates.txt')]:
                result = subprocess.run(['ros2', 'service', 'call', '/nvblox_node/'+service,
                                         'nvblox_msgs/srv/FilePath',
                                         '{file_path: '+str(output/filename)+'}'],
                                        env=env, text=True, capture_output=True, timeout=30)
                (output/(service+'.log')).write_text(result.stdout+result.stderr)
                record[service+'_returncode'] = result.returncode
                record[service+'_success'] = result.returncode == 0 and 'success=True' in result.stdout
                if not record[service+'_success']:
                    raise RuntimeError(service + ' 未成功，见对应服务日志')
    finally:
        for child in children:
            if child.args and 'bag' in str(child.args) and 'play' in str(child.args):
                stop(child)
        time.sleep(2)
        for child in reversed(children):
            stop(child)
        for stream in streams:
            stream.close()
        record['child_returncodes'] = [child.returncode for child in children]
        record['completed_at'] = time.time()
        (output/'measurement.json').write_text(json.dumps(record, indent=2)+'\n')


if __name__ == '__main__':
    main()
