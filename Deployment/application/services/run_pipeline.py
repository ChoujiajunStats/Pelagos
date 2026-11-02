#!/usr/bin/env python3
"""独立运行四个算法进程，冻结输入并在任一退出时结束整次建图。"""
import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import secrets
import shutil
import signal
import subprocess
import sys
import time


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='milliseconds')


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest(path, manifest):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n',
                         encoding='utf-8')
    temporary.replace(path)


def hardware_summary():
    info = {'uname': list(platform.uname()), 'python': platform.python_version()}
    for label, source in [('os_release', '/etc/os-release'),
                          ('jetson_release', '/etc/nv_tegra_release'),
                          ('device_model', '/proc/device-tree/model')]:
        try:
            info[label] = Path(source).read_text().strip('\x00\n')[:4096]
        except OSError:
            pass
    return info


def snapshot_inputs(root, profile_dir, checkpoint, run_dir, mapper='wavemap', navigation=False):
    configuration = run_dir / 'configuration'
    shutil.copytree(profile_dir, configuration, ignore=shutil.ignore_patterns('__pycache__'))
    if mapper not in ('wavemap', 'nvblox'):
        raise ValueError(f'不支持的建图后端：{mapper}')
    if navigation and mapper != 'wavemap':
        raise ValueError('Wavestar 导航只支持 Wavemap 后端')
    required = ['okvis2.yaml', 'stereo-geometry.json', f'{mapper}.yaml']
    if navigation:
        required.append('navigation.yaml')
    for name in required:
        if not (configuration / name).is_file() or not (configuration / name).stat().st_size:
            raise FileNotFoundError(f'缺少算法配置：{configuration / name}')
    frozen_weight = run_dir / 'weights' / checkpoint.name
    frozen_weight.parent.mkdir(mode=0o700)
    before = sha256(checkpoint)
    shutil.copy2(checkpoint, frozen_weight)
    if sha256(checkpoint) != before or sha256(frozen_weight) != before:
        raise RuntimeError('复制期间 LAS2 权重发生变化，结束本次启动')
    files = {}
    for path in sorted(configuration.rglob('*')):
        if path.is_file():
            files[str(path.relative_to(run_dir))] = sha256(path)
    files[str(frozen_weight.relative_to(run_dir))] = before
    source_lock = root / 'Deployment/artifacts/locks/algorithm-sources.json'
    provenance = {'profile_source': str(profile_dir), 'checkpoint_source': str(checkpoint),
                  'snapshot_sha256': files, 'algorithm_inputs_are_snapshots': True,
                  'camera_configuration_mode': '相机由独立服务运行；这里只记录原配置哈希'}
    ingress = profile_dir / 'zed-x.yaml'
    if ingress.is_file():
        provenance['camera_configuration_source'] = str(ingress)
        provenance['camera_configuration_sha256_at_start'] = sha256(ingress)
    if source_lock.is_file():
        shutil.copy2(source_lock, run_dir / 'source-lock.json')
        provenance['source_lock_sha256'] = sha256(run_dir / 'source-lock.json')
    if navigation:
        navigation_lock = root / 'Deployment/contexts/navigation/provenance/source-lock.json'
        shutil.copy2(navigation_lock, run_dir / 'navigation-source-lock.json')
        provenance['navigation_source_lock_sha256'] = sha256(run_dir / 'navigation-source-lock.json')
        binary = root / 'Deployment/infrastructure/ros2_ws/install-navigation/underwater_navigation_ros/lib/underwater_navigation_ros/wavemap_navigation_node'
        provenance['navigation_binary_sha256'] = sha256(binary)
    return configuration, frozen_weight, provenance


def process_commands(root, configuration, checkpoint, run_dir, mapper='wavemap', navigation=False):
    if mapper not in ('wavemap', 'nvblox'):
        raise ValueError(f'不支持的建图后端：{mapper}')
    if navigation and mapper != 'wavemap':
        raise ValueError('Wavestar 导航只支持 Wavemap 后端')
    deployment = root / 'Deployment'
    geometry = str(configuration / 'stereo-geometry.json')
    commands = [
        ('okvis2', ['/usr/bin/bash', str(deployment / 'contexts/localization/scripts/start_okvis2.sh'),
                    '-p', f'config_filename:={configuration / "okvis2.yaml"}']),
        ('pose_adapter', ['/usr/bin/bash', str(deployment / 'contexts/localization/scripts/start_pose_adapter.sh'),
                          '--extrinsics', geometry]),
        ('las2', ['/usr/bin/bash', str(deployment / 'contexts/depth/scripts/start_las2.sh'),
                  '--calibration', geometry, '--checkpoint', str(checkpoint)]),
    ]
    if mapper == 'nvblox':
        commands.extend([
            ('nvblox_pose_tf', ['/usr/bin/bash', str(deployment / 'contexts/mapping/scripts/start_nvblox_pose_tf.sh')]),
            ('nvblox', ['/usr/bin/bash', str(deployment / 'contexts/mapping/scripts/start_nvblox.sh'),
                        str(configuration / 'nvblox.yaml')]),
        ])
    elif navigation:
        commands.append(('wavemap', ['/usr/bin/bash', str(deployment / 'contexts/navigation/scripts/start_navigation.sh'),
                         str(run_dir / 'map.wvmp'), str(configuration / 'navigation.yaml'), geometry]))
    else:
        commands.append(('wavemap', ['/usr/bin/bash', str(deployment / 'contexts/mapping/scripts/start_wavemap.sh'),
                         str(run_dir / 'map.wvmp'), str(configuration / 'wavemap.yaml')]))
    return commands


class PipelineSupervisor:
    def __init__(self, root, run_dir, manifest, grace_seconds=30.0):
        self.root = Path(root)
        self.run_dir = Path(run_dir)
        self.manifest = manifest
        self.manifest_path = self.run_dir / 'manifest.json'
        self.grace_seconds = grace_seconds
        self.children = []
        self.stop_signal = None

    def request_stop(self, signum, _frame):
        self.stop_signal = signum

    def save(self):
        write_manifest(self.manifest_path, self.manifest)

    def signal_groups(self, signum):
        for _, process, _, _ in self.children:
            try:
                os.killpg(process.pid, signum)
            except ProcessLookupError:
                pass

    def wait_groups(self, timeout):
        deadline = time.monotonic() + timeout
        while True:
            alive = False
            for _, process, _, _ in self.children:
                process.poll()
                try:
                    os.killpg(process.pid, 0)
                    alive = True
                except ProcessLookupError:
                    pass
            if not alive:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.1)

    def stop_children(self):
        before_shutdown = {name: process.poll()
                           for name, process, _, _ in self.children}
        self.manifest['status'] = 'stopping'
        try:
            self.save()
        except OSError as error:
            print(f'停止时无法更新清单，继续回收进程：{error}', file=sys.stderr, flush=True)
        self.signal_groups(signal.SIGINT)
        if not self.wait_groups(self.grace_seconds):
            self.manifest['shutdown_escalated_to'] = 'SIGTERM'
            self.signal_groups(signal.SIGTERM)
            if not self.wait_groups(3.0):
                self.manifest['shutdown_escalated_to'] = 'SIGKILL'
                self.signal_groups(signal.SIGKILL)
                self.wait_groups(3.0)
        abnormal_exits = []
        controlled_exit_codes = (130, 143, -signal.SIGINT, -signal.SIGTERM)
        for name, process, log, record in self.children:
            code = process.poll()
            record['returncode'] = code
            record['observed_stopped_at'] = utc_now()
            controlled_stop = before_shutdown[name] is None
            if code != 0 and not (controlled_stop and code in controlled_exit_codes):
                abnormal = {'process': name, 'returncode': code,
                            'phase': 'during_shutdown' if controlled_stop else 'before_shutdown'}
                if code is not None and code < 0:
                    try:
                        abnormal['signal'] = signal.Signals(-code).name
                    except ValueError:
                        abnormal['signal'] = str(-code)
                if code is None:
                    abnormal['error'] = 'process_still_running'
                abnormal_exits.append(abnormal)
            log.close()
        self.manifest['shutdown_clean'] = not abnormal_exits
        self.manifest['shutdown_abnormal_exits'] = abnormal_exits

    def run(self, commands):
        previous_handlers = {}
        for signum in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
            previous_handlers[signum] = signal.signal(signum, self.request_stop)
        exit_code = 1
        try:
            self.manifest['status'] = 'starting'
            self.manifest['processes'] = {}
            self.save()
            for name, command in commands:
                if self.stop_signal is not None:
                    break
                log_path = self.run_dir / 'logs' / (name + '.log')
                log = log_path.open('ab', buffering=0)
                try:
                    process = subprocess.Popen(command, cwd=self.root, stdin=subprocess.DEVNULL,
                                               stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)
                except BaseException:
                    log.close()
                    raise
                record = {'command': command, 'pid': process.pid, 'pgid': process.pid,
                          'log': str(log_path.relative_to(self.run_dir)), 'started_at': utc_now()}
                self.children.append((name, process, log, record))
                self.manifest['processes'][name] = record
                self.save()
            self.manifest['status'] = 'running'
            self.save()
            print(f'算法运行目录：{self.run_dir}', flush=True)
            critical_exited = False
            while self.stop_signal is None:
                for name, process, _, record in self.children:
                    code = process.poll()
                    if code is not None:
                        record['returncode'] = code
                        self.manifest['stop_reason'] = {'kind': 'critical_process_exited',
                                                        'process': name, 'returncode': code}
                        exit_code = min(code, 255) if code > 0 else (128 - code if code < 0 else 1)
                        # 只有受控停止才允许服务将 130/143 视为成功。
                        if exit_code in (130, 143):
                            exit_code = 1
                        critical_exited = True
                        break
                if critical_exited:
                    break
                time.sleep(0.2)
            if not critical_exited:
                self.manifest['stop_reason'] = {'kind': 'signal', 'signal': self.stop_signal}
                exit_code = 128 + self.stop_signal
        except BaseException as error:
            self.manifest['stop_reason'] = {'kind': 'supervisor_error',
                                            'error': f'{type(error).__name__}: {error}'}
            print(f'算法启动或监管失败：{error}', file=sys.stderr, flush=True)
            exit_code = 1
        finally:
            self.stop_children()
            if not self.manifest['shutdown_clean']:
                exit_code = 1
            provenance = self.manifest.get('inputs', {})
            changed_snapshots = []
            for relative, expected in provenance.get('snapshot_sha256', {}).items():
                try:
                    actual = sha256(self.run_dir / relative)
                except OSError:
                    actual = None
                if actual != expected:
                    changed_snapshots.append(relative)
            provenance['snapshot_inputs_unchanged_at_stop'] = not changed_snapshots
            provenance['changed_snapshot_inputs'] = changed_snapshots
            source = provenance.get('camera_configuration_source')
            if source:
                try:
                    after = sha256(source)
                except OSError:
                    after = None
                provenance['camera_configuration_sha256_at_stop'] = after
                provenance['camera_configuration_unchanged'] = (
                    after == provenance.get('camera_configuration_sha256_at_start'))
            self.manifest['status'] = 'stopped'
            self.manifest['stopped_at'] = utc_now()
            self.manifest['supervisor_returncode'] = exit_code
            try:
                self.save()
            finally:
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)
        return exit_code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument('--runs-dir', type=Path)
    parser.add_argument('--profile-dir', type=Path)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--mapper', choices=('wavemap', 'nvblox'), default='wavemap')
    parser.add_argument('--navigation', action='store_true', help='启用原生地图管理与 Wavestar（仅 Wavemap）')
    args = parser.parse_args()
    root = args.root.resolve()
    runs_dir = (args.runs_dir or root / 'runs').resolve()
    profile_dir = (args.profile_dir or root / 'Deployment/config/profiles/orin-nano').resolve()
    checkpoint = (args.checkpoint or root / 'Deployment/artifacts/weights/LAS2_S.pth').resolve()
    runs_dir.mkdir(parents=True, exist_ok=True)
    with (runs_dir / '.algorithm-pipeline.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print('已有算法主运行单元在运行，拒绝重复启动。', file=sys.stderr)
            return 1
        run_name = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        run_dir = runs_dir / (run_name + '-' + secrets.token_hex(4))
        run_dir.mkdir(mode=0o700)
        (run_dir / 'logs').mkdir(mode=0o700)
        manifest = {'schema_version': 1, 'started_at': utc_now(), 'status': 'preparing',
                    'root': str(root), 'run_dir': str(run_dir), 'hardware': hardware_summary(),
                    'ros_environment': {key: os.environ.get(key) for key in
                                        ('ROS_DOMAIN_ID', 'RMW_IMPLEMENTATION', 'ROS_LOCALHOST_ONLY')}}
        manifest_path = run_dir / 'manifest.json'
        write_manifest(manifest_path, manifest)
        try:
            configuration, frozen_weight, provenance = snapshot_inputs(
                root, profile_dir, checkpoint, run_dir, args.mapper, args.navigation)
            manifest['inputs'] = provenance
            manifest['mapper'] = args.mapper
            manifest['navigation'] = args.navigation
            commands = process_commands(root, configuration, frozen_weight, run_dir, args.mapper, args.navigation)
            return PipelineSupervisor(root, run_dir, manifest).run(commands)
        except (OSError, ValueError, RuntimeError) as error:
            manifest.update(status='stopped', stopped_at=utc_now(), supervisor_returncode=1,
                            stop_reason={'kind': 'preparation_error', 'error': str(error)})
            write_manifest(manifest_path, manifest)
            print(f'准备运行快照失败：{error}；详情：{manifest_path}', file=sys.stderr)
            return 1


if __name__ == '__main__':
    raise SystemExit(main())
