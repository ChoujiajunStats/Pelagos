#!/usr/bin/env python3
"""只读采样板端进程、资源和现有地图日志，不新增图像订阅或启停服务。"""
import argparse
import datetime as dt
import json
import math
from pathlib import Path
import re
import signal
import time


MAP_LINE = re.compile(r'Published ([0-9]+) occupied cells after ([0-9]+) integrated frames [(]([0-9]+) skipped[)]')


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec='milliseconds')


def process_state(pid):
    directory = Path('/proc') / str(pid)
    try:
        fields = directory.joinpath('stat').read_text().rsplit(') ', 1)[1].split()
        values = {}
        for line in directory.joinpath('status').read_text().splitlines():
            key, _, raw = line.partition(':')
            if key in ('VmRSS', 'VmHWM', 'VmSwap', 'Threads'):
                parts = raw.split()
                values[key] = int(parts[0]) * (1024 if len(parts) > 1 and parts[1] == 'kB' else 1)
        return dict(pid=pid, present=True, state=fields[0], start_ticks=int(fields[19]),
                    argv=directory.joinpath('cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0'),
                    status=values)
    except (OSError, ValueError, IndexError):
        return dict(pid=pid, present=False)


def descendants(pid):
    result, pending, seen = [], [pid], {pid}
    while pending:
        current = pending.pop()
        try:
            children = Path(f'/proc/{current}/task/{current}/children').read_text().split()
        except OSError:
            continue
        for raw in children:
            child = int(raw)
            if child not in seen:
                seen.add(child)
                pending.append(child)
                result.append(process_state(child))
    return result


def resources():
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        name, _, raw = line.partition(':')
        parts = raw.split()
        if parts:
            memory[name] = int(parts[0]) * (1024 if len(parts) > 1 and parts[1] == 'kB' else 1)
    temperatures = {}
    for zone in sorted(Path('/sys/class/thermal').glob('thermal_zone*')):
        try:
            name = zone.joinpath('type').read_text().strip()
            value = float(zone.joinpath('temp').read_text()) / 1000
            if math.isfinite(value):
                temperatures[name + ':' + zone.name] = value
        except (OSError, ValueError, TypeError):
            pass
    return dict(mem_available_bytes=memory.get('MemAvailable'),
                mem_total_bytes=memory.get('MemTotal'),
                swap_used_bytes=memory.get('SwapTotal', 0) - memory.get('SwapFree', 0),
                loadavg=Path('/proc/loadavg').read_text().strip(), temperatures_c=temperatures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--seconds', type=float, default=300)
    parser.add_argument('--interval', type=float, default=5)
    parser.add_argument('--stall-seconds', type=float, default=15)
    args = parser.parse_args()
    if any(not math.isfinite(value) or value <= 0 for value in (args.seconds, args.interval, args.stall_seconds)):
        parser.error('时间参数必须为有限正数')
    run_dir = args.run_dir.resolve()
    manifest = json.loads((run_dir / 'manifest.json').read_text())
    tracked = {name: record['pid'] for name, record in manifest['processes'].items()}
    identities = {name: process_state(pid).get('start_ticks') for name, pid in tracked.items()}
    output = run_dir / 'stability-observation.json'
    result = dict(schema_version=1, run_dir=str(run_dir), started_at=utc_now(), requested_seconds=args.seconds,
                  sample_interval_s=args.interval, stall_threshold_s=args.stall_seconds,
                  map_source_zh='读取 Wavemap 现有约五秒节流日志；融合和跳帧计数为日志时刻累计值，未新增 DDS 订阅。',
                  monitored_pids=tracked, samples=[], mapping_events=[], interrupted=False, completed=False)
    log = (run_dir / 'logs/wavemap.log').open('r', encoding='utf-8', errors='replace')
    initial = log.read()
    baseline = list(MAP_LINE.finditer(initial))
    previous = None
    if baseline:
        previous = dict(elapsed_s=0.0, occupied_cells=int(baseline[-1][1]),
                        integrated_frames=int(baseline[-1][2]), skipped_frames=int(baseline[-1][3]), baseline=True)
        result['mapping_events'].append(previous)
    started = time.monotonic()
    stopped = False
    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True
        result['interrupted'] = True
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    def save():
        temporary = output.with_suffix('.tmp')
        temporary.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(output)

    next_sample = 0.0
    try:
        while True:
            elapsed = time.monotonic() - started
            for match in MAP_LINE.finditer(log.read()):
                event = dict(elapsed_s=elapsed, occupied_cells=int(match[1]), integrated_frames=int(match[2]),
                             skipped_frames=int(match[3]), baseline=False)
                result['mapping_events'].append(event)
                previous = event
            if elapsed >= next_sample or elapsed >= args.seconds or stopped:
                processes = {}
                for name, pid in tracked.items():
                    state = process_state(pid)
                    state['same_process_alive'] = (state.get('present') and state.get('state') != 'Z'
                                                   and state.get('start_ticks') == identities[name]
                                                   and identities[name] is not None)
                    state['descendants'] = descendants(pid)
                    processes[name] = state
                result['samples'].append(dict(elapsed_s=elapsed, wall_time=utc_now(), processes=processes,
                                              resources=resources(), latest_mapping=previous))
                next_sample = elapsed + args.interval
                result['observed_seconds'] = elapsed
                save()
            if elapsed >= args.seconds or stopped:
                break
            time.sleep(min(0.25, max(0.01, args.seconds - elapsed)))
    finally:
        log.close()
        duration = time.monotonic() - started
        events = result['mapping_events']
        increases = [(right['elapsed_s'], right['integrated_frames'] - left['integrated_frames'])
                     for left, right in zip(events, events[1:])
                     if right['integrated_frames'] > left['integrated_frames']]
        decreases = sum(right['integrated_frames'] < left['integrated_frames']
                        for left, right in zip(events, events[1:]))
        progress_times = [0.0] + [elapsed for elapsed, _ in increases] + [duration]
        gaps = [dict(start_s=left, end_s=right, duration_s=right-left)
                for left, right in zip(progress_times, progress_times[1:])]
        stalls = [gap for gap in gaps if gap['duration_s'] >= args.stall_seconds]
        all_alive = bool(result['samples']) and all(
            process['same_process_alive'] for sample in result['samples']
            for process in sample['processes'].values())
        memory = [sample['resources']['mem_available_bytes'] for sample in result['samples']]
        swap = [sample['resources']['swap_used_bytes'] for sample in result['samples']]
        temperatures = {}
        for sample in result['samples']:
            for name, value in sample['resources']['temperatures_c'].items():
                temperatures[name] = max(value, temperatures.get(name, value))
        result.update(completed=not result['interrupted'] and duration >= args.seconds,
                      completed_at=utc_now(), observed_seconds=duration)
        result['summary'] = dict(all_tracked_processes_alive_at_every_sample=all_alive,
                                 mapping_first=events[0] if events else None, mapping_last=events[-1] if events else None,
                                 integrated_frames_increase=sum(delta for _, delta in increases), counter_decreases=decreases,
                                 skipped_frames_increase=(events[-1]['skipped_frames'] - events[0]['skipped_frames']) if events else None,
                                 maximum_observed_progress_gap_s=max((gap['duration_s'] for gap in gaps), default=None),
                                 observed_stall_intervals=stalls, mem_available_min_bytes=min(memory) if memory else None,
                                 swap_used_start_bytes=swap[0] if swap else None, swap_used_end_bytes=swap[-1] if swap else None,
                                 swap_used_max_bytes=max(swap) if swap else None, temperature_max_c=temperatures)
        result['short_continuity_observed'] = bool(result['completed'] and all_alive and increases
                                                    and not decreases and not stalls)
        result['qualified'] = False
        result['limits_zh'] = ['只证明本观察窗口的进程与地图计数连续性；不替代水中运动、轨迹尺度和地图质量验收。',
                               '地图日志约五秒发布一次；停滞时长按日志到达采样估算，不能替代逐帧时间分析。',
                               '进程存活按采样检查；温度仅包含当前用户能够读取的温区。',
                               '未启停或重配算法服务；资源变化同时包含 RViz/网络等外部负载。']
        save()
    print(str(output), flush=True)
    return 0 if result['short_continuity_observed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
