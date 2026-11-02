#!/usr/bin/env python3
import sys
from pathlib import Path as TaskLocation
for _role in ('core', 'adapters', 'services'):
    sys.path.insert(0, str(TaskLocation(__file__).resolve().parents[1] / _role))
"""相机独占诊断：导出本机 SDK 标定、IMU 噪声参数和批量采样时序。"""
import argparse
import json
from pathlib import Path
import time
import threading
import pyzed.sl as sl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--mode', choices=('grab', 'read'), default='grab')
    parser.add_argument('--sensor-poll-rate', type=float, default=0.0)
    args = parser.parse_args()
    init = sl.InitParameters()
    init.set_from_serial_number(42312885)
    init.camera_resolution = sl.RESOLUTION.HD1080
    init.camera_fps = 15
    init.coordinate_units = sl.UNIT.METER
    init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP_X_FWD
    init.depth_mode = sl.DEPTH_MODE.NONE
    init.camera_disable_self_calib = True
    camera = sl.Camera()
    status = camera.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        raise RuntimeError(str(status))
    try:
        info = camera.get_camera_information(sl.Resolution(960, 540))
        sensors = info.sensors_configuration

        def intrinsics(cam):
            return dict(fx=cam.fx, fy=cam.fy, cx=cam.cx, cy=cam.cy, distortion=list(cam.disto))

        def calibration(cal):
            return dict(left=intrinsics(cal.left_cam), right=intrinsics(cal.right_cam),
                        stereo_transform=cal.stereo_transform.m.tolist())

        def sensor(par):
            return dict(noise_density=par.noise_density, random_walk=par.random_walk,
                        sampling_rate=par.sampling_rate, sensor_unit=str(par.sensor_unit),
                        sensor_range=list(par.sensor_range), resolution=par.resolution)

        result = dict(serial_number=info.serial_number, sdk=sl.Camera.get_sdk_version(),
                      coordinate_system=str(init.coordinate_system), coordinate_units=str(init.coordinate_units),
                      width=960, height=540, acquisition_mode=args.mode,
                      rectified=calibration(info.camera_configuration.calibration_parameters),
                      raw=calibration(info.camera_configuration.calibration_parameters_raw),
                      camera_imu_transform=sensors.camera_imu_transform.m.tolist(),
                      gyroscope=sensor(sensors.gyroscope_parameters),
                      accelerometer=sensor(sensors.accelerometer_parameters))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        stamps = []
        batches = []
        stop = threading.Event()

        def collect():
            batch = []
            batch_status = camera.get_sensors_data_batch(batch)
            if batch_status == sl.ERROR_CODE.SUCCESS:
                values = [x.get_imu_data().timestamp.get_nanoseconds() for x in batch]
                batches.append(values)
                stamps.extend(values)

        def poll():
            while not stop.is_set():
                collect()
                stop.wait(1.0 / args.sensor_poll_rate)

        worker = None
        if args.sensor_poll_rate > 0:
            worker = threading.Thread(target=poll)
            worker.start()
        started = time.monotonic()
        try:
            while time.monotonic() - started < 5:
                acquire = camera.grab if args.mode == 'grab' else camera.read
                if acquire() == sl.ERROR_CODE.SUCCESS and worker is None:
                    collect()
        finally:
            stop.set()
            if worker is not None:
                worker.join()
        unique = sorted(set(stamps))
        result['imu_batch_diagnostic'] = dict(
            duration_s=time.monotonic() - started, batches=batches, unique_count=len(unique),
            sensor_poll_rate=args.sensor_poll_rate,
            unique_rate_hz=((len(unique)-1)*1e9/(unique[-1]-unique[0]) if len(unique)>1 else None),
            batches_ordered=all(all(a < b for a,b in zip(row,row[1:])) for row in batches))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps({key: value for key, value in result.items()
                          if key not in ('raw', 'rectified', 'imu_batch_diagnostic')}, indent=2))
        print('IMU 批量采样唯一计数:', len(unique))
        print('记录:', args.output)
    finally:
        camera.close()


if __name__ == '__main__':
    main()
