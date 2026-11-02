#!/usr/bin/env python3
"""Build a run from the installed Stonefish workspace's own M2 scene generator."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import secrets
import shutil
import sys
import xml.etree.ElementTree as ET

import yaml
from ament_index_python.packages import get_package_share_directory
from uw_robot.profiles import load_robot
from uw_simulations.m2_scene import generate_m2_scene

sys.path.insert(0, '/pelagos/Simulation/adapters')
from geometry import matrix4, optical_transform
from manual_envelope import manual_operating_envelope
from joystick_core import COMMAND_LIMITS
from mapping_config import configure_wavemap
from autonomy_config import nbv_config,source_hash

WORKSPACE = Path('/stonefish-workspace')
PELAGOS = Path('/pelagos')
ASSET_LOCK = Path('/opt/uw/asset-lock.json')


class OpenCvYamlDumper(yaml.SafeDumper):
    def increase_indent(self, flow=False, indentless=False):
        return super().increase_indent(flow, False)


def scalar_lists(dumper, seq):
    return dumper.represent_sequence('tag:yaml.org,2002:seq', seq,
                                     flow_style=all(isinstance(x, (int, float, str, bool)) for x in seq))


OpenCvYamlDumper.add_representer(list, scalar_lists)


def calibration(profile, out):
    left = profile['cameras']['left']
    right = profile['cameras']['right']
    for key in ('width', 'height', 'horizontal_fov_deg', 'rpy_frd'):
        if left[key] != right[key]:
            raise ValueError(f'stereo cameras disagree on {key}')
    width, height = int(left['width']), int(left['height'])
    fx = width / (2 * math.tan(math.radians(float(left['horizontal_fov_deg'])) / 2))
    cx, cy = (width - 1) / 2, (height - 1) / 2
    tl, tr = matrix4(optical_transform(left)), matrix4(optical_transform(right))
    baseline = float(profile['stereo_baseline_m'])
    if abs((left['xyz_frd'][1] - right['xyz_frd'][1]) + baseline) > 1e-8:
        raise ValueError('stereo baseline does not match the camera positions')
    k = [[fx, 0.0, cx], [0.0, fx, cy], [0.0, 0.0, 1.0]]
    identity = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    pl = [[fx, 0.0, cx, 0.0], [0.0, fx, cy, 0.0], [0.0, 0.0, 1.0, 0.0]]
    pr = [[fx, 0.0, cx, -fx * baseline], [0.0, fx, cy, 0.0], [0.0, 0.0, 1.0, 0.0]]
    geometry = dict(schema_version=1, width=width, height=height, baseline_m=baseline,
                    image_time_offset_s=0.0, camera_frame='cam0_rect_optical',
                    world_frame='world', body_frame='body', body_is_imu=True,
                    translation_unit='m', K_left=k, K_right=k, D_left=[0.0]*5,
                    D_right=[0.0]*5, R_left=identity, R_right=identity,
                    P_left=pl, P_right=pr, T_body_left_optical=tl,
                    T_body_right_raw_optical=tr)
    (out / 'stereo-geometry.json').write_text(json.dumps(geometry, indent=2) + '\n')
    template = (PELAGOS / 'Deployment/config/templates/okvis2-template.yaml').read_text()
    config = yaml.safe_load(template.removeprefix('%YAML:1.0\n'))
    for index, transform in enumerate((tl, tr)):
        config['cameras'][index].update(T_SC=[x for row in transform for x in row],
            image_dimension=[width, height], distortion_coefficients=[0.0]*4,
            focal_length=[fx, fx], principal_point=[cx, cy])
    config['camera_parameters']['image_delay'] = 0.0
    config['imu_parameters']['g0'] = [0.0, 0.0, 0.0]
    config['imu_parameters']['a0'] = [0.0, 0.0, 0.0]
    # Reuse OKVIS's own tracked observations, landmark IDs, reprojections and
    # tracking-quality display, rather than detecting a second set of points.
    config['output_parameters'].update(display_matches=True, display_overhead=True)
    (out / 'okvis2.yaml').write_text('%YAML:1.0\n---\n' + yaml.dump(
        config, Dumper=OpenCvYamlDumper, default_flow_style=None, sort_keys=False))
    return geometry


def add_depth_camera(scene, profile):
    tree = ET.parse(scene)
    robot = tree.getroot().find('robot')
    camera = profile['cameras']['left']
    depth = ET.SubElement(robot, 'sensor', name='camera_left_depth', type='depthcamera', rate='-1')
    ET.SubElement(depth, 'link', name=robot.find('base_link').get('name'))
    ET.SubElement(depth, 'origin', xyz=' '.join(map(str, camera['xyz_frd'])),
                  rpy=' '.join(map(str, camera['rpy_frd'])))
    ET.SubElement(depth, 'specs', resolution_x=str(camera['width']),
                  resolution_y=str(camera['height']), horizontal_fov=str(camera['horizontal_fov_deg']),
                  depth_min='0.05', depth_max='12.0')
    ET.SubElement(depth, 'noise', depth='0')
    ET.SubElement(depth, 'ros_publisher', topic='sim/raw/left_depth')
    ET.indent(tree, space='  ')
    tree.write(scene, encoding='utf-8', xml_declaration=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--scene', choices=('fixture', 'porth'), default='porth')
    parser.add_argument('--ros-domain', type=int, default=42)
    parser.add_argument('--autonomy', action='store_true')
    args = parser.parse_args()
    out = args.run_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    profile = load_robot(WORKSPACE / 'robot/config', 'bluerov2_six_disaware')
    config = yaml.safe_load((WORKSPACE / 'config/run.m2.yaml').read_text())
    config.update(run_id=out.name, namespace='rov01', robot_profile='bluerov2_six_disaware',
                  scene_profile='visual_fixture' if args.scene == 'fixture' else 'porth_sump9',
                  ros_domain_id=args.ros_domain, state_source='ground_truth_debug',
                  evaluation_phase='manual', case_id='manual', fixed_fixture=False,
                  control_mode='body_velocity')
    if not args.autonomy:
        config['manual_position_limits'] = 'disabled'
        config['manual_velocity_limits'] = list(COMMAND_LIMITS)
    if args.autonomy:
        if args.scene!='porth':raise ValueError('Autonomy bounds are approved for the Porth scene only')
        config.update(schema_version=4, state_source='okvis2_stereo_imu',
                      control_state_source='okvis2_stereo_imu', estimator_profile='okvis2',
                      controller_backend='jepa_mppi', planner_profile='ethz_nbvplanner',
                      observation_access='ESTIMATED', evaluation_phase='exploration')
        config['planning_only']=os.environ.get('PELAGOS_PLANNING_ONLY','0')=='1'
    (out / 'resolved_config.yaml').write_text(yaml.safe_dump(config, sort_keys=False))
    (out / 'robot_profile.yaml').write_text(yaml.safe_dump(profile, sort_keys=False))
    session = {'terminal_secret': secrets.token_hex(24)}
    (out / 'session.json').write_text(json.dumps(session) + '\n')
    (out / 'session.json').chmod(0o600)
    shutil.copy2(WORKSPACE / 'robot/config/thrusters_six_disaware.yaml', out / 'thrusters.yaml')
    controller = yaml.safe_load((WORKSPACE / 'controller/config/defaults.yaml').read_text())
    if args.autonomy:
        controller['backend'] = 'jepa_mppi'
        settings = yaml.safe_load((WORKSPACE / 'controller/config/jepa_mppi.yaml').read_text())
        settings.update(repository_root='/pelagos/TP/Controller',
                        checkpoint='/pelagos/TP/Controller/disaware_learning/checkpoints/stage2_zhat_response_potential_full4096_four_modal_e5.pt',
                        mppi_config='disaware_learning/config/jepa_mppi_feedback_search.json',
                        state_source='OKVIS2_STEREO_IMU',run_dir=str(out))
        settings['fluid_density_kg_m3']=yaml.safe_load((out/'thrusters.yaml').read_text())['density_kg_m3']
        selection=yaml.safe_load((PELAGOS / 'Simulation/config/nbv.yaml').read_text())
        settings['mppi_overrides']=dict(selection.get('jepa_mppi_overrides',{}))
        (out / 'jepa-settings.yaml').write_text(yaml.safe_dump(settings, sort_keys=False))
        navigation=yaml.safe_load((WORKSPACE/'navigation/config/defaults.yaml').read_text())
        source_horizontal_speed=float(navigation['horizontal_speed_m_s'])
        speed_multiplier=float(selection.get('horizontal_speed_multiplier',1.0))
        if not 0<speed_multiplier< float('inf'):raise ValueError('Invalid approved horizontal speed multiplier')
        navigation['horizontal_speed_m_s']=source_horizontal_speed*speed_multiplier
        (out/'navigation-parameters.yaml').write_text(yaml.safe_dump(navigation,sort_keys=False))
        asset_metadata=json.loads(Path('/porth/asset.json').read_text())
        nbv,provenance=nbv_config(PELAGOS,selection,asset_metadata,profile['cameras']['left'])
        nbv['pelagos_map_source']=selection.get('map_source','wavemap')
        nbv['pelagos_start_unknown_policy']=selection.get('startup_unknown_policy','initial_self_only')
        nbv['pelagos_obstacle_response']=selection.get('obstacle_response','turn_then_cruise')
        nbv['pelagos_obstacle_plan_distance_m']=float(selection.get('obstacle_plan_distance_m',3.0))
        nbv['pelagos_forward_task']=selection.get('forward_task_constraint')=='initial_heading'
        nbv['pelagos_allow_self_unknown']=nbv['pelagos_forward_task']
        nbv['pelagos_start_min_depth_m']=float(yaml.safe_load((PELAGOS/'Simulation/config/wavemap.yaml').read_text())['/**']['ros__parameters']['min_range_m'])
        if nbv['pelagos_map_source'] not in ('wavemap','native_depth'):raise ValueError('Unsupported NBV map source')
        provenance['map_source']=dict(owner=nbv['pelagos_map_source'],
            native_depth_fusion_enabled=nbv['pelagos_map_source']=='native_depth',
            octomap_resolution_m=nbv['resolution'],
            projection='any occupied subcell wins; free requires complete observed-free coverage; otherwise unknown',
            classification='Wavemap OccupancyClassifier default: log odds 0; unknown abs(log odds)<1e-3',
            frame='world -> rov01/odom must be identity; no GT alignment',
            documentation='Simulation/nbv/WAVEMAP_ADAPTATION.md',
            sources={relative:source_hash(PELAGOS/relative) for relative in (
                'Simulation/adapters/wavemap_nbv_relay.py','Simulation/nbv/native.cpp',
                'Simulation/nbv/patch_sources.py',
                'Deployment/infrastructure/ros2_ws/src/underwater_wavemap_ros/include/underwater_wavemap_ros/octomap_projection.hpp',
                'TP/Mapping/wavemap/library/cpp/include/wavemap/core/utils/query/occupancy_classifier.h',
                'TP/Mapping/wavemap/library/cpp/include/wavemap/core/utils/query/impl/occupancy_classifier_inl.h',
                'TP/Planner/nbvplanner/volumetric_mapping/octomap_world/src/octomap_world.cc')})
        provenance['startup']=dict(policy=nbv['pelagos_start_unknown_policy'],
            obstacle_response=nbv['pelagos_obstacle_response'],
            execution_approval='User explicitly requested NBV turn selection followed by continuous forward cruise until the next approximately 3 m obstacle; no source endpoint tracking or sampled-view observation wait',
            obstacle_plan_distance_m=nbv['pelagos_obstacle_plan_distance_m'],
            min_depth_m=nbv['pelagos_start_min_depth_m'],max_distance_m=None,
            approval='2026-11-01: user explicitly requested continuous straight motion through FOV-limited startup unknown space until about 3 m before an obstacle, then local planning; discard incompatible initialization conditions',
            map_free_fabrication=False,known_occupied_rejected=True,
            view_heading_gate=False,body_known_gate=False,
            collision_source_sha256=source_hash(PELAGOS/'Simulation/nbv/forward_obstacle.hpp'))
        algorithm=PELAGOS/'TP/Controller/disaware_learning/config/jepa_mppi_feedback_search.json'
        shutil.copy2(algorithm,out/'jepa-algorithm-source.json')
        provenance['jepa']=dict(source=str(algorithm),sha256=source_hash(algorithm),
            overrides=settings['mppi_overrides'],
            override_approval=selection.get('jepa_override_approval','User approved sample_count 256 -> 128 on 2026-10-30; all other values retained'),
            override_evidence_run=selection.get('jepa_override_evidence_run','20261030T064446Z-67d66a1b'))
        for relative in ('control/body_twist_pid.py','collection/jepa_mppi_rollout.py','deployment/protocol.py'):
            source=PELAGOS/'TP/Controller/disaware_learning'/relative
            provenance['jepa'].setdefault('execution_sources',{})[relative]=source_hash(source)
        provenance['navigation_adapter_source']=str(WORKSPACE/'navigation/config/defaults.yaml')
        provenance['navigation_adapter_sha256']=source_hash(WORKSPACE/'navigation/config/defaults.yaml')
        provenance['navigation_speed']=dict(source_horizontal_m_s=source_horizontal_speed,
            multiplier=speed_multiplier,requested_horizontal_m_s=navigation['horizontal_speed_m_s'],
            approval='2026-11-01: user explicitly requested four times the current driving speed',
            vertical_and_yaw_unchanged=True)
        provenance['control_adapter_source']=str(WORKSPACE/'controller/config/jepa_mppi.yaml')
        provenance['control_adapter_sha256']=source_hash(WORKSPACE/'controller/config/jepa_mppi.yaml')
        provenance['control_position_limits']='disabled'
        provenance['manual_takeover']=dict(
            feedback='OKVIS2 native body odometry; no GT',
            controller_source=str(WORKSPACE/'controller/uw_controller/core.py'),
            controller_sha256=source_hash(WORKSPACE/'controller/uw_controller/core.py'),
            parameters_source=str(WORKSPACE/'controller/config/defaults.yaml'),
            parameters_sha256=source_hash(WORKSPACE/'controller/config/defaults.yaml'),
            request_limits=list(COMMAND_LIMITS),
            limits_source='Previously approved Simulation/adapters/joystick_core.py manual profile',
            authority='One original Guard, one controller output publisher, one actuator terminal')
        provenance['execution_adaptations']=[
            'Source scan waypoints relative to initial OKVIS pose; advance after source hold and tracker arrival',
            'No RotorS tracking-bias z offset on the ROV',
            'Interpolate OKVIS feedback on checkpoint model clock; consume actual IMU, pressure and applied actions',
            'Use source JEPA normalized roll PD and mixer instead of generic force attitude loop',
            'Derive motor direction from native allocation geometry and model-axis definitions; invert on feedback',
            'Prewarm one real-history optimizer call without sending actuator commands',
            'Pressure-depth water density matches the same frozen thruster profile used to generate the scene']
        (out/'autonomy-provenance.json').write_text(json.dumps(provenance,indent=2))
        (out / 'nbv.yaml').write_text(yaml.safe_dump(nbv, sort_keys=False))
    (out / 'controller-parameters.yaml').write_text(yaml.safe_dump(controller, sort_keys=False))
    guard = yaml.safe_load((WORKSPACE / 'guard/config/defaults.yaml').read_text())
    guard['safety_envelope'] = manual_operating_envelope(guard['safety_envelope'])
    if args.autonomy:
        # Admit the explicitly requested navigation speed without changing
        # state/command deadlines, measured-motion envelopes or motor limits.
        guard['velocity_limits'][:2]=[max(float(value),navigation['horizontal_speed_m_s'])
                                      for value in guard['velocity_limits'][:2]]
    if not args.autonomy:
        guard['velocity_limits'] = list(COMMAND_LIMITS)
        # The requested yaw target is now 1.25 rad/s. Retaining the old
        # 1.2 rad/s actual-rate guard would stop a valid full-stick turn.
        guard['safety_envelope']['angular_speed_peak_rad_s'] = max(
            guard['safety_envelope']['angular_speed_peak_rad_s'],2*COMMAND_LIMITS[3])
    (out / 'guard-parameters.yaml').write_text(yaml.safe_dump(guard, sort_keys=False))
    geometry = calibration(profile, out)
    template = yaml.safe_load((PELAGOS / 'Simulation/config/wavemap.yaml').read_text())
    mapping = configure_wavemap(template,geometry,out)
    if args.autonomy and nbv.get('pelagos_map_source')=='wavemap':
        mapping['/**']['ros__parameters'].update(octomap_topic='/mapping/octomap',
            octomap_resolution_m=float(nbv['resolution']),localization_status_topic='/rov01/localization/status')
    (out / 'wavemap.yaml').write_text(yaml.safe_dump(mapping,sort_keys=False))
    upstream = Path(get_package_share_directory('stonefish_bluerov2'))
    scene = out / 'scene.scn'
    assets = json.loads(ASSET_LOCK.read_text())
    generate_m2_scene(upstream, WORKSPACE / 'simulations/scenarios/empty_water.scn', profile,
                      'rov01', scene, assets, config, session, out / 'thrusters.yaml')
    if args.scene == 'porth':
        from uw_simulations.porth_scene import load_asset, add_cave
        asset = load_asset('/porth')
        add_cave(scene, '/porth', asset)
    add_depth_camera(scene, profile)
    evidence_path = scene.with_suffix('.assets.json')
    evidence = json.loads(evidence_path.read_text())
    evidence['scene_sha256'] = hashlib.sha256(scene.read_bytes()).hexdigest()
    evidence['pelagos_depth_sensor'] = 'left camera, native optical Z, render acquisition timestamp'
    evidence_path.write_text(json.dumps(evidence, indent=2) + '\n')
    print(json.dumps({'scene': str(scene), 'namespace': 'rov01',
                      'manual_position_limits': config.get('manual_position_limits'),
                      'manual_velocity_limits': config.get('manual_velocity_limits'),
                      'motion_feedback': ('planning only; no controller or actuator adapter' if config.get('planning_only')
                                          else 'OKVIS2 body + pressure + IMU; JEPA-MPPI' if args.autonomy
                                          else 'simulation truth, manual diagnostic only')}, indent=2))


if __name__ == '__main__':
    main()
