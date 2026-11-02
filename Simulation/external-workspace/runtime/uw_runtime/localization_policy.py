"""Explicit navigation tuning, shared by state validation and motion authority."""

SESSION_KEYFRAMES = 'session_keyframes_v1'


def estimated_state_source(config):
    vins=config.get('estimator_profile')=='vins_fusion_m3'
    if config.get('state_fusion_profile')=='rtabmap_ekf':
        return 'VINS_FUSION_RTABMAP_EKF' if vins else 'OPENVINS_RTABMAP_EKF'
    return 'VINS_FUSION_STEREO_IMU' if vins else 'OPENVINS_STEREO_IMU'


def localization_policy(config):
    name = config.get('planner_localization_policy', 'strict')
    if name not in ('strict', SESSION_KEYFRAMES):
        raise ValueError('Unknown navigation localization policy')
    session = name == SESSION_KEYFRAMES
    if session and not (config.get('state_fusion_profile') == 'rtabmap_ekf'
                        and config.get('planner_restore_map')
                        and config.get('scene_profile') == 'porth_sump9'
                        and config.get('controller_backend') in ('jepa_mppi','tinympc')):
        raise ValueError('Session keyframes require restored Porth visual-map fusion')
    return dict(name=name, session_keyframes=session,
                maximum_position_variance_m2=1. if session else .25,
                maximum_visual_gap_sec=180. if session else 90.,
                maximum_pose_correction_m=.5 if session else .15,
                nominal_position_variance_m2=.25,
                nominal_visual_gap_sec=10.)
