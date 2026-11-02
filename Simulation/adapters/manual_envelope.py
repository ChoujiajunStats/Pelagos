"""Manual simulation settings and readable state diagnostics."""
import math


def manual_operating_envelope(original):
    # The retained Guard accepts numerical lower/upper bounds. Infinity
    # explicitly disables position checks without substituting another
    # arbitrary exploration box. Finite-state, attitude, speed and lease
    # validation remain in the existing Guard.
    envelope=dict(original)
    envelope['depth_enu_m']=[-math.inf,math.inf]
    envelope['horizontal_position_abs_m']=math.inf
    return envelope


def state_diagnostic(position,quaternion,velocity,envelope):
    x,y,z,w=quaternion
    roll=math.degrees(math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)))
    pitch=math.degrees(math.asin(max(-1.,min(1.,2*(w*y-z*x)))))
    low,high=envelope['depth_enu_m']
    violations=[]
    if not low<position[2]<high:
        violations.append(f'Z={position[2]:.3f} m outside ({low:.3f}, {high:.3f})')
    horizontal=envelope['horizontal_position_abs_m']
    if max(abs(v) for v in position[:2])>=horizontal:
        violations.append(f'XY={position[:2]} outside +/-{horizontal} m')
    if max(abs(roll),abs(pitch))>=envelope['roll_pitch_peak_deg']:
        violations.append(f'roll/pitch={roll:.1f}/{pitch:.1f} deg exceeds {envelope["roll_pitch_peak_deg"]}')
    if max(abs(v) for v in velocity[:3])>=envelope['linear_speed_peak_m_s']:
        violations.append('linear_speed_limit')
    if max(abs(v) for v in velocity[3:])>=envelope['angular_speed_peak_rad_s']:
        violations.append('angular_speed_limit')
    return dict(position_enu_m=list(position),roll_pitch_deg=[roll,pitch],
                body_velocity=list(velocity),violations=violations)
