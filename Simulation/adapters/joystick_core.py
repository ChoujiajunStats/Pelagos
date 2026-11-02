"""Gamepad mapping and lease checks, independent of ROS and input devices."""
import math

# Five times the original .15/.15/.10/.25 body-velocity targets.
COMMAND_LIMITS=(.75,.75,.50,1.25)


def deadzone(value, threshold=.12):
    if not math.isfinite(value) or not 0<=threshold<1:
        raise ValueError('invalid axis or deadzone')
    value=max(-1.,min(1.,float(value)))
    return 0. if abs(value)<=threshold else math.copysign((abs(value)-threshold)/(1-threshold),value)


def select_mapping(axis_codes, button_codes):
    """Use Linux ABS/BTN codes so D-input and X-input axis indices both work."""
    if 0 not in axis_codes or 1 not in axis_codes:
        raise ValueError('手柄没有 ABS_X / ABS_Y 左摇杆')
    right=(3,4) if 3 in axis_codes and 4 in axis_codes else (2,5)
    if any(code not in axis_codes for code in right):
        raise ValueError('无法识别右摇杆；运行 --inspect 查看轴映射')
    required={'deadman':0x136,'stop':0x131,'reset':0x13b}
    if any(code not in button_codes for code in required.values()):
        raise ValueError('无法识别 LB、B 或 Start；运行 --inspect 查看按键映射')
    return dict(forward_axis=axis_codes.index(1),left_axis=axis_codes.index(0),
                up_axis=axis_codes.index(right[1]),yaw_axis=axis_codes.index(right[0]),
                **{name+'_button':button_codes.index(code) for name,code in required.items()})


def command(axes, buttons, mapping, threshold=.12):
    # ROS body FLU: +X forward, +Y left, +Z up, +yaw counterclockwise.
    values=[-deadzone(axes[mapping[axis+'_axis']],threshold)*limit
            for axis,limit in zip(('forward','left','up','yaw'),COMMAND_LIMITS)]
    return dict(velocity=values,
                deadman=bool(buttons[mapping['deadman_button']]),
                stop=bool(buttons[mapping['stop_button']]),
                reset=bool(buttons[mapping['reset_button']]))


def packet_error(packet, now_ns, session=None, previous_sequence=0):
    if not isinstance(packet,dict):return 'invalid_packet'
    if session is not None and packet.get('session')!=session:return 'input_session_changed'
    sequence=packet.get('sequence')
    stamp=packet.get('issued_ns')
    if type(sequence) is not int or sequence<=previous_sequence:return 'replayed_input'
    if type(stamp) is not int or not 0<=now_ns-stamp<=250_000_000:return 'input_expired'
    velocity=packet.get('velocity')
    if not isinstance(velocity,list) or len(velocity)!=4:return 'invalid_velocity'
    if any(type(v) not in (int,float) or not math.isfinite(v) or abs(v)>limit+1e-8
           for v,limit in zip(velocity,COMMAND_LIMITS)):return 'velocity_limit'
    if any(type(packet.get(k)) is not bool for k in ('deadman','stop','reset','connected')):
        return 'invalid_buttons'
    if not isinstance(packet.get('session'),str) or not packet['session']:return 'invalid_session'
    return None
