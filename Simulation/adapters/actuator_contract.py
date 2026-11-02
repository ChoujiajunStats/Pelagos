"""Derive the model-motor / native-shaft polarity from the actual allocation."""


def model_motor_polarity(allocation,mixer):
    # Model axes: forward, right, up, clockwise yaw, positive body roll.
    # The allocator reports forces/torques in body FLU.
    expected=((0,1),(1,-1),(2,1),(5,-1),(3,1))
    signs=[]
    for axis,(row,direction) in enumerate(expected):
        command=[0.]*4;roll=0.
        if axis==4:roll=1.
        else:command[axis]=1.
        motor,_=mixer(command,roll=roll)
        effect=float(allocation.predict(motor)[row])
        if abs(effect)<1e-8:raise ValueError('模型动作在原生推进器上没有对应控制权限')
        signs.append(1. if effect*direction>0 else -1.)
    if len(set(signs))!=1:raise ValueError('模型到原生电机不是统一符号转换，需要核对通道顺序')
    return signs[0]
