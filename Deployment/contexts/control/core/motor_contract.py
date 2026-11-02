"""Vendor 24-byte BB frames; BlueROV model order is distinct from wire order."""
import math

CONTRACT_ID='vendor_bb_bluerov6_right_first_l4_r4_1500_v2'
WIRE_ORDER=('L1','L2','L3','L4','R1','R2','R3','R4')
# Training actuator order: FrontRight, FrontLeft, BackRight, BackLeft,
# DiveRight, DiveLeft. Vendor L/R labels are physical left/right.
MODEL_ORDER=('R1','L1','R3','L3','R2','L2')


def modal_from_motors(values):
    if len(values)!=6 or not all(math.isfinite(v) and -1<=v<=1 for v in values):
        raise ValueError('Expected six normalized model-order motor demands')
    a,b,c,d,e,f=values
    return [(-a-b+c+d)/4,(a-b+c-d)/4,-(e+f)/2,(a-b-c+d)/4,(e-f)/2]


def decode_motor_frame(frame,channels=MODEL_ORDER,signs=(1,1,1,1,1,1)):
    if len(frame)!=24 or frame[0]!=255 or frame[23]!=187:
        raise ValueError('Expected vendor FF ... BB motor frame')
    if tuple(channels)!=MODEL_ORDER or len(signs)!=6 or any(s not in (-1,1) for s in signs):
        raise ValueError('Expected four horizontal then two vertical model channels')
    slots={name:int.from_bytes(frame[1+2*i:3+2*i],'big') for i,name in enumerate(WIRE_ORDER)}
    if any(not 1000<=v<=2000 for v in slots.values()):raise ValueError('Motor PWM outside vendor range')
    if slots['L4']!=1500 or slots['R4']!=1500:raise ValueError('Unused L4/R4 must be neutral')
    motors=[(slots[name]-1500)/500*sign for name,sign in zip(channels,signs)]
    modal=modal_from_motors(motors)
    return dict(protocol='motor',motor_pwm=slots,model_motor_normalized=motors,
                modal_action=modal,command=modal[:4],armed=bool(frame[21]),gear=1,
                heading_hold=False,depth_hold=False,actuation_contract=CONTRACT_ID)
