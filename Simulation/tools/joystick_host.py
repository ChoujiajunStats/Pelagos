#!/usr/bin/env python3
"""Read the desktop's Linux gamepad and stream bounded snapshots to docker exec.

No device pass-through, host ROS installation or additional Python packages.
Kernel ABI: https://docs.kernel.org/input/joydev/joystick-api.html
"""
import array
import fcntl
import json
import os
from pathlib import Path
import select
import struct
import subprocess
import sys
import time
import uuid

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'adapters'))
from joystick_core import command, select_mapping


def devices():
    found=sorted(Path('/sys/class/input').glob('js*'))
    if not found:
        print('未检测到游戏手柄：请确认手柄已开机，接收器已插入，或重新连接 USB。',flush=True)
        return False
    for item in found:
        try:name=(item/'device/name').read_text().strip()
        except OSError:name='unknown'
        path=Path('/dev/input')/item.name
        print(f'{path}: {name}' + ('' if path.exists() else '（当前执行环境不可访问设备节点）'))
    return True


def ioctl_read(fd, number, size):
    buffer=bytearray(size)
    # _IOR('j', number, buffer): Linux joystick.h on the desktop's x86_64 ABI.
    fcntl.ioctl(fd,0x80000000 | (size<<16) | (ord('j')<<8) | number,buffer,True)
    return buffer


class Gamepad:
    def __init__(self,path):
        self.fd=os.open(path,os.O_RDONLY|os.O_NONBLOCK)
        try:
            self.name=bytes(ioctl_read(self.fd,0x13,128)).split(b'\0',1)[0].decode(errors='replace')
            na,nb=ioctl_read(self.fd,0x11,1)[0],ioctl_read(self.fd,0x12,1)[0]
            self.axis_codes=list(ioctl_read(self.fd,0x32,64)[:na])
            codes=array.array('H');codes.frombytes(ioctl_read(self.fd,0x34,1024))
            self.button_codes=list(codes[:nb])
            self.axes=[0.]*na;self.buttons=[False]*nb
            self.pressed_events=set()
            self.mapping=select_mapping(self.axis_codes,self.button_codes)
        except BaseException:
            os.close(self.fd);raise

    def read(self):
        # Drain all queued changes before publishing the latest state.
        for _ in range(128):
            try:buffer=os.read(self.fd,8*64)
            except BlockingIOError:return
            if not buffer:raise OSError('手柄已断开')
            if len(buffer)%8:raise OSError('手柄事件长度错误')
            for _,value,kind,index in struct.iter_unpack('=IhBB',buffer):
                kind &= ~0x80
                if kind==1 and index<len(self.buttons):
                    self.buttons[index]=bool(value)
                    if value:self.pressed_events.add(index)
                elif kind==2 and index<len(self.axes):self.axes[index]=max(-1.,value/32767.)

    def close(self):os.close(self.fd)


def main(device='/dev/input/js0',list_only=False,inspect=False,threshold=.12,container='pelagos-stonefish'):
    if list_only:return 0 if devices() else 1
    if not 0<=threshold<.5:raise ValueError('deadzone 必须在 0..0.5 之间')
    try:pad=Gamepad(device)
    except PermissionError:
        raise RuntimeError(f'当前用户不能读取 {device}；在桌面会话检查该设备的读权限')
    except FileNotFoundError:
        devices()
        raise RuntimeError(f'无法访问 {device}，请在连接手柄的桌面终端执行此命令')
    process=None
    session=uuid.uuid4().hex;sequence=0
    print(f'手柄：{pad.name}\n轴代码：{pad.axis_codes}\n按键代码：{pad.button_codes}\n映射：{pad.mapping}',flush=True)
    print('左摇杆：前后/横移；右摇杆：升降/转向；按住 LB 使能，松开停止；B 停止，Start 清除故障。',flush=True)
    print('速度上限（原来的 5 倍）：前后/横移 0.75 m/s，升降 0.50 m/s，转向 1.25 rad/s。',flush=True)
    try:
        if not inspect:
            process=subprocess.Popen(['docker','exec','-i',container,'/bin/bash',
                                      '/pelagos/Simulation/docker/command.sh','joystick'],
                                     stdin=subprocess.PIPE,text=True,bufsize=1)
        deadline=time.monotonic()
        while process is None or process.poll() is None:
            select.select([pad.fd],[],[],max(0.,deadline-time.monotonic()))
            pad.read()
            now=time.monotonic()
            if now<deadline:continue
            sequence+=1
            controls=command(pad.axes,pad.buttons,pad.mapping,threshold)
            # Preserve a quick B/Start tap even if pressed and released between
            # two 20 Hz snapshots. Deadman continues to represent held state.
            for button in ('stop','reset'):
                controls[button] |= pad.mapping[button+'_button'] in pad.pressed_events
            packet=dict(session=session,sequence=sequence,issued_ns=time.monotonic_ns(),connected=True,
                        name=pad.name,**controls)
            if inspect:
                print(json.dumps(dict(axes=[round(v,3) for v in pad.axes],buttons=pad.buttons,
                                      **{k:v for k,v in packet.items() if k in ('velocity','deadman','stop','reset')})),flush=True)
            else:
                process.stdin.write(json.dumps(packet,allow_nan=False)+'\n');process.stdin.flush()
            pad.pressed_events.clear()
            deadline=now+(.2 if inspect else .05)
        return process.returncode
    except KeyboardInterrupt:return 0
    except BrokenPipeError:
        raise RuntimeError('仿真手柄接收进程已退出，检查终端输出和 joystick.log')
    finally:
        pad.close()
        if process is not None:
            if process.poll() is None:
                try:
                    sequence+=1
                    packet=dict(session=session,sequence=sequence,issued_ns=time.monotonic_ns(),connected=False,
                                velocity=[0.]*4,deadman=False,stop=True,reset=False)
                    process.stdin.write(json.dumps(packet)+'\n');process.stdin.flush()
                except (BrokenPipeError,OSError):pass
            try:process.stdin.close()
            except (BrokenPipeError,OSError):pass
            try:process.wait(timeout=4)
            except subprocess.TimeoutExpired:process.terminate()
