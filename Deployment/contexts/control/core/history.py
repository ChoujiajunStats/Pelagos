"""Real 10 Hz observations and interval-mean successful motor/channel TX demands."""
from collections import deque
import numpy as np
from motor_contract import CONTRACT_ID,modal_from_motors


def command_to_modal(command):
    surge,sway,heave,yaw=command
    motor=np.clip([yaw-surge+sway,-yaw-surge-sway,-yaw+surge+sway,yaw+surge-sway,-heave,-heave],-1,1)
    a,b,c,d,e,f=motor
    return np.array([(-a-b+c+d)/4,(a-b+c-d)/4,-(e+f)/2,(a-b-c+d)/4,(e-f)/2],dtype=np.float32)


class RealHistory:
    def __init__(self, steps=26, hz=10):
        self.steps=steps;self.dt=1/hz;self.observations=deque(maxlen=256);self.tx=deque(maxlen=1024)
        self.last_tx_sequence=0
        self.action_contract_verified=False

    def append_tx(self, events):
        for e in events:
            if e['sequence']<=self.last_tx_sequence:continue
            if self.last_tx_sequence and e['sequence']!=self.last_tx_sequence+1:
                self.tx.clear();self.observations.clear()
            self.last_tx_sequence=e['sequence'];self.tx.append(e)

    def append(self, stamp, observation, position):
        if self.observations and stamp<=self.observations[-1][0]:return
        if self.observations and stamp-self.observations[-1][0]>.3:self.observations.clear()
        self.observations.append((stamp,np.asarray(observation),np.asarray(position)))

    def arrays(self):
        self.action_contract_verified=False
        if len(self.observations)<2 or not self.tx:raise ValueError('waiting_for_real_history')
        end=np.floor(self.observations[-1][0]/self.dt)*self.dt
        times=end-np.arange(self.steps-1,-1,-1)*self.dt
        stamps=np.array([x[0] for x in self.observations]);events=list(self.tx)
        if stamps[0]>times[0] or events[0]['stamp_ns']*1e-9>times[0]-self.dt:
            raise ValueError('waiting_for_real_history')
        if events[-1]['stamp_ns']*1e-9<end-.15:raise ValueError('successful_tx_history_stale')
        values=np.array([x[1] for x in self.observations]);positions=np.array([x[2] for x in self.observations])
        obs=np.column_stack([np.interp(times,stamps,values[:,i]) for i in range(16)])
        pos=np.column_stack([np.interp(times,stamps,positions[:,i]) for i in range(3)])
        for row in obs:
            a=row[:3];a/=np.linalg.norm(a);b=row[3:6];b-=a*np.dot(a,b);b/=np.linalg.norm(b)
        actions=[];ts=np.array([e['stamp_ns']*1e-9 for e in events]);verified=True
        for t in times:
            start=t-self.dt;i=max(0,np.searchsorted(ts,start,side='right')-1);total=np.zeros(5)
            while start<t-1e-8:
                e=events[i];stop=min(t,ts[i+1] if i+1<len(ts) else t)
                if e['armed'] and (e['gear']!=1 or e['heading_hold'] or e['depth_hold']):
                    raise ValueError('action_history_contains_unmodelled_hold_or_gear')
                if not e['armed']:modal=np.zeros(5)
                elif e.get('protocol')=='motor' and e.get('actuation_contract')==CONTRACT_ID:
                    modal=np.asarray(modal_from_motors(e['model_motor_normalized']))
                else:
                    verified=False;modal=command_to_modal(e['command'])
                total+=modal*(stop-start)
                start=stop;i+=1
            actions.append(total/self.dt)
        self.action_contract_verified=verified
        return obs.astype(np.float32),np.asarray(actions,dtype=np.float32),pos[-1].astype(np.float32),times[-1]
