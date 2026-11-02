"""Four-timestamp clock offset for cross-host sensor/action history alignment."""
from copy import deepcopy


class ClockAlignment:
    def __init__(self):
        self.offset_ns=None
        self.round_trip_ns=None

    def align(self, packet, received_ns):
        sent=packet['robot_send_ns']
        h1=packet.get('echo_host_send_ns',0);r2=packet.get('robot_receive_ns',0)
        if not h1 or not r2:return None  # Require measured round trip; no invented alignment.
        rtt=(received_ns-h1)-(sent-r2)
        if not 0<=rtt<100_000_000:return None  # Ignore delayed samples; data freshness still expires.
        offset=((h1-r2)+(received_ns-sent))//2
        if self.offset_ns is not None and abs(offset-self.offset_ns)>200_000_000:
            raise ValueError('cross_host_clock_step_restart_required')
        self.offset_ns=offset if self.offset_ns is None else int(.95*self.offset_ns+.05*offset)
        self.round_trip_ns=rtt
        result=deepcopy(packet)
        for key in ('body','imu','reference'):
            if result.get(key):result[key]['stamp_ns']+=self.offset_ns
        for sample in result.get('samples',[]):
            sample['body']['stamp_ns']+=self.offset_ns
            sample['imu']['stamp_ns']+=self.offset_ns
        for key in ('visual_stamp_ns','depth_stamp_ns','now_ns'):
            if result.get(key):result[key]+=self.offset_ns
        result['clock_offset_ns']=self.offset_ns
        result['clock_round_trip_ns']=rtt
        return result
