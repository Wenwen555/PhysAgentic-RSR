"""Small JSON/array RPC worker mounted in an isolated policy filesystem.

No backend, MuJoCo model, scene path, evaluator or privileged-World object is
passed to the worker. stdin/stdout are protocol-only; policy prints go to stderr.
"""
import base64
import json
import struct
import sys
sys.path.insert(0, '/packages')
import numpy as np

MAX_PACKET=16*1024*1024


def encode(value):
    if isinstance(value,np.ndarray):
        a=np.ascontiguousarray(value)
        if a.dtype.kind not in 'biuf':raise TypeError('Only numeric arrays cross RPC')
        return {'__array__':base64.b64encode(a.tobytes()).decode(),'dtype':a.dtype.str,'shape':list(a.shape)}
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {str(k):encode(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [encode(v) for v in value]
    return value


def decode(value):
    if isinstance(value,dict):
        if '__array__' in value:
            dt=np.dtype(value['dtype'])
            if dt.kind not in 'biuf':raise ValueError('Invalid array dtype')
            return np.frombuffer(base64.b64decode(value['__array__']),dtype=dt).reshape(value['shape']).copy()
        return {k:decode(v) for k,v in value.items()}
    if isinstance(value,list):return [decode(v) for v in value]
    return value


def packet(value):
    payload=json.dumps(encode(value),allow_nan=False,separators=(',',':')).encode()
    if len(payload)>MAX_PACKET:raise ValueError('RPC payload too large')
    return struct.pack('!I',len(payload))+payload


class RobotAPI:
    __slots__=('channel_in','channel_out')
    def __init__(self,inp,out):self.channel_in,self.channel_out=inp,out
    def _call(self,method,*args,**kwargs):
        self.channel_out.write(packet({'method':method,'args':args,'kwargs':kwargs}));self.channel_out.flush()
        header=self.channel_in.read(4)
        if len(header)!=4:raise RuntimeError('RPC closed')
        size=struct.unpack('!I',header)[0]
        if size>MAX_PACKET:raise RuntimeError('Invalid response size')
        response=decode(json.loads(self.channel_in.read(size)))
        if not response['ok']:raise RuntimeError(response['error'])
        return response['value']
    def observe(self,**kwargs):return self._call('observe',**kwargs)
    def phase(self,name):return self._call('phase',name)
    def move_joints(self,positions_rad,*,duration_s=2.0):
        return self._call('move_joints',positions_rad,duration_s=duration_s)
    def move_ee(self,position_m,quaternion_wxyz,*,duration_s=1.0):
        return self._call('move_ee',position_m,quaternion_wxyz,duration_s=duration_s)
    def move_arc(self,center_m,axis_world,angle_rad,*,duration_s=2.0):
        return self._call('move_arc',center_m,axis_world,angle_rad,duration_s=duration_s)
    def set_gripper(self,width_m,*,duration_s=.6):
        return self._call('set_gripper',width_m,duration_s=duration_s)
    def hold(self,duration_s):return self._call('hold',duration_s)


def main():
    inp,out=sys.stdin.buffer,sys.stdout.buffer
    sys.stdout=sys.stderr
    try:
        namespace={'__name__':'robot_policy'}
        with open('/policy.py',encoding='utf-8') as f:source=f.read()
        exec(compile(source,'/policy.py','exec'),namespace)
        result=namespace['run'](RobotAPI(inp,out))
        response={'done':True,'value':result}
    except BaseException as exc:
        response={'done':True,'error':f'{type(exc).__name__}: {exc}'}
    out.write(packet(response));out.flush()

if __name__=='__main__':main()
