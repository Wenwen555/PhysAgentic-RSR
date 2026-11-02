"""Standalone sandbox guest: standard library only, robot RPC over stdio."""
import contextlib
import json
import random
import sys
import traceback

random.seed(0)
wire=sys.stdout


def send(message):
    wire.write(json.dumps(message,allow_nan=False)+'\n');wire.flush()


class Robot:
    def call(self,method,params):
        send(dict(event='call',method=method,params=params))
        reply=json.loads(sys.stdin.readline())
        if not reply['ok']:raise RuntimeError(reply['error'])
        return reply.get('result')
    def observe(self):return self.call('observe',{})
    def describe(self):return self.call('describe',{})
    def hold(self,duration_s):return self.call('hold',dict(duration_s=duration_s))
    def checkpoint(self,state):return self.call('checkpoint',dict(state=state))
    def open_gripper(self):return self.call('open_gripper',{})
    def close_gripper(self):return self.call('close_gripper',{})
    def goto_pose(self,position,quaternion_wxyz,z_approach=0.,speed=None):
        return self.call('goto_pose',dict(position=position,quaternion_wxyz=quaternion_wxyz,z_approach=z_approach,speed=speed))


try:
    source=open(sys.argv[1]).read()
    namespace={'__name__':'generated_cap'}
    with contextlib.redirect_stdout(sys.stderr):
        exec(compile(source,'generated_cap.py','exec'),namespace)
        returned=namespace['run'](Robot())
    completion=dict(event='done')
    if len(sys.argv)>2 and sys.argv[2]=='--capture-return':
        completion['return_value']=returned
    send(completion)
except BaseException:
    send(dict(event='error',error=traceback.format_exc()[-5000:]))
    sys.exit(2)
