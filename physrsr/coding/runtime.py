"""OS-isolated generated Python, with a narrow shared simulation/hardware API."""
from dataclasses import dataclass,field
import hashlib
import json
import os
from pathlib import Path
import resource
import selectors
import subprocess
import tempfile
import time
import numpy as np

from .agent import system_mounts


class ProgramError(RuntimeError):pass


class RobotActionError(RuntimeError):
    """Known controller failure with measured state; guest may recover boundedly."""
    pass


def _limits():
    resource.setrlimit(resource.RLIMIT_CPU,(5,5))
    resource.setrlimit(resource.RLIMIT_AS,(256*1024**2,256*1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE,(1024**2,1024**2))
    resource.setrlimit(resource.RLIMIT_NOFILE,(64,64))
    resource.setrlimit(resource.RLIMIT_CORE,(0,0))


def execute_source(source,dispatch,timeout=120,*,capture_return=False,max_calls=96):
    """Child cannot see repository/auth/World data or use host networking."""
    with tempfile.TemporaryDirectory(prefix='physrsr-code-') as directory:
        folder=Path(directory);(folder/'program.py').write_text(source)
        command=['bwrap','--unshare-all','--die-with-parent','--new-session','--cap-drop','ALL',
                 *system_mounts(),'--proc','/proc','--dev','/dev','--tmpfs','/tmp',
                 '--ro-bind',str(folder),'/program','--ro-bind',str(Path(__file__).with_name('guest.py').resolve()),'/guest.py',
                 '--chdir','/program','/usr/bin/python3','-I','-u','/guest.py','/program/program.py']
        if capture_return:command.append('--capture-return')
        with (folder/'stderr.log').open('w+') as error_log:
            process=subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=error_log,
                                     preexec_fn=_limits,bufsize=0,env={'PYTHONHASHSEED':'0'})
            selector=selectors.DefaultSelector();selector.register(process.stdout,selectors.EVENT_READ)
            buffer=b'';calls=0;deadline=time.monotonic()+timeout
            try:
                while True:
                    remaining=deadline-time.monotonic()
                    if remaining<=0:raise ProgramError('Program wall-time limit exceeded')
                    if not selector.select(min(remaining,2)):
                        if process.poll() is not None:raise ProgramError(f'Sandbox exited {process.returncode} without completion')
                        continue
                    chunk=os.read(process.stdout.fileno(),65536)
                    if not chunk:
                        process.wait(timeout=3)
                        error_log.flush();error_log.seek(0)
                        detail=error_log.read()[-2000:]
                        if 'bwrap:' in detail:
                            raise RuntimeError('Sandbox infrastructure failed: '+detail)
                        raise ProgramError(f'Sandbox ended before completion (rc={process.returncode}): {detail}')
                    buffer+=chunk
                    if len(buffer)>65536:raise ProgramError('Program protocol output exceeded limit')
                    while b'\n' in buffer:
                        line,buffer=buffer.split(b'\n',1)
                        try:message=json.loads(line)
                        except (ValueError,UnicodeError) as exc:raise ProgramError('Invalid program RPC') from exc
                        if not isinstance(message,dict):raise ProgramError('Invalid program RPC object')
                        if message.get('event')=='done':
                            try:process.wait(timeout=3)
                            except subprocess.TimeoutExpired as exc:raise ProgramError('Program descendants did not exit') from exc
                            if process.returncode:raise ProgramError('Sandbox reported completion but failed')
                            result=dict(calls=calls)
                            if capture_return:result['return_value']=message.get('return_value')
                            return result
                        if message.get('event')=='error':raise ProgramError(str(message.get('error'))[:5000])
                        if set(message)!={'event','method','params'} or message['event']!='call':raise ProgramError('Invalid robot call')
                        calls+=1
                        if calls>max_calls:raise ProgramError('Robot call budget exceeded')
                        # Dispatch exceptions propagate: trusted API failures are
                        # infrastructure errors; explicit contract violations are ProgramError.
                        try:
                            result=dispatch(message['method'],message['params'])
                            reply=dict(ok=True,result=result)
                        except RobotActionError as exc:
                            reply=dict(ok=False,error=str(exc))
                        try:process.stdin.write((json.dumps(reply,allow_nan=False)+'\n').encode())
                        except BrokenPipeError as exc:raise ProgramError('Program closed robot RPC') from exc
            finally:
                selector.close()
                if process.poll() is None:process.kill()
                process.wait()


@dataclass
class CodeProgram:
    id: str
    source: str
    trace: list = field(default_factory=list)
    error_kind: str = 'infrastructure'
    runtime: dict = field(default_factory=dict)

    def specification(self):
        return dict(id=self.id,source_sha256=hashlib.sha256(self.source.encode()).hexdigest())

    def run(self,env,api,physics=None):
        from physrsr.candidate_programs import rod_pose,down_grasp_quat
        from physrsr.m21r.core import sense
        self.trace.clear();self.error_kind='infrastructure'
        self.runtime=dict(rpc_counts={},guest_sha256=hashlib.sha256(Path(__file__).with_name('guest.py').read_bytes()).hexdigest(),python_random_seed=0)
        centre,_=rod_pose(env);centre=np.asarray(centre)
        def dispatch(method,params):
            counts=self.runtime['rpc_counts'];counts[method]=counts.get(method,0)+1
            if not isinstance(params,dict):raise ProgramError('Robot parameters must be a dictionary')
            if method=='observe' and not params:
                pos,axis=rod_pose(env);_,tcp=env.get_eef_pose()
                result=dict(rod_position=np.asarray(pos).tolist(),rod_axis=np.asarray(axis).tolist(),
                            grasp_quat=down_grasp_quat(env).tolist(),tcp_position=np.asarray(tcp).tolist())
            elif method in ('open_gripper','close_gripper') and not params:
                getattr(api,method)();result=None
            elif method=='goto_pose' and set(params)=={'position','quaternion_wxyz','z_approach','speed'}:
                try:
                    p=np.asarray(params['position'],float);q=np.asarray(params['quaternion_wxyz'],float)
                    z=float(params['z_approach']);speed=params['speed']
                    if p.shape!=(3,) or q.shape!=(4,) or not np.isfinite(p).all() or not np.isfinite(q).all():raise ValueError()
                    if abs(np.linalg.norm(q)-1)>.01 or not 0<=z<=.12:raise ValueError()
                    if np.any(abs(p[:2]-centre[:2])>.25) or not -.02<=p[2]-centre[2]<=.20:raise ValueError()
                    if speed is not None and (not isinstance(speed,(int,float)) or not .1<=speed<=1):raise ValueError()
                except (ValueError,TypeError):raise ProgramError('Invalid pose/speed or workspace constraint violation') from None
                api.goto_pose(p,q,z_approach=z,speed=speed);result=None
            else:raise ProgramError(f'Robot method not allowed: {method}')
            self.trace.append(sense(env))
            return result
        self.trace.append(sense(env))
        try:self.runtime.update(execute_source(self.source,dispatch))
        except ProgramError:
            self.error_kind='program';raise
