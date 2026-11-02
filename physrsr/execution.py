"""OS-isolated generated Python, with a narrow shared simulation/hardware API."""
import json
import os
from pathlib import Path
import resource
import selectors
import subprocess
import tempfile
import time

from .coding.agent import system_mounts


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
                 '--ro-bind',str(folder),'/program','--ro-bind',str((Path(__file__).with_name('coding')/'guest.py').resolve()),'/guest.py',
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
