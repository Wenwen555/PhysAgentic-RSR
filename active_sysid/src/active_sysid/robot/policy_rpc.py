"""Run self-contained CaP in bubblewrap; only sensor/action RPC crosses boundary."""
from pathlib import Path
import json
import os
import selectors
import shutil
import signal
import select
import struct
import subprocess
import sys
import time
import numpy as np
from .policy_worker import MAX_PACKET, packet, decode


class PolicyError(RuntimeError): pass


def run_policy(path,dispatch,log_path,*,timeout_s=120):
    if shutil.which('bwrap') is None:
        raise RuntimeError('bubblewrap is required; refusing in-process fallback')
    args=['bwrap','--unshare-all','--die-with-parent','--new-session','--clearenv']
    # Deliberately no repository, World XML, run history, HOME or network mount.
    for path_to_mount in ['/usr','/lib','/lib64','/etc/ld.so.cache',sys.base_prefix]:
        if Path(path_to_mount).exists():args+=['--ro-bind',path_to_mount,path_to_mount]
    args+=['--ro-bind',str(Path(np.__file__).resolve().parents[1]),'/packages',
           '--proc','/proc','--dev','/dev','--tmpfs','/tmp','--chdir','/tmp',
           '--setenv','HOME','/tmp','--setenv','PATH','/usr/bin:/bin',
           '--setenv','OPENBLAS_NUM_THREADS','1','--setenv','PYTHONHASHSEED','0',
           '--ro-bind',str(Path(__file__).with_name('policy_worker.py')),'/worker.py',
           '--ro-bind',str(Path(path).resolve()),'/policy.py',
           '--',str(Path(sys.executable).resolve()),'-I','-B','/worker.py']
    deadline=time.monotonic()+timeout_s
    with open(log_path,'wb') as log:
        proc=subprocess.Popen(args,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,
                              bufsize=0,start_new_session=True,env={})
        os.set_blocking(proc.stdin.fileno(),False)
        selector=selectors.DefaultSelector();selector.register(proc.stdout,selectors.EVENT_READ)
        def read_exact(size):
            chunks=[];remaining=size
            while remaining:
                timeout=deadline-time.monotonic()
                if timeout<=0 or not selector.select(timeout):raise PolicyError('Policy wall-clock budget exceeded')
                chunk=os.read(proc.stdout.fileno(),remaining)
                if not chunk:raise PolicyError('Policy worker exited without result; see worker.log')
                chunks.append(chunk);remaining-=len(chunk)
            return b''.join(chunks)
        try:
            for _ in range(5000):
                size=struct.unpack('!I',read_exact(4))[0]
                if not 0<size<=MAX_PACKET:raise PolicyError('Invalid RPC request size')
                request=decode(json.loads(read_exact(size)))
                if request.get('done'):
                    if 'error' in request:raise PolicyError(request['error'])
                    proc.wait(timeout=3)
                    return request.get('value')
                result=dispatch(request['method'],request.get('args',[]),request.get('kwargs',{}))
                response=packet({'ok':True,'value':result})
                # FileIO writes can be short for large RGB responses.
                view=memoryview(response)
                while len(view):
                    timeout=deadline-time.monotonic()
                    if timeout<=0 or not select.select([], [proc.stdin], [], timeout)[1]:
                        raise PolicyError('Policy stopped reading observations')
                    try:count=os.write(proc.stdin.fileno(),view[:65536])
                    except BlockingIOError:continue
                    view=view[count:]
            raise PolicyError('Policy RPC call budget exceeded')
        finally:
            selector.close()
            if proc.poll() is None:
                os.killpg(proc.pid,signal.SIGKILL);proc.wait()
            proc.stdin.close();proc.stdout.close()
