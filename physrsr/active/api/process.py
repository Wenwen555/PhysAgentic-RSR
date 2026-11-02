"""HTTP service lifecycle and audited calls; no policy or experiment logic."""
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time

from .client import HTTPClient


class LoggedAPI:
    def __init__(self, client, path):
        self.client, self.path = client, Path(path)

    def call(self, method, params):
        with self.path.open("a") as stream:
            stream.write(json.dumps(dict(event="request", method=method, params=params), ensure_ascii=False, allow_nan=False)+"\n")
        result = self.client.call(method, params)
        with self.path.open("a") as stream:
            stream.write(json.dumps(dict(event="response", method=method, result=result), ensure_ascii=False, allow_nan=False)+"\n")
        return result


@contextmanager
def service(arguments, port, log_path):
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
    environment = dict(os.environ, MUJOCO_GL="egl", OPENBLAS_NUM_THREADS="1", OMP_NUM_THREADS="1")
    with Path(log_path).open("w") as log:
        process = subprocess.Popen([sys.executable, "-u", "-m", "physrsr.active",
            *arguments, "--host", "127.0.0.1", "--port", str(port)], stdout=log,
            stderr=subprocess.STDOUT, env=environment)
        try:
            deadline = time.monotonic()+20
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"Service exited {process.returncode}: {log_path}")
                try:
                    with socket.create_connection(("127.0.0.1", port), timeout=.2):
                        break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError(f"Service startup timed out: {log_path}")
                    time.sleep(.1)
            yield HTTPClient(f"http://127.0.0.1:{port}")
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
