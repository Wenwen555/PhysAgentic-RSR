"""Task selection stays in composition; generic services import no Rod backend."""
from pathlib import Path
import subprocess
import sys

from physrsr.active.tasks.registry import create_adapters
from physrsr.active.tasks.rod_lift import RodMeasuredAdapter


def test_default_and_explicit_task_registration_select_same_adapter():
    for specification in (None,["rod_lift"],["physrsr.active.tasks.rod_lift:RodMeasuredAdapter"]):
        adapters=create_adapters(specification)
        assert len(adapters)==1 and isinstance(adapters[0],RodMeasuredAdapter)


def test_generic_prediction_and_measured_api_do_not_load_rod_task_modules(tmp_path):
    # Fresh interpreter avoids task modules already imported by other tests.
    script="""import sys
import physrsr.active.runtime.predictive
import physrsr.active.runtime.instrumented
assert 'physrsr.active.tasks.rod_lift' not in sys.modules
assert 'physrsr.active.tasks.rod_lift_world' not in sys.modules
assert 'physrsr.active.runtime.simulator' not in sys.modules
assert 'physrsr.active.runtime.backends' not in sys.modules
"""
    result=subprocess.run([sys.executable,"-c",script],cwd=tmp_path,capture_output=True,text=True,timeout=20)
    assert result.returncode==0,result.stderr
