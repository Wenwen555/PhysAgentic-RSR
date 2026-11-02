"""Historical CodeProgram adapter; current execution lives in physrsr.execution."""
from dataclasses import dataclass, field
import hashlib
from pathlib import Path
import numpy as np
from physrsr.execution import ProgramError, RobotActionError, execute_source

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
