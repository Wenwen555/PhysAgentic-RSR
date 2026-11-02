"""Bounded deterministic nonphysical checks, not a proof of arbitrary Python.

AST syntax/API checks plus synthetic kinematic contract replay through the same
isolated guest. Synthetic tilt and sag scenarios expose known math/state bugs.
No MuJoCo, learned policy, World outcomes, or physical parameters are used.
"""
import ast
import numpy as np
from scipy.spatial.transform import Rotation
from physrsr.coding.runtime import execute_source,ProgramError

REQUIREMENTS='''Perform ALL these static checks, whether execution evidence is supplied or not:
Python syntax and allowed robot API; consistent robot-base positions/object-axis CoM transform;
quaternion normalization, rotation axis/sign must reduce tilt under rigid tracking;
loop termination and RPC limit; separate successful grasp/hold from failed recovery;
never reopen an elevated retained object just because a hold loop expired; lower before recovery;
task constraints, exception handling, valid workspace/speed/quaternion ranges.
First state observed facts separately from possible causes. Identify provable code/math errors,
the exact code locations to change, and the expected behavioral effect. Prefer minimal necessary
changes and preserve validated functions. Return complete executable code and rationale describing
observations, hypotheses, edit locations and expected effects. No extra tools or hidden data.'''


class ContractRobot:
    def __init__(self,tilt,sag):
        self.rod=np.array([.5,0.,0.]);self.tcp=np.array([.5,0.,.08])
        self.axis=np.array([np.sqrt(1-tilt*tilt),0.,tilt]);self.q=np.array([0.,0.,1.,0.])
        self.closed=False;self.sag=sag;self.findings=[];self.calls=0
    def dispatch(self,method,p):
        self.calls+=1
        if method=='observe':return dict(rod_position=self.rod.tolist(),rod_axis=self.axis.tolist(),
            tcp_position=self.tcp.tolist(),grasp_quat=[0.,0.,1.,0.])
        if method=='open_gripper':
            if self.closed and self.rod[2]>.04:self.findings.append('release_while_lifted')
            self.closed=False;return
        if method=='close_gripper':self.closed=True;return
        if method!='goto_pose':raise ProgramError('Unsupported robot API')
        target=np.array(p['position']);q=np.array(p['quaternion_wxyz'])
        speed=p.get('speed')
        if target.shape!=(3,) or q.shape!=(4,) or not np.isfinite(target).all() or not np.isfinite(q).all() or abs(np.linalg.norm(q)-1)>.01:
            self.findings.append('invalid_pose');return
        if speed is not None and not .1<=speed<=1:self.findings.append('invalid_speed')
        if np.any(np.abs(target[:2]-[.5,0.])>.25) or not -.02<=target[2]<=.20:self.findings.append('workspace')
        if self.closed:
            rot=Rotation.from_quat(q[[1,2,3,0]])*Rotation.from_quat(self.q[[1,2,3,0]]).inv()
            new_axis=rot.apply(self.axis)
            if abs(new_axis[2])>abs(self.axis[2])+.003:self.findings.append('leveling_rotation_increases_tilt')
            self.axis=new_axis
            delta=target-self.tcp
            self.rod+=delta
            if np.linalg.norm(delta)<.006 and self.rod[2]>.04:self.rod[2]-=self.sag
        self.tcp=target;self.q=q


def check(source):
    try:
        tree=ast.parse(source);compile(tree,'<cap>','exec')
    except SyntaxError as e:return dict(passed=False,findings=[dict(type='syntax',line=e.lineno)],coverage='bounded')
    findings=[]
    if not any(isinstance(n,ast.FunctionDef) and n.name=='run' for n in tree.body):findings.append(dict(type='missing_run'))
    allowed={'observe','goto_pose','open_gripper','close_gripper'}
    for node in ast.walk(tree):
        if isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name) and node.value.id=='robot' and node.attr not in allowed:
            findings.append(dict(type='api',line=node.lineno,method=node.attr))
        if isinstance(node,(ast.Import,ast.ImportFrom)):
            names=[n.name for n in node.names] if isinstance(node,ast.Import) else [node.module]
            if any(n!='math' for n in names):findings.append(dict(type='unsupported_import',line=node.lineno))
    scenarios=[]
    for tilt,sag in ((.1,0.),(-.1,0.),(0.,.0008)):
        robot=ContractRobot(tilt,sag)
        error=None
        try:execute_source(source,robot.dispatch,timeout=3)
        except ProgramError as exc:
            error=str(exc)
            # A counterexample that exceeds the SAME RPC contract is actionable.
            robot.findings.append('contract_execution_error')
        scenarios.append(dict(tilt=tilt,sag=sag,calls=robot.calls,error=error,findings=sorted(set(robot.findings))))
        findings.extend(dict(type=f,scenario=len(scenarios)-1) for f in sorted(set(robot.findings)))
    return dict(passed=not findings,findings=findings,scenarios=scenarios,
                coverage='Syntax/API and three deterministic kinematic contract scenarios, not physical simulation or general verification')
