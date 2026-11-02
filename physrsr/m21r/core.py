"""Shared robot/hardware control templates; no World physics access."""
from dataclasses import dataclass, field
import numpy as np
from scipy.spatial.transform import Rotation


def sense(env):
    pose = np.asarray(env.get_observation()['cube_poses']['primary'], float)
    _, tcp = env.get_eef_pose()
    return dict(pose=pose.tolist(), tcp=np.asarray(tcp, float).tolist())


@dataclass(frozen=True)
class Program:
    id: str
    offset: float
    speed: float | None = .35
    stages: tuple[float, ...] = (.1,)
    trace: list = field(default_factory=list, compare=False, repr=False)

    def __post_init__(self):
        if not np.isfinite(self.offset) or abs(self.offset) > .12:
            raise ValueError('Grasp outside rod-end clearance envelope')
        if self.speed is not None and not 0 < self.speed <= 1:
            raise ValueError('Invalid speed')
        if not self.stages or any(not 0 < z <= .1 for z in self.stages):
            raise ValueError('Invalid lift stages')
        if list(self.stages) != sorted(set(self.stages)):
            raise ValueError('Stages must increase')

    def run(self, env, api, physics=None):
        from physrsr.candidate_programs import rod_pose, down_grasp_quat
        self.trace.clear()
        centre, axis = rod_pose(env)
        gp = centre + self.offset * axis
        q = down_grasp_quat(env)
        api.open_gripper()
        api.goto_pose(gp, q, z_approach=.1)
        api.goto_pose(gp, q)
        api.close_gripper()
        self.trace.append(sense(env))
        for dz in self.stages:
            api.goto_pose(gp + np.array([0., 0., dz]), q, speed=self.speed)
            self.trace.append(sense(env))

    def specification(self):
        return dict(id=self.id, offset=self.offset, speed=self.speed,
                    stages=self.stages[:])

    def source(self):
        return ("from physrsr.m21r.core import Program\n"
                f"program = Program({self.id!r}, {self.offset!r}, {self.speed!r}, {self.stages!r})\n"
                "def run(env, api):\n    program.run(env, api)\n")


PROBES = (Program('probe_left', -.025, .35, (.02, .04, .06)),
          Program('probe_right', .025, .35, (.02, .04, .06)))


def response(trace):
    """Translation-invariant tracking response with SIGNED object pitch.

    Object and TCP poses are both robot-base quantities. No privileged success,
    contact, force, mass or MJCF fields enter identification.
    """
    if len(trace) != 4:
        raise ValueError('Expected contact plus three tracking snapshots')
    first = trace[0]
    out = []
    for t in trace[1:]:
        pose = np.asarray(t['pose']); q = pose[3:]
        axis = Rotation.from_quat(q[[1, 2, 3, 0]]).apply([1, 0, 0])
        delta = pose[:3] - np.asarray(first['pose'])[:3]
        slip = delta - (np.asarray(t['tcp']) - first['tcp'])
        out.extend([delta[2], slip[2], axis[2]])
    return np.asarray(out)


def identify(observed, library_specs, library_responses):
    """Common inverse-simulation identifier, used equally by Direct and Sim.

    Fixed scales, MAP grid model. Near-optimal loss support is uncertainty, not
    calibrated confidence. This estimates both CoM and effective friction.
    """
    obs = np.asarray(observed)
    pred = np.asarray(library_responses)
    if pred.ndim != 3 or obs.shape != pred.shape[1:] or not np.isfinite(pred).all() or not np.isfinite(obs).all():
        raise ValueError('Invalid identification response arrays')
    scale = np.tile([.005, .005, .05], 3)
    loss = np.mean(((pred - obs) / scale)**2, axis=(1, 2))
    best = int(np.argmin(loss))
    support = np.flatnonzero(loss <= loss[best] + 1.)
    values = np.array([s['com_offset'] for s in library_specs])
    mus = np.array([s['friction'] for s in library_specs])
    # Continuous inversion of a regular response library; no test-specific grid.
    from scipy.interpolate import RegularGridInterpolator
    from scipy.optimize import least_squares
    xs, fs = np.unique(values), np.unique(mus)
    if len(xs)*len(fs) != len(values) or len(set(zip(values,mus))) != len(values) or len(xs)<2 or len(fs)<2:
        raise ValueError('Identifier requires complete rectangular model grid')
    grid = np.empty((len(xs), len(fs)) + obs.shape)
    for i, (x, mu) in enumerate(zip(values, mus)):
        grid[np.searchsorted(xs,x),np.searchsorted(fs,mu)] = pred[i]
    interpolator = RegularGridInterpolator((xs, fs), grid, bounds_error=True)
    def residual(theta):
        return ((interpolator(np.asarray(theta)[None,:])[0]-obs)/scale).ravel()
    fits = [least_squares(residual, [values[i], mus[i]],
                         bounds=([xs[0],fs[0]],[xs[-1],fs[-1]]), max_nfev=100)
            for i in np.argsort(loss)[:3]]
    fit = min(fits, key=lambda f:float(f.fun@f.fun))
    return dict(com_offset=float(fit.x[0]), friction=float(fit.x[1]),
                com_support=[float(min(values[support].min(),fit.x[0])),
                             float(max(values[support].max(),fit.x[0]))],
                support_indices=support.tolist(), best_index=best, loss=loss.tolist(),
                interpolated_loss=float(np.mean(fit.fun**2)))


def generate_candidates(task, physics_estimate, uncertainty, control_constraints):
    if task != 'rod_lift':
        raise ValueError('Unknown task')
    centre = float(physics_estimate['com_offset'])
    limit = min(.12, float(control_constraints.get('max_offset', .105)))
    offsets = control_constraints.get('deltas', (-.02, -.01, 0., .01, .02))
    speeds = control_constraints.get('speeds', (.35, None))
    candidates = []
    for delta in offsets:
        x = round(centre + float(delta), 6)
        if abs(x) > limit:
            continue
        validator = control_constraints.get('workspace_check')
        if validator is not None and not validator(x):
            continue
        for speed in speeds:
            candidates.append(Program(f'g{x:+.6f}_s{speed}', x, speed))
    if not candidates:
        raise ValueError('No candidate within static reach/rod-clearance envelope')
    return candidates


def select_sim(candidates, records, reliability=.95):
    rates = {p.id:np.mean([r['record']['status']=='success' for r in records if r['motion']==p.id]) for p in candidates}
    times = {p.id:np.mean([r['record']['sim_duration_s'] for r in records if r['motion']==p.id]) for p in candidates}
    if not all(np.isfinite(list(rates.values()))):
        raise ValueError('Missing model predictions')
    eligible = [p for p in candidates if rates[p.id] >= reliability]
    # Ties keep candidate order; no access to World outcomes.
    if eligible:
        chosen = min(eligible, key=lambda p:(times[p.id], candidates.index(p)))
    else:
        chosen = max(candidates, key=lambda p:(rates[p.id], -candidates.index(p)))
    return chosen, dict(rates={k:float(v) for k,v in rates.items()}, execution_s={k:float(v) for k,v in times.items()},
                        empirical_pass=bool(eligible))
