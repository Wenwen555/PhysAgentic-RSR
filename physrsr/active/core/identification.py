"""Task-independent experimental design and measured robot capability contract."""
from .contracts import IDENTIFICATION_API_VERSION


ROBOT_API = """Define run(robot); import math only. All spatial data use robot_base,
metres, seconds, newtons, newton-metres, quaternions wxyz. No hidden environment,
files, networking or physical parameter truth. The same source runs on hardware.
robot.describe() returns capabilities: workspace, gravity, geometry, force/torque
limits and sensor provenance. Limits are not measured object properties.
robot.describe()['identification'] is the CURRENT instance's measured history,
not parameter truth. It can contain estimates, trials from the latest completed
probe, and trial_history records with source_probe, source_evidence_id, status
and trials. Use .get() for optional fields. Trials contain the executed settings,
actual responses and validity/recovery results. Continue an unfinished search
from this evidence instead of restarting a known uninformative setting after
every prepared episode. Never pool distinct instances or branch on instance IDs.
robot.observe() returns:
  time_s, coordinate_frame;
  object: position[3], quaternion_wxyz[4], axes[3][3] (body x,y,z in base),
    dimensions_m[3], grasp_quaternion_wxyz[4], support_clearance_m;
  tcp: position[3], quaternion_wxyz[4];
  wrench: force_n[3], torque_nm[3], frame='robot_base', about='tcp',
    valid, provenance (raw wrist transducer including tool weight; NOT object-only);
  joints: position_rad[7], actuator_torque_nm[7]; gripper: opening_m.
support_clearance_m is a visual geometry/plane calculation, not a contact sensor.
Wrist measurements need unloaded tare at matching orientation and settled motion.
Table support and acceleration confound force/torque parameter inference.
robot.open_gripper(); robot.close_gripper();
robot.goto_pose(position, quaternion_wxyz, z_approach=0., speed=None);
Motion completion uses bounded measured TCP feedback per capabilities.motion_contract.
The published tracking tolerance is different from sensor resolution. Diagnostic
motion smaller than that tolerance cannot establish a commanded motion direction.
robot.hold(duration_s): captures and holds measured joints with unchanged gripper command,
  0 < duration_s <= 2; actual controller time advances. Returns no observations;
  call observe afterwards. observe itself does not advance simulation time.
robot.describe()['probe_execution'] contains this frozen program's sha256,
remaining_rpc_calls and resume_state (None on its first execution). Read it again
when accounting for RPCs. resume_state belongs only to this instance and this
exact source. Persist search progress (attempted settings, brackets, next setting),
not stale live TCP poses or a tare valid only in the previous episode.
robot.checkpoint(state): save a JSON object of at most 4 KiB at a measured stable
supported, open-gripper boundary. First restore support, open and hold/observe;
then checkpoint as the LAST physical step and return status='paused' or
'budget_exhausted'. All actions including checkpoint count toward 192 RPCs.
The host preserves progress even if later code fails, but acknowledges it as
resumable only after this safe ending. Recovery failures cannot be resumed.
Reserve enough calls to restore support and save progress BEFORE exhausting the
hard limit. A continuation executes the same source, remeasures geometry/tare,
and selects an untried setting from resume_state instead of starting over.
Speed 0.1..1.0. At most 192 total RPCs, including describe, hold and observe.
The guest protocol has a 64 KiB line limit. Return a compact result, targeting
under 8 KiB: status, selected quantities with source IDs, and a bounded list of
scalar diagnostics. Do not copy full observations or identification history
into the return value; the host separately archives every API observation.
Workspace and recovery rules in capabilities and experiment_budget are mandatory.
Do not release until the object is stably supported in the allowed rest region.
An exact initial position is not required unless the task explicitly requires it;
do not force the tool through a supported object merely to reproduce an old TCP.
Gripper force limits cannot be changed. Joint torque headroom is not the gripper's
contact moment capacity: friction and contact geometry govern wrench transmission.
"""


DEFAULT_BUDGET = dict(max_rpc_calls=192, max_duration_s=360., max_distinct_settings=9,
    max_local_excitation_translation_m=.04, max_local_excitation_rotation_rad=.35,
    setting_range="Use observed geometry and workspace; local excitation amplitude does not limit the search range of contact configurations",
    normal_speed=.15, recovery_reserved_calls=20)


def public_goal(task):
    """Semantic goal only; neither benchmark identifier nor task-specific advice."""
    return dict(instruction=task["instruction"], success=task["success"], constraints=task["constraints"])


def packet(policy):
    if policy.api_version != IDENTIFICATION_API_VERSION:
        raise ValueError("Identification programs require the measured capability API")
    return policy.wire()
