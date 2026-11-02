"""Same control/observation contract and task across all four arms."""
TASK = '''Write executable Python defining run(robot). You control a Panda holding a 0.30 x 0.03 x 0.03 m rod.
Task: grasp, lift and hold the rod, remaining grasped and approximately horizontal
(abs rod long-axis vertical component <=0.25). Nominal transport target is a 0.10 m rise.
The unchanged legacy task grade requires centre height > tabletop +0.04 m, grasped and level
for ten additional closed-grip hold steps. Actual rise and nominal-target error are separately recorded.
Known control command: each finger force cap is 2 N. Hidden object CoM/friction are not available
unless a later message explicitly provides them. Minimize execution time subject to reliable success.
Allowed robot API (same in simulation and hardware):
  robot.observe() -> dict with rod_position [x,y,z], rod_axis unit vector along object +x,
      grasp_quat [w,x,y,z] (downward, yaw aligned across rod short axis), tcp_position [x,y,z].
      All coordinates use robot base frame. Observations are available throughout execution.
  robot.open_gripper(); robot.close_gripper()
  robot.goto_pose(position, quaternion_wxyz, z_approach=0.0, speed=None)
      position/quaternion are Python lists. speed None means full controller speed;
      speed in [0.1,1] rate-limits joint commands (reference 2.5 rad/s).
No env, physics, MJCF, filesystem, network, imports from project, or hidden state access.
You may use Python standard library math, helper functions, loops and observation-driven conditions.
Robot method calls are limited to 96. Target xy must stay within .25 m of initial rod centre,
height within initial centre z minus .02 to plus .20 m; pre-approach <=.12 m.
Do not implement a no-op or raise instead of attempting the task when it may be infeasible.
Do not change grip force or success conditions. Return JSON {"code":...,"rationale":...}.
Generate actual Python manipulation logic; there is no fixed candidate/template pool.'''


def initial_messages():
    return [dict(role='user',content=TASK)]


def information_message(estimate=None):
    if estimate is None:
        return 'No physical identification information is available. Review the initial program using task geometry, general physics and static code inspection only.'
    import json
    from capx.envs.simulators.robosuite_rod_lift import rod_mass_properties
    mass,com,inertia=rod_mass_properties((.15,.015,.015),1000.,estimate['com_offset'])
    informed=dict(estimate)
    informed['shared_model_prior']=dict(
        geometry_half_extents_m=[.15,.015,.015],base_density_kg_m3=1000.,
        mass_distribution='Uniform box plus hidden end counterweight; mass/inertia covary with CoM in this benchmark family.',
        predicted_mass_kg=float(mass),predicted_principal_inertia_kg_m2_approx=sorted(inertia.tolist()),
        gravity_m_s2=9.81,grip_command_per_finger_N=2.,
        contact_model='MuJoCo soft contact; rod surface dominates sliding friction. Object parameters are estimates except in Oracle arm.')
    return ('Physical identification information (object +x CoM offset in metres, friction, '
            'and uncalibrated CoM support), plus SAME known model assumptions available to the Simulator: '+json.dumps(informed)+
            '\nUse this information to improve the initial manipulation code. Do not assume support is a calibrated confidence interval.')


def static_review(code):
    try:compile(code,'<generated>','exec');result='Python syntax valid.'
    except SyntaxError as exc:result=f'SyntaxError line {exc.lineno}: {exc.msg}'
    return 'Static review only (no Simulator or World task outcome): '+result+' Reassess and return the final program, or keep it if appropriate.'


def simulation_review(feedback):
    import json
    return ('Internal Simulator execution feedback, NOT World ground truth. Model may be wrong. '
            'Diagnose failures and revise manipulation logic, grasp point, speed, stages, or observation conditions as appropriate. '
            'If already reliable, avoid unnecessary changes.\n'+json.dumps(feedback))
