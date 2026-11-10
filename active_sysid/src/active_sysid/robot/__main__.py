"""Explicitly separated Panda fixture and original double-door World runners."""
import argparse
import importlib.util
import json
from pathlib import Path
from .fixture_backend import ROOT
from .fixture_evaluator import evaluate_robot_policy
from .fixture_policy import reach_and_grasp, run

WORLD_COMMANDS = ('world-inspect', 'world-smoke', 'world-baseline',
                  'world-evaluate', 'world-ablate-no-rotate')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('inspect', 'smoke', 'baseline', 'evaluate', *WORLD_COMMANDS))
    parser.add_argument('--scene', choices=('oracle', 'reconstructed'), default='oracle')
    parser.add_argument('--out', type=Path)
    parser.add_argument('--policy', type=Path)
    parser.add_argument('--mechanism-mjcf', type=Path)
    parser.add_argument('--observations', choices=('robot', 'privileged'), default='robot')
    parser.add_argument('--video', action='store_true')
    parser.add_argument('--max-steps', type=int, default=30000)
    parser.add_argument('--binding', type=Path, help='World robot/door binding YAML')
    parser.add_argument('--bundle', type=Path, help='Frozen Sim policy bundle for World evaluation')
    parser.add_argument('--target', choices=('lever', 'round'), default='lever',
                        help='Target door (fixed world policy supports lever)')
    args = parser.parse_args()
    if args.command == 'world-evaluate':
        if args.policy is not None or args.observations == 'privileged' or args.binding is not None:
            parser.error('Direct/privileged World policy development is disabled. Use stages in Sim, then --bundle.')
        if args.bundle is None:
            parser.error('world-evaluate requires a frozen Sim --bundle; use robot.sim2sim freeze first')
        from .sim2sim import evaluate_frozen
        result = evaluate_frozen(args.bundle, args.out or Path('runs/world_final'), video=args.video)
        print(json.dumps(result, indent=2))
        return
    if args.command in WORLD_COMMANDS:
        from .reference_world_backend import WorldRobotBackend
        from .reference_oracle_policy import reach_and_grasp as world_grasp, run as world_run, run_without_rotation
        from .reference_world_evaluator import evaluate_world_policy
        import hashlib
        args.out = args.out or Path('runs/m3r_real_world') / args.command
        if args.command == 'world-inspect':
            backend = WorldRobotBackend(mechanism=args.target,
                                        **({'binding': args.binding} if args.binding else {}))
            model = backend.model
            report = {
                'scene_role': 'reference_world', 'scene': str(backend.scene),
                'sha256': hashlib.sha256(backend.scene.read_bytes()).hexdigest(),
                'model': {'nbody': model.nbody, 'njnt': model.njnt, 'ngeom': model.ngeom,
                          'nu': model.nu, 'neq': model.neq, 'ncam': model.ncam},
                'robot_joints': backend.cfg['arm_joints'],
                'robot_actuators': backend.cfg['arm_actuators'],
                'finger_joints': backend.cfg['finger_joints'],
                'tcp_site': backend.cfg['tcp_site'],
                'selected_door': args.target,
                'door_joint': backend.mechanism['door_joint'],
                'handle_joint': backend.mechanism['handle_joint'],
                'handle_collision_geom': backend.mechanism['handle_geom'],
                'handle_bolt_constraint': backend.mechanism['constraint'],
                'constraint_active': bool(model.eq_active0[backend.equality_id]),
                'initial_handle_center_m': backend.data.geom_xpos[backend.handle_geom].tolist(),
            }
            args.out.mkdir(parents=True, exist_ok=True)
            (args.out / 'inspection.json').write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n')
            print(json.dumps(report, indent=2, ensure_ascii=False))
            return
        if args.command in ('world-smoke', 'world-baseline', 'world-ablate-no-rotate'):
            if args.target != 'lever':
                parser.error('the frozen fixed baseline currently targets only the lever door')
            policy = (world_grasp if args.command == 'world-smoke' else
                      run_without_rotation if args.command == 'world-ablate-no-rotate' else world_run)
            acceptance, observations = ('r0' if args.command == 'world-smoke' else 'r1'), 'privileged'
        else:
            if args.policy is None: parser.error('world-evaluate requires --policy')
            spec = importlib.util.spec_from_file_location('evaluated_world_policy', args.policy.resolve())
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            policy, acceptance, observations = module.run, 'task', args.observations
        result = evaluate_world_policy(policy, out_dir=args.out, binding=args.binding,
                                       mechanism=args.target, observations=observations,
                                       acceptance=acceptance, video=args.video,
                                       max_steps=args.max_steps)
        if args.command == 'world-ablate-no-rotate':
            result['negative_control_pass'] = (result['execution_error'] is None and
                result['final_door_angle_rad'] < 0.05)
            (args.out / 'metrics.json').write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')
        print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
        raise SystemExit(0 if result.get('negative_control_pass', result['success']) else 1)
    args.out = args.out or Path('runs/m3r') / args.command
    scene = ROOT / 'assets/fixtures/panda_single_door' / f'{args.scene}_robot_door.xml'
    if args.command == 'inspect':
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(scene))
        print(json.dumps({'scene':str(scene),'scene_role':'execution_fixture','nq':model.nq,'nv':model.nv,'nu':model.nu},indent=2))
        return
    if args.command == 'evaluate':
        if args.policy is None: parser.error('evaluate requires --policy')
        spec = importlib.util.spec_from_file_location('evaluated_robot_policy', args.policy.resolve())
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        policy, acceptance, observations = module.run, 'task', args.observations
    else:
        policy = reach_and_grasp if args.command == 'smoke' else run
        acceptance, observations = ('r0' if args.command == 'smoke' else 'r1'), 'privileged'
    result = evaluate_robot_policy(policy, out_dir=args.out, scene=scene,
        mechanism_mjcf=args.mechanism_mjcf, observations=observations,
        acceptance=acceptance, video=args.video, max_steps=args.max_steps)
    print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
    raise SystemExit(0 if result['success'] else 1)

if __name__ == '__main__': main()
