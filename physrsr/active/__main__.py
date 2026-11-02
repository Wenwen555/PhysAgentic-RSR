"""Offline config validation is separate from explicit service/run commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from physrsr.paths import OUTPUT_ROOT, output_path

from .api.client import HTTPClient
from .core.contracts import read_config
from .core.policy import Policy


def main(argv=None):
    parser = argparse.ArgumentParser(description=f"PhysRSR active loop; outputs: {OUTPUT_ROOT}; no work starts without a subcommand")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate-config", help="Offline only; no model/robot/API calls")
    check.add_argument("config", type=Path)
    run = commands.add_parser("run", help="One World, one predictive ActiveLoop; all scene objects share this loop")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--out", type=output_path, required=True, metavar="RUN")
    run.add_argument("--cases", type=Path, help=argparse.SUPPRESS)
    run.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    debug = commands.add_parser("debug", help="Explicitly author or execute one program in one World; never certify a task")
    debug.add_argument("action",choices=("design-probe","repair-probe","synthesize-policy","execute-probe","execute-policy"))
    debug.add_argument("--config",type=Path,required=True)
    debug.add_argument("--out",type=output_path,required=True,metavar="RUN")
    debug.add_argument("--context",type=Path,help="Public Agent context for authoring")
    debug.add_argument("--bundle",type=Path,help="Frozen debug bundle to repair/execute, or previous policy to revise")
    debug.add_argument("--belief",type=Path,help="Measured belief for execution, including acknowledged Probe checkpoints")
    debug.add_argument("--resume",action="store_true",help="Continue the identical Probe from a safe checkpoint")
    execute_cmd = commands.add_parser("execute", help="Explicitly execute the frozen identical policy on a target API")
    execute_cmd.add_argument("--bundle", type=Path, required=True)
    execute_cmd.add_argument("--target-url", required=True)
    for name, port in (("serve-simulator", 8121), ("serve-legacy-simulator",8121), ("serve-agent", 8122), ("serve-physic", 8124), ("serve-target", 8123), ("serve-sim-target", 8123), ("serve-world", 8123), ("serve-measured-world", 8123), ("serve-measured-target",8123)):
        command = commands.add_parser(name)
        command.add_argument("--host", default="127.0.0.1")
        command.add_argument("--port", type=int, default=port)
        if name in ("serve-agent", "serve-physic"):
            command.add_argument("--out", type=output_path, required=True, metavar="RUN")
        if name == "serve-target":
            command.add_argument("--robot-url", required=True, help="Shared robot/episode API, backed by measured hardware or simulation")
        if name in ("serve-sim-target", "serve-world", "serve-measured-world"):
            command.add_argument("--private-config", type=Path, required=True)
        if name == "serve-measured-world":
            command.add_argument("--task",choices=("rod_lift","three_rod_lift_return"),default="rod_lift",
                help="Physical World task; three rods coexist in one scene for three_rod_lift_return")
        if name == "serve-simulator":
            command.add_argument("--adapter",action="append",default=[],metavar="MODULE:FACTORY",
                help="Register a built-in task name or trusted MODULE:FACTORY; repeat for additional tasks")
        if name == "serve-measured-target":
            command.add_argument("--factory",required=True,help="Trusted MODULE:FACTORY returning a measured robot API adapter")
    args = parser.parse_args(argv)
    if args.command == "run" and args.cases is not None:
        parser.error("run uses one World and a predictive Simulator. Put all objects in the same target scene; --cases is no longer supported")
    if args.command == "run" and args.resume:
        parser.error("run does not support process-level --resume. Same-source Probe continuation inside ActiveLoop uses resume_probe")
    if args.command == "debug":
        from .tools.stages import author_program, execute_program
        authoring=args.action in ("design-probe","repair-probe","synthesize-policy")
        if authoring and args.context is None:parser.error("Debug authoring requires --context with public evidence")
        if not authoring and args.context is not None:parser.error("Execution uses the frozen bundle, not --context")
        if args.action in ("repair-probe","execute-probe","execute-policy") and args.bundle is None:
            parser.error("Repair/execution requires --bundle")
        if args.action=="design-probe" and args.bundle is not None:parser.error("Use repair-probe to revise an existing Probe")
        if authoring and args.belief is not None:parser.error("--belief is only for execution")
        if args.resume and args.action!="execute-probe":parser.error("--resume is only for execute-probe")
        payload=json.loads(args.config.read_text())
        task,_,_=read_config(payload)
        kind="probe" if args.action.endswith("probe") else "policy"
        endpoint_name=("physic" if kind=="probe" else "agent") if authoring else "target"
        if endpoint_name not in payload.get("endpoints",{}):parser.error(f"Debug action requires {endpoint_name} endpoint")
        endpoint=HTTPClient(payload["endpoints"][endpoint_name])
        if authoring:
            result=author_program(args.out,kind=kind,task=task,context=json.loads(args.context.read_text()),
                agent=endpoint,previous=args.bundle)
        else:
            result=execute_program(args.out,kind=kind,bundle=args.bundle,target=endpoint,
                belief=json.loads(args.belief.read_text()) if args.belief else None,resume=args.resume)
        print(json.dumps(result,ensure_ascii=False,indent=2))
        return 0 if authoring or (result["status"]=="completed" and
            result.get("program_status") not in ("recovery_failed","program_error","execution_error") and
            (result.get("task_result") is None or result["task_result"]["success"])) else 2
    if args.command in ("validate-config", "run"):
        payload = json.loads(args.config.read_text())
        task, config, models = read_config(payload)
        names = ("agent","physic","simulator","target")
        if any(name not in payload.get("endpoints", {}) for name in names):
            parser.error("Unified loop requires agent, physic, simulator and v2 target endpoints")
        endpoints = {name: HTTPClient(payload["endpoints"][name]) for name in names}
        if args.command == "validate-config":
            print(json.dumps(dict(valid=True, task=task.id, models=len(models), external_calls=0), indent=2))
            return 0
        from .workflows.loop import ActiveLoop
        result = ActiveLoop(task, config, models, output=args.out, **endpoints).run()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "task_sufficient" else 2
    if args.command == "execute":
        manifest = json.loads((args.bundle / "manifest.json").read_text())
        world_accepted=(manifest.get("status")=="frozen" and manifest.get("one_shared_policy") is True
            and manifest.get("recorded_world_cases",0)>0
            and manifest.get("recorded_world_cases")==manifest.get("recorded_world_successes"))
        if manifest.get("status") != "task_sufficient" and not world_accepted:
            raise ValueError("Only a frozen policy with complete recorded acceptance can be executed")
        policy = Policy.from_wire(dict(source=(args.bundle / "policy.py").read_text(),
            sha256=manifest["sha256"], api_version=manifest["api_version"]))
        result = HTTPClient(args.target_url).call("target.execute_policy", dict(policy=policy.wire()))
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["status"] == "completed" else 2
    from .api.server import serve
    if args.command == "serve-simulator":
        from .runtime.predictive import PredictiveSimulator
        from .tasks.registry import create_adapters
        service = PredictiveSimulator(create_adapters(args.adapter or None))
    elif args.command == "serve-legacy-simulator":
        from .runtime.simulator import RodSimulatorService
        service = RodSimulatorService()
    elif args.command == "serve-agent":
        from .agents.coding import CodingService
        service = CodingService(args.out)
    elif args.command == "serve-physic":
        from .agents.physic import PhysicService
        service = PhysicService(args.out)
    elif args.command == "serve-measured-world":
        if args.task == "three_rod_lift_return":
            from .tasks.rod_scene_world import ThreeRodWorld
            service = ThreeRodWorld(json.loads(args.private_config.read_text()))
        else:
            from .tasks.rod_lift_world import InstrumentedWorld
            service = InstrumentedWorld(json.loads(args.private_config.read_text()))
    elif args.command == "serve-measured-target":
        from .core.factories import load_factory
        from .runtime.instrumented import InstrumentedTarget
        service = InstrumentedTarget(load_factory(args.factory))
    elif args.command == "serve-world":
        from .runtime.world import WorldService
        service = WorldService(json.loads(args.private_config.read_text()))
    elif args.command == "serve-sim-target":
        from .runtime.backends import SimulatedRobotService
        from .runtime.robot import TargetService
        service = TargetService(SimulatedRobotService(json.loads(args.private_config.read_text())))
    else:
        from .runtime.robot import TargetService
        service = TargetService(HTTPClient(args.robot_url))
    serve(service, args.host, args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
