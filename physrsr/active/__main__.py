"""Offline config validation is separate from explicit service/run commands."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from physrsr.paths import OUTPUT_ROOT, output_path

from .api.client import HTTPClient
from .core.contracts import read_config, IDENTIFICATION_API_VERSION
from .core.policy import Policy


def main(argv=None):
    parser = argparse.ArgumentParser(description=f"PhysRSR active loop; outputs: {OUTPUT_ROOT}; no work starts without a subcommand")
    commands = parser.add_subparsers(dest="command", required=True)
    check = commands.add_parser("validate-config", help="Offline only; no model/robot/API calls")
    check.add_argument("config", type=Path)
    run = commands.add_parser("run", help="Probe-first identification and revision; --cases evaluates one shared policy in private Worlds")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--out", type=output_path, required=True, metavar="RUN")
    run.add_argument("--cases",type=Path,help="Private independent World cases; probe first, one shared task policy")
    run.add_argument("--resume",action="store_true",help="Resume the saved instance-scoped batch controller; requires --cases")
    initial = commands.add_parser("author-initial", help="Legacy task-only CaP authoring; not the default loop entry")
    initial.add_argument("--config", type=Path, required=True)
    initial.add_argument("--out", type=output_path, required=True, metavar="RUN")
    probe = commands.add_parser("author-probe", help="Organize archived initial-CaP evidence and author an active probe without executing it")
    probe.add_argument("--experiment", type=output_path, required=True, metavar="RUN")
    probe.add_argument("--round", type=int, default=1)
    probe.add_argument("--port", type=int, default=18124)
    identification = commands.add_parser("identification", help="Measured, task-independent physical identification workflow")
    identification.add_argument("stage", choices=("design", "redesign", "probe", "synthesize", "revise-task", "evaluate"))
    identification.add_argument("--experiment", type=output_path, required=True, metavar="RUN")
    identification.add_argument("--public-context", type=Path)
    identification.add_argument("--private-cases", type=Path)
    identification.add_argument("--initial-source", type=Path)
    identification.add_argument("--round", type=int, default=1)
    identification.add_argument("--review", type=Path)
    identification.add_argument("--previous-round", type=int)
    identification.add_argument("--instances",nargs="+",help="Execute only these independent instance IDs")
    identification.add_argument("--resume",action="store_true",help="Resume the same frozen Probe from safe checkpoints, or retain completed task evaluations")
    identification.add_argument("--repair",action="store_true",help="For redesign: code repair only; never executes the repaired Probe")
    for name, help_text, default_port in (
        ("probe-world", "Execute the frozen probe in the two archived failing Worlds", 18123),
        ("revise-cap", "Make one shared CaP revision from archived World probe evidence", 18122),
        ("compare-caps", "Re-execute initial and revised CaPs in paired failing Worlds", 18123),
    ):
        command = commands.add_parser(name, help=help_text)
        command.add_argument("--experiment", type=output_path, required=True, metavar="RUN")
        command.add_argument("--round", type=int, default=1)
        command.add_argument("--port", type=int, default=default_port)
    smoke = commands.add_parser("smoke", help="Execute one policy on three CoM configurations at fixed mass")
    smoke.add_argument("--out", type=output_path, required=True, metavar="RUN")
    smoke.add_argument("--initial-cap", type=Path)
    smoke.add_argument("--mass-kg", type=float, default=.27)
    smoke.add_argument("--seed", type=int, default=19001)
    smoke.add_argument("--friction", type=float, default=1.5)
    smoke.add_argument("--base-port", type=int, default=18121)
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
        if name == "serve-simulator":
            command.add_argument("--adapter",action="append",default=[],metavar="MODULE:FACTORY",
                help="Register a built-in task name or trusted MODULE:FACTORY; repeat for additional tasks")
        if name == "serve-measured-target":
            command.add_argument("--factory",required=True,help="Trusted MODULE:FACTORY returning a measured robot API adapter")
    args = parser.parse_args(argv)
    if args.command == "identification":
        from .workflows.identification import (author_identification, redesign_identification, execute_identification,
            synthesize_identified_policy, revise_identified_task, evaluate_identified_policy)
        if args.repair and args.stage!="redesign":parser.error("--repair is only for identification redesign")
        if args.resume and args.stage not in ("probe","evaluate"):parser.error("--resume is only for probe/evaluate")
        if args.instances and args.stage not in ("probe","evaluate"):parser.error("--instances is only for probe/evaluate")
        if args.stage == "design":
            if not all((args.public_context, args.private_cases)):
                parser.error("identification design requires --public-context and --private-cases")
            result = author_identification(args.experiment, public_context=json.loads(args.public_context.read_text()),
                private_cases=json.loads(args.private_cases.read_text()), initial_source=args.initial_source)
        elif args.stage == "redesign":
            if args.review is None:
                parser.error("identification redesign requires --review with public design findings")
            review=json.loads(args.review.read_text())
            if args.repair:review=dict(review,mode="repair_probe")
            result = redesign_identification(args.experiment,review,round_index=args.round,previous_round=args.previous_round)
        elif args.stage == "revise-task":
            if args.round < 2:
                parser.error("revise-task requires --round >= 2")
            result = revise_identified_task(args.experiment,
                previous_round=args.previous_round or args.round-1, round_index=args.round)
        else:
            kwargs=dict(round_index=args.round)
            if args.stage in ("probe","evaluate"):kwargs.update(instance_ids=args.instances,resume=args.resume)
            result = {"probe": execute_identification, "synthesize": synthesize_identified_policy,
                "evaluate": evaluate_identified_policy}[args.stage](args.experiment,**kwargs)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command in ("probe-world", "revise-cap", "compare-caps"):
        from .workflows.probe_revision import execute_world_probes, revise_cap_once, evaluate_revision_pair
        workflow = {"probe-world": execute_world_probes, "revise-cap": revise_cap_once,
            "compare-caps": evaluate_revision_pair}[args.command]
        result = workflow(args.experiment, round_index=args.round, port=args.port)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "smoke":
        from .workflows.smoke import run
        run(args.out, initial_cap=args.initial_cap, mass_kg=args.mass_kg, seed=args.seed,
            friction=args.friction, base_port=args.base_port)
        return 0
    if args.command == "author-probe":
        from .workflows.probe_authoring import author_active_probe
        result = author_active_probe(args.experiment, round_index=args.round, port=args.port)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command in ("validate-config", "run", "author-initial"):
        payload = json.loads(args.config.read_text())
        task, config, models = read_config(payload)
        if args.command=="run" and args.cases is not None:
            from .workflows.batch import run_probe_first_batch,continue_probe_first,ExplorationBudget
            cases=json.loads(args.cases.read_text())
            budget=ExplorationBudget(max_probe_cost_s=config.max_probe_cost,max_probe_executions=config.max_probes,
                probe_cost_weight=config.probe_cost_weight,min_net_value=config.min_net_value)
            if args.resume:
                root=args.out/"run"
                if json.loads((root/"evaluator_only_cases.json").read_text())!=cases:
                    raise ValueError("Resume cannot change the independent World instances")
                from .core.identification import public_goal
                if json.loads((root/"active_probe_1/context.json").read_text())["goal"]!=public_goal(task.public()):
                    raise ValueError("Resume cannot change the task goal")
                result=continue_probe_first(root,max_iterations=config.max_rounds,budget=budget)
            else:result=run_probe_first_batch(args.out,task,cases,max_iterations=config.max_rounds,budget=budget)
            print(json.dumps(result,ensure_ascii=False,indent=2))
            return 0 if result["status"]=="accepted" else 2
        if args.command=="run" and args.resume:parser.error("run --resume requires --cases")
        names = ("agent","physic","simulator","target") if args.command in ("run","validate-config") else ("agent","target")
        if any(name not in payload["endpoints"] for name in names):
            parser.error("Unified loop requires agent, physic, simulator and v2 target endpoints")
        endpoints = {name: HTTPClient(payload["endpoints"][name]) for name in names}
        if args.command == "validate-config":
            print(json.dumps(dict(valid=True, task=task.id, models=len(models), external_calls=0), indent=2))
            return 0
        if args.command == "author-initial":
            from .workflows.initial_cap import generate_initial_cap
            policy = generate_initial_cap(task, target=endpoints["target"], agent=endpoints["agent"], output=args.out,
                api_version=IDENTIFICATION_API_VERSION)
            print(json.dumps(dict(status="authored", sha256=policy.sha256,
                policy=str(args.out / "bundle/policy.py"), simulator_executed=False), indent=2))
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
