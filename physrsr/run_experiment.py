"""Paired Direct-CaP vs Sim-CaP comparison (Step 4, M1 oracle physics).

Core question: when both agents know mass / CoM / friction / inertia, does
letting one agent run the Simulator improve program selection?

Protocol (paired design): for each physics configuration both methods see the
same physics and the same candidate pool, and *every* candidate is scored on
the same independent ground-truth seeds. That yields, per case:

* the ground-truth success rate of each candidate (with uncertainty),
* the choice each method makes,
* the best achievable candidate on ground truth (oracle upper bound) and the
  regret of each method,
* per-seed paired outcomes for direct vs sim, enabling a within-seed
  (paired) comparison instead of a noisy two-sample one.

M1 uses an oracle simulator, i.e. the simulator *is* the ground-truth model.
It is an upper-bound validation harness, not evidence that simulation helps in
general. M2 (parameter uncertainty + active probing) is the real test.
"""

from __future__ import annotations

import argparse
from physrsr.paths import OUTPUT_ROOT, output_path
import json
import multiprocessing as mp
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path

import numpy as np

from physrsr.candidate_programs import CANDIDATES
from physrsr.physics_spec import (
    CaseResult,
    PhysicsSpec,
    RolloutRecord,
    TestCase,
)

# Physics grid for M1. Three difficulty axes, each with a *measured* live range:
#
# * com_offset: a centred CoM is always graspable; the centre grasp starts
#   failing at |com| ~ 0.065 (mu=1) / ~ 0.09 (mu=2). Bracketing that boundary
#   with +-0.09 and +-0.03 keeps both "any grasp works" and "only a CoM-aware
#   grasp works" cases in the grid.
# * friction: only observable when the grip is near its friction limit (with
#   the robot's 20 N grip the grasp force exceeds friction by >10x). mu=2.0 is
#   also the regime where transport speed decides the CoM boundary.
# * grip_force: ``None`` (robot limit) or a weak 2 N "delicate grip" that puts
#   the grasp at the friction limit.
DEFAULT_COM_OFFSETS = (-0.09, -0.03, 0.0, 0.03, 0.09)
DEFAULT_FRICTIONS = (0.7, 2.0)
DEFAULT_GRIP_FORCES = (None, 2.0)
DEFAULT_DENSITY = 1000.0


class DirectAgent:
    """Rule-based baseline: reasons about physics *without* simulator rollout.

    With oracle physics it knows the CoM offset, so the "physically informed"
    choice is the program that grasps at the CoM.
    """

    def select(self, physics: PhysicsSpec, candidates):
        for p in candidates:
            if p.where == "com":
                return p
        return candidates[0]


class SimAgent:
    """Rule-based agent: selects using simulator-rollout feedback."""

    def select(self, physics: PhysicsSpec, candidates, feedback: dict[str, float]):
        # ``max`` keeps the first candidate on ties -> deterministic.
        return max(candidates, key=lambda p: feedback.get(p.id, 0.0))


def build_cases(
    com_offsets=DEFAULT_COM_OFFSETS,
    frictions=DEFAULT_FRICTIONS,
    density: float = DEFAULT_DENSITY,
    grip_forces=DEFAULT_GRIP_FORCES,
    candidate_ids=tuple(CANDIDATES),
) -> list[TestCase]:
    cases = []
    for com in com_offsets:
        for mu in frictions:
            for grip in grip_forces:
                physics = PhysicsSpec(
                    com_offset=float(com),
                    friction=float(mu),
                    density=density,
                    grip_force=None if grip is None else float(grip),
                )
                cases.append(
                    TestCase(
                        case_id=physics.case_id,
                        physics=physics,
                        candidate_ids=tuple(candidate_ids),
                    )
                )
    return cases


# ---------------------------------------------------------------- parallel run


def _run_task(task: tuple) -> list[RolloutRecord]:
    """Run one (physics, program, role) over its seeds in a fresh worker."""
    # Imported lazily so a spawn worker only pays for it when it has real work.
    from physrsr.worker import RolloutWorker

    physics, program_id, seeds, hold_steps, role, level_tolerance = task
    worker = RolloutWorker(physics, hold_steps=hold_steps, level_tolerance=level_tolerance)
    try:
        return worker.rollout_many(CANDIDATES[program_id], seeds, role=role)
    finally:
        worker.close()


def _make_tasks(cases, sim_seeds, gt_seeds, hold_steps, level_tolerance):
    tasks = []
    for case in cases:
        for pid in case.candidate_ids:
            tasks.append((case.physics, pid, tuple(sim_seeds), hold_steps, "sim", level_tolerance))
            tasks.append((case.physics, pid, tuple(gt_seeds), hold_steps, "gt", level_tolerance))
    return tasks


def _rate(records) -> float:
    return sum(r.ok for r in records) / len(records) if records else 0.0


def _outcomes_by_seed(records, seeds) -> list[int]:
    by_seed = {r.seed: int(r.ok) for r in records}
    missing = [s for s in seeds if s not in by_seed]
    if missing:
        raise RuntimeError(f"missing rollouts for seeds {missing}")
    return [by_seed[s] for s in seeds]


def _assemble_case(case, sim_records, gt_records, sim_seeds, gt_seeds) -> CaseResult:
    candidates = [CANDIDATES[cid] for cid in case.candidate_ids]

    sim_feedback = {p.id: _rate(sim_records[p.id]) for p in candidates}
    gt_rates = {p.id: _rate(gt_records[p.id]) for p in candidates}

    direct_choice = DirectAgent().select(case.physics, candidates)
    sim_choice = SimAgent().select(case.physics, candidates, sim_feedback)
    oracle_choice = max(candidates, key=lambda p: gt_rates[p.id])

    errors = Counter(
        r.reason
        for recs in list(sim_records.values()) + list(gt_records.values())
        for r in recs
        if r.status == "error"
    )
    return CaseResult(
        case_id=case.case_id,
        com_offset=case.physics.com_offset,
        friction=case.physics.friction,
        density=case.physics.density,
        grip_force=case.physics.grip_force,
        direct_choice=direct_choice.id,
        sim_choice=sim_choice.id,
        oracle_choice=oracle_choice.id,
        direct_gt_rate=_rate(gt_records[direct_choice.id]),
        sim_gt_rate=_rate(gt_records[sim_choice.id]),
        oracle_gt_rate=gt_rates[oracle_choice.id],
        direct_gt_outcomes=_outcomes_by_seed(gt_records[direct_choice.id], gt_seeds),
        sim_gt_outcomes=_outcomes_by_seed(gt_records[sim_choice.id], gt_seeds),
        gt_seeds=list(gt_seeds),
        sim_feedback=sim_feedback,
        gt_rates=gt_rates,
        sim_outcomes={p.id: _outcomes_by_seed(sim_records[p.id], sim_seeds) for p in candidates},
        gt_outcomes={p.id: _outcomes_by_seed(gt_records[p.id], gt_seeds) for p in candidates},
        n_sim_errors=sum(r.status == "error" for p in candidates for r in sim_records[p.id]),
        n_gt_errors=sum(r.status == "error" for p in candidates for r in gt_records[p.id]),
        error_messages=dict(errors),
    )


def preflight(hold_steps: int, level_tolerance: float) -> None:
    """Cheap health checks before spending minutes of rollouts.

    Verifies (a) the physics knobs reach the compiled MuJoCo model, (b) a
    rollout actually executes (IK server up, program runs) and (c) the same
    (program, seed) is reproducible. Without (b) a dead IK server would look
    like a uniform zero success rate.
    """
    from physrsr.worker import RolloutWorker

    print("[preflight] friction readback")
    for mu in (0.2, 1.0):
        worker = RolloutWorker(
            PhysicsSpec(com_offset=0.0, friction=mu),
            hold_steps=hold_steps,
            level_tolerance=level_tolerance,
        )
        try:
            values = worker.read_rod_friction()
        finally:
            worker.close()
        if not values or not all(abs(v - mu) < 1e-6 for v in values):
            raise RuntimeError(f"friction readback failed: target {mu}, got {values}")
        print(f"  mu={mu}: contact geoms {values}")

    print("[preflight] reset state reproducible")
    physics = PhysicsSpec(com_offset=0.0, friction=1.0)
    worker = RolloutWorker(physics, hold_steps=hold_steps, level_tolerance=level_tolerance)
    try:
        f1 = worker.reset_fingerprint(7)
        f2 = worker.reset_fingerprint(7)
        if not np.array_equal(f1, f2):
            worst = int(np.argmax(np.abs(f1 - f2)))
            raise RuntimeError(
                "reset is nondeterministic: same seed gave different initial state "
                f"(max diff {abs(f1[worst] - f2[worst]):.3e} at index {worst})"
            )
        print(f"  reset(seed=7): rod+joints identical across 2 resets (n={f1.size})")

        print("[preflight] rollout executes (IK server reachable) and repeats")
        for pid, seed in (("P1", 7), ("P5", 7)):
            a = worker.rollout(CANDIDATES[pid], seed, role="preflight")
            b = worker.rollout(CANDIDATES[pid], seed, role="preflight")
            if a.status == "error" or b.status == "error":
                raise RuntimeError(f"{pid}/seed{seed} errored: {a.reason!r} / {b.reason!r}")
            if (a.status, a.reason, a.failure_step) != (b.status, b.reason, b.failure_step):
                raise RuntimeError(
                    f"{pid}/seed{seed} not reproducible: "
                    f"{(a.status, a.reason, a.failure_step)} vs "
                    f"{(b.status, b.reason, b.failure_step)}"
                )
            print(f"  {pid}/seed{seed}: status={a.status} reason={a.reason or '-'} (x2 identical)")
    finally:
        worker.close()
    print("[preflight] PASS")


# ---------------------------------------------------------------------- driver


def run(cases, sim_seeds, gt_seeds, hold_steps, level_tolerance, jobs: int):
    tasks = _make_tasks(cases, sim_seeds, gt_seeds, hold_steps, level_tolerance)
    by_case: dict[str, dict[str, dict[str, list[RolloutRecord]]]] = {
        c.case_id: {"sim": {}, "gt": {}} for c in cases
    }

    print(f"[run] {len(cases)} cases, {len(tasks)} tasks, jobs={jobs}")
    t0 = time.time()
    with ProcessPoolExecutor(
        max_workers=jobs, mp_context=mp.get_context("spawn")
    ) as pool:
        futures = {}
        for task in tasks:
            physics, pid, _seeds, _h, role, _lt = task
            futures[pool.submit(_run_task, task)] = (physics.case_id, pid, role)
        for i, fut in enumerate(as_completed(futures), 1):
            key = futures[fut]
            recs = fut.result()
            by_case[key[0]][key[2]][key[1]] = recs
            errs = sum(r.status == "error" for r in recs)
            print(
                f"  [{i:>3}/{len(tasks)}] {key[0]:<28} {key[1]} {key[2]} "
                f"rate={_rate(recs):.2f} n={len(recs)} errors={errs}"
            )
    print(f"[run] finished in {time.time() - t0:.0f}s")

    all_records = [
        r for case in by_case.values() for role in case.values() for recs in role.values() for r in recs
    ]
    n_err = sum(r.status == "error" for r in all_records)
    if n_err:
        msg = Counter(r.reason for r in all_records if r.status == "error")
        print(f"[run] WARNING {n_err}/{len(all_records)} rollouts errored:")
        for reason, count in msg.most_common(5):
            print(f"  {count:>5}x {reason}")
        if n_err > 0.05 * len(all_records):
            raise RuntimeError(
                "more than 5% of rollouts errored - fix infrastructure before interpreting results"
            )

    results = [
        _assemble_case(c, by_case[c.case_id]["sim"], by_case[c.case_id]["gt"], sim_seeds, gt_seeds)
        for c in cases
    ]
    return results, all_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sim-seeds", type=int, default=8)
    parser.add_argument("--gt-seeds", type=int, default=32)
    parser.add_argument("--hold-steps", type=int, default=10)
    parser.add_argument(
        "--level-tolerance",
        type=float,
        default=0.25,
        help="max |z| component of the rod long axis (0.25 ~= 15 deg tilt)",
    )
    parser.add_argument("--com-offsets", type=float, nargs="+", default=list(DEFAULT_COM_OFFSETS))
    parser.add_argument("--frictions", type=float, nargs="+", default=list(DEFAULT_FRICTIONS))
    parser.add_argument(
        "--grip-forces",
        type=str,
        nargs="+",
        default=["robot" if g is None else str(g) for g in DEFAULT_GRIP_FORCES],
        help="Per-finger force cap in newtons; 'robot' keeps the robot's own limit.",
    )
    parser.add_argument("--density", type=float, default=DEFAULT_DENSITY)
    parser.add_argument("--jobs", type=int, default=min(32, mp.cpu_count()))
    parser.add_argument("--no-preflight", action="store_true")
    parser.add_argument("--out", type=output_path, default=OUTPUT_ROOT / "physics_cap/results.json")
    parser.add_argument(
        "--records", type=output_path, default=OUTPUT_ROOT / "physics_cap/rollouts.jsonl"
    )
    args = parser.parse_args()

    sim_seeds = list(range(1, 1 + args.sim_seeds))
    gt_seeds = list(range(1001, 1001 + args.gt_seeds))

    if not args.no_preflight:
        preflight(args.hold_steps, args.level_tolerance)

    grip_forces = [
        None if g.lower() in {"robot", "none", "default"} else float(g)
        for g in args.grip_forces
    ]
    cases = build_cases(args.com_offsets, args.frictions, args.density, grip_forces)
    results, all_records = run(
        cases, sim_seeds, gt_seeds, args.hold_steps, args.level_tolerance, args.jobs
    )

    args.records.parent.mkdir(parents=True, exist_ok=True)
    with args.records.open("w") as fh:
        for r in all_records:
            fh.write(json.dumps(asdict(r)) + "\n")
    print(f"wrote {args.records} ({len(all_records)} records)")

    from physrsr.evaluate import summarize

    summary = summarize(results, all_records)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "sim_seeds": sim_seeds,
            "gt_seeds": gt_seeds,
            "hold_steps": args.hold_steps,
            "level_tolerance": args.level_tolerance,
            "density": args.density,
            "com_offsets": list(args.com_offsets),
            "frictions": list(args.frictions),
            "grip_forces": grip_forces,
            "candidates": {k: asdict(v) for k, v in CANDIDATES.items()},
        },
        "summary": summary,
        "results": [asdict(r) for r in results],
    }
    args.out.write_text(json.dumps(payload, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
