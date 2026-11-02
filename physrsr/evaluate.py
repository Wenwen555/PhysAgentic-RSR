"""Metrics for the Direct-CaP vs Sim-CaP comparison.

For M1 the deliverable is a *validation harness*, not a claimed result: an
oracle simulator that shares the GT model is an upper bound on simulator
usefulness. This module therefore reports quantities that stay meaningful even
under that upper bound:

* per-candidate ground-truth success rates with Wilson intervals,
* the ground-truth-optimal candidate (oracle) and each method's *regret*,
* a within-seed paired bootstrap of ``sim - direct`` success (resampling cases
  and seeds, since both choices are scored on the same GT seeds),
* simulator-vs-ground-truth agreement (rate error and rank correlation), which
  is the actual question M1 can answer: does simulator feedback track the true
  ordering?
* a breakdown of physical failure modes and infrastructure errors.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence

import numpy as np

from physrsr.physics_spec import CaseResult, RolloutRecord


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (better than normal at n<30)."""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def paired_cluster_bootstrap(
    results: Sequence[CaseResult], n_boot: int = 5000, seed: int = 0
) -> tuple[float, float, float]:
    """Mean paired ``sim - direct`` GT success, with a 95% bootstrap CI.

    Direct and sim are scored on the *same* GT seeds within a case, so the
    per-seed difference is paired. Two sources of variance remain: which cases
    we happened to draw (cluster) and which seeds within a case, so both are
    resampled: cases with replacement, then seed indices within each case.
    """
    diffs = [
        np.asarray(r.sim_gt_outcomes, dtype=float) - np.asarray(r.direct_gt_outcomes, dtype=float)
        for r in results
    ]
    diffs = [d for d in diffs if d.size]
    if not diffs:
        return (0.0, 0.0, 0.0)
    point = float(np.mean([d.mean() for d in diffs]))
    rng = np.random.default_rng(seed)
    n_cases = len(diffs)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        case_idx = rng.integers(0, n_cases, n_cases)
        per_case = []
        for i in case_idx:
            d = diffs[int(i)]
            per_case.append(d[rng.integers(0, d.size, d.size)].mean())
        stats[b] = float(np.mean(per_case))
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return point, float(lo), float(hi)


def split_half_oracle(results: Sequence[CaseResult]) -> float | None:
    """Out-of-sample estimate of the best candidate in the pool.

    ``max`` over noisy GT rates is optimistically biased (the oracle rate in
    ``CaseResult``). Picking the best candidate on one half of the GT seeds and
    scoring it on the *other* half gives an unbiased estimate of what "select
    the best program" is worth at this sample size. Both directions are used.
    """
    if not results or not results[0].gt_outcomes:
        return None
    scores = []
    for r in results:
        n = len(next(iter(r.gt_outcomes.values())))
        if n < 4:
            continue
        first = list(range(0, n, 2))
        second = list(range(1, n, 2))
        for select_idx, eval_idx in ((first, second), (second, first)):
            pid = max(
                sorted(r.gt_outcomes),
                key=lambda p: sum(r.gt_outcomes[p][i] for i in select_idx) / len(select_idx),
            )
            scores.append(sum(r.gt_outcomes[pid][i] for i in eval_idx) / len(eval_idx))
    return float(np.mean(scores)) if scores else None


def _fmt_rate(rate: float, n: int) -> str:
    k = round(rate * n)
    lo, hi = wilson_ci(k, n)
    return f"{rate:.2f} [{lo:.2f},{hi:.2f}]"


def _spearman(a: dict[str, float], b: dict[str, float]) -> float:
    try:
        from scipy.stats import spearmanr

        keys = sorted(a)
        rho = spearmanr([a[k] for k in keys], [b[k] for k in keys]).statistic
        return float(rho) if np.isfinite(rho) else float("nan")
    except ImportError:  # pragma: no cover - scipy is a project dependency
        return float("nan")


def summarize(
    results: Iterable[CaseResult], records: Iterable[RolloutRecord] | None = None
) -> dict:
    rows = list(results)
    recs = list(records) if records is not None else None
    n_gt = len(rows[0].gt_seeds) if rows else 0

    header = (
        f"{'case':<26} {'d/s/o':<10} {'direct_gt':>18} {'sim_gt':>18} "
        f"{'oracle_gt':>10} {'regret d/s':>12} {'agree':>6}"
    )
    print(header)
    print("-" * len(header))

    for r in rows:
        agree = "yes" if r.sim_choice == r.oracle_choice else (
            "tie" if r.sim_gt_rate == r.oracle_gt_rate else "no"
        )
        print(
            f"{r.case_id:<26} "
            f"{r.direct_choice + '/' + r.sim_choice + '/' + r.oracle_choice:<10} "
            f"{_fmt_rate(r.direct_gt_rate, n_gt):>18} "
            f"{_fmt_rate(r.sim_gt_rate, n_gt):>18} "
            f"{r.oracle_gt_rate:>10.2f} "
            f"{r.oracle_gt_rate - r.direct_gt_rate:>5.2f}/"
            f"{r.oracle_gt_rate - r.sim_gt_rate:<5.2f} "
            f"{agree:>6}"
        )

    # ---------------------------------------------------------------- aggregate
    n_cases = len(rows)
    mean_direct = float(np.mean([r.direct_gt_rate for r in rows])) if rows else 0.0
    mean_sim = float(np.mean([r.sim_gt_rate for r in rows])) if rows else 0.0
    mean_oracle = float(np.mean([r.oracle_gt_rate for r in rows])) if rows else 0.0
    regret_direct = float(np.mean([r.oracle_gt_rate - r.direct_gt_rate for r in rows])) if rows else 0.0
    regret_sim = float(np.mean([r.oracle_gt_rate - r.sim_gt_rate for r in rows])) if rows else 0.0
    diff, diff_lo, diff_hi = paired_cluster_bootstrap(rows)
    oracle_oos = split_half_oracle(rows)

    sim_oracle = sum(r.sim_choice == r.oracle_choice for r in rows)
    direct_oracle = sum(r.direct_choice == r.oracle_choice for r in rows)

    per_candidate = defaultdict(list)
    abs_err = []
    rhos = []
    for r in rows:
        for pid, rate in r.gt_rates.items():
            per_candidate[pid].append(rate)
            abs_err.append(abs(r.sim_feedback.get(pid, 0.0) - rate))
        rho = _spearman(r.sim_feedback, r.gt_rates)
        if np.isfinite(rho):
            rhos.append(rho)

    print("\nAGGREGATE")
    print(
        f"  cases={n_cases}  gt_seeds/case={n_gt}  candidates/case="
        f"{len(rows[0].gt_rates) if rows else 0}"
    )
    print(f"  mean GT success rate   direct={mean_direct:.3f}  sim={mean_sim:.3f}  oracle={mean_oracle:.3f}")
    if oracle_oos is not None:
        print(f"  oracle (split-half, out-of-sample) = {oracle_oos:.3f}")
    print(f"  mean regret            direct={regret_direct:.3f}  sim={regret_sim:.3f}  (vs in-sample oracle)")
    print(
        f"  paired sim-direct      {diff:+.3f}  95% CI [{diff_lo:+.3f}, {diff_hi:+.3f}]"
    )
    print(
        f"  sim picks oracle       {sim_oracle}/{n_cases}   "
        f"direct picks oracle {direct_oracle}/{n_cases}"
    )
    print(
        f"  sim-vs-GT mean |rate error| = {np.mean(abs_err):.3f}   "
        f"mean Spearman rho = {np.nanmean(rhos) if rhos else float('nan'):.3f}"
    )

    print("\nPER-CANDIDATE mean GT success (averaged over cases)")
    for pid in sorted(per_candidate):
        rates = per_candidate[pid]
        print(f"  {pid}: {np.mean(rates):.3f}  (min {np.min(rates):.2f}, max {np.max(rates):.2f})")

    # ------------------------------------------------------------- diagnostics
    err_total = sum(r.n_sim_errors + r.n_gt_errors for r in rows)
    if err_total:
        print(f"\nINFRASTRUCTURE ERRORS: {err_total} rollouts errored")
        for r in rows:
            for msg, count in r.error_messages.items():
                print(f"  {r.case_id}: {count}x {msg}")

    if recs is not None:
        _print_failure_modes(recs)

    phase_counts = (
        Counter(r.failure_phase for r in recs if r.status == "fail" and r.failure_phase)
        if recs
        else Counter()
    )
    grasp_rate = (
        float(np.mean([bool(r.grasp_established) for r in recs])) if recs else None
    )

    summary = {
        "n_cases": n_cases,
        "n_gt_seeds": n_gt,
        "mean_direct_gt": mean_direct,
        "mean_sim_gt": mean_sim,
        "mean_oracle_gt": mean_oracle,
        "oracle_split_half": oracle_oos,
        "mean_regret_direct": regret_direct,
        "mean_regret_sim": regret_sim,
        "paired_diff": diff,
        "paired_diff_ci": [diff_lo, diff_hi],
        "sim_picks_oracle": sim_oracle,
        "direct_picks_oracle": direct_oracle,
        "sim_gt_mean_abs_rate_error": float(np.mean(abs_err)) if abs_err else None,
        "sim_gt_mean_spearman": float(np.nanmean(rhos)) if rhos else None,
        "per_candidate_mean_gt": {k: float(np.mean(v)) for k, v in sorted(per_candidate.items())},
        "n_errors": err_total,
        "failure_phases": dict(phase_counts),
        "grasp_established_rate": grasp_rate,
    }

    print(
        "\nNOTE: M1 is an upper-bound harness (oracle sim == GT model). "
        "Its question is whether sim feedback tracks the true ordering; "
        "the causal effect of simulation needs M2 (parameter uncertainty)."
    )
    return summary


def _print_failure_modes(records: Iterable[RolloutRecord]) -> None:
    fails: dict[str, Counter] = defaultdict(Counter)
    status = Counter()
    by_program = defaultdict(Counter)
    by_phase: Counter = Counter()
    by_phase_program: dict[str, Counter] = defaultdict(Counter)
    grasp_by_program: dict[str, list[bool]] = defaultdict(list)
    approach_err: list[float] = []
    for r in records:
        status[r.status] += 1
        by_program[r.program_id][r.status] += 1
        grasp_by_program[r.program_id].append(bool(r.grasp_established))
        for sample in r.phases:
            if sample.get("phase") == "pre_grasp" and sample.get("tcp_error") is not None:
                approach_err.append(float(sample["tcp_error"]))
        if r.status == "fail":
            fails[r.program_id][r.reason] += 1
            if r.failure_phase:
                by_phase[r.failure_phase] += 1
                by_phase_program[r.program_id][r.failure_phase] += 1

    total = sum(status.values())
    print(f"\nROLLOUT OUTCOMES (n={total})")
    for s, c in status.most_common():
        print(f"  {s:<8} {c:>6}  ({c / total:.1%})")

    print("\nFAILURE MODES by program (first violated hold condition)")
    for pid in sorted(by_program):
        n_fail = sum(fails[pid].values())
        modes = ", ".join(f"{k}={v}" for k, v in fails[pid].most_common()) or "-"
        print(f"  {pid}: {n_fail:>4} fails  {modes}")

    if by_phase:
        n_fail = sum(by_phase.values())
        print(f"\nFAILURE PHASE (which pipeline stage broke, n={n_fail} failures)")
        for phase, c in by_phase.most_common():
            print(f"  {phase:<10} {c:>6}  ({c / n_fail:.1%})")

    print("\nEFFECTIVE GRASP RATE by program (both finger pads on the object)")
    for pid in sorted(grasp_by_program):
        vals = grasp_by_program[pid]
        print(f"  {pid}: {sum(vals) / len(vals):.2f}  (n={len(vals)})")

    if approach_err:
        arr = np.asarray(approach_err)
        print(
            f"\nAPPROACH TCP ERROR |commanded-achieved|: mean {arr.mean():.4f} m  "
            f"p95 {np.percentile(arr, 95):.4f} m  max {arr.max():.4f} m  "
            f"(fraction > 1 cm: {float((arr > 0.01).mean()):.2%})"
        )


def lower_bound(k, n):
    """One-sided exact 95% binomial lower bound (individual comparison)."""
    from scipy.stats import beta
    return float(beta.ppf(.05, k, n-k+1)) if k else 0.


def cluster_ci(values, seed=20261101):
    x = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(len(x), size=(10000, len(x)))].mean(axis=1)
    return [float(v) for v in np.quantile(means, [.025, .975])]
