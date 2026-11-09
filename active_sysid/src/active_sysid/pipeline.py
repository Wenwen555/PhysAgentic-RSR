"""主链路: 把 Planner / Probe / Identification / Simulator / CaP 串成闭环。

    hypotheses = planner.propose(task, scene, history)
    probe      = planner.select_probe(hypotheses, library, task)
    evidence   = executor.execute(probe, world)
    model      = mechanism_identifier.update(hypotheses, evidence)
    params     = parameter_identifier.fit(model, history + [evidence])
    simulator.update(model, params)
    result     = validator.evaluate(simulator, task)
    planner.update(result)

停止条件与下游任务绑定: 当当前仿真能够预测保留的 Probe 结果, 且进一步
物理辨识对 CaP 的成功预测改善有限时, 停止探索并执行 CaP。
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from active_sysid.cap.evaluator import CaPEvaluator
from active_sysid.config import require_section
from active_sysid.identification.cma_es import CMASysID
from active_sysid.identification.mechanism import MechanismIdentifier
from active_sysid.planner.probe_planner import ProbePlanner
from active_sysid.probes.executor import ProbeExecutor, WorldInterface
from active_sysid.probes.library import ProbeLibrary
from active_sysid.simulator.model_editor import ModelEditor
from active_sysid.simulator.validator import Validator
from active_sysid.types import LoopReport, TaskSpec


class ProbeLoop:
    def __init__(
        self,
        planner: ProbePlanner,
        library: ProbeLibrary,
        executor: ProbeExecutor,
        mechanism_identifier: MechanismIdentifier,
        parameter_identifier: CMASysID,
        model_editor: ModelEditor,
        validator: Validator,
        cap_evaluator: CaPEvaluator | None = None,
        config: dict[str, Any] | None = None,
        source_mjcf: str | Path | None = None,
        output_dir: str | Path | None = None,
    ):
        loop_cfg = require_section(
            (config or {}).get("probe_loop"), ("max_probes", "min_probes"),
            "probe_loop",
        )
        self.max_probes = int(loop_cfg["max_probes"])
        self.min_probes = int(loop_cfg["min_probes"])
        self.planner = planner
        self.library = library
        self.executor = executor
        self.mechanism_identifier = mechanism_identifier
        self.parameter_identifier = parameter_identifier
        self.model_editor = model_editor
        self.validator = validator
        self.cap_evaluator = cap_evaluator or CaPEvaluator()
        self.source_mjcf = Path(source_mjcf) if source_mjcf else None
        self.output_dir = Path(output_dir) if output_dir else None

    # ------------------------------------------------------------------
    def run(self, task: TaskSpec, scene: dict[str, Any],
            world: WorldInterface) -> LoopReport:
        hypotheses = self.planner.propose(task, scene)
        evidences: list = []
        executed_keys: set[str] = set()
        report = LoopReport()
        sim_path: Path | None = None

        for step in range(self.max_probes):
            probe = self.planner.select_probe(
                hypotheses, self.library, task, executed_keys
            )
            executed_keys.add(probe.key())
            evidence = self.executor.execute(probe, world, index=step)
            evidences.append(evidence)
            hypotheses.update(evidence)
            report.belief_history.append(hypotheses.posterior())

            model = self.mechanism_identifier.update(hypotheses, evidences)
            fit = self.parameter_identifier.fit(model, evidences)

            if self.source_mjcf is not None and self.output_dir is not None:
                sim_path = self.output_dir / "repaired" / f"scene_step{step + 1:02d}.mjcf"
                self.model_editor.apply(self.source_mjcf, model,
                                        fit.params, sim_path)

            validator_report = self.validator.evaluate(
                sim_path, model, heldout=evidences[-1:]
            )
            self.planner.update(validator_report)
            report.best_hypothesis = {
                "hid": model.hid,
                "structure": model.structure,
                "params": model.params,
            }
            report.fitted_params = dict(fit.params)

            stop, reason = self.planner.should_stop(
                hypotheses, validator_report,
                n_probes=len(evidences),
                min_probes=self.min_probes, max_probes=self.max_probes,
            )
            if stop:
                report.stop_reason = reason
                break
        else:
            report.stop_reason = "max_probes"

        report.n_probes = len(evidences)
        report.probe_keys = sorted(executed_keys)
        report.predictive = bool(
            report.best_hypothesis.get("params", {}).get("theta_unlock") is not None
        )
        report.cap_result = {"status": "not_run"}

        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            (self.output_dir / "report.json").write_text(
                json.dumps(asdict(report), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        return report
