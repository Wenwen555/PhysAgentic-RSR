"""Persist authored probes, reviews and analyses beside initial-CaP evidence."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil

from ..agents.physic import DEFAULT_PROBE_BUDGET, PHYSIC_SCHEMA, author_messages, prompt_for, validate_proposal
from ..core.artifacts import save_json
from ..core.paths import PROMPTS_ROOT
from ..api.process import LoggedAPI, service
from .evidence import prepare_initial_evidence

def analysis_markdown(payload):
    lines = ["# Active Probe：任务相关物理不确定性与动作原语", "",
             "状态：仅生成，未执行；信息价值为定性判断，成本为估计。", "", "## 观测事实", ""]
    for fact in payload["observed_facts"]:
        lines.append(f"- {fact['statement']}（证据：{', '.join(fact['evidence_ids'])}）")
    lines += ["", "## Task-relevant Physical Uncertainty", ""]
    for item in payload["task_relevant_uncertainties"]:
        lines += [f"### {item['factor']}", "", item["task_decision_affected"], "",
            "证据：" + ", ".join(item["supporting_evidence_ids"]), "",
            "替代解释：" + "；".join(item["alternative_explanations"]), "",
            "当前 API 的可辨识边界：" + item["identifiable_with_current_api"], ""]
    lines += ["## 候选交互与成本", "", "| 候选 | 描述 | 预期信息价值 | 估计时间 | 估计 RPC |",
              "| --- | --- | --- | --- | --- |"]
    for candidate in payload["candidate_probes"]:
        lines.append(f"| {candidate['id']} | {candidate['description']} | {candidate['qualitative_information_value']} | {candidate['expected_cost_seconds']} s | {candidate['estimated_rpc_calls']} |")
    for candidate in payload["candidate_probes"]:
        lines += ["", f"**{candidate['id']}**：{candidate['selection_tradeoff']}", "",
            "预期区分信号：" + "；".join(candidate["expected_discriminating_observations"]), "",
            "改变的任务决策：" + candidate["policy_decision_if_outcomes_differ"], "",
            "风险：" + "；".join(candidate["risks"])]
    lines += ["", "## 选中原语", "", payload["selected_probe_id"], "", payload["selection_reason"], "",
              "完整代码：[probe.py](probe.py)。", "", "## 观测与解释规则", ""]
    for rule in payload["interpretation_rules"]:
        lines += [f"- 条件：{rule['condition']}", f"  - 支持：{rule['hypotheses_supported']}",
            f"  - 未解决：{rule['hypotheses_not_resolved']}", f"  - 后续任务决策：{rule['policy_change']}"]
    lines += ["", "## 停止与恢复", "", *[f"- {s}" for s in payload["stop_and_recovery"]],
              "", "## 限制", "", *[f"- {s}" for s in payload["limitations"]], ""]
    return "\n".join(lines)


def author_active_probe(experiment, *, round_index=1, agent=None, port=18124):
    experiment = Path(experiment).resolve()
    stage = experiment / f"active_probe_{round_index}"
    if round_index < 1:
        raise ValueError("Probe round must be positive")
    if stage.exists():
        raise FileExistsError(f"Refusing to overwrite probe authoring: {stage}")
    context = prepare_initial_evidence(experiment)
    context["probe_budget"] = dict(DEFAULT_PROBE_BUDGET)
    stage.mkdir()
    save_json(stage / "context.json", context)
    save_json(stage / "schema.json", PHYSIC_SCHEMA)
    (stage / "prompt.txt").write_text(prompt_for(context))
    (stage / "prompt_template.txt").write_text((PROMPTS_ROOT / "physic_probe.txt").read_text())
    if agent is None:
        with service(["serve-physic", "--out", str(stage / "inference")], port, stage / "service.log") as endpoint:
            payload = LoggedAPI(endpoint, stage / "api.jsonl").call("physic.propose_probe", context)
    else:
        payload = LoggedAPI(agent, stage / "api.jsonl").call("physic.propose_probe", context)
    return record_proposal(stage, payload, context, round_index)


def record_proposal(stage, payload, context, round_index, **metadata):
    save_json(stage / "response.json", payload)
    policy = validate_proposal(payload, context)
    (stage / "probe.py").write_bytes(policy.source.encode())
    analysis = {key: value for key, value in payload.items() if key != "code"}
    save_json(stage / "analysis.json", analysis)
    (stage / "analysis.md").write_text(analysis_markdown(payload))
    manifest = dict(stage=f"active_probe_{round_index}", status="authored_not_executed",
        parent_policy_sha256=context["initial_policy"]["sha256"], probe_sha256=policy.sha256,
        input_sha256=hashlib.sha256((stage / "context.json").read_bytes()).hexdigest(),
        selected_probe_id=payload["selected_probe_id"], probe_budget=context["probe_budget"],
        information_value_kind=payload["information_value_kind"],
        source_validation_passed=True, dynamics_verified=False, probe_executed=False,
        real_robot_connected=False, replay_added_physical_interactions=0)
    manifest.update(metadata)
    save_json(stage / "manifest.json", manifest)
    return manifest


def revise_active_probe(experiment, review_findings, *, round_index=1, agent=None, port=18124):
    experiment = Path(experiment).resolve()
    stage = experiment / f"active_probe_{round_index}"
    previous = json.loads((stage / "response.json").read_text())
    old_manifest = json.loads((stage / "manifest.json").read_text())
    if old_manifest["probe_executed"]:
        raise ValueError("This source review path only handles unexecuted probes")
    context = prepare_initial_evidence(experiment)
    context["probe_budget"] = dict(DEFAULT_PROBE_BUDGET)
    drafts = stage / "drafts"
    drafts.mkdir(exist_ok=True)
    index = len(list(drafts.glob("draft_*")))+1
    draft = drafts / f"draft_{index:03d}"
    draft.mkdir()
    for name in ("probe.py", "analysis.json", "analysis.md", "response.json", "context.json", "prompt.txt", "prompt_template.txt", "manifest.json"):
        shutil.copy2(stage / name, draft / name)
    request = dict(context=context, previous_proposal=previous, review_findings=review_findings)
    save_json(stage / f"review_request_{index:03d}.json", request)
    if agent is None:
        with service(["serve-physic", "--out", str(stage / f"inference_review_{index:03d}")], port,
                stage / f"review_service_{index:03d}.log") as endpoint:
            payload = LoggedAPI(endpoint, stage / f"review_api_{index:03d}.jsonl").call("physic.revise_probe", request)
    else:
        payload = LoggedAPI(agent, stage / f"review_api_{index:03d}.jsonl").call("physic.revise_probe", request)
    validate_proposal(payload, context)
    save_json(stage / "context.json", context)
    messages = author_messages(context, previous, review_findings)
    (stage / "prompt.txt").write_text("\n\n".join(f"[{m['role']}]\n{m['content']}" for m in messages))
    (stage / "prompt_template.txt").write_text((PROMPTS_ROOT / "physic_probe.txt").read_text())
    return record_proposal(stage, payload, context, round_index,
        source_review_count=index, previous_probe_sha256=old_manifest["probe_sha256"],
        review_kind="API/source contract review; no new physical feedback")
