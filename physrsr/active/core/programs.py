"""Deterministic source contracts shared by agents, workflows and executors."""
import ast

from .contracts import IDENTIFICATION_API_VERSION
from .policy import Policy


def source_policy(source):
    policy=Policy(source,api_version=IDENTIFICATION_API_VERSION)
    methods={"describe","observe","hold","checkpoint","goto_pose","open_gripper","close_gripper"}
    for node in ast.walk(ast.parse(source)):
        if (isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name)
                and node.value.id=="robot" and node.attr not in methods):
            raise ValueError("Source uses an unavailable measured API method")
    return policy


def resumable_probe_source(source):
    policy=source_policy(source)
    tree=ast.parse(source)
    methods={node.attr for node in ast.walk(tree) if isinstance(node,ast.Attribute)
        and isinstance(node.value,ast.Name) and node.value.id=="robot"}
    literals={node.value for node in ast.walk(tree) if isinstance(node,ast.Constant) and isinstance(node.value,str)}
    if not {"checkpoint","describe"}<=methods or not {"probe_execution","resume_state"}<=literals:
        raise ValueError("Probe must read the same-source resume_state and publish a safe budget checkpoint")
    return policy


def task_candidate(source):
    policy=source_policy(source)
    # Bound robot methods may be passed through a local RPC-counting helper.
    # This excludes no-op certifiers, not a proof of reachability/completion.
    if not any(isinstance(node,ast.Attribute) and isinstance(node.value,ast.Name)
            and node.value.id=="robot" and node.attr in {"goto_pose","open_gripper","close_gripper"}
            for node in ast.walk(ast.parse(source))):
        raise ValueError("A task candidate must include a state-changing public robot action; a no-op certifier is not synthesis")
    return policy
