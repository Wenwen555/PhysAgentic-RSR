"""Context generation is deterministic and never starts a coding agent."""
import json
import pytest
from active_sysid.robot.stage_freeze import sha256,MANIFEST
from scripts.prepare_cap_context import prepare
from scripts.launch_cap_agent import command


def test_context_handoff_prepared_without_execution(tmp_path):
    result=prepare(tmp_path/'context')
    assert result['status']=='prepared_not_executed'
    assert result['model']=='gpt-6-astra'
    for i,stage in enumerate(('privileged_exact','visual_exact','visual_randomized'),1):
        folder=tmp_path/'context'/f'stage{i}_{stage}'
        manifest=json.loads((folder/'input/manifest.json').read_text())
        status=json.loads((folder/'output/status.json').read_text())
        assert manifest['frozen_execution_sha256']==sha256(MANIFEST)
        assert status['status']=='not_started'
        assert not manifest['outputs_generated']
        assert not list((folder/'output').glob('*.py'))
        assert manifest['coding_agent']['model']=='gpt-6-astra'
    args=command(tmp_path/'context','privileged_exact')
    assert args[:4]==['codex','exec','-m','gpt-6-astra']
    with pytest.raises(ValueError,match='Previous stage'):
        command(tmp_path/'context','visual_exact')
    with pytest.raises(FileExistsError):
        prepare(tmp_path/'context')
