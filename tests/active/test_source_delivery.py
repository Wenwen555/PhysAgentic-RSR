"""Source archives carry first-party tests and exact dependency content records."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

from physrsr.active.core.provenance import git_revision
from tools import export_source


def repository(path):
    path.mkdir(parents=True)
    subprocess.run(['git','init','-q',str(path)],check=True)
    (path/'capx').mkdir();(path/'capx/__init__.py').write_text('fixture dependency\n')
    subprocess.run(['git','-C',str(path),'add','.'],check=True)
    subprocess.run(['git','-C',str(path),'-c','user.name=Fixture','-c','user.email=fixture@example.invalid',
        'commit','-qm','Fixture content'],check=True)
    return subprocess.check_output(['git','-C',str(path),'rev-parse','HEAD'],text=True).strip()


def test_zip_contains_current_tests_and_verified_recursive_content_manifest(tmp_path,monkeypatch):
    root=tmp_path/'project';root.mkdir()
    for folder in ['physrsr','tests/active','tools']:(root/folder).mkdir(parents=True)
    (root/'physrsr/__init__.py').write_text('"""fixture"""\n')
    (root/'tests/active/test_fixture.py').write_text('def test_fixture(): assert True\n')
    sha=repository(root/'src/cap-x')
    lock=dict(repositories=[dict(path='src/cap-x',commit=sha,required=True,url='https://example.invalid/capx'),
        dict(path='src/cap-x/optional',commit='a'*40,required=False,url='https://example.invalid/optional')])
    for name in ['README.md','AGENTS.md','pyproject.toml','.gitmodules']:(root/name).write_text('fixture\n')
    (root/'dependencies.lock.json').write_text(json.dumps(lock))
    monkeypatch.setattr(export_source,'ROOT',root)
    result=export_source.export(tmp_path/'archive')
    with zipfile.ZipFile(tmp_path/'archive/source.zip') as archive:
        assert 'tests/active/test_fixture.py' in archive.namelist()
        for path,digest in result['sha256'].items():assert hashlib.sha256(archive.read(path)).hexdigest()==digest
    dependency=result['dependencies']['src/cap-x']
    assert dependency['commit']==sha and dependency['files'][0]['object']
    assert result['dependencies']['src/cap-x/optional']['source_available'] is False


def test_unpacked_source_does_not_inherit_an_unrelated_parent_git_revision(tmp_path):
    repository(tmp_path/'parent')
    unpacked=tmp_path/'parent'/'source_zip';unpacked.mkdir()
    assert git_revision(unpacked)==dict(commit=None,dirty=None)
