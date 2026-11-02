"""Export first-party sources/tests plus pinned recursive dependency content lists.

Optional --include-capx adds tracked CaP-X and its required Robosuite source.
No credentials, environments, model weights or experimental outputs are exported.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from physrsr.paths import output_path


def tracked_tree(root):
    output=subprocess.check_output(['git','-C',str(root),'ls-tree','-r','HEAD'],text=True)
    return [dict(zip(('mode','kind','object','path'),line.replace('\t',' ',1).split(' ',3))) for line in output.splitlines()]


def export(destination,include_capx=False):
    destination.mkdir(parents=True,exist_ok=False)
    paths=[]
    for folder in ('physrsr','tests','tools'):
        paths.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts
            and p.suffix in ('.py','.txt','.json','.md'))
    paths.extend(ROOT/name for name in ('README.md','AGENTS.md','pyproject.toml','.gitmodules','dependencies.lock.json'))
    dependency_inventory={}
    for entry in json.loads((ROOT/'dependencies.lock.json').read_text())['repositories']:
        repo=ROOT/entry['path']
        available=(repo/'.git').exists()
        if entry['required'] and not available:
            raise RuntimeError('Initialize the locked required dependency before exporting: '+str(repo))
        inventory=tracked_tree(repo) if available else []
        dependency_inventory[entry['path']]=dict(commit=entry['commit'],source_available=available,files=inventory)
        if include_capx and entry['required']:
            paths.extend(repo/item['path'] for item in inventory if item['kind']=='blob'
                and Path(item['path']).suffix not in ('.pt','.pth','.ckpt','.safetensors','.onnx'))
    hashes={}
    with zipfile.ZipFile(destination/'source.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(set(paths)):
            relative=path.relative_to(ROOT).as_posix()
            payload=os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
            hashes[relative]=hashlib.sha256(payload).hexdigest()
            info=zipfile.ZipInfo(relative)
            if path.is_symlink():info.create_system=3;info.external_attr=0o120777<<16
            info.compress_type=zipfile.ZIP_DEFLATED
            archive.writestr(info,payload)
        from physrsr.active.core.provenance import git_revision
        manifest=dict(format='physrsr.source.v1',project=git_revision(ROOT),sha256=hashes,dependencies=dependency_inventory)
        archive.writestr('source_manifest.json',json.dumps(manifest,indent=2))
    (destination/'source_manifest.json').write_text(json.dumps(manifest,indent=2))
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=output_path,required=True)
    parser.add_argument('--include-capx',action='store_true')
    args=parser.parse_args()
    result=export(args.out,args.include_capx)
    print(json.dumps(dict(archive=str(args.out/'source.zip'),files=len(result['sha256']))))
