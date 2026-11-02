"""Restore exact required source dependencies from a GitHub ZIP without .git."""
import argparse
import json
from pathlib import Path
import subprocess

ROOT=Path(__file__).resolve().parents[1]


def restore(include_optional=False):
    lock=json.loads((ROOT/'dependencies.lock.json').read_text())
    for repository in lock['repositories']:
        if not repository['required'] and not include_optional:continue
        target=ROOT/repository['path']
        if target.exists() and any(target.iterdir()):
            # Exported dependency trees can have no .git: verify their manifest
            # rather than replacing potentially edited source directories.
            if not (target/'.git').exists():
                manifest_path=ROOT/'source_manifest.json'
                if not manifest_path.is_file():raise RuntimeError('Existing dependency has no verifiable source manifest: '+str(target))
                import hashlib
                manifest=json.loads(manifest_path.read_text())
                entries={name:digest for name,digest in manifest['sha256'].items() if name.startswith(repository['path']+'/')}
                if not entries or any(not (ROOT/name).is_file() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest for name,digest in entries.items()):
                    raise RuntimeError('Exported dependency content mismatch: '+str(target))
                continue
            actual=subprocess.check_output(['git','-C',str(target),'rev-parse','HEAD'],text=True).strip()
            if actual!=repository['commit']:raise RuntimeError('Existing dependency revision differs: '+str(target))
            continue
        target.parent.mkdir(parents=True,exist_ok=True)
        subprocess.run(['git','clone','--filter=blob:none','--no-checkout',repository['url'],str(target)],check=True)
        subprocess.run(['git','-C',str(target),'checkout','--detach',repository['commit']],check=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--include-optional',action='store_true')
    args=parser.parse_args();restore(args.include_optional)
