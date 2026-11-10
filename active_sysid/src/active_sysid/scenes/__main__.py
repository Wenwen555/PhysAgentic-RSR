"""Import the real source World without replacing robot or door mechanisms."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import mujoco


def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('import','inspect'))
    parser.add_argument('--source',type=Path)
    parser.add_argument('--out',type=Path)
    parser.add_argument('--scene',type=Path)
    args=parser.parse_args()
    if args.command=='import':
        if not args.source or not args.out: parser.error('import needs --source and --out')
        source=args.source.resolve()
        original=mujoco.MjModel.from_xml_path(str(source))
        if args.out.exists(): parser.error('destination already exists; choose a new directory')
        # Preserve relative include/mesh/texture paths by copying the complete source tree.
        shutil.copytree(source.parent,args.out,symlinks=False)
        copied=args.out/source.name
        restored=mujoco.MjModel.from_xml_path(str(copied))
        assert (original.nbody,original.njnt,original.neq,original.nu,original.ncam)==(restored.nbody,restored.njnt,restored.neq,restored.nu,restored.ncam)
        report={'scene_role':'reference_world','source':str(source),'scene':str(copied),
                'sha256':digest(copied),'nbody':restored.nbody,'njnt':restored.njnt,
                'neq':restored.neq,'nu':restored.nu,'ncam':restored.ncam}
        (args.out/'scene_manifest.json').write_text(json.dumps(report,indent=2))
    else:
        if not args.scene: parser.error('inspect needs --scene')
        scene=args.scene/'scene.xml' if args.scene.is_dir() else args.scene
        model=mujoco.MjModel.from_xml_path(str(scene))
        report={'scene':str(scene),'sha256':digest(scene),'nbody':model.nbody,
                'njnt':model.njnt,'neq':model.neq,'nu':model.nu,'ncam':model.ncam}
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
