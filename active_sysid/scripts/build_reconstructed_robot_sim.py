"""Bundle the existing visual reconstruction + legacy inferred mechanisms.

Never reads a Ground-Truth World scene. The source already embeds the calibrated
robot. Geometry and coordinates are retained, including reconstruction errors.
"""
from pathlib import Path
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT.parent/'output/unidoor_real2sim_20261108/stage3_final/scene.xml'
OUT=ROOT/'assets/reconstructed_sim/visual_stage3_v1'


def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()


def build():
    OUT.mkdir(parents=True,exist_ok=True)
    tree=ET.parse(SOURCE); root=tree.getroot(); compiler=root.find('compiler')
    dependencies={}
    for el in root.findall('./asset/*[@file]'):
        file=Path(el.get('file'))
        if not file.is_absolute():
            directory=compiler.get('texturedir' if el.tag=='texture' else 'meshdir','')
            file=SOURCE.parent/directory/file
        file=file.resolve()
        # A standalone simulator must not depend on private World assets at runtime.
        if '/assets/reference_world/' in str(file) or '/world_experiment/' in str(file):
            raise ValueError(f'Unexpected World asset dependency: {file}')
        rel=Path('resources')/(sha(file)[:16]+'_'+file.name)
        dest=OUT/rel;dest.parent.mkdir(exist_ok=True);shutil.copy2(file,dest)
        el.set('file',str(rel));dependencies[str(rel)]={'source':str(file),'sha256':sha(file)}
    compiler.set('meshdir','.');compiler.set('texturedir','.')
    root.set('model','reconstructed_robot_sim_not_world')
    root.find('./visual/global').set('offwidth','640')
    root.find('./visual/global').set('offheight','480')
    root.find(".//camera[@name='stage3_camera_0']").set('name','overview')
    # Placeholder constraint disabled in Initial. M1/M2 activate only their learned
    # functional latch, not the private World's hidden bolt/coupling geometry.
    eq=ET.SubElement(root,'equality')
    ET.SubElement(eq,'weld',name='sim_latch_lever',body1='LeverDoor_Leaf',
                  body2='common_world_frame',active='false',solref='0.01 1',solimp='0.98 0.999 0.001')
    ET.indent(root); tree.write(OUT/'scene.xml',encoding='unicode')
    mechanisms=OUT/'mechanisms';mechanisms.mkdir(exist_ok=True)
    mechanism_sources={}
    for label,rel in [('m1','runs/m1_structure_20261109_175027/repaired_door.mjcf'),
                       ('m2','runs/m2_parameters_20261110_001716/repaired_m2.mjcf')]:
        p=ROOT/rel
        for suffix in ('','.repair.json'):
            src=Path(str(p)+suffix);dst=mechanisms/(label+'.mjcf'+suffix);shutil.copy2(src,dst)
            mechanism_sources[str(dst.relative_to(OUT))]={'source':str(src),'sha256':sha(src)}
    manifest={'role':'development_sim','source_scene':str(SOURCE),'source_scene_sha256':sha(SOURCE),
      'bundle_scene_sha256':sha(OUT/'scene.xml'),'dependencies':dependencies,'mechanisms':mechanism_sources,
      'provenance_note':'Visual Stage3 reconstruction with calibrated robot; M1/M2 seeds are from prior OBJECT-SPACE identification, NOT native double-door robot-contact identification.',
      'geometry_status':'Retained source coordinates and collision proxies; no ground-truth alignment or hidden bolt copied.',
      'adaptations':['relative resource paths','stage3_camera_0 aliased as overview','640x480 buffer','inactive functional latch slot'],
      'known_gaps':['Lever reconstruction contains hub geometry but no full lever bar.',
                    'Reconstructed environment poses differ from the reference World; registration is not corrected here.',
                    'Contact parameters and door-frame geometry have not passed Robot-Space Sim R1.']}
    (OUT/'provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'scene':str(OUT/'scene.xml'),'resources':len(dependencies),'role':manifest['role']},indent=2))

if __name__=='__main__':build()
