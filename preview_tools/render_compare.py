import sys
from pathlib import Path
import mujoco
import numpy as np

def render(xml, out, cam=None):
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    if m.nkey:
        mujoco.mj_resetDataKeyframe(m, d, next((i for i in range(m.nkey) if m.key(i).name=='home'), 0))
    mujoco.mj_forward(m, d)
    r = mujoco.Renderer(m, m.vis.global_.offheight, m.vis.global_.offwidth)
    if cam is not None:
        r.update_scene(d, camera=cam)
    else:
        r.update_scene(d)
    img = r.render()
    import PIL.Image; PIL.Image.fromarray(img).save(out)
    print("saved", out, "| nbody:", m.nbody, "nq:", m.nq, "nu:", m.nu)

render(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv)>3 else None)
