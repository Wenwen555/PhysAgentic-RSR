from pathlib import Path
import mujoco
import mujoco.viewer
model=mujoco.MjModel.from_xml_path(str(Path(__file__).resolve().parent/'scene.xml'))
data=mujoco.MjData(model)
mujoco.mj_resetDataKeyframe(model,data,model.key('home').id)
mujoco.mj_forward(model,data)
mujoco.viewer.launch(model,data)
