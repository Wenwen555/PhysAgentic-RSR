# M3-R 机器人执行 R0/R1（当前布局）

已完成原始双门 World 的机器人绑定、真实接触固定开门程序及逐步物理验收。

- **当前命令、实验结果、原 World 与 Panda 样例的区别：** [Robot execution layout](robot_execution_layout.md)
- **历史首次交付的 Panda 单门样例说明：** [Initial delivery archive](archive/m3r_r0_r1_initial_delivery.md)。该文件包含旧目录与运行命令，仅作历史留存，不要将其命令当作当前工作区入口。

当前 R1 主命令：

```bash
cd /home/wen/experiment/PhyAgent-RSR/active_sysid
MUJOCO_GL=egl .venv/bin/python -m active_sysid.robot world-baseline \
  --out runs/m3r_real_world/r1 --video
```
