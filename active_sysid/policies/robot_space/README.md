# Robot-Space policies

策略在独立 Sim 中开发，不在原始 World 中开发。入口是自包含文件 `run(api)`；项目 helper 和 World 文件不会挂载到策略进程。这里尚未编写开门 CaP。

请先阅读 `active_sysid/docs/ROBOT_SPACE_CAP_HANDOFF.md` 和 `SIM_WORLD_BOUNDARY.md`。三个 Stage 使用 `python -m active_sysid.robot.stages ...`；许可 World 取证和冻结迁移使用 `python -m active_sysid.robot.sim2sim ...`。

注意：`finger_contact_normal_n` 是指端对所有外部接触的合计，不包含“已抓到目标把手”的标签。World 始终不提供 `privileged`。

当前新增两个固定验证程序：`reference_initial_direct_pull.py` 对应初始 Sim 的直接拉动先验；`reference_grounded_turn_pull.py` 对应经过 Probe 更新后的旋转—拉动序列。它们用于验证执行与迁移，不是通用视觉 CaP。未来 CaP 默认在 `robot_m2` 开发；全过程参见 `docs/ROBOT_SPACE_M0_M2_IMPLEMENTATION.md`。
