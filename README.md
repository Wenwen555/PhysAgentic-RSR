# PhyAgent-RSR 工作区

| 目录 | 内容 |
| - | - |
| `agentic_rsr/` | 上游论文代码库 (git submodule -> github.com/HY-LiYihan/agentic_rsr；本地未提交的修改不在本仓库内) |
| `active_sysid/` | Robot-Space Sim2Sim 主线与原 Object-Space 实验：独立重建 Sim、机器人 Probe、CaP 开发、冻结 World 评测 |
| `preview_tools/` | 双门 World 预览与对比脚本 |
| `output/` | 重建/采集的中间输出 (体积大, 仅保留在本地, 不入库) |

新主线入口：[active_sysid/README.md](active_sysid/README.md) ·
[目录与角色](active_sysid/docs/REPO_LAYOUT.md) ·
[Robot-Space M0–M2 实施记录](active_sysid/docs/ROBOT_SPACE_M0_M2_IMPLEMENTATION.md) ·
[CaP 复用与独立验收](active_sysid/docs/CAP_PIPELINE_REUSE.md)

主线数据流：`agentic_rsr` 的视觉重建 → `active_sysid/assets/reconstructed_sim/`
→ Robot-Space Probe 证据 → M0-R/M1-R/M2-R 任务模型 → Sim 中开发并冻结 CaP
→ 独立 `assets/reference_world/` 评测。

原 Object-Space M0–M2 位于 `active_sysid/worlds/legacy_object_space/`，Panda 单门执行样例位于
`active_sysid/assets/fixtures/panda_single_door/`。这两者均不代表真实双门 World 的 Sim2Sim 结果。
