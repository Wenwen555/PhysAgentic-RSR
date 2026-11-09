# PhyAgent-RSR 工作区

| 目录 | 内容 |
| - | - |
| `agentic_rsr/` | 上游论文代码库 (Real2Sim 重建与场景导出, 独立 git 仓库, 勿混入新代码) |
| `active_sysid/` | 新研究框架: Task-Driven Active Physical System Identification (主动物理辨识闭环) |
| `preview_tools/` | 双门 World 预览与对比脚本 |
| `output/` | 重建/采集的中间输出 |

新框架入口: [active_sysid/README.md](active_sysid/README.md) ·
研究方案: [active_sysid/docs/research_plan.md](active_sysid/docs/research_plan.md)

数据流: `agentic_rsr` 导出重建 MJCF → `active_sysid/worlds/reconstructed/`
→ 主动 Probe 辨识 → 修复后仿真 + CaP → `agentic_rsr` 的 World 验证。
