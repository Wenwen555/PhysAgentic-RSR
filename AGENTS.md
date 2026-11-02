# 实验产物目录

所有新实验输出、运行日志、实验分析、图表和结果报告统一放在：
`/data/wjt/PhyAgent-RSR/output/<实验名>/`。

使用 `physrsr.paths.OUTPUT_ROOT` 或 `output_path()` 定位输出，避免依赖当前工作目录。
一次实验按 `task_context/`、`active_probe_1/`、`candidate_cap_1/`、后续修订等阶段组织，prompt、生成源码、观测、分析和日志保存在对应实验内。
源码 README 只写接口和使用方式，不写实验结果。

现有有效历史对照也已迁至 `/data/wjt/PhyAgent-RSR/output/`。任务失败本身不代表实验无效；不得因结果不佳删除有效对照。
