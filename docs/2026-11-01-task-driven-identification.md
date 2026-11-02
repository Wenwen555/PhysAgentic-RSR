# 通用物理辨识与共享 Policy 验收

高层流程已改为：从公共任务与证据选择待辨识因素，生成多设置实验与数值估计代码，将当前实例估计写入 API 记忆，再合成、执行和修订同一份任务策略。

通用 Physic prompt 没有 Rod Lift 专用规则、质心搜索模板或真实参数。模型生成了多抓点、同姿态空载去皮、静态力/力矩反演与跨抓点复测代码，使用夹持力上限、指垫几何、关节力矩和实测腕部载荷。

| COM | 实际物体抓取次数 | 估计 COM | 最终 World 结果 |
| --- | ---: | ---: | --- |
| 0 cm | 2 | +0.004 cm | success |
| −4 cm | 4 | −3.996 cm | success |
| +8 cm | 3 | +8.003 cm | success |

质量 0.27 kg、COM 坐标系惯量、摩擦、夹爪力和 seed 固定。三个场景共享同一 Probe 和同一最终 Policy。最终任务标准沿用原判定。

开发期还修复了测量坐标、传感刷新和运动精度，并依据已知配置的任务反馈修订了控制逻辑。因此这是功能验收，不能把成功全部归因于 Probe，也不是独立配置泛化结果。推移与重定向仅做了相同提示下的另一类 Probe 生成检查；真机未执行。

- [验收报告与保留证据](/data/wjt/PhyAgent-RSR/output/task_driven_identification_20261101/report.zh.md)
- [最终共享策略](/data/wjt/PhyAgent-RSR/output/task_driven_identification_20261101/final_cap/policy.py)
- [生成的多抓点 Probe](/data/wjt/PhyAgent-RSR/output/task_driven_identification_20261101/active_probe_2/probe.py)
- [通用提示模板](../physrsr/active/prompts/physical_identification.txt)
- [通用实现说明](../physrsr/active/docs/identification.md)

结果单独存档，未写入源码 README。
