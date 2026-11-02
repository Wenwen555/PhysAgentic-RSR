# 首版 CaP：同一份 policy 的固定质量质心对照

本记录独立于代码 README。实验输出位于
[`initial_cap_single_policy_fixed_mass_20261101`](/data/wjt/PhyAgent-RSR/output/initial_cap_single_policy_fixed_mass_20261101/)。

## 实验协议

- 使用同一份冻结的首版 CaP，分别在质心 0 cm、−4 cm、+8 cm 的仿真场景中各执行一次，总共三次有效 rollout。
- 程序复用此前按时间最先生成的初版源码，没有针对配置重新生成或修订；本批次没有新的 Coding 调用。
- 总质量固定为 **0.270 kg**，质心处主惯量固定为 `[0.0000405, 0.00204525, 0.00204525] kg·m²`。
- 几何尺寸、摩擦 1.5、每指夹爪力上限 2 N、初始化种子 19001 保持一致。
- 使用显式 MJCF `inertial`，只改变物体系的质心坐标 `body_ipos`，不添加配重。
- 不运行 Active Probe，不把仿真结果反馈给 Coding Agent，不连接真机。

唯一源码：[policy.py](/data/wjt/PhyAgent-RSR/output/initial_cap_single_policy_fixed_mass_20261101/initial_cap/bundle/policy.py)。

SHA-256：`2bb9d9c4091c6b4ed4d50cf73ec6f57a26d9975ed316d91014ba9d560f2f094d`。

成功判定沿用原 Rod Lift 标准：杆中心高于桌面 4 cm、保持抓持、近似水平，并连续通过十个额外闭合夹爪控制步。

## 结果

| 质心 | 任务结果 | 观测最大升高 | 最终升高 | 最大 abs(axis_z) | 仿真时间 |
| --- | --- | --- | --- | --- | --- |
| 0 cm | 成功 | 10.23 cm | 9.91 cm | 0.0039 | 21.65 s |
| −4 cm | 失败，`not_lifted` | 4.56 cm | 0.14 cm | 0.2990 | 54.00 s |
| +8 cm | 失败，`not_lifted` | 4.50 cm | 0.14 cm | 0.2947 | 46.80 s |

两个偏心案例都曾建立抓持并抬起物体；约 4.5 cm 升高时，观测到倾斜超过 0.25，程序执行回落恢复，最终未通过保持判定。
末态 `not_lifted` 不表示全过程从未抬起物体。该程序重试时仍选择物体几何中心，未辨识质心或调整质心相关的抓取偏移。

![同一份 policy、固定质量的轨迹](/data/wjt/PhyAgent-RSR/output/initial_cap_single_policy_fixed_mass_20261101/trajectories.png)

上述最大升高和倾斜来自 API 边界采样，连线仅连接采样点，不代表连续控制步轨迹。
每个配置只有一次执行，本记录用于功能检查，不足以估计稳健成功率。

## 参数与源码核对

MuJoCo 编译参数读回确认：

- 三个场景的质量均为 0.270 kg。
- 三个场景的质心分别为 `[0, 0, 0]`、`[-0.04, 0, 0]`、`[0.08, 0, 0]` m。
- 三个场景的主惯量及惯性坐标系一致。
- 三次实际评估请求包含逐字节相同的 policy 源码，源码哈希与冻结记录一致。

证据：

- [原始结果](/data/wjt/PhyAgent-RSR/output/initial_cap_single_policy_fixed_mass_20261101/results.json)
- [完成状态](/data/wjt/PhyAgent-RSR/output/initial_cap_single_policy_fixed_mass_20261101/completion.json)
- [参数与源码一致性检查](/data/wjt/PhyAgent-RSR/output/initial_cap_single_policy_fixed_mass_20261101/protocol_checks.json)
- [冻结输入](/data/wjt/PhyAgent-RSR/output/initial_cap_single_policy_fixed_mass_20261101/frozen.json)
- [运行目录内报告](/data/wjt/PhyAgent-RSR/output/initial_cap_single_policy_fixed_mass_20261101/report.zh.md)

## 后续记录

违反同一份策略与固定质量要求的早期运行、已否定的保护性 Probe 分支及其专用报告和日志已删除。
本页的初版固定质量对照及全部原始任务失败保留。后续有效流程见 [通用物理辨识验收](2026-11-01-task-driven-identification.md)。
