# 通用物理辨识与共享策略

自动入口已统一为 `run`，模型提交与 v2 服务启动见 [自动闭环](architecture.md)。
下列分阶段函数仍用于单阶段检查与显式修订；自动状态机使用相同的 Physic/Coding API 和执行合同。

Physic Agent 根据语义目标、视觉姿态/几何和机器人能力，选择影响任务决策的因素。
无需初版 CaP 或未辨识 Simulator 的预测。这里的视觉输入是感知 API 的结构化输出，不是直接输入 RGB 图像。
通用 prompt 不包含 benchmark 名称、预设待辨识参数或任务专用动作模板。
任务适配器负责感知字段与原始成功判定；实验设计和估计代码由 Agent 生成。

## 数据流

1. `physic.design_identification` 输出因素排序、受控变量、多个实验设置、测量方程、可辨识分量、误差区间计算与完整 `run(robot)`。
2. `target.execute_probe` 执行冻结源码，保存动作参数和实测 API 边界观测。空载校准动作与物体交互分别统计。
3. Probe 返回带来源的参数估计。控制失败状态和已获得的有效物理证据分别保留，每个实例独立存储。
4. `target.update_identification` 写入当前实例的估计记忆；生成程序通过 `robot.describe()['identification']` 读取。
5. `agent.synthesize_identified_policy` 生成同一份任务候选。控制精度、名义航点和任务目标区域分别处理，候选通过实际执行验证。
6. 仍有物理歧义时使用 `identification redesign`；参数已有足够证据而失败来自控制/目标处理时使用 `identification revise-task`，避免无必要地继续 Probe。

`run --cases` 首先由 Physic Agent 判断逐实例证据需求，再选择交互或任务程序合成。
有可用证据的实例可以先生成并验证共享候选，不等待所有其他实例完成辨识。
源码修复不自动执行，后续交互须有独立的决策与成本理由，详见 [实验调度合同](exploration.md)。
每个实例保留独立的 `estimates`、最新 `trials` 和带证据来源的 `trial_history`，
后续恢复失败不会抹掉此前有效测量；读取历史估计不构成新的独立测量。
准备新 episode 后仍可根据此前失败设置继续搜索，不能跨实例混合参数。
批量入口只在全部配置的原 World 判定成功后生成 `final_cap/`；部分执行记录不能触发交付。
已有任务候选时，新增 Probe 后仍保留候选源码和原失败轨迹，将新测量一同交给 Coding Agent 修订。
程序错误（例如返回值超过协议限制）保留为 `error` 并进入修订；基础设施错误独立报错，不计为普通物理失败。

参数名和估计模型是生成结果。代码不按任务名称分发质心搜索、摩擦测试或其他固定辨识器。
`examples/identification_acceptance.py` 是 benchmark 数据准备，独立于 Agent 与通用工作流。

## 公开 API 与本体信息

`physrsr.robot.v2` 使用 `observe`、`describe`、`goto_pose`、`hold`、`checkpoint`、`open_gripper`、`close_gripper`。
完整合同位于 `core/identification.py`。v1 的字段和预算保持兼容。
隔离执行的协议行上限为 64 KiB；生成程序返回值以 8 KiB 以下的状态、所选参数和少量标量诊断为目标。
不要回传完整观测或全部历史记忆，完整 API 轨迹由宿主单独存档。

`observe` 返回物体完整姿态和几何、实测 TCP、腕部力/力矩、关节位置/力矩、夹爪开度和传感时间。
力/力矩在 `robot_base` 表达、关于实测 TCP；工具自重包含在原始信号中。
静态反演需要同姿态空载去皮、静置和合理的支撑状态证据。

`describe` 返回重力、工作空间参考点、夹持力上限、指垫几何、关节力矩限位、传感误差分析下限与运动合同。
力上限不等于实际接触法向力，关节力矩限位不等于夹爪可传递力矩容量；未辨识的容量显式保留为未知。
工程误差区间不声称是统计置信区间。

运动合同公布定位/姿态容差及有限次实测纠正。
Franka 使用显式精度配置向相同的仿真/真机关节控制接口传递收敛容差；其他调用者保留原默认值。
已知的数值未到位反馈到生成程序作有界恢复，通信/感知基础设施错误独立处理。
保持动作采集实测关节配置并推进控制时间，重复 `observe` 不推进仿真时间。

## 仿真与真机接口

`SimulatedMeasuredRobot` 是现有 benchmark 的传感适配器，读取真正的仿真腕部传感器。
它和 `InstrumentedWorld` 位于 `tasks/rod_lift_world.py`；通用 `runtime/instrumented.py` 只维护共享执行与真机接口。
split stepping 后刷新传感计算，不推进时间，不用隐藏质量或质心构造信号。
测量 TCP 的坐标轴与命令 TCP 保持一致，保留腕部测量到 TCP 的力矩参考点转换。

真机使用：

```python
robot = MeasuredRobot(control, perception_api, proprioception_api, setup_api, capabilities)
target = InstrumentedTarget(robot)
```

- `perception.object` 提供实测物体姿态、几何、支撑面距离与采集时间。
- `robot.state` 提供同步的实测 TCP、关于 TCP 的腕部力/力矩、编码器和夹爪信息。
- `robot.hold` 由真实控制器保姿并推进实测时间；不会以 sleep 伪造动力学保持。
- `episode.prepare` 明确确认支持起点、就绪状态及不变的夹持力上限。

共享执行器在准备后以真实 `hold` 和观测确认物体、TCP、夹爪连续稳定，记录设置成本与诊断。
等待超过有限次数仍未就绪会明确报错；不把 reset 调用返回等同于执行器已静止。
恢复到任务允许的静止支撑区域即可，是否需要精确回原位由任务约束决定。
零去皮腕力不能作为所有接触交接的统一前提，释放仍须满足执行器的支撑检查。

`target.execute_probe` 与 `target.execute_policy` 走同一隔离执行器，使用相同源码与 API。
执行完成不等同任务成功，独立评分保留在 World/任务评估服务中。
缺失真实传感或设置服务时显式失败，不使用仿真数据或默认假测量代替。

## 替换任务环境

每个 case 可提供兼容的 `endpoint`。工作流仅调用公开目标/World API，不向 Agent 传递私有配置。
没有外部 endpoint 时，本仓库通过私有配置启动已有的 Rod Lift World。
其他任务需要自己的感知和成功判定适配器，通用 Physic prompt、schema 和工作流无需修改。

分阶段和批量 World 流程将估计交给策略消费者；`ActiveLoop` 单实例模式另外重放拟合证据、提交 Simulator 参数并检查新动作预测。
`run --cases` 当前检验的是实测反馈下的共享程序闭环，不包含独立的预测 Simulator 更新/验证；不能用其 World 成功替代该项验证。
任务候选的有限次成功执行仍不能替代模型预测能力的独立验证。
