# Coding Agent：代码修订与失败归因小型审计

本审计只读取 `physics_cap_coding_final` 的已冻结代码、实际提示及完整 rollout：4 个物理 case、
每 case 2 次代码生成、2 个 Model 种子（8201/8202）、4 个独立 World 种子（9101–9104）。
没有新增 LLM 调用或物理执行，没有重选代码或改动原结果。样本较小，以下是逐试验核对，不是新的泛化结论。

**主要发现：6 次 AST 变化中只有 4 次改变控制行为；普通 Sim 修订在 Model 中修复 2 次失败，
在 World 中修复 0 次、破坏 0 次。Oracle Model 修订修复 1 次、破坏 1 次，净收益为零。**
反馈包含有用的末态，但会漏掉掉落前的过程，且存在诊断坐标系未标注一致的问题。

## Physics-informed 与 Physics-informed + Sim 的代码究竟有什么不同

两组是同一个 Astra low Agent，在完全相同 Probe 估计、模型先验和已生成 `physics_initial` 代码上分叉。
8/8 分支的最后一次调用之前，完整消息前缀逐项相同。最后一次调用分别给静态检查与 Model 执行反馈；
两组都有一次修订机会、相同控制 API 和运行限制，都可以在执行时 observe、纠偏、重抓。
因此 Physics-informed 本身也是反馈控制代码；区别是开发时能否得到内部仿真的执行证据。

最终源码有 7/8 对 AST 不同，case1/gen0 的最终 AST 相同。具体差异如下（case/gen 均从 0 编号）：

| case/gen | Physics-informed 最终代码 | Physics-informed + Sim 最终代码 |
|---|---|---|
| 0/0 | 改成每个保持步骤都观测，10 步；偏离即中止/重试 | 保留初始控制，12 次 close、间隔观测 |
| 0/1 | 中间高度判定从仅下界改为双侧误差约束 | 保留初始的单侧高度检查 |
| 1/0 | 重命名局部变量、移除未用初始化 | 同样整理，最终 AST 相同 |
| 1/1 | 单次抓取，最多 44 次运输/保持循环，连续保持计数 | 新增滑移检测、最多 3 次抓取与逐次降速；重抓时包含 25mm 抓点 |
| 2/0 | 保留初始抓取/抬升门限与控制 | 加 TCP 收敛检查，放宽初抬门限，再验证跟随、纠正高度 |
| 2/1 | 保持 12→10 步；恢复抓取后也使用短抬升 | 控制不变，仍保持 12 步，仅变量重命名 |
| 3/0 | **静态审查修正调平旋转方向** | 加到达与耦合检查，但**保留错误调平方向**，并改变速度/增量 |
| 3/1 | 保留三段抬升，保持判定改为已抬起且水平 | 改成五段绝对抬升、降速、固定保持目标并收紧跟随判定 |

最有解释力的例子是 case3/gen0。对正的杆轴竖直分量，调平四元数的旋转轴应为
`[-axis[1], axis[0], 0]`。Direct 静态审查修正了这个符号，Sim 根据末尾日志把问题理解成接近/抓取建立，
增加检查却留下相反方向 `[axis[1], -axis[0], 0]`。简单旋转代数核对（轴 z=.2，旋转 .1rad）得到
旧方向 z=.29682、修正方向 z=.10118，支持“旧方向会放大倾斜”的代码级判断；它不是新的 rollout。

对应源码：[Direct](/data/wjt/PhyAgent-RSR/output/physics_cap_coding_final/case_03_generation_00/physics_final/program.py)、
[Sim](/data/wjt/PhyAgent-RSR/output/physics_cap_coding_final/case_03_generation_00/sim_final/program.py)、
[实际收到的反馈](/data/wjt/PhyAgent-RSR/output/physics_cap_coding_final/case_03_generation_00/sim_final/messages.json)。
不能由“Sim 生成了更多条件判断”推断它修复了正确的问题。

## 6 次 AST 变化的语义分类与配对结果

下面比较共享 `physics_initial` → `sim_final`，不是把 Direct 最终代码当成修订前代码。
“参数/顺序/反馈”允许重叠；代码整理不算控制变化。

| case/gen | 修改类别及实质 | Model 成功数 /2，前→后 | World 成功数 /4，前→后 |
|---|---|---|---|
| 1/0 | 仅整理：`last_rise` 改名，删除未用初始化 | 2→2 | 4→4 |
| 1/1 | 参数+顺序+反馈：固定 attachment 参考、滑移门控、短抬升、最多三次抓取及降速 | 1→2 | 4→4 |
| 2/0 | 参数+顺序+反馈：初抬最小升高 12→4mm、容许相对位移 8→25mm；到达检查/重复收敛/高度纠偏 | 1→2 | 3→3 |
| 2/1 | 仅整理：`previous_com_offset` 改名；算式、门限、控制不变 | 2→2 | 3→3 |
| 3/0 | 参数+顺序+反馈：到达/跟随检查、未耦合禁止运输；运输增量 25→35mm，速度 .25→.3 | 1→1 | 0→0 |
| 3/1 | 参数+顺序+反馈：三段相对抬升→五段绝对 waypoint；降速、收紧 lag、保持目标稳定 | 1→1 | 3→3 |

另两份 case0 代码只改注释，AST 不变。因此“6/8 AST 变化”不能表述成“6/8 有效控制修订”。
逐份原始差异保存在各分支 `simulation_revision.diff`，人工语义标签与计数见
[机器可读报告](/data/wjt/PhyAgent-RSR/output/physics_cap_coding_revision_audit/report.json)。

| 配对对比 | 原成功仍成功 | 失败→成功 | 成功→失败 | 失败仍失败 |
|---|---:|---:|---:|---:|
| 普通 Model：初始→Sim 修订，16 对 | 12 | 2 | 0 | 2 |
| World：共享初始→Sim 修订，32 对 | 25 | 0 | 0 | 7 |
| World：Direct 最终→Sim 最终，32 对 | 25 | 0 | 1 | 6 |
| Oracle Model：初始→修订，16 对 | 11 | 1 | 1 | 3 |

普通 Model 被修复的是 **case1/gen1/8201、case2/gen0/8201**。前者的 World 初始代码本就 4/4，
没有待修复的 World 失败；后者的 World 失败种子 9103 修订后仍失败。
case1/gen1 修订的两个 Model 末态升高仅 78.2/76.2mm，虽然通过旧任务成功标准，仍未达到名义 100mm 目标。

相对共享初始，World 的 7 个失败完全没有变成成功，25 个成功也完全没有变成失败；
case3/gen1/9103 的失败标签从 `not_lifted` 变成 `not_level`，两者都失败，并非轨迹不变。
相对 Direct 最终代码少的那次成功是 **case3/gen0/9104**；它在共享初始中本来失败，
所以准确表述是“Sim 未获得 Direct 静态修订修复的成功”，不能说“Sim 破坏了共享初始的成功”。

普通 Sim 的 Model SR 为 12/16→14/16，但 World 仍为 25/32；
Direct 最终为 26/32。Sim 相对共享初始平均执行时间 +2.281s，相对 Direct +5.231s。
后一个成功率差 −.03125 的既有 case-cluster bootstrap 95% CI 为 [−.09375, 0]；
四个 case 不足以宣称 Sim 普遍有害。

## 仿真反馈是否明确指向真实失败原因

核对的是 `sim_final/messages.json` 中 Agent **实际收到**的内容，不是事后拥有的完整轨迹。
执行器只发送最后 4 个 phase 样本、末态升高、终局失败标签与抓取摘要。

| 修订前失败，均为 Model seed8201 | 完整日志支持的过程 | 反馈质量与 Agent 归因 |
|---|---|---|
| Sim 1/1 | 最高升高 109.7mm，倾斜分量最高 .753，之后掉回 1.4mm、失去抓取 | 末尾全是未抓住/已落桌；Agent 推断保持丢失与完整过程一致，但反馈没给出失稳起点 |
| Sim 2/0 | 第一抓未建立；第二抓已接触，最高升高 9.1mm、末态 8.4mm，程序结束 | 小升高与抓取末态支持检查提前拒绝；“接触 seating”仍是假设，不能由日志唯一确认 |
| Sim 3/0 | **确实抓住并抬升 99.3mm**，倾斜逐渐增大到 .493 后掉落 | 摘要 `grasp_established=false`，末尾全为未抓住；Agent 改进接近/抓取，与主要已记录失稳过程不符 |
| Sim 3/1 | 已抬升 100.8mm，倾斜 .653 后掉落 | 最后四点含掉落前样本，Agent 的“抬起后丢失保持”有直接证据；根本物理/控制诱因仍未唯一确定 |
| Oracle 3/0 | 末态仍抬升 100.5mm 且抓住，但倾斜分量 .690 | `not_level` 与轨迹一致，代码也确有反向调平；这条诊断较明确，但后续修改未修复同种子结果 |

`not_lifted` 表示终局条件没有满足，不能解释成“从未抬起”。
`grasp_established` 汇总与 phase 名称来自旧模板的启发式跟踪器；重抓、反复 close 会使摘要不能代表整个程序。
例如 Sim 3/0 摘要为 false，但较早样本明确记录了双指接触、抓住及升高。

另有一个确定的反馈封装问题：归档实现的 `phase_tracker.rod_position` 直接读取 MuJoCo world 坐标，
而 `achieved` 来自机器人基座坐标 TCP；`model_feedback` 将两者原样放在同一条样本里，未注明变换。
实际 Agent 控制 API 的 `observe()` 则将杆与 TCP 都表达在基座坐标，控制接口本身没有这个混用。
因此反馈中的两列绝对位置不能直接相减解释滑移；升高等同帧时间差和 tilt 仍可使用。
本审计记录问题，保留原提示与结果，没有事后修正日志重评分。

结论是：反馈通常报告了终局症状，有几次支持正确的过程判断，但**没有稳定地明确指出根因**。
尤其不能依据尾部低高度/未抓住，给早期接近控制或物体摩擦作确定归因。

## Oracle：物理参数准确，为何仍没有净收益

先界定证据边界：**Oracle 初始代码没有 World rollout**，只有初始/最终的配对 Model rollout 和最终 World 结果。
所以无法从现有日志计算“Oracle 修订前后 World 增益”；Oracle 最终 26/32 与 Direct 的 26/32 相等，
也不是 Oracle 修订无效的配对因果估计。可以确定的是 Oracle **Model 修订净收益为零**。

| Oracle case/gen | Model 前→后 /2 | 修订及诊断 | 最终 World /4 |
|---|---|---|---:|
| 0/0 | 2→2 | 仅注释 | 4 |
| 0/1 | 2→2 | 用 .30 常数替代本来就求值为 .30 的速度表达式，控制不变 | 4 |
| 1/0 | 1→2 | 加 attachment 检查、慢速重抓和保持反馈；修复 8201 | 4 |
| 1/1 | 1→0 | 加滑移/失耦合退出、重抓与固定保持；**破坏 8202** | 0 |
| 2/0 | 1→1 | 到达收敛、接触等待、第二抓加深 3mm；8201 仍从未建立抓取 | 3 |
| 2/1 | 2→2 | 仅注释 | 4 |
| 3/0 | 1→1 | 修正调平符号，同时更改速度、抬升幅度和保持循环；8201 从倾斜失败变成掉落失败 | 3 |
| 3/1 | 2→2 | 仅注释 | 4 |

Oracle 1/1/8202 是最明确的代码退化。修订前成功；修订后第一次抓取仍将杆抬到 100.1mm，
倾斜始终很小（最高 .017），并保持抓住。固定保持时缓慢下沉，超过 4mm 名义高度容差会清零保持计数；
一次高度纠偏后又从头计数。源码 `for step in range(21)` 耗尽后没有保护已建立的抓取，
直接进入第二次 attempt 的 `open_gripper()`。完整日志显示随后杆落桌、第二次未抓住，最终失败。
这支持“保持判据/循环限额与无条件重抓冲突”的程序原因，而不是未知 CoM 导致首次抓取不可行。

证据：[Oracle 修订代码](/data/wjt/PhyAgent-RSR/output/physics_cap_coding_final/case_01_generation_01/oracle_final/program.py)、
[修订前](/data/wjt/PhyAgent-RSR/output/physics_cap_coding_final/case_01_generation_01/oracle_before.jsonl)、
[修订后](/data/wjt/PhyAgent-RSR/output/physics_cap_coding_final/case_01_generation_01/oracle_after.jsonl)。

准确的动力学允许执行暴露错误，却不会自动保证 Agent 的修订正确。本协议只有一次修订；
即使 after-Model 已显示 Oracle 1/1 退化到 0/2，最终仍按预定规则执行修订代码，**没有接受/回滚门控**。
多个控制因素同时修改，也不能把 Oracle 3/0 的结果单独归因于符号修正。
2 个开发 Model 种子还不足以证明对独立种子的可靠性；现有证据不能断言搜索越多越容易利用仿真误差。

## 冻结与复现

保留当前负结果，不推进 Active Probe 或扩展搜索。下一轮若重启，应先修复反馈坐标标注、记录首次失稳窗口及
累计抓取历史，并在独立开发种子上检验修订接受/回滚和上述循环退出问题。
这些是后续实验建议，本审计没有修改控制器、提示、选择规则或重跑旧评估。

一次性审计脚本仅保存在输出目录，没有增加常驻 audit 模块或测试文件：

```bash
cd /data/wjt/PhyAgent-RSR-v1/src/cap-x
python3 outputs/physics_cap_coding_revision_audit/audit.py
```

脚本逐项检查 8 个相同生成前缀、160 条 World 记录的源码 hash、64 条 Model 记录、完整配对键与实际发送的尾部样本；
核对并确认 145 个输入文件 hash 在审计期间未变。配对计数含明确断言。
执行时源码以原运行 `source.tar.gz` 和 config hash 为准；当前 `experiment.py` 比归档多了异常中止处理，
不会将这一后续变化当成旧运行的实现证据。其他被核对的观测/跟踪代码与归档相同。

交付：[audit.py](/data/wjt/PhyAgent-RSR/output/physics_cap_coding_revision_audit/audit.py)、
[report.json](/data/wjt/PhyAgent-RSR/output/physics_cap_coding_revision_audit/report.json)。
语义类别为人工源码审查结论；自动 AST 比较、成功计数与轨迹摘要是可重算的事实。

后续 [M2.1-S](2026-11-01-m21s.md) 另做了新的受控实验，补齐了单一已知 Oracle case1/gen1 的
World 修订前后反事实（3/4→0/4），并验证 gate 保留旧程序。本文“Oracle 初始 World 缺失”的判断
描述当时冻结日志的范围；新数据没有回填或改写历史结果，其余 Oracle 初始程序仍未完整配对评分。
