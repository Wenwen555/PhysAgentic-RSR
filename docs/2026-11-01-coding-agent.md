# PhysRSR：真实 Codex Coding Agent 四组实验

## 实现范围

新增 `physrsr/coding/`，保留冻结的模板基线和结果，不在旧候选池中选ID。
使用本机 Codex CLI 0.166.0，明确调用 **gpt-6-astra / low**。已真实完成生成和代码执行，
不是 mock，也不是由宿主人工替换生成代码。测试中的mock仅用于检查配对协议。

`CodexAgent.generate(messages, destination)` 返回完整Python与rationale，保留请求、响应、代码、
事件、耗时及token。登录使用本机已有Codex认证；不复制凭据进入代码／实验归档。
Codex CLI在干净命名空间运行，仅挂载可执行文件、系统依赖、认证及当前临时工作目录；
不挂载仓库、实验数据、用户config／skills，禁用native shell/apps/multi-agent/code-mode。
模型调用失败则中止，不换模型、不制造程序。官方接口参考：
https://developers.openai.com/codex/noninteractive/ 。

## 代码与物理信息配对

每case/generation先生成一份无物理信息的公共初始程序。
从同一初始程序做3种适配：无物理静态检查、辨识物理信息、Oracle物理信息。
**辨识后的程序和完整上下文逐字节复制为 Physics-informed 与 Sim-CaP 的共同起点。**
最后四组各有1次生成／修订调用：

|方法|信息|最终调用|
|---|---|---|
|Vanilla CaP|无Probe或GT物理参数|静态代码审查|
|Physics-informed CaP|共同辨识θ、不确定支持和已知模型先验|静态代码审查|
|Sim-CaP|完全相同的θ、支持、先验及初始代码|内部Simulator执行反馈|
|Oracle Sim-CaP|参考真θ及相同格式先验|Oracle Simulator反馈|

每条方法路径均为3次同模型同effort调用；由于共享前缀，每case/generation总计8个逻辑调用节点。
不声称实际token数相等，全部记录实际用量。每组均可用正常在线物体跟踪、本体观测和条件纠偏，
Vanilla没有被剥夺反馈能力。Oracle只改变物理信息／内部Model，不增加控制权限。

四组共享 `run(robot)` 接口，支持Python标准库、函数、循环、分段运动、反馈判断和重抓。
真实代码经 `robot.observe/open_gripper/close_gripper/goto_pose` 执行；无模板参数覆盖生成结果。
提供的pose/TCP均为机器人基坐标，物体长轴方向和抓取四元数为通用几何观测，不含CoM／摩擦。

## 执行隔离与回归

生成Python在单独bubblewrap PID/user/network/filesystem命名空间中执行，不能看到仓库、认证、
MuJoCo或隐藏参数，不能连接宿主网络；唯一控制入口为白名单RPC。
所有组同样限制workspace、四元数、速度、96次RPC、5秒guest CPU、120秒host墙钟。
Python random固定seed为0；隔离解释器不保证任意hash-order／熵源／clock驱动程序确定性。
本批实际生成代码仅使用math，完整物理重放另行验证。
World与Model使用不同spawn进程池。模型反馈只给任务／观测字段，不给编码物理条件的case_id。
错误分别记录program与infrastructure；后者中止实验，不能变成任务失败。

保留IK cold start、环境与placement sampler seed、hard-reset物理恢复和稳定hold判据。
真实Codex代码在mu=.7/2.0下做4次QC rollout：实际rod–finger混合摩擦正确；同seed完整物理记录
及允许的跟踪观测重放一致。QC代码与4条完整记录归档在 `outputs/physics_cap_coding_quality/`。

## 协议版本

正式版本补全了全部组一致的 episode horizon 处理及物理先验信息配对。
已废弃的 pilot、中止运行及物理信息不匹配版本的输出和日志已删除；正式版本自含其实际使用的 prompt、响应和源码副本，原复用来源记录仍保留。

## 正式评估协议

4个连续正常配置 × 2次独立代码生成 × 4个World测试种子。
配置来自固定8个正常+2个低摩擦负对照序列的前4个；本轮**未覆盖后两个负对照**，不能泛化到完整参数域。
Probe=8901，Model=8201/8202，World=9101–9104，离线共同辨识库seed41。
每份Sim/Oracle代码均在修订前后各运行两个Model种子；无额外World任务反馈或模型调参。
所有程序生成／修订和代码哈希先冻结，才开始统一World评分。

主四组各32条World rollout；额外32条评估共同的 identified initial 程序，区分“额外静态review”
与“仿真修订”增益，共160条。另有8条World Probe、64条Model任务rollout。
成功率按意向执行统计，程序error算未成功但另报error类型，物理fail独立记录。
case级bootstrap，不能把同一物体的两个代码生成或多个seed当作独立物理case夸大显著性。

主成功条件保持旧Rod Lift：中心高于桌面.04m、grasped、level并连续保持10步。
.10m为名义搬运目标，另记录真实rise／target error；不能把legacy SR直接称为严格10cm成功率。
循环／重抓的旧phase归因只作启发式，需结合原始状态、代码和日志判断真正失败原因。

运行命令和接口说明见 `physrsr/coding/README.md`。

## 正式结果（信息匹配版本）

原始结果：`outputs/physics_cap_coding_final/report.json`。以下均为 legacy stable-hold SR，
不是“所有trial严格抬升10cm”的成功概率。

|方法|World SR（32 trials）|case bootstrap 95% CI|平均执行物理时间s|平均rise m|
|---|---:|---|---:|---:|
|Vanilla CaP|.21875|[.03125,.50000]|83.009|.0350|
|Physics-informed CaP|.81250|[.62500,1.00000]|42.538|.0892|
|Sim-CaP|.78125|[.53125,1.00000]|47.769|.0845|
|Oracle Sim-CaP|.81250|[.59375,.96875]|42.564|.0819|
|Shared identified initial（阶段诊断）|.78125|[.53125,1.00000]|45.487|.0831|

- Physics − Vanilla = **+.59375 [.37500,.81250]**；32个配对trial中救回20个，损失1个。
- Sim − Physics = **−.03125 [−.09375,0]**；救回0个，损失1个。
- Sim − shared identified initial = **0 [0,0]**；25个共同成功、7个共同失败，没有trial状态改变。
  这个退化CI不是总体等效证明。
- Sim − Physics执行时间 = **+5.231 s [.806,12.350]**。没有观察到可靠性或效率的额外收益。

World全160条和Model全64条均无程序error或基础设施error。所有组可使用实时反馈。
Oracle是信息真值条件，**不是全局最优策略上界**，因此不必优于另一个随机生成的代码。
四case、两代码重复及四初始种子不足以证明普遍LLM能力或≥.95可靠性；bootstrap条件于当前生成和状态。

### 代码是否因物理信息／仿真反馈而改变？

- 8/8份physics_initial均不同于同公共起点的无物理review结果。
  常见变化是把首抓点从几何中心／通用探索改为 `centre + com_offset * rod_axis`，
  并根据质量、夹持命令和摩擦余量选择分段慢速抬升／条件纠偏。
- Sim修订8/8文本发生变化，**6/8 AST发生变化**；2/8只是文本变化，不能称为逻辑修订。
  AST变化也不自动等于有益的语义变化，以下实例需结合原代码和diff。
- case1/gen1：Model初始1 fail+1 success，修改为短抬验证attachment、检测相对滑移，避免把滑移
  吸收到attachment估计后继续向上追赶，并允许两次较慢重抓；循环2→3、条件3→7。
- case2/gen0：Model初始1 fail+1 success，放宽初始接触settling后再要求持续随动；条件7→12，
  函数4→5。Model中的成功率升至2/2，但本批独立World状态并未增加成功trial。
- case3/gen0：增加实测TCP到位检查、接近修正和物体随动验证，Model仍1 fail+1 success，
  World仍0/4。这里CoM估计偏差约−17.75mm，错误的模型／任务信息和代码逻辑都不能仅靠这轮反馈修复。

总体Sim程序的Model SR由 .750升至 .875，而World shared-initial与最终Sim同为.78125。
**模型内程序表现改善不等于World收益。** 差异文件保存于每case/generation目录的
`physical_information.diff`、`simulation_revision.diff`，完整代码和rationale均保留。

### 模型与代码修订的不同局限

最终Sim程序的Model→World成功率MAE=.15625；被模型判2/2通过的6份程序，其World trial失败率8.33%。
Oracle对应MAE=.0625，5份2/2通过程序没有World失败；这些仍是小样本经验结果，不是可靠性证书。

case3的CoM估计误差−17.75mm且支持范围不能覆盖真值；Model反馈无法凭自身执行发现World模型失配。
另一方面，Oracle case1/gen1的修订使Model由1/2降至0/2，World也为0/4，说明拿到真实参数也不保证
Coding Agent在一次修订预算内产出正确控制代码。本协议按预定规则执行最终revision，
**没有根据after-model结果回滚／选择旧代码**，不能事后把更好的版本替换进主表。

事后 evaluator-only 阳性对照：4个case均采用准确CoM、speed=.35的旧固定控制，在各前两个配对种子上
为2/2（共8次）。它不是新增LLM组、不进入提示或主评分，仅说明这些配置至少存在可行控制，
Oracle LLM不是数学上界。不同seed数量也不支持把该8/8直接当作与32-trial LLM组的显著差异。

### 成本与质量检查

每个需辨识的对象共享两个Probe，平均物理交互时间24.338s。
单次部署按Probe+任务记：Vanilla83.009s，Physics66.875s，Sim72.106s，Oracle42.564s。
Oracle参考免费获得真值，不能当作可部署的低成本方法。
每程序平均RPC次数依次58.81、52.59、47.34、50.88；close计数包含hold，不等于独立抓取次数。

正式结果包含64个逻辑Codex节点：40次新推理、24份匹配输入缓存；每方法路径仍有3个相同模型／effort的
推理slot。逻辑token为 input989399/output81477；本次新增input640968/output53101，
其中cached-input440320。reasoning-output为output子集，不重复累加。美元费用不可获取，记null。
开发／中止运行的独立输出和日志已清理；其历史成本不合并到正式 SR，也不作为“免费”推理隐去。

`protocol_checks.json` 验证8组共享前缀、实际源代码hash、统一guest版本、统一Astra/low/CLI0.166.0；
原生CLI工具调用为0。CLI每次记录一个“Code Mode host disabled”的能力通知（事件type=error），
这是主动禁用工具的非致命通知；64次生成均returncode0并返回有效schema，不是实验错误。
另外4对AST相同的before/after Model执行在相同seed上物理记录逐项相同。
真实Codex代码的4次QC同时验证实际接触摩擦和完整状态／观测重放。
最终相关回归测试 **26 passed**（30.32s，2项既有依赖弃用警告）。

## 本轮结论

已建立真正的LLM代码生成和仿真反馈修订闭环，验证物理信息包对这批程序生成有明显收益。
物理信息包包含Probe估计和共同模型先验，本实验没有分别消融CoM、摩擦与先验贡献。
Sim反馈会改变操作逻辑，但本轮没有额外World成功率收益，还增加执行时间。
负结果不能被代码变长、AST变化、Model成功率提高或Oracle标签代替。

下一步优先考虑独立开发验证下的修订接受／回滚机制、参数不确定校准及Coding Agent控制正确性；
需要新测试协议才能评价这些改进。尚未检验Active Probe，不推进额外复杂Agent架构或RL训练。


交付版另补强了失败中止：一个并行生成分支失败时，取消待运行分支，并在后续推理／Model调用边界
停止，避免异常后继续排队消耗预算；相关回归测试已覆盖。该变化只涉及错误路径，正式数据无错误。
报告源代码后续改动仅补充统计／验证，不改变已执行的策略、物理或选择。


补充的一次调用Vanilla诊断（事后执行全部8份已冻结initial代码，不新增推理、不改变主表）：
32个World trial成功率 **.25000 [.03125,.59375]**，Physics相对它为
**+.56250 [.34375,.78125]**。因此主表 .21875 是三调用预算匹配版，不是把单次初始程序混称为同一条件。
该补充只用于检查盲目self-review是否改变基线；所有代码在正式World评分前已经冻结，不按成绩选择。
补充32次World执行和阳性对照8次、QC4次另记，不计入主实验160次。

补充只读[代码修订与失败归因审计](2026-11-01-coding-revision-audit.md)：6 次普通 Sim AST 变化中
4 次改变控制行为；Model 修复 2 次失败，World 修复/破坏均为 0。Oracle Model 修复 1 次、破坏 1 次，
保持计数与重抓循环冲突造成明确代码退化。反馈截断会掩盖已抬起后掉落的过程，诊断样本还混用了
world/base 坐标。Oracle 初始代码没有 World 评分，不能补称已有 Oracle 修订前后 World 因果对照。
本次不新增执行或推理，不改变冻结代码和原结果。

后续 [M2.1-S 协议重构与独立评估](2026-11-01-m21s.md) 已在独立目录完成：
共享静态审查、base-frame 控制步反馈、独立验证与回滚均已实现。新独立集四组均 15/32，未获得额外 World 修复。
该新研究另补齐了已知 Oracle case1/gen1 的配对 World 反事实：旧修订前 3/4、修订后 0/4。
这些新增数据不改写本文的旧四组结果或当时的审计数据范围。
