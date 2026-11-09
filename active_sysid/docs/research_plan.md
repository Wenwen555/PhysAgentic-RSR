# 研究方案: Task-Driven Active Physical System Identification for Interactive Simulation Reconstruction

目标不是把重建场景的所有物理参数恢复到真实值, 而是让 Agent 根据 CaP 的
执行需求, 主动设计能够区分物理机制、辨识关键参数的交互实验, 并利用机器
人本体感受更新可交互仿真, 使其能预测真实任务中的动作结果。

## 1. 研究问题

> How can an agent actively discover and identify task-critical physical
> mechanisms missing from a reconstructed simulator, using minimal robot
> interactions to build a simulation that reliably predicts downstream
> code-based manipulation?

## 2. 两类物理误差 (双门场景)

- **Structural Model Error**: sim 缺少闩锁、弹簧及把手-门栓耦合。即使把
  质量、摩擦全部优化正确, 门也仍能在未解锁时被拉开。
- **Parametric Model Error**: sim 具有门铰链, 但门质量相差 48 倍、阻尼
  不同、限位错误。能产生类似运动, 却不能正确预测完成动作所需的力与
  动态响应。

CMA-ES 定位为**连续参数辨识器**, 而非唯一的物理辨识器。

## 3. 辨识链路

```
Task + Reconstructed Simulator + CaP   当前任务/场景/程序轨迹与失败信息
        ↓
Agentic Probe Planner                 提出假设 · 选择辨识属性 · 评估信息增益
        ↓
Probe Library                          Pull · Rotate · Release · Sweep · Perturb
        ↓
Probe Generator                        组合、实例化与调整实验动作
        ↓
World Interaction                      Joint Pos / Torque / Contact / Vision
        ↓
Mechanism Inference                    闩锁、弹簧、隐藏约束、运动模式
        ↓
CMA-ES System ID                       质量、惯量、摩擦、刚度、阻尼
        ↓
Simulator Repair + CaP Validation      更新 MJCF → 重放 → 预测未见交互 → 验证
```

核心设计: Probe 不仅用于估计参数, 也用于发现当前仿真**没有表示出来的**
物理机制。

## 4. 相关工作

| 工作 | 已解决 | 对本项目的价值 |
| - | - | - |
| Ditto (CVPR 2022) | 交互前后点云恢复部件几何/关节类型与轴线 | 新关节结构假设 |
| ASID (ICLR 2024) | 主动探索收集高信息量数据辨识参数 | Probe 信息增益优化 |
| PhysCaP (2026) | Planner 决定探索与停止, 本体感受测质量/刚度 | Agentic Planner + CaP 接口 |
| ForceTwin (2026) | 交互姿态与力辨识惯量/摩擦/阻尼及机构动态响应 | 门机构动力学辨识 |
| EMPIRIC (2026) | 实验识别引擎缺失机制并写成可执行模型 | 结构性误差修复 |
| SPI-Active (CoRL 2025) | CMA-ES 匹配真实/仿真轨迹 + Fisher 信息设计探索 | 优化器与实验设计 |

另有 Sturm 等 *Learning Kinematic Models for Articulated Objects* 的候选
运动学模型比较, 以及 ICRA 2015 *Task-oriented Planning for Manipulating
Articulated Mechanisms under Model Uncertainty* 的模型区分与任务执行结合。

差距: 四篇核心工作没有完成"任务驱动结构发现 + Probe Library + 机构修复
+ CaP 验证"的完整闭环。贡献组织为:

1. **Task-Driven Mechanism Discovery** — 从 CaP 执行失败出发, 主动发现
   重建仿真缺失的隐藏机构;
2. **Mechanism-Aware Active Probe Planning** — 物理原语库 + 机构假设库,
   联合结构信息增益、参数可辨识性与下游任务收益;
3. **Program-Ready Interactive Simulation Repair** — 证据转化为可执行机构
   与参数, 以未见交互预测与 CaP 的 World 执行作为评价。

## 5. 两层 Library

Physical Probe Library (configs/probes.yaml, 描述实验) 与 Mechanism
Library (configs/mechanisms.yaml, 描述物理规律) 分开管理。

### 5.1 Physical Probe Library (9 个参数化模板)

| Probe | 操作 | 主要识别目标 |
| - | - | - |
| `gentle_pull` | 小力低速拉动 | 锁定/阻挡/接触有效性 |
| `rotate_handle` | 多角度缓慢旋转把手 | 旋转关节/扭转刚度/解锁阈值 |
| `hold_and_pull` | 保持把手角度轻拉门叶 | 把手-门隐式耦合 |
| `release_and_observe` | 放开并记录回弹 | 回位弹簧/阻尼/预载 |
| `quasi_static_sweep` | 低速近似匀速开关门 | 库仑/静摩擦、运动约束 |
| `velocity_sweep` | 多速度短程操作 | 速度相关阻尼、摩擦模型 |
| `acceleration_pulse` | 短加速度扰动 | 转动惯量、等效质量 |
| `bidirectional_probe` | 反方向尝试 | 单向限位、棘轮、止挡 |
| `limit_search` | 逐步接近行程终点 | 运动限位、末端碰撞 |

Probe 是参数化动作模板而非固定轨迹 (实例化 API 见 `types.ProbeSpec`)。
激励设计参考 ForceTwin: 低加速度 → 库仑摩擦, 速度变化 → 黏性阻尼,
高加速度 → 惯量, 被动跟随 → 机构作用力。

### 5.2 Mechanism Library

| Mechanism | 模型形式 | 需要的观测 |
| - | - | - |
| Revolute / Prismatic Joint | 轴 + 角度/位移 + 限位 | 运动轨迹 |
| Torsional Spring | τ = −k(θ−θ₀) | 角度-扭矩、释放轨迹 |
| Coulomb + Viscous Friction | τ_f = μ_c·sgn(q̇) + b·q̇ | 多速度运动与外力 |
| Coupled Joint | q₂ = a·q₁ + b | 部件关联运动 |
| Latch / Lock | 状态相关运动约束 | 不同把手状态下的拉门 |
| Joint Stop | q ∈ [q_min, q_max] | 边界处力与位移 |

机构假设用模板组合生成 (门 = 铰链 + 摩擦阻尼 + 可选闩锁[把手旋转 +
回位弹簧 + 滑动门栓 + 把手-门栓耦合])。当观察到"门拉不动但把手能转",
Planner 应提出复合机构假设, 而不是无限提高门铰链静摩擦。

## 6. 双门场景的 Probe 执行计划

任务: 打开左侧房门, 使机器人能够进入下一个房间。初始 sim 只有门叶、
门框、把手与推断的铰链, 不知道是否存在闩锁。

### 第一轮: 判断门为什么打不开

1. **Gentle Pull**: 有明显作用力但门基本不动 → 保留闩锁/机械止挡/接触
   打滑等竞争假设, 不直接认定"门很重"。
2. **Handle Rotation**: 扭矩随角度上升、释放后回弹 → 回位弹簧假设。
3. **Hold and Pull** (多个把手角度下重复小幅拉门): 只有超过某角度才能
   拉开 → "把手控制门锁释放"的直接因果证据。

核心成果: Door Openable ⇔ θ_h ≥ θ_unlock (当前 sim 完全缺少的结构性
知识)。再做两项验证: 松手观察回位; 改变把手状态复拉, 排除偶然打滑。

### 第二轮: 确认机构并估计参数 (真值仅供评测)

| Probe | 估计内容 | 参考真值 |
| - | - | - |
| 转动并释放把手 | 弹簧刚度/阻尼/平衡位置 | 0.5 N·m/rad、0.08 |
| 不同角度 Hold and Pull | 解锁阈值 | 0.556 rad |
| 解锁后低速拉门 | 静摩擦与黏性阻尼 | 门铰链阻尼 0.4 |
| 解锁后短加速度扰动 | 门叶绕铰链转动惯量 | 由 12 kg 门叶决定 |
| 正反方向小幅操作 | 单向限位与碰撞约束 | 0–1.508 rad |

### 两级重建目标

仅凭外部力与位姿观测不能保证恢复隐藏门栓的 0.025 m 行程与 0.045 耦合
系数 (不同内部机构可产生相同外部行为), 因此:

- **Functional Mechanism Recovery**: 恢复"旋转把手到阈值才能开门"的
  行为约束 (第一阶段实现, 通常已足以修复 CaP 关键错误);
- **Explicit Mechanism Recovery**: 证据充分时再恢复滑动门栓、等式耦合
  与几何限位。

## 7. Planner: 任务条件信息增益

维护假设集 H_t = {H₁(重/高摩擦), H₂(弹簧/阻尼), H₃(隐藏闩锁), H₄(几何
卡滞)}, 每个假设含结构模型 M_i、连续参数 θ_i 与可信度。

**结构信息增益** (区分机构):

    IG_struct(a) = H(H|D_t) − E_{o|a,D_t}[ H(H|D_t,a,o) ]

例: 再拉一次门对区分高摩擦/闩锁帮助很小; 旋把+拉门能直接区分 → 后者
结构增益更高。

**参数信息增益** (机构已知时, Fisher 信息):

    I(a,θ) = Σ_t J_t^T Σ_o^{-1} J_t,   J_t = ∂ô_t/∂θ

惯量辨识需要高加速度轨迹; 黏性阻尼需要不同速度。ASID 用 Fisher 信息
构建主动探索目标, SPI-Active 用 CMA-ES 优化探索输入。

**评分**:

    a* = argmax_a [ λ_s·IG_struct(a) + λ_p·IG_param(a)
                  + λ_T·ΔV_CaP(a) − λ_C·C(a) ]

ΔV_CaP 通过在候选仿真模型中重放同一份 CaP 近似估计。它回答关键决策:
门打不开时, 优先辨识质量还是优先判断是否上锁 → 应优先闩锁 Probe
(辨识出 12 kg 也无法解释"门完全没有开度")。

## 8. CMA-ES 与本体感受

**记录数据** D_i = {u_t, q_t, q̇_t, τ_t, T_t^ee, T_t^obj, F̂_t}: 关节
位置/速度/命令、测量扭矩、夹爪与末端位姿、动力学模型估计的外部接触力、
视觉估计的把手/门叶角度与接触状态。外力估计需扣除机器人自身重力/惯性/
摩擦并经接触雅可比转换, 禁止把 `qfrc_actuator` 直接当接触力。

**拟合参数** θ = [I_d, b_d, μ_d, k_h, b_h, θ_{h,0}, θ_unlock]; 证据充分
时再扩展门栓行程与耦合系数。优化目标:

    θ* = argmin_θ Σ_i [ w_q·L_q^i + w_F·L_F^i + w_e·L_e^i ] + λ·R(θ)

L_q/L_F 为归一化 Huber 损失, L_e 为锁定/解锁/限位等离散事件不一致损失。

**分阶段拟合**: ① 机构模型选择 → ② 低速响应 (静摩擦/库仑摩擦/刚度) →
③ 动态响应 (黏性阻尼/惯量) → ④ 全量记录联合精修 → ⑤ 未见轨迹验证。

注意: 直接辨识得到的是绕铰链的转动惯量, 结合可信的门叶几何与质量分布
假设才能换算为 12 kg 质量。

**位置**: 结构由 Agent 提议及模型比较确定 (model_selection_score 含复杂
度与未见实验预测, 防止用复杂机构拟合噪声), 参数由 CMA-ES 优化。不能把
整套问题交给一个大维度 CMA-ES —— 它能优化弹簧刚度, 却不会凭空生成缺失
的门栓关节。

## 9. Simulator Update

**Structure Repair**: 显式机构 (门栓滑动关节 + 耦合等式 + strike 接触)
或功能性机构 (离散锁定状态, 把手转角超过阈值时解除门运动约束)。第一阶
段用功能性实现闭环 (`model_editor.py` 的 `functional_latch`), 之后升级
显式机构。参考 EMPIRIC 让引擎通过新增程序表示缺失机制/约束/隐藏状态。

**Parameter Repair** 的 MJCF 映射: `body/inertial`←I_d, `joint/damping`
←b_d/b_h, `joint/frictionloss`←μ_d, `joint/stiffness`+`springref`←k_h/θ₀,
`joint/range`←限位, `equality`/自定义状态约束←联动。状态相关机构力可参
考 ForceTwin 每步施加额外关节力, 但能用显式机构表示的物理规律优先显式
建模, 不用黑箱 residual force 掩盖已识别的闩锁。

## 10. 六类偏差的分工

| 偏差 | 处理方式 | 优先级 |
| - | - | - |
| 闩锁结构缺失 | Mechanism Hypothesis + Active Probe + Structure Repair | P0 |
| 质量/惯量/摩擦/阻尼错误 | Proprioception + CMA-ES | P0 |
| 全局坐标旋转错位 | SE(3) Registration (R,t 刚体配准, 纯平移不够) | P0 |
| SAM3 外观匹配失败 | 几何/功能部件定位与跨域视觉匹配 | P1 |
| Panda 与 FR3 不一致 | Robot/Controller Calibration + 统一末端动作接口 | P0 |
| 铰链轴/碰撞/限位错误 | 运动学 Probe + 几何接触修订 | P1 |

坐标系问题必须先于任何 Physical Probe 解决; 双机器人场景先统一机器人
模型与控制接口 (或先分别校准执行器), 否则 CMA-ES 会把执行器增益差异
错误归因于门叶质量或摩擦。

## 11. 代码映射

| 方案模块 | 代码 |
| - | - |
| 假设集合 H_t | `src/active_sysid/planner/hypothesis.py` |
| 评分选择 Probe | `src/active_sysid/planner/probe_planner.py` |
| Physical Probe Library | `configs/probes.yaml` + `probes/library.py` |
| World 交互与记录 | `probes/executor.py` (WorldInterface 契约) |
| Mechanism Library | `configs/mechanisms.yaml` |
| 结构模型选择 | `identification/mechanism.py` |
| CMA-ES 拟合 | `identification/cma_es.py` |
| Structure/Parameter Repair | `simulator/model_editor.py` |
| 预测验证 | `simulator/validator.py` |
| CaP 评测 | `cap/evaluator.py` |
| 主链路与停止条件 | `pipeline.py` |

停止条件: 当前仿真能预测保留 Probe 结果, 且进一步辨识对 CaP 成功预测
改善有限时, 停止探索并执行 CaP (优于预先规定"执行五次后停止")。
