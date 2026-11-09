# 实验设计: Hidden Mechanism Recovery for CaP Transfer

## 主实验

任务: 在未提供 World 物理真值的条件下, 通过少量物理交互重建带有隐藏
闩锁及回位弹簧的门, 并使重建仿真中验证成功的 CaP 能在 World 中成功执行。

研究内容: 主动物理实验能否同时恢复缺失的机构结构与关键连续参数, 提高
仿真对接触丰富任务的可预测性。先做清楚物理误差的分解实验, 不急着扩展
任务数量。

## 对比方法

| 方法 | 结构发现 | 参数辨识 | 主动实验选择 |
| - | - | - | - |
| Visual Reconstruction | 无 | 视觉先验 | 无 |
| Passive SysID | 无 | CMA-ES | 固定轨迹 |
| Library + Random Probe | 模板假设 | CMA-ES | 随机 |
| Library + Active Probe | 模板假设 | CMA-ES | 信息增益 |
| Full Agentic RSR | Agent 生成与选择机构假设 | CMA-ES | 任务驱动信息增益 |
| Oracle Structure + SysID | 真实结构 | CMA-ES | 主动 |
| Oracle Physics | 真实结构 | 真实参数 | 不需要 |

Oracle 组仅作评测上界。

## 指标

- **Mechanism Accuracy**: 闩锁存在性、弹簧、单向关节、联动关系的结构
  识别准确率;
- **Physical Error**: 解锁阈值误差、惯量误差、力-角度曲线误差、轨迹
  预测误差 (对 `configs/eval_truth.yaml`, 仅评测时读取);
- **World Success Rate**: 更新后 sim 中验证的 CaP 迁移至 World 的任务
  成功率;
- **Probe Efficiency**: 达到物理预测要求所需的交互次数、时间与累计
  接触代价。

## 核心展示: 因果链

最有价值的不是最终成功率数字, 而是完整因果链:

```
无闩锁模型 → 拉门失败 → Agent 提出机构假设 → 选择把手实验
→ 检测解锁条件 → 修复仿真 → CaP 自动改变动作顺序 → World 成功
```

它比"物理参数 MAE 降低"更能证明系统价值。

## 随机化 (防止 Probe 过拟合唯一一扇门)

在以下维度做组合随机化 (范围见 `configs/eval_truth.yaml`):

- 门叶质量、铰链阻尼
- 闩锁阈值、把手弹簧刚度
- 机构存在性 (有/无闩锁)

## 消融

1. Active vs Random vs Fixed probe 选择 (Probe Efficiency 曲线);
2. 功能性闩锁 vs 显式门栓机构 (对 CaP 迁移的充分性);
3. 结构信息增益与参数信息增益的解耦 (λ_s / λ_p 扫描);
4. 停止条件: 任务绑定停止 vs 固定 Probe 次数;
5. 本体感受通道: 去/留力估计对惯量与摩擦辨识的影响。

## 前置修复 (在主实验之前)

1. SE(3) 全局配准 (三维对应点或相机位姿), 只用平移 offset 无法解决
   整体旋转错位;
2. 机器人统一: sim CaP 执行与 World 验证使用一致的本体与控制接口,
   或先分别校准执行器, 把探测数据转换到统一的末端作用力与物体运动
   表示下 —— 否则 CMA-ES 会把执行器增益差异归因于门叶质量/摩擦。
