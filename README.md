# PhysAgentic-RSR

任务驱动的主动物理辨识与共享机器人策略：由 Physic Agent 分析任务和视觉输入，设计并执行 Probe，依据测量修订模型与操作程序。仿真与真机使用同一份策略源码，通过公开 API 获取观测和执行动作。

## 获取代码

```bash
git clone https://github.com/Wenwen555/PhysAgentic-RSR.git
cd PhysAgentic-RSR
git submodule update --init src/cap-x
```

`src/cap-x` 固定引用本项目使用的修改版本，完整源码位于本仓库的 [cap-x 分支](https://github.com/Wenwen555/PhysAgentic-RSR/tree/cap-x)。
该分支基于上游 [capgym/cap-x](https://github.com/capgym/cap-x/tree/53e9966d7a8e2fa7494676772bccc35280f5c0ed)，保留原项目许可证及第三方 submodule 声明，并包含本项目的仿真、控制接口修改和回归测试。

CaP-X 的依赖安装与第三方模块初始化见 [CaP-X README](src/cap-x/README.md) 和其 `.gitmodules`。第三方库按所需环境安装，虚拟环境、模型权重和运行输出不随本项目提交。

准备好 CaP-X Python 环境后，在主仓库安装 PhysRSR：

```bash
uv pip install --python src/cap-x/.venv/bin/python 'jsonschema>=4.18,<5'
uv pip install --python src/cap-x/.venv/bin/python --no-deps -e .
src/cap-x/.venv/bin/python -m physrsr --help
src/cap-x/.venv/bin/python -m physrsr validate-config physrsr/active/examples/rod_lift.json
```

## 代码与接口

- [physrsr/](physrsr/README.md)：物理辨识、模型修订、程序生成和执行。
- [主动辨识模块](physrsr/active/README.md)：统一 CLI 和服务入口。
- [实验调度合同](physrsr/active/docs/exploration.md)：源码修复与物理交互分离、单 World 证据判断、决策收益理由和同源码断点续执行。
- [Agent 合同](physrsr/active/docs/agent_contracts.md)：物理设计、CaP 合成与 Workflow 编排的职责边界。
- [单次诊断](physrsr/active/docs/debug.md)：显式生成或执行一份程序，不构成第二条工作流。
- [机器人 API](physrsr/active/docs/identification.md)：视觉、本体、腕部力/力矩、保持和进度接口。
- [架构说明](physrsr/active/docs/architecture.md)：任务适配器、预测 Simulator 与 World 评估的职责。

`run` 只使用一条 `ActiveLoop`：一个 World、一套机器人、一份 CaP，经过模型更新、预测评估与独立验证。多个长条属于同一场景，用对象 ID 区分观测与物理估计，不拆成独立 World。任务相关物理绑定与整场景评分由任务适配器实现。旧的自动闭环及批量入口已从代码包删除；历史实验产物保留。

多对象观测、辨识记忆和预测接口见 [单 World 场景合同](physrsr/active/docs/single_world.md)。内置 `three_rod_lift_return` 在一个 MuJoCo 环境中构造三根长条及共享机器人，按逐根抬升、保持和放回评分；原 `rod_lift` 单杆任务保留。

## 离线检查

```bash
MUJOCO_GL=egl src/cap-x/.venv/bin/python -m pytest src/cap-x/tests/active -q
```

该测试集包含接口 fixtures、真实三杆场景编译、接触与物理步进检查，不调用 LLM、IK 服务或真机。只有显式执行相应实验命令才会开始交互。

实验产物保存在本地输出目录；路径由 `physrsr.paths` 和项目 `AGENTS.md` 管理，不纳入 Git。
