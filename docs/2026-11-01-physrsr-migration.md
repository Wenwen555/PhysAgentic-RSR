# PhysRSR 目录迁移

源码目录从 `src/cap-x/experiments/physics_cap/` 移至仓库根目录的 `physrsr/`，Python 命名空间同步改为 `physrsr`。
当前入口见 [physrsr/README.md](../physrsr/README.md)。

导入、子进程服务入口、提示词与示例路径、控制器资源位置和源码快照路径已随目录迁移更新。
根目录新增 `pyproject.toml`，在已有 CaP-X 环境中以 editable 方式安装；CaP-X 仍位于 `src/cap-x/`。

从仓库根目录运行：

```bash
uv pip install --python src/cap-x/.venv/bin/python --no-deps -e .
src/cap-x/.venv/bin/python -m physrsr --help
src/cap-x/.venv/bin/python -m physrsr.active validate-config physrsr/active/examples/rod_lift.json
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 src/cap-x/.venv/bin/python -m pytest src/cap-x/tests/active -q
```

迁移检查：82 项测试通过，两个已有依赖弃用提示。帮助和离线配置检查通过，配置检查的外部调用数为 0。
服务元数据接口已分别从根目录、CaP-X 目录和临时目录验证；未启动任务 rollout、模型调用或真机连接。

历史实验输出、冻结源码、提示词、哈希和归档保留原样，仍位于原来的输出目录。
这些冻结文件保留当时的导入路径，当前命令请使用新入口。
原最终 policy 与第二轮 probe 的 SHA-256 均与迁移前一致。
详细迁移清单见 [manifest](physics-cap-migration-manifest.json)。

后续按用户指定范围删除了错误及废弃尝试；有效历史对照保留。新产物目录为 `/data/wjt/PhyAgent-RSR/output/`，清理清单位于该目录的 `maintenance/cleanup_20261101.json`。
