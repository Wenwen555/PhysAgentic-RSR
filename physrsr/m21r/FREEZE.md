# M2.1-R v2 — code-layout cleanup

本版本仅精简主目录与常规测试，不改变 v1 研究数据、生成器、控制器、估计器或选择规则；
不新增 World/Model rollout，不进入 M2.2。

- 当前代码和数据校验：`outputs/physics_cap_m21r_cleanup_v2/SHA256SUMS`
- 当前源码快照：该目录的 `source.tar.gz`
- 退役文件：该目录的 `legacy_sources.tar.gz` 和 `migration.json`
- 原 v1 冻结快照／清单：`outputs/physics_cap_m21r_freeze_v1/`，保持原样。

从仓库根目录校验当前版本：

```bash
sha256sum --quiet -c outputs/physics_cap_m21r_cleanup_v2/SHA256SUMS
```

v1 的源码路径清单针对旧目录结构，不能直接用它检查已整理的 v2 工作树。
复现 v1 时在独立仓库副本中解压 v1 快照；该快照包含已移出的审计工具和原测试。
当前依赖仍包含已有未提交改动，以源码归档和哈希定义版本，而不只依赖 Git HEAD。
