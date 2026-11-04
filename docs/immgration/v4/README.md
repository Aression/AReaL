# v4 迁移：在 ascend 新 HEAD 上重放 Score Centering

> 本目录是 **v4 迁移**的工作基线。目标是以最快路径把 Score Centering（SC）合入 `ascend`，先在内网把 tis/mis
> 与多轮推理两个已知问题修完，再回蓝区重放并走 CI。
>
> 上一版 `v3`（Megatron 后端实验）标记为**后续工作**，不影响本版推进。

______________________________________________________________________

## 1. 为什么重开 v4

PR #1797（`feature/score-centering`）原先基于 `ascend` 的 `abe0fb0a`，之后 `ascend` 前进了 22 个提交到
`ce444f9f`，PR 变成 `mergeable_state = dirty`、`rebaseable = false`。 继续在旧基线上解冲突的收益低于在新 HEAD
上重放，因此：

- 以 `ce444f9f` 为基线重放两个提交，得到 `debug/score_centering_v4`；
- 内网修复直接在 v4 上做，产出增量 patch；
- 蓝区按 v4 README 的顺序重放 patch + 跑 CI 验收。

______________________________________________________________________

## 2. 基线分支与提交

| 项         | 值                                                                         |
| ---------- | -------------------------------------------------------------------------- |
| 基线       | `areal-project/AReaL:ascend` @ `ce444f9f`                                  |
| 工作分支   | `Aression/AReaL:debug/score_centering_v4`                                  |
| 分支头     | `e220b02e`（与 `exp/score_centering_megatron_fsdp` 同一 tree，CPU 已验证） |
| 对比文件量 | 18 文件，+1899 / −13                                                       |

分支上的两个提交：

| #   | 提交                                                                                             | Patch                                                             |
| --- | ------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------- |
| 1   | `feat(trainer): add Score Centering drift correction for off-policy RL`                          | `0001-feat-trainer-add-Score-Centering-drift-correction-fo.patch` |
| 2   | `fix(engine): use None attention_mask for packed sequences to enable block-diagonal causal mask` | `0002-fix-engine-use-None-attention_mask-for-packed-sequen.patch` |

______________________________________________________________________

## 3. 重放时的冲突与处理（相对旧基线）

只有 `areal/trainer/ppo/actor.py` 冲突，3 处；其余文件自动合并。

| #   | 位置                | 处理                                                                       |
| --- | ------------------- | -------------------------------------------------------------------------- |
| 1   | import 区           | 同时保留上游 `ReduceType` 与本特性的 `score_centering` 导入                |
| 2   | `grpo_loss_fn` 签名 | 同时保留上游 `vocab_mean_logits` / `vocab_norm_logits` 与 6 个 `sc_*` 参数 |
| 3   | 统计日志区          | 删除本地重复的版本陈旧度日志（上游已移到 `_ppo_update`），仅保留 SC 统计   |

新基线还引入了两处**语义冲突**，已在提交 1 中加了 fail-fast：

1. **`importance_sampling` 与 `sc_loss_mode='reinforce'` 互斥**：Plan A 替换了承载该 修正的 PPO
   surrogate，同时开启会让 IS 修正被静默丢弃；`ppo_additive` 仍可共存。
1. **SC 需要完整 vocab、完整序列的 logits**：
   `check_score_centering_parallel_supported(use_sc, sp_size, tp_size)` 拒绝
   `sp_size > 1`（Ulysses/CP 本地切片）与 `tp_size > 1`（TP vocab 分片）。

______________________________________________________________________

## 4. CPU 验收（蓝区前的一次性确认）

在 fork 上复用上游 `ascend-pr-tests.yml` 的 `pre-commit` + `cpu-unit-tests`：

| 检查                            | 结果                                                             |
| ------------------------------- | ---------------------------------------------------------------- |
| pre-commit（含 CLI 文档一致性） | 通过                                                             |
| CPU 单测                        | 3886 passed / 12 skipped / 223 deselected，163 模块，22 分 01 秒 |

因为 `debug/score_centering_v4` 与已验证分支 tree 相同，该结果直接适用于本分支。

______________________________________________________________________

## 5. 本轮待修复问题（内网）

以下两项在 v4 基线上修复，修复后以增量 patch 形式回传本目录（编号从 `0003` 起）。

### 5.1 tis / mis 的权重语义

现状（`areal/utils/score_centering.py`）：

- `tis`：`ratio.clamp(min=low, max=high)`，双侧截断；
- `mis`：`where(low <= ratio <= high, ratio, 0)`，区间外归零；
- `sc_mode='none'`：恒为 1，纯 SC。

待确认/修复：

1. `tis` 下界是否应作用于 head 与 tail 两处（当前 tail 用 `weight_fn(1/rho)`，下界截断会 改变 `alpha` 的语义）；
1. `mis` 在 `low=0` 时退化为"高位截断 + 其余保留"，与 IcePop 的预期是否需要显式区分；
1. 默认值组合（`sc_clip_low=0.5`、`sc_clip_high=inf`）在 `tis`/`mis` 下都会触发配置
   校验失败，需要给出可直接使用的默认组合或更清晰的报错；
1. 三个模式在数值基准下的期望值需要同步更新 `tests/test_score_centering.py` 的独立手算表（当前只覆盖到 `[0.5, 2.0]`
   区间的一组）。

### 5.2 多轮推理

现状：`chat_template_type='concat'` 且存在 parent 时，`to_tensor_dict()` 抛
`NotImplementedError`——父轨迹的 sampler top-k 没有向下传递。

待完成：

1. 在 concat 子轨迹中继承父轨迹的 `sampler_topk_token_ids` / `sampler_topk_logps`；
1. 保证 k 维一致（父为 prompt 段，子为新增段），并在长度不匹配时给出明确报错；
1. 补一条多轮用例：两轮不同分布 → 组装后每个有效训练位置的 top-k 与来源轮次一致。

______________________________________________________________________

## 6. 蓝区重放流程

蓝区（无 torch / 无 GPU / 无 NPU 的基础 Python 环境）执行：

```bash
git checkout <ascend 基线 ce444f9f 或蓝区等价基线>
git apply --check docs/immgration/v4/0001-*.patch
git apply        docs/immgration/v4/0001-*.patch
git apply --check docs/immgration/v4/0002-*.patch
git apply        docs/immgration/v4/0002-*.patch
# 内网修复补齐后，继续按编号顺序 apply 0003 及以后
```

静态验收：

```bash
python3 docs/immgration/v1/verify_sc_math.py          # 12 项纯 Python 数学检查
python3 docs/immgration/v1/verify_attention_mask_fix.py <repo>   # 3 项源码模式检查
```

> `verify_sc_math.py` 与 `verify_attention_mask_fix.py` 仍是 v1 版本，只校验公式与源码 模式；tis/mis
> 的权重语义变更后需要同步扩展这两个脚本或新增 v4 版本。

CI 验收（fork 侧，无需上游审批）：

1. 以重放后的分支为 `ref` 触发 `pre-commit`；
1. 复用上游 `scripts/select_cpu_tests.py` + marker 过滤跑完整 CPU 套件；
1. 期望：pre-commit 通过，CPU 全绿（基线 3886 passed）。

______________________________________________________________________

## 7. 验收清单

| #   | 项目                                           | 状态                |
| --- | ---------------------------------------------- | ------------------- |
| 1   | 两个提交在 `ce444f9f` 重放成功                 | 完成                |
| 2   | 冲突 3 处按第 3 节处理                         | 完成                |
| 3   | IS 互斥、并行布局 fail-fast                    | 完成                |
| 4   | 完整 CPU 套件 + pre-commit                     | 完成（3886 passed） |
| 5   | tis / mis 权重语义修复                         | 待内网              |
| 6   | 多轮（concat）top-k 传递                       | 待内网              |
| 7   | 增量 patch 回传并蓝区重放                      | 待内网              |
| 8   | 蓝区 CI 全绿                                   | 待蓝区              |
| 9   | 回到 PR：#1797 base 切到 `ce444f9f` 并更新描述 | 待 CI 后            |
| 10  | （后续）Megatron 后端实验，见 v3               | 延后                |

______________________________________________________________________

## 8. 与 v3 的关系

- `v3` 记录的是在 **Megatron 后端**上接入 SC 的实验与待验证项（logits 注入点、TP/CP、 chunked LM
  head、数值一致性、性能与显存）。
- 本轮为最快推进合入，暂不阻塞在 Megatron 上；v3 的内容保留，待 FSDP 路径合入后再启动。
