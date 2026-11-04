# Score Centering 在 ascend 新基线上的重实现记录

> 本文档记录 `exp/score_centering_megatron_fsdp` 分支的由来、rebase 过程中的冲突与 处理方式，以及需要在内网 Megatron
> 环境完成的待验证项目。

______________________________________________________________________

## 1. 前因：PR #1797 与新基线分叉

PR #1797（`Aression/AReaL:feature/score-centering` → `areal-project/AReaL:ascend`）最初 基于
`ascend` commit `abe0fb0a`，随后 `ascend` 前进到 `5c633b70`。当时 CI 全绿：

- pre-commit：通过
- 上游完整 CPU 套件：3739 passed / 12 skipped
- 上游 NPU 套件：137/137 passed

之后 `ascend` 继续前进到 `ce444f9f`（含 22 个新提交），PR 状态变为
`mergeable_state = dirty`，`rebaseable = false`，即无法直接合并。

为不阻塞后续实验，另起 `exp/score_centering_megatron_fsdp` 分支，在 **当前 `ascend` HEAD（`ce444f9f`）**
上重新实现同一份改动，而不是在旧基线上继续解冲突。

______________________________________________________________________

## 2. 新基线带来的变化

从 `abe0fb0a` / `5c633b70` 到 `ce444f9f`，与本特性相关的上游改动：

| 上游提交                                                     | 影响                                                                                                 |
| ------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| `77d57cfc` feat(trainer): MiMo-inspired RL metrics (#1736)   | 引入 `stat_compact` / `ReduceType` 紧凑统计；把版本陈旧度日志从 `grpo_loss_fn` 移到 `_ppo_update`    |
| `20bb2db8` feat(trainer): separate importance-sampling stage | `ppo_actor_loss_fn` 新增独立 `importance_sampling` 修正阶段                                          |
| `42ac71b3` / `6a924588`                                      | CP-safe vocab 统计、`vocab_mean_logits` / `vocab_norm_logits` 指标；CP 只聚合标量而非全 vocab logits |
| `1801143e` / `f40f1f24` / `5e981ad4`                         | NPU chunked-logprobs、fp32 LM head、`logprobs_chunk_size` 配置                                       |
| `8d86a2b1` perf(trainer): critic-free GAE fast path          | GRPO 无 critic 时用 cumsum 计算 reward-to-go                                                         |
| `cc2b8ad7` 等                                                | Qwen3.5 加入 `attention_mask = None` 分支                                                            |

______________________________________________________________________

## 3. 冲突与处理

只有 `areal/trainer/ppo/actor.py` 冲突，共 3 处；其余 5 个文件自动合并成功。

| #   | 位置                | 冲突内容                                                                 | 处理                                       |
| --- | ------------------- | ------------------------------------------------------------------------ | ------------------------------------------ |
| 1   | import 区           | 上游 `ReduceType` vs 本特性 `score_centering` 导入                       | 两者都保留                                 |
| 2   | `grpo_loss_fn` 签名 | 上游 `vocab_mean_logits` / `vocab_norm_logits` vs 6 个 `sc_*` 参数       | 两者都保留                                 |
| 3   | 统计日志区          | 上游已把版本陈旧度日志移到 `_ppo_update`，本特性在同一位置追加了 SC 统计 | 删除本地重复的版本陈旧度块，仅保留 SC 统计 |

______________________________________________________________________

## 4. 重实现时新增的两处语义守卫

自动合并成功不代表语义正确。新基线的两个能力与本特性存在真实冲突，已加 fail-fast：

### 4.1 `importance_sampling` 与 `sc_loss_mode='reinforce'` 互斥

Plan A（`reinforce`）整体替换 PPO clipped surrogate，而上游新增的 `importance_sampling` 修正是在该
surrogate 内部计算的。两者同时开启会让 IS 修正 被静默丢弃，因此在 `PPOActorConfig.__post_init__` 中拒绝该组合；Plan B
（`ppo_additive`）保留 PPO loss，仍可与 `importance_sampling` 共存。

### 4.2 SC 需要完整 vocab、完整序列的 logits

`check_score_centering_parallel_supported(use_sc, sp_size, tp_size)` 现在同时拒绝：

- `sp_size > 1`：Ulysses / CP 注入的是本地序列切片，而 `logprobs` / `loss_mask` 已 all-gather
  到全局序列，长度不一致；
- `tp_size > 1`：TP 的 vocab 分片 LM head 下，采样 token id 可能根本不落在本地 vocab 分片内。

纯 DP 复制（`sp_size == tp_size == 1`）不受影响。

______________________________________________________________________

## 5. 分支内容

`exp/score_centering_megatron_fsdp`（基于 `ce444f9f`）包含两个提交：

| Commit                                                                                           | 内容                                                                     |
| ------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------ |
| `feat(trainer): add Score Centering drift correction for off-policy RL`                          | SC 完整实现 + 端到端 top-k 数据链路 + 配置校验 + 6 处语义守卫 + 三套测试 |
| `fix(engine): use None attention_mask for packed sequences to enable block-diagonal causal mask` | 非 Qwen3 分支的 attention mask 修复                                      |

`fix` 提交仍然必要：上游虽把 Qwen3.5 加入 `None` 分支，`else` 分支依旧使用
`dict(full_attention=None, sliding_attention=None)`，Qwen2.5 / Llama 等模型仍受影响。

### 测试覆盖

| 文件                                     | 内容                                                                                                     |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------- |
| `tests/test_score_centering.py`          | 手算分布下的 correction 值/梯度、weighting 边界、padding 不变量、分块 autograd、配置校验、并行与 IS 守卫 |
| `tests/test_score_centering_actor.py`    | 四条 loss 路径的最终 loss 与解析梯度（含 PPO clipping）、mask 不变性、归约分母、缺失输入 fail-fast       |
| `tests/test_score_centering_pipeline.py` | vLLM parser → RLVR/OpenAI 组装 → actor next-token 对齐 → packing 的位置级校验                            |

核心测试做了变异验证：correction 符号取反、weighting 失效、MIS 退化为 TIS、 scatter 覆盖梯度、跳过末 chunk 均能触发失败。

______________________________________________________________________

## 6. 已验证项（CPU）

在 fork 上复用上游 `ascend-pr-tests.yml` 的 `pre-commit` 与 `cpu-unit-tests` 两个 job（去掉需要专用
runner 的 NPU job），对 `ce444f9f` + 本分支两个提交跑完整套件：

| 检查                                | 结果                                                                            |
| ----------------------------------- | ------------------------------------------------------------------------------- |
| pre-commit（含 CLI 文档生成一致性） | 通过                                                                            |
| CPU 单测                            | **3886 passed / 12 skipped / 223 deselected**，163 个测试模块，耗时 22 分 01 秒 |

测试选择使用上游 `scripts/select_cpu_tests.py` 与 marker 过滤，口径与上游一致。 作为参考，同一套件在 rebase 前的旧
PR（`242c16fb`）上是 3739 passed，增量来自新基线 自身新增的用例与本分支新增的 SC 测试。

______________________________________________________________________

## 7. 待验证项（内网 Megatron 环境）

以下项目无法在 CPU CI 覆盖，需要在内网 Megatron 后端完成：

1. **Megatron trainer 注入路径** `fsdp_engine` 中的 logits 注入只覆盖 FSDP。Megatron 引擎需要把训练侧 logits
   传入 `grpo_loss_fn`；当前实现下 `use_score_centering=True` 会因缺少 `logits` 触发
   `RuntimeError`。需确认 Megatron 侧接入点（`mcore` forward 输出、vocab 分片与 `logprobs_chunk_size`
   的关系）并补齐。

1. **TP / CP 组合** `tp_size > 1` 与 `sp_size > 1` 当前被显式拒绝。若要在 Megatron TP/CP 下使用 SC， 需要实现按
   vocab 分片 gather 与跨 CP 序列拼接，或明确保持拒绝。

1. **chunked LM head 下的 logits 可用性** 上游新增 `logprobs_chunk_size`、fp32 LM head 与 CP
   标量聚合路径。需确认这些路径 是否仍然 materialize 完整 `[tokens, vocab]` logits；若否，SC 的 gather 需要改为 沿同一条
   chunked/vocab-sharded 通道计算。

1. **数值一致性** 同一 batch 下比较：FSDP 与 Megatron 的 `train_head_logp`、`correction`、`rho`
   是否在容差内一致（bf16 vs fp32 head）。

1. **性能开销** 在固定 batch、`sc_top_k=128` 下测量 `timeperf/sc_gather_train_head_logp` 与
   `timeperf/sc_correction` 占比，并与 SC-off 的 `timeperf/train_step` 对比；注意 剔除首步编译预热（此前 NPU
   E2E 中首步 `train_step` 45.6s、稳态 1.67s）。

1. **显存峰值** chunked autograd 的目标是压峰值而非降时间，需确认在 Megatron 下峰值增量可接受。

1. **端到端训练** Geometry3K / GSM8K 小步数（2~10 step）跑通，检查 `sc_active=1`、 `sc_rho` 与
   `sc_p_mass` 在合理范围、无 NaN/Inf。

______________________________________________________________________

## 8. 后续流程

- 内网 Megatron 实验产生的改动，以 patch 形式推送到本 docs 分支（`docs/score-centering`）， 放在
  `docs/immgration/v3/` 下，并在本文档第 7 节勾掉对应项。
- 待 Megatron 侧接入完成且验证通过后，再统一回到 PR #1797，处理与 `ascend` 的合并。
