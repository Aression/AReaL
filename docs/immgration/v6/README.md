# v6 迁移：Score Centering（FSDP 路径）重放说明

> 本目录是 v6 迁移的**可重放基线**。三个 patch 依次应用到 `areal-project/AReaL:ascend` 的 `b3f55ed9`，结果与分支
> `Aression/AReaL:debug/score_centering_v6` 的 tree **完全一致** （已用干净 worktree 校验）。

______________________________________________________________________

## 1. 基线

| 项       | 值                                                   |
| -------- | ---------------------------------------------------- |
| 上游基线 | `areal-project/AReaL:ascend` @ `b3f55ed9`            |
| 结果分支 | `Aression/AReaL:debug/score_centering_v6`            |
| 分支头   | `711a2c86`                                           |
| PR       | `areal-project/AReaL#1797`（head 已指向 `711a2c86`） |

## 2. Patch 与顺序

必须按下列顺序应用：

```bash
BASE=b3f55ed9
git checkout "$BASE"
git apply docs/immgration/v6/src.patch      # 生产代码，13 文件
git apply docs/immgration/v6/tests.patch    # 测试，6 文件
git apply docs/immgration/v6/docs.patch     # CLI 参考 + 示例说明，3 文件
```

| Patch         | 范围                                                                                                 |
| ------------- | ---------------------------------------------------------------------------------------------------- |
| `src.patch`   | `areal/` 13 个文件：SC 配置、引擎注入、actor 损失、数据链路、`score_centering.py`、`topk_padding.py` |
| `tests.patch` | `tests/` 6 个文件：unit / guards / objective / pipeline / tis + rollout 版本回归                     |
| `docs.patch`  | \`docs/en                                                                                            |

## 3. v6 的设计要点

- **权重模式由 SC 自己掌管**：`sc_mode = none | tis | mis`，边界为 `sc_clip_low` / `sc_clip_high`（默认
  `0.5` / `5.0`）。开启 SC 时 `rejection_sampling` 与 `importance_sampling` 必须为
  `null`，否则配置加载报错。
- **head/tail 质量都做掩码**：`_masked_head_masses` 用 `isfinite(sampler)` 同时屏蔽 trainer 与
  sampler 两侧，`residual * train_head_logp` 再按 valid 置零，因此 `-inf` 槽位既不贡献质量也不贡献 score。
- **尾部比值**：`rho = clamp(1-q_sum, eps) / clamp(1-p_sum, eps)`，两条尾都设下限。
- **指标**：`sc_correction`、`sc_loss_term`、`sc_tis_weight`、`sc_tis_clamped_fraction` （由
  `band_mask` 判定区间，`mode='none'` 时不上报）、`sc_rho`、`sc_q_mass`、 `sc_p_mass`；另有 `loss_mode`
  / `sc_active` 标量。
- **fail-fast**：非 fsdp 后端、tree training、critic、SAPO/CISPO/decoupled/m2、 非 token 级
  IS、distillation、缺失 `sampler_head_*`、`sp_size/tp_size > 1` 一律拒绝；采样前置要求
  `temperature=1`、`recompute_logprob=False`、 `top_p=1`、`top_k` 无界、无 penalty、非 greedy。
- **完整日志告警**：`_warn_sc_head_padding` 只在生成位置（按 `loss_mask` 过滤）出现 补位或零质量行时告警一次。

## 4. 本次重放时修正的两处

1. **`tests/test_score_centering_tis.py` 未过 ruff-format**：重放时多了一个空行， 已格式化。`tests.patch`
   已按格式化后的结果重新生成（内容与原 patch 一致，仅该空行 不同）。
1. **`examples/math/README.md` 文案与 v6 契约矛盾**：原 patch 仍指导用户用
   `actor.rejection_sampling.{level,metric,action,upper}` 配置 TIS，而 v6 在 SC 开启时 禁止
   `rejection_sampling`/`importance_sampling`。已改写为 `sc_mode` / `sc_clip_low` /
   `sc_clip_high` 的写法，并补充边界校验条件与指标清单。 `docs.patch` 已按改写后的结果重新生成。

> `src.patch` 未做任何内容修改，仅保留上游作者的原始上下文粒度。

## 5. 已验证项（CPU）

在 fork 上复用上游 `ascend-pr-tests.yml` 的 `pre-commit` 与 `cpu-unit-tests` （去掉需要专用 runner 的
NPU job）：

| 检查                                                     | 结果                                                                 |
| -------------------------------------------------------- | -------------------------------------------------------------------- |
| pre-commit（mdformat / generate-cli-docs / ruff / SPDX） | 通过                                                                 |
| CPU 单测                                                 | **3953 passed / 47 skipped / 223 deselected**，165 模块，19 分 17 秒 |

运行记录：`Aression/AReaL` actions run `42792338821`。

## 6. 待验证项（内网 / NPU / Megatron）

1. **NPU 端到端**：Geometry3K 或 GSM8K 2~10 步，确认 `sc_active=1`、`sc_rho`/`sc_p_mass` 在合理范围、无
   NaN/Inf。
1. **tis / mis 数值语义**：`sc_mode=tis|mis` 下 `sc_tis_weight` 的 min/max 是否落在 band
   内，`sc_tis_clamped_fraction` 是否与离线统计一致。
1. **多轮（`chat_template_type='concat'`）**：当前仍抛 `NotImplementedError`，需决定是否 在 v7 实现父轨迹
   head 继承。
1. **Megatron 后端**：`score_centering.py` 已预留 `vocab_parallel_selected_logprobs`， 但引擎入口仍拒绝
   `tp_size > 1`；接入点与数值一致性待验证（详见 `../v3/README.md`）。
1. **性能 / 显存**：`sc_gather_train_head_logp` 已改为 checkpoint 分块，需实测 `train_step` 增量与峰值显存。
1. **attention-mask 修复**：v6 不含该修复，依赖上游
   [#1558](https://github.com/areal-project/AReaL/pull/1558)；若上游先合并，需重新评估 是否仍需单独提交。
