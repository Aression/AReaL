# Prompt 03 — 纯 Python 验收 SC 数学 + 代码结构

## 你的角色

你是一个验收 agent。你的任务是用纯 Python（无 torch 依赖）验证 Score Centering (SC) 的核心数学正确性，以及 attention mask 修复的代码模式正确性。

## 前置条件

1. **Prompt 01 已完成**：SC 基础设施 patch 已应用
2. **Prompt 02 已完成**：Attention mask 修复 patch 已应用
3. 环境有 `python3`（**不需要 torch**，不需要 NPU/GPU）

## 验证脚本

本目录包含两个纯 Python 验证脚本：

| 脚本 | 验证内容 | 测试数 | 依赖 |
|------|---------|-------|------|
| `verify_sc_math.py` | SC 核心数学公式（纯 Python 重实现 + 12 项断言） | 12 | 仅标准库 |
| `verify_attention_mask_fix.py` | Attention mask 修复的代码模式（grep + 正则） | 3 | 仅标准库 |

---

## 第一部分：SC 核心数学验证

### 验证原理

`verify_sc_math.py` 用纯 Python 重新实现了 `score_centering.py` 的核心数学：

1. **`pad_topk_ids` / `pad_topk_logps`**：padding 函数（id 填 0，logprob 填 -inf）
2. **`build_weight_fn`**：IS 权重函数（none/tis/mis 三种模式）
3. **`score_centering_correction`**：SC correction 完整公式
   - `q = exp(sampler_topk_logps)` × mask
   - `p = exp(train_head_logp)` × mask
   - `rho = (1 - q_mass) / (1 - p_mass)`（尾质量比）
   - `alpha = rho * weight_fn(1/rho)`（尾 IS 权重）
   - `head_weights = weight_fn(p/q) * mask`
   - `residual = q * head_weights - alpha * p`
   - `correction = sum(residual * train_head_logp)`
   - `s_bar = sum(q * train_head_logp)`

### 12 项验证测试

| # | 测试名 | 验证的数学性质 |
|---|--------|--------------|
| 1 | `pad_topk_ids` | padding id 为 0，长度正确 |
| 2 | `pad_topk_logps` | padding logprob 为 -inf（非 0.0），长度正确 |
| 3 | `pad_topk_logps_truncation` | 超长输入截断到 k |
| 4 | `on_policy_correction_is_zero` | p==q 时 correction=0（所有模式） |
| 5 | `on_policy_rho_is_one` | p==q 时 rho=1.0 |
| 6 | `q_mass_leq_one` | -inf padding 时 q_mass ≤ 1.0 |
| 7 | `q_mass_inflated_by_zero_padding` | 0.0 padding 时 q_mass > 1.0（Bug 2 复现） |
| 8 | `off_policy_nonzero` | p≠q 时 correction ≠ 0 |
| 9 | `weight_fn_none` | pure SC 权重恒为 1.0 |
| 10 | `weight_fn_tis` | tis 权重 clamp 到 sc_clip_high |
| 11 | `weight_fn_mis` | mis 权重 band-mask |
| 12 | `invalid_mode_raises` | 非法 sc_mode 抛 ValueError |

### 执行命令

```bash
python3 /path/to/verify_sc_math.py
```

### 预期输出

```
============================================================
  Score Centering Pure-Python Math Verification
============================================================

  ✅ PASS  pad_topk_ids
          pad_topk_ids([1,2,3],5)=[1, 2, 3, 0, 0]
  ✅ PASS  pad_topk_logps
          pad_topk_logps([-0.1],3)=[-0.1, -inf, -inf]
  ✅ PASS  pad_topk_logps_truncation
          truncated to [-0.1, -0.2]
  ✅ PASS  on_policy_correction_is_zero
          All modes: correction=0 when p==q
  ✅ PASS  on_policy_rho_is_one
          rho=1.0
  ✅ PASS  q_mass_leq_one
          q_mass=0.6219 <= 1.0
  ✅ PASS  q_mass_inflated_by_zero_padding
          q_mass=4.9407 > 1.0 confirms 0.0 padding inflates (Bug 2 reproduced)
  ✅ PASS  off_policy_nonzero
          correction=-1.598214, s_bar=-2.7623, rho=1.0000
  ✅ PASS  weight_fn_none
          All ratios -> 1.0
  ✅ PASS  weight_fn_tis
          clamp(max=5.0) correct
  ✅ PASS  weight_fn_mis
          band-mask [0.2, 5.0] correct
  ✅ PASS  invalid_mode_raises
          ValueError raised correctly

  Results: 12 passed, 0 failed, 0 warnings

  ✅ ALL CHECKS PASSED — SC core math is correct.
```

### 关键数学性质说明

**测试 4（on-policy correction=0）** 是最重要的验证：当 p==q（on-policy），`residual = q*1 - 1*p = q - p = 0`，所以 `correction = 0`。这验证了 SC 在 on-policy 场景下是精确 no-op（不引入多余梯度）。

**测试 6+7（q_mass ≤ 1.0 vs Bug 2）** 验证了 -inf padding 的必要性。`q_mass = sum(exp(top_k_logprobs))`，log_softmax 的 exp 之和 ≤ 1.0。如果 padding 用 0.0，则 `exp(0)=1.0` 会在每个 padding 槽位添加 1.0 的假概率质量，使 q_mass 远超 1.0，破坏 SC 尾模型。

---

## 第二部分：Attention Mask 修复验证

### 3 项验证测试

| # | 测试名 | 验证内容 |
|---|--------|---------|
| 1 | `no_dict_full_attention` | `mb["attention_mask"] = dict(full_attention=None)` 模式已消除 |
| 2 | `none_assignment` | `mb["attention_mask"] = None` 赋值存在（≥4 处） |
| 3 | `comment_explains_fix` | 修复注释包含解释性关键词（create_causal_mask, block-diagonal, packed sequence） |

### 执行命令

```bash
python3 /path/to/verify_attention_mask_fix.py /path/to/AReaL
```

注意：第二个参数是 AReaL 仓库根目录路径。

### 预期输出

```
============================================================
  Attention Mask Fix Verification
============================================================
  File: /path/to/AReaL/areal/engine/fsdp_engine.py

  ✅ PASS  no_dict_full_attention
          No dict(full_attention=None) assignment found in attention_mask path
  ✅ PASS  none_assignment
          Found 4 None assignments for attention_mask
  ✅ PASS  comment_explains_fix
          Found explanatory phrases: ['create_causal_mask', 'block-diagonal', 'packed sequence']

  Results: 3 passed, 0 failed

  ✅ ALL CHECKS PASSED — attention mask fix is correctly applied.
```

---

## 第三部分：代码结构完整性检查

### 步骤 1：SC 数据流完整性

```bash
# 验证 sampler top-k 数据流贯穿 6 个文件
for f in areal/api/io_struct.py areal/engine/vllm_remote.py areal/infra/remote_inf_engine.py areal/workflow/rlvr.py areal/experimental/openai/types.py areal/trainer/ppo/actor.py; do
    echo "=== $f ==="
    grep -c "sampler_topk\|output_topk" $f
done
```

**预期输出**：每个文件至少 1 个匹配。

### 步骤 2：SC 配置字段完整性

```bash
# 验证 cli_args.py 中 SC 配置字段
grep -c "use_score_centering\|sc_top_k\|sc_mode\|sc_clip_low\|sc_clip_high\|sc_reinforce_baseline" areal/api/cli_args.py
```

**预期输出**：≥10（字段定义 + 校验逻辑中的引用）。

### 步骤 3：SC 校验逻辑完整性

```bash
# 验证校验逻辑中的关键检查
grep -n "raise ValueError\|raise NotImplementedError" areal/api/cli_args.py | grep -i "sc_\|score_centering"
```

**预期输出**：≥5 处 raise（sc_mode 校验 + sc_clip 校验 + tis/mis 校验 + recompute_logprob warning + sc_reinforce_baseline 校验）。

### 步骤 4：Python 导入验证（如果有 torch）

```bash
# 如果环境有 torch，可以验证导入成功
python3 -c "from areal.utils.score_centering import score_centering_correction, gather_train_head_logp, pad_topk_ids, pad_topk_logps; print('Import OK')"
```

**注意**：如果环境没有 torch，跳过此步骤——纯 Python 验证（第一、二部分）已足够确认 patch 应用正确。

---

## 验收标准

| # | 验收项 | 通过标准 |
|---|-------|---------|
| M1 | SC 数学验证 | `verify_sc_math.py` 通过 12/12 |
| M2 | Attention mask 验证 | `verify_attention_mask_fix.py` 通过 3/3 |
| M3 | SC 数据流完整性 | 6 文件各 ≥1 个 `sampler_topk`/`output_topk` 匹配 |
| M4 | SC 配置字段完整性 | cli_args.py ≥10 个 SC 字段匹配 |
| M5 | SC 校验逻辑完整性 | cli_args.py ≥5 处 SC raise |

---

## 完成标志

当且仅当以上 5 项验收全部通过时，Prompt 03 完成。

如果 M4/M5 的阈值有偏差（例如基线代码版本不同导致行数略有差异），请报告实际数值并说明原因。核心要求是 M1（数学验证 12/12）和 M2（attention mask 验证 3/3）必须全部通过。

---

## 如果验证失败

### SC 数学验证失败

- **测试 4（on-policy correction=0）失败**：检查 `score_centering_correction` 的 residual 计算。`residual = q * head_weights - alpha * p`，当 p==q 时 `head_weights=weight_fn(1)=1`，`alpha=rho*weight_fn(1/rho)=1*1=1`，所以 `residual = q - p = 0`。如果失败，说明 patch 应用后代码有语法/逻辑错误。
- **测试 6（q_mass ≤ 1.0）失败**：检查 `data.py` 的 -inf padding 白名单是否正确应用。
- **测试 7（Bug 2 复现）失败**：这是预期行为——0.0 padding 应该使 q_mass > 1.0。如果这个测试通过但测试 6 失败，说明 -inf padding 逻辑有问题。

### Attention mask 验证失败

- **测试 1 失败**：说明 patch 未正确应用，`dict(full_attention=None)` 仍在。重新应用 patch。
- **测试 2 失败**：说明 `None` 赋值不存在。检查 patch 是否应用到正确的文件位置。
- **测试 3 失败**：说明修复注释缺失。这可能是基线版本差异导致的 context 偏移。
