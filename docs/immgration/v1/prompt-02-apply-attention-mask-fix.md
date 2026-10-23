# Prompt 02 — 应用 Attention Mask 修复 Patch

## 你的角色

你是一个代码迁移执行 agent。你的任务是在已应用 SC 基础设施的 AReaL 代码上，应用 attention mask 修复 patch，并验证修复正确。

## 前置条件

1. **Prompt 01 已完成**：SC 基础设施 patch 已成功应用
2. patch 文件 `02-attention-mask-fix.patch` 在可访问路径
3. 环境有 `git` 和 `python3`

## 背景：这个修复解决什么问题

### TIM（Training-Inference Mismatch）根因

AReaL 的 FSDP engine 在 `_prepare_mb_list` 中，对非 Qwen3 模型（如 Qwen2.5）设置：

```python
mb["attention_mask"] = dict(full_attention=None, sliding_attention=None)
```

`Qwen2Model.forward` 检测到 `isinstance(attention_mask, dict)` 为 True 时，**跳过 `create_causal_mask`**，直接将 `None` 传给 `eager_attention_forward`。`eager_attention_forward` 在 `attention_mask is None` 时不加任何 mask——导致 packed 序列中所有 token 互相 attend（全局 attention），而非正确的 causal attention。

### 修复前的症状

| 指标 | 修复前（dict(None) bug） | 修复后（None） | 噪声底 |
|------|-------------------------|---------------|--------|
| Jaccard（trainer top-k vs vLLM top-k） | 0.093 | 0.605 | — |
| TIM \|Δlogp\| | 6.72 nats | 0.68 | ~0.49（bf16 噪声底） |
| Spike rate（\|Δ\| > 2） | 94% | 9.8% | — |
| Trainer argmax 集中度 | 77.1% 集中在单一 token | 5.2% | — |

### 修复方式

将 `dict(full_attention=None, sliding_attention=None)` 改为 `None`。`attention_mask=None` 让 HF transformers 的 `create_causal_mask` 通过 `find_packed_sequence_indices`（基于 `position_ids` 在序列边界 reset 为 0）自动检测 packed 格式，构建正确的 block-diagonal causal mask。

## Patch 信息

| 维度 | 值 |
|------|-----|
| Patch 文件 | `02-attention-mask-fix.patch` |
| 对应 commit | `05212ef`（`fix(engine): use None attention_mask for packed sequences to enable block-diagonal causal mask`） |
| 日期 | 2026-10-22 |
| 净改动 | 1 文件（`areal/engine/fsdp_engine.py`），+11/-4 行 |
| 类型 | `git format-patch` |

### Patch 的关键 diff

```diff
-                mb["attention_mask"] = dict(full_attention=None, sliding_attention=None)
-                padded_mb["attention_mask"] = dict(
-                    full_attention=None, sliding_attention=None
-                )
+                # Pass None so that the HF model's create_causal_mask auto-detects
+                # packed sequences from position_ids (which reset to 0 at each sequence
+                # boundary via amend_position_ids) and builds a correct block-diagonal
+                # causal mask.  Using dict(full_attention=None, ...) here bypasses the
+                # mask creation path in Qwen2Model.forward (isinstance check on dict),
+                # which causes eager_attention_forward to receive None and skip the mask
+                # entirely -- making every token attend to every other token across
+                # sequence boundaries.  This was the root cause of the TIM
+                # (Training-Inference Mismatch) that broke Score Centering's tail model.
+                mb["attention_mask"] = None
+                padded_mb["attention_mask"] = None
```

---

## 执行步骤

### 步骤 1：确认 SC patch 已应用

```bash
# 确认 score_centering.py 存在（SC patch 已应用的标志）
ls areal/utils/score_centering.py
```

**预期**：文件存在。如果不存在，先执行 Prompt 01。

### 步骤 2：确认 attention mask 修复尚未应用

```bash
# 检查当前代码是否还有 bug 模式
grep -n "dict(full_attention=None" areal/engine/fsdp_engine.py
```

**预期输出**：找到 2 行（`mb["attention_mask"] = dict(...)` 和 `padded_mb["attention_mask"] = dict(...)`），说明 bug 仍存在，需要应用修复。

**如果输出为空**：说明修复已应用（可能基线已包含修复），跳到步骤 5 做验证。

### 步骤 3：记录修复前状态（可选，用于对比）

```bash
# 保存修复前的 fsdp_engine.py 片段（attention mask 赋值区域）
grep -n -A 5 "attention_mask" areal/engine/fsdp_engine.py | head -30
```

### 步骤 4：应用 patch

```bash
# 方式 A：git apply（推荐）
git apply /path/to/02-attention-mask-fix.patch

# 方式 B：git am（创建 commit）
git am /path/to/02-attention-mask-fix.patch
```

**如果报错**：
- 检查 SC patch 是否已应用（attention mask patch 的 context 依赖 SC patch 后的 fsdp_engine.py 状态）
- 尝试 `--ignore-whitespace` 或 `--3way`

### 步骤 5：验证 bug 模式已消除

```bash
# 不应再找到 dict(full_attention=None) 赋值
grep -n 'mb\["attention_mask"\].*dict' areal/engine/fsdp_engine.py
```

**预期输出**：空（bug 模式已消除）。

### 步骤 6：验证 None 赋值存在

```bash
# 应找到 mb["attention_mask"] = None（在 else 分支中）
grep -n 'mb\["attention_mask"\].*=\s*None' areal/engine/fsdp_engine.py
```

**预期输出**：至少 4 行（Qwen3 分支 2 行 + 非 Qwen3 分支 2 行）。

### 步骤 7：验证修复注释存在

```bash
# 应找到解释修复原因的注释
grep -n "block-diagonal\|create_causal_mask\|packed sequence" areal/engine/fsdp_engine.py
```

**预期输出**：至少 2 行匹配（注释中的关键词）。

### 步骤 8：运行纯 Python 验证脚本

```bash
python3 /path/to/verify_attention_mask_fix.py /path/to/AReaL
```

**预期输出**：
```
  ✅ PASS  no_dict_full_attention
  ✅ PASS  none_assignment
  ✅ PASS  comment_explains_fix
  Results: 3 passed, 0 failed
  ✅ ALL CHECKS PASSED — attention mask fix is correctly applied.
```

### 步骤 9：验证完整 diff（两个 patch 累计）

```bash
# 对比基线 5faead9，查看累计改动
git diff 5faead9 --stat
```

**预期输出**：12 文件（SC patch 12 个 + attention mask patch 1 个重叠文件 = 12 文件），总计 +894/-10 行。

---

## 验收标准

| # | 验收项 | 通过标准 | 命令 |
|---|-------|---------|------|
| S4 | Patch 应用成功 | `git apply` 退出码 0 | `git apply --check 02-attention-mask-fix.patch` |
| S5a | `dict(full_attention=None)` 消除 | grep 无匹配 | `grep 'mb\["attention_mask"\].*dict' fsdp_engine.py` |
| S5b | `None` 赋值存在 | ≥4 行匹配 | `grep 'mb\["attention_mask"\].*None' fsdp_engine.py` |
| S5c | 修复注释存在 | ≥2 行匹配 | `grep 'block-diagonal\|create_causal_mask' fsdp_engine.py` |
| S5d | 验证脚本通过 | 3/3 通过 | `python3 verify_attention_mask_fix.py` |

---

## 完成标志

当且仅当以上 5 项验收全部通过时，Prompt 02 完成。输出验收总结报告。

---

## 关键注意事项

### 这个 bug 是 baseline pre-existing 的

`dict(full_attention=None)` 模式在 `ma-ascend-v1.0.5` 基线代码中就存在（commit `5faead9`），不是 SC 引入的。SC 代码（commit `f8e0b28`）只修改了 `fsdp_engine.py` 的 logits 注入部分，没有碰 attention mask。修复是在 commit `05212ef` 中单独做的。

### 为什么 SC 需要这个修复

SC 的尾模型（rho+alpha+residual）依赖 `train_head_logp`（trainer 对 sampler top-k token 的 logprob）与 `sampler_topk_logps` 的分布一致性。当 attention mask 缺失时，trainer forward 计算的是全局 attention 而非 causal attention，产出的 logits 与 vLLM sampler 的 causal attention logits 严重偏离（Jaccard=0.093），SC 尾模型的前提（top-k 重叠）被彻底违反。

### 上游已知此 bug

AReaL 上游仓库有 PR #1558（`fix(engine): use model-native packed attention masks`，2026-07-22 提交，state=open，未合并）和 issue #1557 / #1442 报告了完全相同的根因。我们的修复方式与 PR #1558 完全一致。
