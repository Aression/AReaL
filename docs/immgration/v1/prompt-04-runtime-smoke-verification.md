# Prompt 04 — NPU 环境运行时验收

## 你的角色

你是一个运行时验收 agent。你的任务是在 NPU 环境上运行 2-step smoke 训练，验证 Score Centering (SC) + attention mask 修复端到端工作正常。

## 前置条件

1. **Prompt 01-03 已全部完成**：两个 patch 已应用，静态验收通过
2. **NPU 环境**：Ascend NPU 机器 + a3 镜像（CANN 9.x，vLLM 0.23.0，transformers 5.5.4）
3. **模型就绪**：Qwen2.5-1.5B-Instruct 模型权重可访问
4. **数据集就绪**：GSM8K 数据集（save_to_disk 格式或可下载）

## 环境要求

| 维度 | 要求 | 说明 |
|------|------|------|
| NPU | Ascend 910（≥3 卡） | 2+1 separation：actor 2 卡 + rollout 1 卡 |
| 驱动 | ≥25.2.1 | a3 镜像（CANN 9.x）在 25.2.1 验证通过 |
| 镜像 | a3（`v1.0.5-a3`） | 含 vLLM 0.23.0 + transformers 5.5.4 |
| 容器配置 | `--runtime=ascend --privileged` | 必须 ascend runtime（runc 导致 `aclInit` 失败） |
| 卡号 | 容器内 local index（0,1,2） | 非 NPU 物理卡号（物理卡号导致 `device_count=0`） |
| attn_impl | `eager` | 镜像无 `flash_attn` 包，transformers 默认 flash_attention_2 会 ImportError |

---

## 执行步骤

### 步骤 1：环境检查

```bash
# 确认 NPU 可见
npu-smi info

# 确认容器内 NPU 数量
python3 -c "import torch; import torch_npu; print(f'NPU count: {torch.npu.device_count()}')"
```

**预期**：NPU count ≥ 3。

### 步骤 2：准备模型和数据

```bash
# 确认模型路径（Qwen2.5-1.5B-Instruct）
ls /path/to/models/Qwen2.5-1.5B-Instruct/

# 准备 GSM8K 数据集（save_to_disk 格式）
# 如果已有预打包数据集，确认路径
ls /path/to/dataset/gsm8k/
```

### 步骤 3：准备 smoke 配置

创建 2-step smoke 配置文件 `gsm8k_grpo_npu_sc_smoke.yaml`，关键配置项：

```yaml
# 模型
actor:
  model:
    path: /path/to/models/Qwen2.5-1.5B-Instruct
  generation_batch_size: 30
  micro_batch_size_per_gpu: 5
  use_score_centering: true           # ← SC 开启
  sc_top_k: 128                       # ← top-k = 128
  sc_mode: tis                        # ← truncated IS
  sc_clip_low: 0.2
  sc_clip_high: 5.0
  # attn_impl 必须是 eager（镜像无 flash_attn）
  attn_impl: eager

rollout:
  model:
    path: /path/to/models/Qwen2.5-1.5B-Instruct
  tensor_parallel_size: 1
  gpu_memory_utilization: 0.5         # ← 注意：不是默认 0.85，避免 OOM

generation:
  sc_top_k: 128                       # ← 采样 top-k logprobs
  temperature: 1.0
  top_p: 1.0
  max_len: 512

# 训练参数
total_train_epochs: 1
max_steps: 2                          # ← 2-step smoke
learning_rate: 1e-6                   # ← 极小 lr，验证流程不训练模型

# 数据
dataset:
  train_files: /path/to/dataset/gsm8k/train.parquet
  val_files: null
```

### 步骤 4：启动 2-step smoke 训练

```bash
# 2+1 separation：actor 用卡 0,1，rollout 用卡 2
ASCEND_RT_VISIBLE_DEVICES=0,1,2 python3 -m areal.main \
    --config gsm8k_grpo_npu_sc_smoke.yaml \
    --entrypoint gsm8k_rl_rlvr.py \
    2>&1 | tee sc_smoke.log
```

**关键注意事项**：
- `ASCEND_RT_VISIBLE_DEVICES=0,1,2` 用容器内 local index，**不是** NPU 物理卡号
- `--runtime=ascend --privileged` 容器启动参数（docker run 时）
- `attn_impl: eager` 必须设置（否则 FlashAttention2 ImportError）
- `gpu_memory_utilization: 0.5`（不是 0.85，避免 compute_logprobs OOM）

### 步骤 5：监控训练日志

```bash
# 持续监控日志
tail -f sc_smoke.log

# 检查关键里程碑
grep -E "Training completes|vLLM.*init|rollout.*generation|ppo_update|weight.*update" sc_smoke.log
```

---

## 验收标准

### R1: 训练完成

```bash
grep "Training completes" sc_smoke.log
```

**通过标准**：日志中出现 `Training completes!`。

**失败处理**：
- 如果崩溃在 vLLM init：检查 `--runtime=ascend`、卡号映射、模型路径
- 如果崩溃在 OverflowError：检查 `return_tokens_as_token_ids: True` 是否在 vLLM 请求中（SC patch 的 `vllm_remote.py` 自动设置）
- 如果崩溃在 `Port XXXXX is in use`：使用 `--network=bridge` 给容器独立网络命名空间
- 如果崩溃在 HBM OOM：确认 2+1 separation（actor 独占 2 卡，rollout 独占 1 卡）

### R2: 0 OverflowError

```bash
grep -c "OverflowError" sc_smoke.log
```

**通过标准**：计数为 0。

### R3: 0 NaN/Inf/HCCL

```bash
# 检查 NaN/Inf
grep -c "NaN\|Inf\|HCCL" sc_smoke.log

# 更精确：检查训练指标中的 NaN/Inf
grep "nan\|inf" sc_smoke.log | grep -v "logprobs.*-inf\|NEG_INF_PAD\|pad.*-inf" | head -5
```

**通过标准**：0 次异常 NaN/Inf/HCCL（排除 -inf padding 的正常日志）。

### R4: SC correction 非零

```bash
# SC correction 应非零（说明 SC 计算路径激活且有效）
grep "sc_correction" sc_smoke.log | head -5
```

**通过标准**：`sc_correction/avg` 非零且非 NaN（典型范围 10-20）。如果 ≈0，可能是 on-policy 场景（recompute_logprob=true）——这是正确的 no-op 行为，但说明 SC 无实际效果（需要 recompute_logprob=false 才能看到 SC 校正效果）。

### R5: q_mass 正常

```bash
# q_mass 应 ≤ 1.0（-inf padding 生效的标志）
grep "sc_q_mass" sc_smoke.log | head -5
```

**通过标准**：`sc_q_mass/avg` ≤ 1.0（典型 0.8-1.0）。如果 > 1.0，说明 -inf padding 白名单未生效（Bug 2 未修复）。

### R6: rho 正常

```bash
# rho 应在 O(1) 量级（1-10），不爆炸
grep "sc_rho" sc_smoke.log | head -5
```

**通过标准**：`sc_rho/avg` 在 1-10 量级。如果 100+，说明尾模型退化（TIM 仍很大或 attention mask 修复未生效）。

### R7: SC 耗时指标

```bash
# SC 耗时打点应出现在日志中
grep "timeperf/sc_correction\|timeperf/sc_gather_train_head_logp" sc_smoke.log | head -5
```

**通过标准**：两个 timeperf 指标出现在日志中，与 `recompute_logp` / `train_step` 同级。典型值 `sc_gather_train_head_logp` ~0.2-0.6ms，`sc_correction` ~0.3-0.5ms。

---

## 验收总结表

| # | 验收项 | 通过标准 | 命令 |
|---|-------|---------|------|
| R1 | 训练完成 | `Training completes!` 出现 | `grep "Training completes" sc_smoke.log` |
| R2 | 0 OverflowError | 计数 = 0 | `grep -c "OverflowError" sc_smoke.log` |
| R3 | 0 NaN/Inf/HCCL | 0 次异常 | `grep -c "NaN\|Inf\|HCCL" sc_smoke.log` |
| R4 | SC correction 非零 | `sc_correction/avg` 非零非 NaN | `grep "sc_correction" sc_smoke.log` |
| R5 | q_mass 正常 | `sc_q_mass/avg` ≤ 1.0 | `grep "sc_q_mass" sc_smoke.log` |
| R6 | rho 正常 | `sc_rho/avg` 在 1-10 量级 | `grep "sc_rho" sc_smoke.log` |
| R7 | SC 耗时指标 | 两个 timeperf 指标出现 | `grep "timeperf/sc_" sc_smoke.log` |

---

## 完成标志

当且仅当 R1-R7 全部通过时，Prompt 04 完成。输出验收总结报告，包含每项的 ✅/❌ 状态和实际日志输出。

**全部 4 个 Prompt 完成后，两个 commit（`f8e0b28` + `05212ef`）已在蓝区基线上完整复现，可以创建 PR。**

---

## 常见问题

### Q: `aclInit` 失败 / NPU 不可见
**A**: 容器启动参数需要 `--runtime=ascend --privileged`。runc runtime 会导致 aclInit 失败。确认 `docker run --runtime=ascend --privileged ...`。

### Q: `device_count=0`
**A**: `ASCEND_RT_VISIBLE_DEVICES` 用容器内 local index（0,1,2），不是 NPU 物理卡号。物理卡号在 NPU group 边界外时导致 device_count=0。

### Q: FlashAttention2 ImportError
**A**: 设置 `attn_impl: eager`。a3 镜像无 `flash_attn` 包，transformers 默认 flash_attention_2 会 ImportError。

### Q: HBM OOM（compute_logprobs 阶段）
**A**: `gpu_memory_utilization: 0.5`（不是 0.85）。vLLM 在 0.85 时预分配 85% HBM 给 KV cache，compute_logprobs 需要分配 logits 张量（vocab=151936 × max_len 在 fp32 下很大）。0.5 留出 50% 给 logprobs 计算。

### Q: sc_correction ≈ 0
**A**: 如果 `recompute_logprob=true`（默认），trainer 用 forward pass 重算 old_logp，TIM≈0，SC correction≈0 是正确的 no-op。要看到 SC 实际效果，需设置 `recompute_logprob=false`。但 2-step smoke 验证的只是"SC 代码路径激活且不崩溃"，correction≈0 在 recompute=true 下是预期行为。

### Q: sc_q_mass > 1.0
**A**: 检查 `areal/utils/data.py` 的 `concat_padded_tensors` 是否正确对 `sampler_topk_logps` key 使用 -inf padding。运行 `verify_sc_math.py` 的测试 6+7 确认 -inf padding 逻辑正确。

### Q: sc_rho 100+
**A**: 这说明 TIM 仍很大（尾模型退化）。检查 attention mask 修复是否生效——运行 `python3 verify_attention_mask_fix.py /path/to/AReaL` 确认 3/3 通过。如果修复已应用但 rho 仍高，可能是 NPU 上 vLLM 的 logprobs 归一化问题（vLLM-ascend 内部实现差异），需进一步排查。
