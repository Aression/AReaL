# Prompt 01 — 应用 SC 基础设施 Patch

## 你的角色

你是一个代码迁移执行 agent。你的任务是在一个干净的 AReaL 基线上，应用 Score Centering (SC) 基础设施 patch，并进行静态验证。

## 前置条件

1. 已有 AReaL 仓库的 `ma-ascend-v1.0.5` 分支（commit `5faead9`）checkout
2. patch 文件 `01-sc-infrastructure.patch` 在可访问路径
3. 环境有 `git` 和 `python3`

## 基线信息

| 维度 | 值 |
|------|-----|
| 仓库 | `https://github.com/HwVanICI/AReaL.git`（或蓝区 mirror） |
| 分支 | `ma-ascend-v1.0.5` |
| Commit | `5faead9`（`fix: remove awex and using awex-ascend`） |
| 日期 | 2026-09-18 |
| vLLM 版本 | v0.23.0 |
| transformers | 5.5.4 |
| 关键特性 | `scale_out` 模块已移除（无环境适配 patch 需求） |

## Patch 信息

| 维度 | 值 |
|------|-----|
| Patch 文件 | `01-sc-infrastructure.patch` |
| 对应 commit | `f8e0b28`（`feat(trainer): add Score Centering drift correction for off-policy RL`） |
| 日期 | 2026-10-21 |
| 净改动 | 12 文件，+883/-6 行 |
| 类型 | `git format-patch`（含 commit message，可用 `git am` 或 `git apply`） |

### Patch 覆盖的 12 个文件

| 文件 | 改动 | 内容概要 |
|------|------|---------|
| `areal/utils/score_centering.py` | **新文件** 278 行 | SC 核心模块：`score_centering_correction`（rho+alpha 尾模型）、`gather_train_head_logp`（chunked fp32 autograd）、`pad_topk_ids/logps`、`check_score_centering_sp_supported` |
| `areal/api/cli_args.py` | +188 行 | 5 个 SC 配置字段（`use_score_centering`, `sc_top_k`, `sc_mode`, `sc_clip_low`, `sc_clip_high`）+ `sc_reinforce_baseline` + 校验逻辑 + `sc_top_k` 传播 |
| `areal/trainer/ppo/actor.py` | +179 行 | `grpo_loss_fn` SC 参数 + SC correction 块（REINFORCE+SC loss 路径 + SC stats） |
| `areal/engine/vllm_remote.py` | +78 行 | `_token_to_id`/`_parse_top_logprobs` 辅助函数 + `logprobs=sc_top_k` 请求 + `parse_generation_response` SC top-k 解析 |
| `areal/engine/fsdp_engine.py` | +11 行 | logits 注入 + SP 校验（`check_score_centering_sp_supported`） |
| `areal/experimental/openai/types.py` | +38 行 | `InteractionWithTokenLogpReward` SC top-k 传播 |
| `areal/workflow/rlvr.py` | +45 行 | `RLVRWorkflow` sampler top-k 数据流（截断 k_dim + padding） |
| `areal/infra/remote_inf_engine.py` | +24 行 | top-k 累积 + `ModelResponse` 传递 |
| `areal/utils/data.py` | +26 行 | `concat_padded_tensors` -inf padding 白名单 + 3D tensor 拆分 |
| `areal/api/io_struct.py` | +7 行 | `ModelResponse` + `HttpGenerationResult` output_topk 字段 |
| `areal/experimental/openai/client.py` | +10 行 | 三个类的 `sc_top_k` 参数和传递 |
| `areal/experimental/openai/proxy/proxy_rollout_server.py` | +1 行 | `sc_top_k` 传递 |

---

## 执行步骤

### 步骤 1：确认基线

```bash
# 确认当前在 ma-ascend-v1.0.5 分支，HEAD = 5faead9
cd /path/to/AReaL
git branch --show-current
git rev-parse HEAD
```

**预期输出**：分支名为 `ma-ascend-v1.0.5`（或 checkout `5faead9`），HEAD 短 hash 为 `5faead9`。

### 步骤 2：确认工作树干净

```bash
git status --short
```

**预期输出**：空（无未提交修改）。如有修改，先 stash 或新建 worktree。

### 步骤 3：应用 patch

```bash
# 方式 A：git apply（推荐，不创建 commit）
git apply /path/to/01-sc-infrastructure.patch

# 方式 B：git am（创建 commit，保留 commit message）
git am /path/to/01-sc-infrastructure.patch
```

**如果 `git apply` 报错**：
- 检查是否在正确的基线上（`5faead9`），patch 的 context 行必须匹配
- 检查是否有行尾符差异（Windows CRLF vs Linux LF）——如在 Windows 上生成 patch 而在 Linux 上 apply，可能需要 `--ignore-whitespace`：
  ```bash
  git apply --ignore-whitespace /path/to/01-sc-infrastructure.patch
  ```
- 如果基线已有其他修改导致 context 偏移，尝试 `--3way`：
  ```bash
  git apply --3way /path/to/01-sc-infrastructure.patch
  ```

### 步骤 4：验证 diff stat

```bash
git diff --stat
```

**预期输出**（12 文件，总计 +883/-6）：
```
 areal/api/cli_args.py                              | 188 ++++++++++++++
 areal/api/io_struct.py                             |   7 +
 areal/engine/fsdp_engine.py                        |  11 +
 areal/engine/vllm_remote.py                        |  78 +++++-
 areal/experimental/openai/client.py                |  10 +
 areal/experimental/openai/proxy/proxy_rollout_server.py | 1 +
 areal/experimental/openai/types.py                 |  38 +++
 areal/infra/remote_inf_engine.py                   |  24 ++
 areal/trainer/ppo/actor.py                         | 179 ++++++++++++-
 areal/utils/data.py                                |  26 +-
 areal/utils/score_centering.py                     | 278 +++++++++++++++++++++
 areal/workflow/rlvr.py                             |  49 +++-
 12 files changed, 883 insertions(+), 6 deletions(-)
```

### 步骤 5：验证新文件存在

```bash
# score_centering.py 是新文件，必须存在
ls -la areal/utils/score_centering.py
wc -l areal/utils/score_centering.py
```

**预期输出**：文件存在，约 278 行。

### 步骤 6：验证关键函数存在

```bash
# 检查 score_centering.py 中的核心函数
grep -n "^def \|^class " areal/utils/score_centering.py
```

**预期输出**：
```
def pad_topk_ids(...)
def pad_topk_logps(...)
def check_score_centering_sp_supported(...)
class _TopKLogProbsFromLogits(...)
def gather_train_head_logp(...)
def _build_weight_fn(...)
def score_centering_correction(...)
```

### 步骤 7：验证 SC 配置字段

```bash
# 检查 cli_args.py 中新增的 SC 配置字段
grep -n "use_score_centering\|sc_top_k\|sc_mode\|sc_clip_low\|sc_clip_high\|sc_reinforce_baseline" areal/api/cli_args.py
```

**预期输出**：应找到上述字段定义（5+ 字段）+ 校验逻辑（`raise ValueError` 多处）。

### 步骤 8：验证 actor.py SC 集成

```bash
# 检查 actor.py 中 SC correction 调用
grep -n "score_centering_correction\|gather_train_head_logp\|sc_correction\|REINFORCE" areal/trainer/ppo/actor.py | head -20
```

**预期输出**：应找到 SC correction 块（`if use_score_centering:`）+ REINFORCE+SC loss 路径。

### 步骤 9：验证数据流传播

```bash
# 检查 SC top-k 数据流（vLLM → io_struct → remote_inf_engine → rlvr/types → actor）
grep -rn "sampler_topk_logps\|sampler_topk_token_ids\|output_topk" areal/ --include="*.py" | grep -v __pycache__ | head -30
```

**预期输出**：在 `io_struct.py`、`vllm_remote.py`、`remote_inf_engine.py`、`rlvr.py`、`types.py`、`actor.py` 中找到 `sampler_topk_logps` / `output_topk` 字段。

### 步骤 10：验证 -inf padding 白名单

```bash
# 检查 data.py 中 -inf padding 白名单
grep -n "sampler_topk_logps\|ref_logprobs\|float.*-inf" areal/utils/data.py
```

**预期输出**：应找到 `sampler_topk_logps`, `ref_logprobs`, `logprobs` 三个 key 的 -inf padding 白名单。

---

## 验收标准

| # | 验收项 | 通过标准 | 命令 |
|---|-------|---------|------|
| S1 | Patch 应用成功 | `git apply` 退出码 0 | `git apply --check 01-sc-infrastructure.patch` |
| S2 | diff stat 正确 | 12 文件，+883/-6 | `git diff --stat` |
| S3 | `score_centering.py` 新文件存在 | 278 行，含 7 个 def/class | `wc -l` + `grep "^def \|^class "` |
| S4 | SC 配置字段存在 | 5+ 字段 + 校验 | `grep use_score_centering cli_args.py` |
| S5 | actor.py SC 集成 | `score_centering_correction` 调用存在 | `grep score_centering_correction actor.py` |
| S6 | 数据流传播完整 | 6 文件含 `sampler_topk_logps` | `grep -rn sampler_topk_logps areal/` |
| S7 | -inf padding 白名单 | 3 个 key | `grep "sampler_topk_logps" data.py` |

---

## 完成标志

当且仅当以上 7 项验收全部通过时，Prompt 01 完成。输出验收总结报告，包含每项的 ✅/❌ 状态和实际输出。

---

## 常见问题

### Q: patch 应用报 "patch does not apply"
**A**: 最常见原因是基线不对。确认 `git rev-parse HEAD` 是 `5faead9`。如果基线有额外 commit（如 gradient decomposition 相关的实验 commit），需要 checkout 到干净的 `5faead9`。

### Q: `score_centering.py` 的 `import torch` 报错
**A**: 这是正常的——本步骤只做静态 patch 应用，不运行代码。torch 依赖在运行时验收（Prompt 04）中验证。

### Q: patch 里的 commit message 提到 "arXiv:2609.20807"，这个论文是什么
**A**: "Score Centering Stabilizes Off-policy Reinforcement Learning"（Marek & Ryabinin），SC 算法的原始论文。patch 实现的是该论文的 additive drift correction 机制。
