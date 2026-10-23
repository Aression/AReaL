# SC 迁移 Prompt 总导航

> 本目录包含将 Score Centering (SC) 特性 + attention mask 修复迁移到蓝区（CodeHub）基线的分步 prompt。
> 每份 prompt 职责清晰、可独立执行，按顺序执行可完整复现两个 commit 的全部改动。

---

## 背景

我们的工作基于 `HwVanICI/AReaL` 的 `ma-ascend-v1.0.5` 分支（commit `5faead9`），在上面添加了两个 commit：

| Commit | 内容 | Patch 文件 |
|--------|------|-----------|
| `f8e0b28` | **SC 基础设施**：Score Centering drift correction 完整实现（12 文件，+883/-6） | `01-sc-infrastructure.patch` |
| `05212ef` | **Attention mask 修复**：packed 序列 attention mask 从 `dict(None)` 改为 `None`（1 文件，+11/-4） | `02-attention-mask-fix.patch` |

**关键约束**：蓝区环境只有基础 Python，没有 torch / 没有 NPU / 没有 GPU。因此验收分两层：
- **静态验收**（无需 torch）：patch 应用 + 文件比对 + 纯 Python 数学验证 + 代码模式检查
- **运行时验收**（需 NPU/GPU 环境）：2-step smoke + SC 指标 + 0 NaN/Inf

---

## Prompt 清单

| # | Prompt 文件 | 职责 | 依赖 |
|---|-----------|------|------|
| 1 | `prompt-01-apply-sc-infrastructure.md` | 在基线上应用 SC 基础设施 patch，静态验证 12 文件改动正确 | 基线 `5faead9` |
| 2 | `prompt-02-apply-attention-mask-fix.md` | 在 SC 基础上应用 attention mask 修复 patch，验证修复正确 | Prompt 1 完成 |
| 3 | `prompt-03-verify-math-and-structure.md` | 纯 Python 验收 SC 核心数学 + attention mask 代码模式（无需 torch） | Prompt 1+2 完成 |
| 4 | `prompt-04-runtime-smoke-verification.md` | NPU 环境运行时验收：2-step smoke + SC 指标 + 0 NaN/Inf | Prompt 1+2+3 完成，NPU 环境 |

---

## 执行流程

```
基线 5faead9（ma-ascend-v1.0.5）
    │
    ▼  Prompt 01: git apply 01-sc-infrastructure.patch
    │  → 静态验证 12 文件
    ▼
SC 基础设施就绪
    │
    ▼  Prompt 02: git apply 02-attention-mask-fix.patch
    │  → 验证 fsdp_engine.py 修复
    ▼
两个 commit 内容完整复现
    │
    ▼  Prompt 03: python3 verify_sc_math.py + verify_attention_mask_fix.py
    │  → 纯 Python 验收（无需 torch）
    ▼
静态验收通过
    │
    ▼  Prompt 04: NPU 2-step smoke（需要 NPU 环境）
    │  → 运行时验收
    ▼
完整验收通过 → 创建 PR
```

## 验收标准总览

### 静态验收（Prompt 01-03，无需 torch）

| # | 验收项 | 通过标准 |
|---|-------|---------|
| S1 | SC patch 应用 | `git apply --check` 退出码 0 |
| S2 | 12 文件改动 | `git diff --stat` 显示 12 文件 +883/-6 |
| S3 | `score_centering.py` 新文件存在 | 278 行，包含 `score_centering_correction` 函数 |
| S4 | Attention mask patch 应用 | `git apply --check` 退出码 0 |
| S5 | `dict(full_attention=None)` 消除 | `verify_attention_mask_fix.py` 通过 3/3 |
| S6 | SC 核心数学 | `verify_sc_math.py` 通过 12/12 |

### 运行时验收（Prompt 04，需 NPU）

| # | 验收项 | 通过标准 |
|---|-------|---------|
| R1 | 训练启动 | `Training completes!`，2-step smoke |
| R2 | 0 OverflowError | 日志中 0 次 OverflowError |
| R3 | 0 NaN/Inf | 日志中 0 次 NaN/Inf/HCCL |
| R4 | SC correction 非零 | `sc_correction/avg` ≈ 14-16（非零，非 NaN） |
| R5 | q_mass 正常 | `sc_q_mass/avg` ≤ 1.0（-inf padding 生效） |
| R6 | rho 正常 | `sc_rho/avg` 在 O(1) 量级（1-10），不爆炸（非 100+） |
| R7 | SC 耗时指标 | `timeperf/sc_correction` + `timeperf/sc_gather_train_head_logp` 出现在日志中 |

---

## 文件清单

```
docs/迁移prompts/v1/
├── README.md                        ← 本文件（总导航）
├── prompt-01-apply-sc-infrastructure.md   ← Prompt 1
├── prompt-02-apply-attention-mask-fix.md  ← Prompt 2
├── prompt-03-verify-math-and-structure.md ← Prompt 3
├── prompt-04-runtime-smoke-verification.md ← Prompt 4
├── 01-sc-infrastructure.patch        ← SC patch（git format-patch，commit f8e0b28）
├── 02-attention-mask-fix.patch       ← Attention mask patch（git format-patch，commit 05212ef）
├── verify_sc_math.py                 ← 纯 Python SC 数学验证脚本（12 测试）
└── verify_attention_mask_fix.py      ← 纯 Python attention mask 验证脚本（3 测试）
```
