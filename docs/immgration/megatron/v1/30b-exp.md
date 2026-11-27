# Score Centering 在 30B MoE 训练中的效果简报

**实验**：`2026-11-25_t2`（有噪声）/ `2026-11-26_t2c`（无噪声）
**模型**：Qwen3-30B-A3B（MoE），LoRA r=16 / α=16
**硬件**：8 卡（两池各 4 卡：c-k4 = davinci 8–11，c-pn = davinci 12–15）
**日期**：2026-11-25 → 2026-11-27

---

## 1. 一句话结论

**当训练与推理之间存在真实失配（`kl_div_direct ≈ 0.10`）时，Score Centering（k=4）带来 +0.045 的 rollout accuracy（相对 +17%，t = +11.1）；失配不存在（`kl ≈ 0.001`）时 SC 无作用（−0.024，t = −1.76，不显著）。SC 的墙钟开销为 0，额外传输 0.4 MB/步。**

---

## 2. 训练数据

### 2.1 来源

| 项 | 内容 |
|---|---|
原始数据集 | `PrimeIntellect/INTELLECT-2-only-math`（7 个 parquet 分片，约 45 MB） |
选用原因 | Score Centering 论文的 30B MoE 实验即用此数据集；GSM8K 在 30B 上准确率约 0.93（headroom 仅 7%），无法观测 SC 效应 |
转换脚本 | `$WS/scripts/build_intellect_math.py` |

### 2.2 筛选与转换

- **筛选带宽**：按参考模型字段 `solve_rate_qwen_r1_distill_7b ∈ [0.30, 0.70]` 过滤，即“参考策略能解一半左右”的题目——既不过易（无梯度信号）也不过难（组内全错，GRPO 优势恒为 0）。原始数据共 79 284 条落在参考带内，本次取 **8000 train / 128 valid**。
- **随机化**：写出前用 `random.Random(0)` 打乱，再取前 8000 条。
- **字段**：仅保留 `question`（str）与 `answer`（str，LaTeX 原样保留），以复用 AReaL 的 `type: rl` gsm8k 加载器。
- **目录布局**（`datasets` 可直接识别）：

```
gsm8k_mathmid/
├── README.md          # config_name: main；train-* / test-* 两个 split
└── main/
    ├── train-00000.parquet   (8000 行)
    └── test-00000.parquet    (128 行)
```

- **命令**：

```bash
python3 build_intellect_math.py \
  --src /ce-efs-hd2-01/l00993886/areal_npu/data/_incoming/mathhard \
  --out /ce-efs-hd2-01/l00993886/areal_npu/data/gsm8k_mathmid \
  --lo 0.30 --hi 0.70 --n-train 8000 --n-valid 128
```

- **奖励**：`areal/reward/gsm8k.py` 走 `math_verify` worker，接受 LaTeX / 数值等价。
- **实际难度**：本批数据上 30B 的 rollout 准确率约 **0.26–0.38**（有噪声 / 无噪声），远低于 GSM8K 的 0.93，headroom 充足。

---

## 3. 噪声构造方法（制造训推失配）

### 3.1 为什么需要它

本次实验的自变量是“训练引擎与推理引擎之间的失配”。自然情况下 30B 的失配只有 `kl_div_direct ≈ 0.001`（太小，SC 无可校正内容）。因此对 **rollout 侧（vLLM）的权重副本**注入相对高斯噪声，人为制造可控的失配。

### 3.2 数学形式

逐元素**乘性**噪声（与 Score Centering 参考实现一致）：

```
w ← w · (1 + σ · z),    z ~ N(0, 1) 逐元素独立
```

- 每个元素按**自身量级**缩放，接近 0 的权重仍接近 0 → 相对扰动在全张量上均匀（约 σ）；
- 对比另一种 `rms_add`（`w ← w + σ·RMS(w)·z`）：Frobenius 范数相同，但 |w| ≪ RMS(w) 的元素会被放大到极大相对扰动，会直接破坏 MoE router。本次采用 `element_mul`。

### 3.3 作用范围与时机

| 项 | 取值/说明 |
|---|---|
作用对象 | **vLLM rollout 侧的权重副本**（被服务的模型），不改变训练侧 |
跳过 LoRA 适配器 | `AREAL_WEIGHT_NOISE_SKIP_LORA=1`（默认）——基座权重被扰动，适配器保持原样 |
生效时机 | 在 rollout 引擎 `load_model()` 时施加**一次** |
模式 | `static`：一次抽样、整个 run 不再刷新（等价于论文的固定失配注入器） |
随机性 | `torch.Generator` 以 `seed + call` 播种，**同 (mode, σ, seed) 可完全复现** |

### 3.4 本次实际使用

| 环境变量 | 值 |
|---|---|
`AREAL_WEIGHT_NOISE_RMS` | **0.12** |
`AREAL_WEIGHT_NOISE_MODE` | `static` |
`AREAL_WEIGHT_NOISE_SHAPE` | `element_mul` |
`AREAL_WEIGHT_NOISE_SEED` | `42` |
（无噪声两臂） | 以上变量全部不设置 |

实现文件：`$WS/patches/noise_lora/areal_weight_noise.py`（含 `static / refresh / ramp / walk` 四种模式，本次只用 `static`）。

### 3.5 剂量—失配标定（本数据集实测）

| σ | `kl_div_direct` | 备注 |
|---|---|---|
0（无噪声） | **0.0010** | 自然失配 |
0.05 | 0.072 | |
0.10 | 0.073 | |
**0.12** | **0.112** | **本次采用，接近 0.1 目标** |
0.15 | 0.194 | 过冲，acc 明显下降 |

---

## 4. 2×2 设计

| 臂 | 噪声 | SC | 训练步数（健康窗口） |
|---|---|---|---|
`T2-off` | σ=0.12，element_mul | off | 207 |
`T2-on` | σ=0.12，element_mul | **on，k=4** | 207 |
`T2-clean-off` | 无 | off | 137 |
`T2-clean-on` | 无 | **on，k=4** | 137 |

其余超参完全一致：`n_samples=8`、`train batch=16`、`max_new_tokens=1024`、`lr=1e-5`（Adam）、LoRA r16/α16、`weight_update_mode=disk`、`vllm.max_logprobs = 4（SC on）/ 20（SC off）`、`sc_mode=none`、`sc_clip_low/high = 0.5/5.0`、`AREAL_SC_FLAT_HEADS=1`、`AREAL_SC_NODETOK=1`。

> 两批实验都在接近 300 步时因 vLLM 服务退出而停滞；下表统计量全部基于**截断到首个异常步之前**的健康窗口。

---

## 5. 效果

| 条件 | `kl_div_direct` | `ρ` | SC off | SC on | 配对差均值 | 中位数 | 截尾均值 | se | **t** |
|---|---|---|---|---|---|---|---|---|---|
**有噪声 σ=0.12** | 0.097 / 0.112 | **4.85** | 0.2603 | **0.3054** | **+0.0451** | +0.0312 | +0.0389 | 0.0041 | **+11.05** |
**无噪声** | 0.00095 | **1.004** | 0.3823 | 0.3588 | −0.0235 | 0.0000 | −0.0217 | 0.0134 | −1.76 |

- **噪声代价**：0.3823 → 0.2603，约 **−0.12 acc**；
- **SC 回收**：+0.045，相当于收回噪声损失的 **约 38%**；
- **机制证据**：`ρ` 在有失配时 4.85（强校正），无失配时恒为 **1.00**（无校正空间）——与 SC 定义一致。

### 效应随训练演化（有噪声条件，20 步分段）

| 步段 | 配对差 |
|---|---|
1–20 | +0.002 |
61–80 | +0.014 |
101–120 | +0.041 |
141–160 | +0.083 |
181–207 | +0.101 |

→ 单调扩大，符合“失配累积 → 校正量累积 → 收益累积”。

### 曲线数据（单文件、分表）

同目录 **`SC_30B_2x2_all.csv`**：一个 CSV 文件包含两张表，用 `table` 列区分。

| `table` | 行数 | 含义 |
|---|---|---|
`acc` | 688 | 四臂逐步曲线（207 + 207 + 137 + 137） |
`paired` | 344 | 两种条件的逐步配对差（on − off，207 + 137） |

| 列 | 含义 |
|---|---|
`table` | `acc` = 单臂曲线；`paired` = 配对差 |
`arm_or_condition` | `acc` 表：`T2-off` / `T2-on` / `T2-clean-off` / `T2-clean-on`；`paired` 表：`noise` / `clean` |
`noise` | `noise`（σ=0.12）/ `clean`（无噪声） |
`sc` | `off` / `on`；`paired` 表为 `on-off` |
`top_k` | SC 的 k（4；off 臂为 0） |
`step` | 训练步号（1 起） |
`acc` | 该步 rollout 准确率 `1 − incorrect_n_seqs / n_seqs` |
`acc_ma10` | 上述 acc 的 10 步滑动平均 |
`kl_direct` | `kl_div_direct/avg`（推理减训练的逐 token logprob 均差） |
`rho` | `sc_rho`（仅 SC on 臂有值） |
`diff` / `diff_ma10` | 仅 `paired` 表：逐步配对差及其 10 步滑动平均 |

---

## 6. 时间与传输损耗

### 6.1 墙钟（中位数）

| 臂 | 每步 | p10 / p90 | steps/h |
|---|---|---|---|
SC off（σ=0.12） | 217 s | 206 / 245 | 16.6 |
SC on（σ=0.12） | **214 s** | 204 / 266 | 16.8 |
SC off（clean） | 214 s | 204 / 293 | 16.8 |
SC on（clean） | **209 s** | 200 / 218 | 17.2 |

**SC 的墙钟开销为 0**（差异 ±2%，运行间噪声）。

### 6.2 分项

| 分项 | off(σ) | on(σ) | off(clean) | on(clean) |
|---|---|---|---|---|
`rollout` | 165.3 s | 165.8 s（+0.3%） | 167.5 s | 163.4 s（−2.4%） |
`train_step` | 40.4 s | 38.2 s（−5.4%） | 34.7 s | 36.2 s（+4.5%） |
LoRA 同步 | 8.75 s | 8.64 s（−1.3%） | 8.72 s | 8.70 s（−0.2%） |

### 6.3 额外传输（top-k logprob）

按每步训练 100 170 条 response token、每条 8 B 估算：

| k | 每步 | 每小时（16.8 步/h） |
|---|---|---|
**4（本次）** | **0.4 MB** | **6.9 MB/h** |
8 | 0.8 MB | 13.7 MB/h |
32 | 3.2 MB | 55 MB/h |
128（论文设置） | 12.8 MB | 220 MB/h |

k 越大，vLLM 侧 logprob 后处理的内存/计算压力越大（0.6B 上 k=128 曾触发 NPU OOM）。

---

## 7. 推荐配置

### 7.1 何时开 SC

| 判据 | 建议 |
|---|---|
`kl_div_direct ≥ ~0.05` 且 `ρ ≥ 2` | **开**（本次 +0.045） |
`kl_div_direct < ~0.01`（`ρ ≈ 1`） | **不开**（无收益） |
不确定时 | 先跑 20 步读 `sc_rho`，`ρ > 1.5` 再开 |

### 7.2 参数

| 参数 | 推荐值 |
|---|---|
`actor.use_score_centering` | true（满足 §7.1） |
`gconfig / rollout .score_centering_top_k` | **4**（30B MoE） |
`vllm.max_logprobs` | **= k（4）** |
`sc_mode` | `none` |
`sc_clip_low / sc_clip_high` | 0.5 / 5.0 |
`AREAL_SC_FLAT_HEADS` / `AREAL_SC_NODETOK` | `1` / `1` |
若需人为失配 | `AREAL_WEIGHT_NOISE_RMS=0.12` + `MODE=static` + `SHAPE=element_mul` + `SEED=42` |

---

## 8. 限制与未确认项

| 项 | 状态 |
|---|---|
单 seed、单 run | 主结论尚未独立复现 |
未跑满 300 步 | 207 / 137 步；两批均因 vLLM 服务退出停滞（`max_new_tokens (-158) is non-positive`，存在超长 prompt） |
两臂 `kl` 不完全相同（0.097 vs 0.112） | 噪声抽样差异；方向不受影响，复现时应固定噪声种子 |
k 的影响未在 30B 上扫描 | 仅测 k=4 |
clean 侧存在 ±0.05 段波动 | 已用中位数/截尾均值处理，仍不能排除“小幅负代价” |

---

## 9. 复现要点

1. 数据：`/ce-efs-hd2-01/l00993886/areal_npu/data/gsm8k_mathmid`（用 `build_intellect_math.py --lo 0.30 --hi 0.70`）
2. 配置：`$WS/configs/T2-{off,on,clean-off,clean-on}.yaml`
3. 队列：`queue_task2.json`（有噪声）/ `queue_task2_clean.json`（无噪声）
4. 归档：`runs/2026-11-25_t2/`、`runs/2026-11-26_t2c/`（含 `attempt1/stdout.log`）
5. 已知坑：`cluster.fileroot` 需按臂名生成；需过滤超长 prompt（否则 `max_new_tokens` 变负）；vLLM 服务退出后 orchestrator 会重启（日志出现 `attempt 2`），统计必须用健康窗口。
