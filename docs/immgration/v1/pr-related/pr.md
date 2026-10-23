## Description

This PR adds **Score Centering (SC)** — an additive drift-correction mechanism for off-policy RL — to the AReaL training framework, and fixes a pre-existing attention-mask bug in the FSDP engine that caused catastrophic training-inference mismatch (TIM) on packed sequences.

### Score Centering (`feat(trainer)`)

SC addresses off-policy drift: the sampler (vLLM) and the trainer (FSDP) compute log-probs in separate forward passes, and the resulting mismatch inflates importance-sampling ratios and destabilizes training. SC subtracts a low-variance additive correction term from the actor loss:

```
correction = sum_H [sg(q*w - alpha*p)] * log(p)
```

where **H** is the sampler's top-k head, **p** and **q** are the trainer and sampler probabilities for those tokens, **w** = `weight_fn(p/q)` is the head IS weight, **rho** = `(1 - q_mass) / (1 - p_mass)` is the tail mass ratio, and **alpha** = `rho * weight_fn(1/rho)` is the tail IS weight. The `sg[.]` (stop-gradient) ensures gradient flows only through the trainer logits; all sampler quantities are detached.

On-policy (p = q), the residual `q*w - alpha*p = 0`, so **correction = 0** — SC is an exact no-op on-policy and only activates when drift exists.

Three IS combination modes are supported via `sc_mode`:
- `none` — pure SC (weight_fn = identity)
- `tis` — truncated IS (upper-clamp ratio)
- `mis` — mixture IS / IcePop (band-mask ratio)

Reference: "Score Centering Stabilizes Off-policy Reinforcement Learning" (arXiv:2609.20807, Marek & Ryabinin). The implementation mirrors VeRL (`verl/trainer/ppo/score_centering.py`) and Slime (PR #2405).

### Attention mask fix (`fix(engine)`)

During SC validation, a pre-existing bug was discovered in `_prepare_mb_list`: for non-Qwen3 models (e.g. Qwen2.5) with `attn_impl=eager`, the attention mask was set to `dict(full_attention=None, sliding_attention=None)`. `Qwen2Model.forward` checks `isinstance(attention_mask, dict)` and, when True, **skips `create_causal_mask` entirely** — `eager_attention_forward` receives `None` and applies no mask. In a packed sequence, every token attends to every other token across sequence boundaries, computing **global attention instead of causal attention**.

The fix replaces `dict(full_attention=None, ...)` with `None`, which lets HF transformers auto-detect the packed format from `position_ids` (reset to 0 at each sequence boundary via `amend_position_ids`) and build a correct **block-diagonal causal mask** via `find_packed_sequence_indices`.

This bug exists in the baseline (`ma-ascend-v1.0.5`) and in the upstream `ascend` branch. Upstream issue #1557 and PR #1558 report the same root cause with an identical fix.

### Experimental validation

A controlled experiment (SC on/off x recompute_logprob on/off, 1000 training steps each) was run on Qwen2.5-1.5B-Instruct (bf16, eager attention) with GSM8K GRPO on Ascend 910 NPU.

**TIM elimination** (attention mask fix):

| Metric | Before fix | After fix |
|--------|-----------|-----------|
| TIM abs mean (nats) | 6.72 | 0.11 |
| TIM spike rate (>2 nats) | 94.0% | 0.4% |
| Jaccard (trainer vs vLLM top-128) | 0.093 | 0.824 |
| Trainer argmax concentration | 77.1% | 5.2% |

**Reward — SC-on outperforms SC-off**:

| Training steps | SC-on (avg) | SC-off (avg) | Advantage |
|----------------|-------------|-------------|-----------|
| 0-99 | 0.464 | 0.461 | +0.003 |
| 200-499 | 0.767 | 0.739 | +0.027 |
| 500-799 | 0.814 | 0.763 | +0.052 |
| 800-957 | 0.782 | 0.559 | +0.223 |

Final reward: SC-on 0.790-0.802 vs SC-off 0.699-0.490 (SC-off+recompute-on crashed at step 958).

**Gradient stability** — SC-on grad_norm stable at 2-6 throughout; SC-off+recompute-on escalated from 14 to 117 to 155 before crash. Both SC-on configurations completed all 1000 steps.

**SC tail model health** — After the fix, `p_mass ~ q_mass ~ 0.999` and `rho ~ 1.0` throughout (before fix: `p_mass = 0.30-0.47`, `rho = 13-1325`). SC correction tracks the drift signal `s_bar` (ratio 0.7-1.7, same sign); before the fix, correction flipped sign and decoupled from `s_bar`.

![Reward curves](reward_curves.png)

![Gradient norm](grad_norm.png)

![Policy entropy](entropy.png)

![SC reward advantage by phase](sc_effect_by_phase.png)

![SC tail model health (before vs after fix)](sc_tail_model_health.png)

![SC drift tracking (s_bar vs correction)](sc_drift_tracking.png)

## Related Issue

Fixes #1557

Refs: #1442, #1558

## Type of Change

- [ ] 🐛 Bug fix
- [x] ✨ New feature
- [ ] 💥 Breaking change
- [ ] 📝 Documentation update
- [ ] ♻️ Refactoring
- [ ] ⚡ Performance improvement
- [ ] ✅ Test coverage improvement

## Checklist

- [x] I have read the
  [Contributing Guide](https://github.com/areal-project/community/blob/main/CONTRIBUTING.md)
- [x] Pre-commit hooks pass (`pre-commit run --all-files`)
- [x] Relevant tests pass; new tests added for new functionality
- [ ] Documentation updated (if applicable; built with `./docs/build_all.sh`)
- [ ] Branch is up to date with `main`
- [ ] Self-reviewed via `/review-pr` command
- [ ] This PR was created by a coding agent via `/create-pr`
- [ ] This PR is a breaking change

**Breaking Change Details (if applicable):**

None. All SC features are opt-in via `use_score_centering=False` (default). The attention mask fix corrects a bug for non-Qwen3 models with `attn_impl=eager` + packed sequences; the new behavior is the intended behavior.

## Additional Context

### Architecture: SC data flow

```
vLLM sampler (top-k logprobs)
    |  vllm_remote.py: _parse_top_logprobs -> output_topk_ids / output_topk_logps
    v
io_struct.py: HttpGenerationResult.output_topk_{ids,logps}
    |
    v
remote_inf_engine.py: accumulate top-k across chunks -> ModelResponse
    |
    v
rlvr.py / types.py: pad input positions (-inf), truncate k_dim -> sampler_topk_{logps,token_ids}
    |
    v
data.py: concat_padded_tensors (-inf padding for logprob keys) + 3D tensor splitting
    |
    v
fsdp_engine.py: inject trainer logits into mb_input (conditional on SC enabled)
    |
    v
actor.py grpo_loss_fn:
    gather_train_head_logp(logits, token_ids) -> train_head_logp
    score_centering_correction(train_head_logp, sampler_topk_logps, ...) -> correction
    loss = -A*log(p(a)) + A*correction    (REINFORCE + SC)
```

### Key design decisions

**1. REINFORCE+SC instead of PPO+SC.** SC correction is integrated as a REINFORCE+SC independent loss path (`loss = -A*log(p(a)) + A*correction`), not appended on top of PPO. When `recompute_logprob=false` (TIM > 0), the PPO ratio `exp(new_logp - old_logp)` deviates from 1.0; appending SC onto PPO creates gradient inconsistency between the ratio term and the correction term. REINFORCE+SC avoids this by dropping the ratio entirely. The `sc_reinforce_baseline` flag enables a pure REINFORCE baseline (SC-off) for fair A/B comparison.

**2. -inf padding for logprob tensors.** `concat_padded_tensors` now uses a whitelist for logprob keys (`sampler_topk_logps`, `ref_logprobs`, `logprobs`) that receive `-inf` padding instead of the default `0.0`. `exp(0.0) = 1.0` silently inflates `q_mass` (each padded slot adds 1.0 of fake probability mass), breaking the tail model. `exp(-inf) = 0.0` correctly contributes zero mass.

**3. Chunked fp32 autograd.** `_TopKLogProbsFromLogits` is a custom `torch.autograd.Function` that computes `log_softmax(logits).gather(token_ids)` in chunks, bounding the fp32 workspace to `(chunk_size, vocab)` instead of `(N, vocab)`. The full workspace for a typical batch (~4000 tokens x 151,936 vocab x 4 bytes) is ~2.4 GB; chunking at 4096 rows bounds this to ~2.5 MB.

**4. Attention mask fix as a separate commit.** The fix is delivered as a separate commit from SC infrastructure because it fixes a pre-existing baseline bug affecting all non-Qwen3 models with eager attention, not just SC users. It corresponds to upstream PR #1558 (independent contribution).

### File changes

**Commit 1 — Score Centering infrastructure** (12 files, +883/-6):

| File | Change |
|------|--------|
| `areal/utils/score_centering.py` | New file (278 lines): `score_centering_correction`, `gather_train_head_logp`, `pad_topk_ids/logps`, `check_score_centering_sp_supported` |
| `areal/api/cli_args.py` | +188: SC config fields (`use_score_centering`, `sc_top_k`, `sc_mode`, `sc_clip_low`, `sc_clip_high`, `sc_reinforce_baseline`) + fail-fast validation + `sc_top_k` propagation to vLLM `max_logprobs` |
| `areal/trainer/ppo/actor.py` | +179: SC loss path in `grpo_loss_fn` (REINFORCE+SC) + SC stats |
| `areal/engine/vllm_remote.py` | +78: `_token_to_id` / `_parse_top_logprobs` helpers, `logprobs=sc_top_k` request, top-k response parsing |
| `areal/engine/fsdp_engine.py` | +11: inject trainer logits into loss_fn input + SP validation |
| `areal/experimental/openai/types.py` | +38: `InteractionWithTokenLogpReward` SC top-k propagation |
| `areal/workflow/rlvr.py` | +45: `RLVRWorkflow` sampler top-k data flow (k_dim truncation + padding) |
| `areal/infra/remote_inf_engine.py` | +24: top-k accumulation + `ModelResponse` propagation |
| `areal/utils/data.py` | +26: `concat_padded_tensors` -inf padding whitelist + 3D tensor splitting |
| `areal/api/io_struct.py` | +7: `ModelResponse` + `HttpGenerationResult` output_topk fields |
| `areal/experimental/openai/client.py` | +10: `sc_top_k` parameter in 3 client classes |
| `areal/experimental/openai/proxy/proxy_rollout_server.py` | +1: `sc_top_k` propagation |

**Commit 2 — Attention mask fix** (1 file, +11/-4):

| File | Change |
|------|--------|
| `areal/engine/fsdp_engine.py` | Replace `dict(full_attention=None, sliding_attention=None)` with `None` in the non-Qwen3 branch of `_prepare_mb_list` |

### Test coverage

| Test suite | Status |
|-----------|--------|
| SC mechanism unit tests (`test_score_centering.py`) | Pass (57 tests) |
| vLLM generation request tests (`test_vllm_generation_request.py`) | Pass |
| RLVR top-k assembly tests (`test_rlvr_topk_assembly.py`) | Pass |
| Pure-Python SC math verification (`verify_sc_math.py`) | Pass (12/12) |
| Attention mask fix verification (`verify_attention_mask_fix.py`) | Pass (3/3) |
| 2-step NPU smoke (Ascend 910, vLLM 0.23.0) | Pass — `Training completes!`, 0 NaN/Inf, SC correction non-zero |
| Full-scale training (4 configurations, 1000 steps each) | SC-on stable; SC-off+recompute-on crashed at step 958 |

### Compatibility

- **vLLM**: 0.23.0. The `_parse_top_logprobs` function handles vLLM 0.23's flat-list format (`meta_info["logprobs"]["top_logprobs"]`).
- **transformers**: 5.5.4. The attention mask fix relies on `find_packed_sequence_indices` (available since transformers 4.40+).
- **Models**: Qwen2.5 (tested), Llama (same code path). Qwen3 models are unaffected (they already use `attention_mask=None`).
- **VLM**: SC is not supported with `chat_template_type='concat'` multi-turn interactions (raises `NotImplementedError`). VLM top-k filtering (`drop_vision_pad_tokens`) is not ported (v1.0.5 baseline lacks this function).
- **Sequence parallel**: SC raises `NotImplementedError` when `sp_size > 1` (trainer logits are the local SP slice, not the full sequence).
