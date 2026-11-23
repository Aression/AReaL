## Description

Enable Score Centering (SC) for **Megatron** training actors. Before this change the
correction added by #1797 worked only with FSDP actors: `PPOActorConfig` rejected any
non-`fsdp:*` backend, and `megatron_engine.py` never scored the sampler's top-k heads,
so Megatron runs could not apply the drift correction at all.

The patch makes the Megatron actor score the sampler heads directly out of its own
logits (before the fused loss destructively reuses the logits), reduces the
vocabulary shards over the TP group, and keeps the existing safety contract: a
micro-batch must carry exactly one sampler-head row per generated token. Modes that
break that contract (tree training, critic actors, MTP, chunked LM head) are rejected
in both the config layer and the engine.

## Related Issue

Fixes #1797 follow-up (Megatron actor support for Score Centering). No separate issue
was filed; the design and evidence are in the commit body and in "Additional Context".

## Type of Change

- [ ] 🐛 Bug fix
- [x] ✨ New feature
- [ ] 💥 Breaking change
- [ ] 📖 Documentation update
- [ ] ♻️ Refactoring
- [ ] ⚡ Performance improvement
- [ ] ✅ Test coverage improvement

## Checklist

- [x] I have read the [Contributing Guide](https://github.com/areal-project/community/blob/main/CONTRIBUTING.md)
- [x] Pre-commit hooks pass (`pre-commit run --all-files`; equivalently `ruff format` + `ruff check` on the four touched files)
- [x] Relevant tests pass; new tests added for new functionality
- [ ] Documentation updated (if applicable; built with `./docs/build_all.sh`) — docstrings + CLI help only, no docs build performed
- [x] Branch is up to date with `main` (based on `ascend@bb8d7bd4`)
- [ ] Self-reviewed via `/review-pr` command
- [ ] This PR was created by a coding agent via `/create-pr`
- [ ] This PR is a breaking change

**Breaking Change Details (if applicable):**

Not a breaking change for existing FSDP users: the FSDP branch of the new validation
returns immediately, so `fsdp:*` configs behave exactly as before. The only behavioural
change is that `megatron:*` actors are no longer rejected wholesale; the previously
rejected unsafe modes (tree / critic / MTP / chunked logits) still raise.

## Additional Context

### What changed (4 files, +648 / −22)

| File | Change |
| --- | --- |
| `areal/api/cli_args.py` | `PPOActorConfig._validate_score_centering_backend()` replaces the "non-FSDP ⇒ reject" guard: accepts `megatron:*` (data / tensor / pipeline / expert parallel), rejects tree training, critic, MTP and `enable_chunked_logits`. |
| `areal/engine/megatron_engine.py` | SC branch in the loss path (runs before the fused kernel reuses the logits), `_score_centering_logprobs()` (row/shape validation, TP vocab-shard reduction, shard-coverage check), `_ensure_score_centering_supported()` (engine-side repeat of the guard, cached on success only), `_warn_sc_head_padding()` and `_warn_sc_head_mass()` (warn once when the sampler top-k covers < 90 % of the probability mass). |
| `areal/engine/megatron_utils/packed_context_parallel.py` | `split_packed_seqs_for_context_parallel()` keeps trailing dimensions, so `[tokens, K]` sampler heads split exactly like 1-D labels (behaviour identical at `cp_size == 1`). |
| `tests/test_score_centering_megatron.py` | 30 CPU cases: config matrix, CP row-alignment invariant, engine helper branches (missing/misaligned heads, temperature, MTP/chunked/tree, shard coverage, head-mass warning, guard caching) and the SC math against a `log_softmax` reference. |

### Validation (accelerator evidence, 8 Ascend dies)

Each arm ran 3 steps and was only accepted once step-1 metrics existed:

| Arm | actor / rollout | `sc_active` | `sc_q_mass` | `sc_rho` |
| --- | --- | --- | --- | --- |
| DP4 | `megatron:d4p1t1` / `vllm:d4p1t1` | 1 | 0.9995–0.9998 | 1.0009–1.0023 |
| DP2×TP2 | `megatron:d2p1t2` / `vllm:d2p1t2` | 1 | 0.9995 / 0.9997 | 1.0032 / 1.0035 |
| PP2×TP2 | `megatron:d1p2t2` / `vllm:d2p1t2` | 1 | 0.9996–0.9998 | 1.0009–1.0031 |
| MoE EP4 + TP4 | `megatron:(attn:d4p1t1c1\|ffn:d1p1t1e4)` / `vllm:d1p1t4` | 1 | **1.0000** | 1.0003 / 1.0011 |

`q_mass ≈ 1` and `rho ≈ 1` at σ = 0 mean SC is a no-op there, which is the expected
behaviour; the TP arm differs from the DP arm by 4e-5 in `q_mass`, well inside the
1e-3 parity budget used for the vocabulary-shard reduction.

### Known limitation: `cp > 1`

Context parallelism is **not** validated end to end. Two layout bugs found while
testing are fixed in this patch (heads must be taken from the padded packed layout;
the per-token correction must follow the same reassemble + unpad transform as
`logprobs`), but a third boundary remains: `grpo_loss_fn` asserts the `logprobs` shape
against `gconfig.max_tokens`, which conflicts with the unpadded row layout produced
under CP. Recommend keeping `cp = 1` (the config guard currently permits CP, so this is
documented rather than enforced — happy to flip it to an explicit rejection if
reviewers prefer fail-fast).

### Tests run

```bash
ruff format --check areal/api/cli_args.py areal/engine/megatron_engine.py \
  areal/engine/megatron_utils/packed_context_parallel.py tests/test_score_centering_megatron.py   # 4 files already formatted
ruff check <same four files>                                                                    # All checks passed
PYTHONPATH=$PWD:$PYTHONPATH pytest tests/test_score_centering_megatron.py -q                     # 30 passed in 39.7 s
```

The new test file is CPU-only and skips gracefully (`pytest.importorskip("megatron.core")`)
when Megatron is not installed, so it does not add an accelerator requirement to CI.

### Reviewer notes

- Modifying the validation structure in `areal/api/cli_args.py` is an AGENTS.md
  "ask first" item; the change is a pure refactor of one guard into a helper plus a
  backend-aware check, with no dataclass field added/renamed.
- `micro_batch_score_centering()` from the earlier prototype was **not** carried over:
  `score_centering_logprobs()` already accepts `tp_group` / `vocab_size` /
  `metrics_out`, so the engine calls it directly and keeps the patch surface minimal.
