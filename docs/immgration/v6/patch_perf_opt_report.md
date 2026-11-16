# perf(engine): cut the sampler-head cost of score centering

## What

Score centering consumes the sampler's top-k head for every generated token. Producing that
head has two costs that are pure representation, not computation:

- the engine detokenizes every candidate token, i.e. `k × tokens` decoder calls per step that
  no consumer reads — the head is consumed as token ids;
- the head is materialised as one Python object per (token, rank) instead of parallel
  primitive lists.

This change lets the vLLM engine skip both, using two `SamplingParams` fields that already
exist upstream:

| Field | Effect |
|---|---|
| `flat_logprobs=True` | the engine keeps the head as parallel primitive lists instead of one dict per position |
| `detokenize=False` | the engine does not decode candidate tokens; only ids and log-probs are produced |

Both are **opt-in** (`AREAL_SC_FLAT_HEADS` / `AREAL_SC_NODETOK`, default off). With neither
set, the request payload is byte-identical to the current one; with `score_centering_top_k=0`
nothing changes either. `detokenize=False` is never sent when stop strings are configured —
vLLM requires detokenization for those, so the pair is guarded.

**No serving-side change is required** and no response schema changes: the head keeps
travelling inside the existing `logprobs` payload.

## Why

Measured on the FSDP actor path (Qwen3-0.6B, GSM8K, 128 responses/step,
`max_new_tokens=1024`, σ=0, 5 steps per arm, same machine and entry point):

| top-k | configuration | rollout (s/step) | vs. baseline |
|---|---|---|---|
| 128 | current behaviour | 91.6 | 1.00× |
| **128** | **`flat_logprobs` + `detokenize=False`** | **44.6** | **2.05×** |
| 32 | current behaviour | 25.3 | 1.00× |
| 32 | `flat_logprobs` + `detokenize=False` | 23.4 | 1.08× |

The training step is unaffected (5.1–6.4 s/step in both arms), so the whole gain lands on
rollout. At `k=32` the head is already cheap, which is why the change only pays off at wide
heads.

## Correctness

- `sc_q_mass` / `sc_p_mass` stay at `0.9995 – 0.9997` (same as the current path; the head
  probability mass stays below one).
- With the switches unset, the request payload is unchanged, so existing behaviour and tests
  are untouched: `tests/test_vllm_generation_request.py` → 27 passed.
- `detokenize=False` is only requested when no stop strings are configured.

## Usage

```bash
# opt in (both are inert by default)
AREAL_SC_FLAT_HEADS=1 AREAL_SC_NODETOK=1 ...
```

Config fields (`InferenceEngineConfig`) for these two switches are a follow-up; this change
keeps them environment-gated so the default path stays byte-identical.

## Out of scope

- Transporting the head as binaries and dropping the per-token objects / JSON — that needs a
  serving-side change and is a separate PR.
- Any change to trainer-side score-centering maths, metrics or config schema.
- Other rollout backends.
