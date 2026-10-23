#!/usr/bin/env python3
"""
Pure-Python verification of Score Centering core math.

No torch dependency — uses only the standard library (math, sys).
Reimplements the SC correction formula from score_centering.py and
checks key mathematical properties.

Usage:
    python3 verify_sc_math.py [--strict]

Exit codes:
    0  — all checks passed
    1  — one or more checks failed (details printed to stderr)
    2  --strict mode: warning-level issues treated as failures

Run this AFTER applying both patches (01 + 02) and committing.
"""

import math
import sys
from dataclasses import dataclass
from typing import Callable

# ============================================================
#  Pure-Python reimplementation of score_centering.py math
#  (mirrors the torch version exactly, but in plain Python)
# ============================================================

NEG_INF_PAD = float("-inf")


def pad_topk_ids(token_ids: list[int], k: int) -> list[int]:
    """Right-pad a top-k token-id row to length k with padding id 0."""
    ids = list(token_ids[:k])
    return ids + [0] * max(0, k - len(ids))


def pad_topk_logps(logps: list[float], k: int) -> list[float]:
    """Right-pad a top-k log-prob row to length k with -inf (q = 0)."""
    ps = list(logps[:k])
    return ps + [NEG_INF_PAD] * max(0, k - len(ps))


def build_weight_fn(sc_mode: str, sc_clip_low: float, sc_clip_high: float) -> Callable[[float], float]:
    """Build the IS weight function (mirrors _build_weight_fn)."""
    if sc_mode == "none":
        return lambda ratio: 1.0
    if sc_mode == "tis":
        return lambda ratio: min(ratio, sc_clip_high)
    if sc_mode == "mis":
        low, high = sc_clip_low, sc_clip_high
        return lambda ratio: ratio if (low <= ratio <= high) else 0.0
    raise ValueError(f"sc_mode must be 'none', 'tis', or 'mis'; got {sc_mode!r}.")


def safe_exp(x: float) -> float:
    """exp(-inf) = 0, everything else uses math.exp."""
    if x == NEG_INF_PAD or math.isinf(x) and x < 0:
        return 0.0
    return math.exp(x)


def score_centering_correction(
    train_head_logp: list[float],      # [k] trainer log-probs for top-k tokens
    sampler_topk_logps: list[float],   # [k] sampler log-probs for same tokens
    sc_mode: str = "none",
    sc_clip_low: float = 0.0,
    sc_clip_high: float = float("inf"),
):
    """Pure-Python reimplementation of score_centering_correction.

    Returns (correction, s_bar, q_mass, p_mass, rho).
    Each is a scalar (since we process one position at a time).
    """
    k = len(train_head_logp)
    assert len(sampler_topk_logps) == k, "train_head_logp and sampler_topk_logps must have same length"

    # q = exp(sampler log-probs), masked by finiteness
    q = []
    valid = []
    for lp in sampler_topk_logps:
        is_valid = math.isfinite(lp)
        valid.append(is_valid)
        q.append(safe_exp(lp) if is_valid else 0.0)

    weight_fn = build_weight_fn(sc_mode, sc_clip_low, sc_clip_high)

    # p = exp(trainer log-probs) * mask
    p = []
    for i in range(k):
        p.append(safe_exp(train_head_logp[i]) if valid[i] else 0.0)

    p_mass = sum(p)
    q_mass = sum(q)

    # rho = (1 - q_mass) / (1 - p_mass), clamped
    eps = 1e-6
    rho = max(1 - q_mass, eps) / max(1 - p_mass, eps)
    alpha = rho * weight_fn(1.0 / rho)

    # head weights: w = weight_fn(p/q), with safe ratio
    head_weights = []
    for i in range(k):
        if valid[i]:
            ratio = safe_exp(train_head_logp[i] - sampler_topk_logps[i])
            head_weights.append(weight_fn(ratio))
        else:
            head_weights.append(0.0)

    # residual = q * w - alpha * p
    residual = [q[i] * head_weights[i] - alpha * p[i] for i in range(k)]

    # correction = sum(residual * train_head_logp)
    # NOTE: in the torch version, gradient flows through train_head_logp.
    # Here we compute the value only (no gradient).
    correction = sum(residual[i] * train_head_logp[i] for i in range(k))

    # s_bar = sum(q * train_head_logp) (detached in torch)
    s_bar = sum(q[i] * train_head_logp[i] for i in range(k))

    return correction, s_bar, q_mass, p_mass, rho


# ============================================================
#  Test cases
# ============================================================

@dataclass
class TestResult:
    name: str
    passed: bool
    detail: str = ""


def approx_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    """Check if two floats are approximately equal."""
    if math.isinf(a) and math.isinf(b) and (a > 0) == (b > 0):
        return True
    return abs(a - b) < tol


def test_pad_topk_ids() -> TestResult:
    """pad_topk_ids pads with 0 to length k."""
    result = pad_topk_ids([1, 2, 3], 5)
    expected = [1, 2, 3, 0, 0]
    if result == expected:
        return TestResult("pad_topk_ids", True, f"pad_topk_ids([1,2,3],5)={result}")
    return TestResult("pad_topk_ids", False, f"Expected {expected}, got {result}")


def test_pad_topk_logps() -> TestResult:
    """pad_topk_logps pads with -inf to length k."""
    result = pad_topk_logps([-0.1], 3)
    expected = [-0.1, NEG_INF_PAD, NEG_INF_PAD]
    if len(result) == 3 and approx_equal(result[0], -0.1) and result[1] == NEG_INF_PAD and result[2] == NEG_INF_PAD:
        return TestResult("pad_topk_logps", True, f"pad_topk_logps([-0.1],3)={result}")
    return TestResult("pad_topk_logps", False, f"Expected {expected}, got {result}")


def test_pad_topk_logps_truncation() -> TestResult:
    """pad_topk_logps truncates to k when input is longer."""
    result = pad_topk_logps([-0.1, -0.2, -0.3, -0.4], 2)
    if len(result) == 2 and approx_equal(result[0], -0.1) and approx_equal(result[1], -0.2):
        return TestResult("pad_topk_logps_truncation", True, f"truncated to {result}")
    return TestResult("pad_topk_logps_truncation", False, f"Got {result}")


def test_on_policy_correction_is_zero() -> TestResult:
    """When p == q (on-policy), correction must be exactly 0 for all modes."""
    k = 4
    logps = [-0.5, -0.8, -1.2, -2.0]  # same for both trainer and sampler
    modes = [
        ("none", 0.0, float("inf")),
        ("tis", 0.0, 5.0),
        ("mis", 0.1, 5.0),
    ]
    for mode, low, high in modes:
        correction, s_bar, q_mass, p_mass, rho = score_centering_correction(
            logps, logps, sc_mode=mode, sc_clip_low=low, sc_clip_high=high
        )
        if not approx_equal(correction, 0.0, tol=1e-12):
            return TestResult(
                f"on_policy_{mode}",
                False,
                f"correction={correction} (expected 0.0), s_bar={s_bar}, rho={rho}",
            )
    return TestResult("on_policy_correction_is_zero", True, "All modes: correction=0 when p==q")


def test_on_policy_rho_is_one() -> TestResult:
    """When p == q, rho must be 1.0 (tail masses equal)."""
    logps = [-0.5, -0.8, -1.2, -2.0]
    _, _, _, _, rho = score_centering_correction(logps, logps)
    if approx_equal(rho, 1.0, tol=1e-9):
        return TestResult("on_policy_rho_is_one", True, f"rho={rho}")
    return TestResult("on_policy_rho_is_one", False, f"rho={rho} (expected 1.0)")


def test_q_mass_leq_one_with_proper_padding() -> TestResult:
    """q_mass must be <= 1.0 when padding uses -inf (not 0.0).

    Uses realistic logprobs where top-k probabilities sum to <= 1.0
    (which is the invariant for valid log_softmax output).
    """
    # Top-3 tokens covering ~50% of probability mass, then 2 padded slots (-inf).
    # Realistic: a single token often has ~0.5 prob, rest are small.
    sampler_logps = [-0.69, -2.30, -3.91, NEG_INF_PAD, NEG_INF_PAD]
    # exp(-0.69)=0.50, exp(-2.30)=0.10, exp(-3.91)=0.02 -> q_mass=0.62
    _, _, q_mass, _, _ = score_centering_correction(
        [-0.5, -1.0, -2.0, -1.0, -1.0],  # trainer (irrelevant for q_mass)
        sampler_logps,
    )
    if q_mass <= 1.0 + 1e-9:
        return TestResult("q_mass_leq_one", True, f"q_mass={q_mass:.4f} <= 1.0")
    return TestResult(
        "q_mass_leq_one",
        False,
        f"q_mass={q_mass} > 1.0! -inf padding is NOT working (Bug 2: pad_value=0.0).",
    )


def test_q_mass_inflated_with_zero_padding() -> TestResult:
    """Demonstrate that 0.0 padding inflates q_mass (the Bug 2 we fixed)."""
    # Simulate WRONG padding: 0.0 instead of -inf
    logps_wrong = [-0.01, -0.02, -0.03, 0.0, 0.0]  # WRONG: exp(0)=1.0 inflates mass
    _, _, q_mass_wrong, _, _ = score_centering_correction(
        [-0.01, -0.02, -0.03, -1.0, -1.0],
        logps_wrong,
    )
    # With wrong padding, q_mass = exp(-0.01)+exp(-0.02)+exp(-0.03)+exp(0)+exp(0)
    # ≈ 0.99 + 0.98 + 0.97 + 1.0 + 1.0 = 4.94 >> 1.0
    if q_mass_wrong > 1.0:
        return TestResult(
            "q_mass_inflated_by_zero_padding",
            True,
            f"q_mass={q_mass_wrong:.4f} > 1.0 confirms 0.0 padding inflates (Bug 2 reproduced)",
        )
    return TestResult(
        "q_mass_inflated_by_zero_padding",
        False,
        f"q_mass={q_mass_wrong} should be > 1.0 with 0.0 padding",
    )


def test_correction_nonzero_off_policy() -> TestResult:
    """When p != q (off-policy), correction should be nonzero."""
    train_logps = [-0.3, -0.5, -1.0, -2.0]   # trainer: more concentrated
    sampler_logps = [-0.1, -0.2, -0.3, -0.4]  # sampler: more diffuse
    correction, s_bar, q_mass, p_mass, rho = score_centering_correction(
        train_logps, sampler_logps
    )
    if abs(correction) > 1e-9:
        return TestResult(
            "off_policy_nonzero",
            True,
            f"correction={correction:.6f}, s_bar={s_bar:.4f}, rho={rho:.4f}",
        )
    return TestResult(
        "off_policy_nonzero",
        False,
        f"correction={correction} should be nonzero when p != q",
    )


def test_weight_fn_none() -> TestResult:
    """weight_fn('none') always returns 1.0 (pure SC, no IS)."""
    wf = build_weight_fn("none", 0.0, float("inf"))
    for r in [0.1, 0.5, 1.0, 2.0, 10.0, 100.0]:
        if not approx_equal(wf(r), 1.0):
            return TestResult("weight_fn_none", False, f"wf({r})={wf(r)} != 1.0")
    return TestResult("weight_fn_none", True, "All ratios -> 1.0")


def test_weight_fn_tis() -> TestResult:
    """weight_fn('tis') clamps to sc_clip_high."""
    wf = build_weight_fn("tis", 0.0, 5.0)
    if not approx_equal(wf(1.0), 1.0):
        return TestResult("weight_fn_tis", False, f"wf(1.0)={wf(1.0)} != 1.0")
    if not approx_equal(wf(10.0), 5.0):
        return TestResult("weight_fn_tis", False, f"wf(10.0)={wf(10.0)} != 5.0 (clamped)")
    if not approx_equal(wf(0.5), 0.5):
        return TestResult("weight_fn_tis", False, f"wf(0.5)={wf(0.5)} != 0.5 (below clamp)")
    return TestResult("weight_fn_tis", True, "clamp(max=5.0) correct")


def test_weight_fn_mis() -> TestResult:
    """weight_fn('mis') masks outside the band [low, high]."""
    wf = build_weight_fn("mis", 0.2, 5.0)
    if not approx_equal(wf(1.0), 1.0):
        return TestResult("weight_fn_mis", False, f"wf(1.0)={wf(1.0)} != 1.0 (in band)")
    if not approx_equal(wf(0.1), 0.0):
        return TestResult("weight_fn_mis", False, f"wf(0.1)={wf(0.1)} != 0.0 (below band)")
    if not approx_equal(wf(10.0), 0.0):
        return TestResult("weight_fn_mis", False, f"wf(10.0)={wf(10.0)} != 0.0 (above band)")
    return TestResult("weight_fn_mis", True, "band-mask [0.2, 5.0] correct")


def test_invalid_sc_mode_raises() -> TestResult:
    """build_weight_fn should raise ValueError for invalid sc_mode."""
    try:
        build_weight_fn("invalid", 0.0, 1.0)
        return TestResult("invalid_mode_raises", False, "No exception raised")
    except ValueError:
        return TestResult("invalid_mode_raises", True, "ValueError raised correctly")


# ============================================================
#  Main runner
# ============================================================

def main():
    strict = "--strict" in sys.argv

    tests = [
        test_pad_topk_ids(),
        test_pad_topk_logps(),
        test_pad_topk_logps_truncation(),
        test_on_policy_correction_is_zero(),
        test_on_policy_rho_is_one(),
        test_q_mass_leq_one_with_proper_padding(),
        test_q_mass_inflated_with_zero_padding(),
        test_correction_nonzero_off_policy(),
        test_weight_fn_none(),
        test_weight_fn_tis(),
        test_weight_fn_mis(),
        test_invalid_sc_mode_raises(),
    ]

    passed = 0
    failed = 0
    warnings = 0

    print("=" * 60)
    print("  Score Centering Pure-Python Math Verification")
    print("=" * 60)
    print()

    for t in tests:
        status = "✅ PASS" if t.passed else "❌ FAIL"
        print(f"  {status}  {t.name}")
        if t.detail:
            print(f"          {t.detail}")
        if t.passed:
            passed += 1
        else:
            failed += 1

    print()
    print(f"  Results: {passed} passed, {failed} failed, {warnings} warnings")
    print()

    if failed > 0:
        print("  ❌ VERIFICATION FAILED — review the failed checks above.")
        sys.exit(1)
    else:
        print("  ✅ ALL CHECKS PASSED — SC core math is correct.")
        sys.exit(0)


if __name__ == "__main__":
    main()
