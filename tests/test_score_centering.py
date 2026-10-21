# SPDX-License-Identifier: Apache-2.0

"""Tests for Score Centering (SC) drift correction.

Covers:
  - _TopKLogProbsFromLogits forward/backward correctness
  - score_centering_correction on-policy no-op (value + gradient)
  - score_centering_correction pad-slot safety (no NaN)
  - gather_train_head_logp equivalence to log_softmax + gather
  - Config validation (sc_loss_mode, sc_mode constraints)
  - data.py concat_padded_tensors: only sampler_topk_logps gets -inf
"""

import pytest
import torch

from areal.utils.data import concat_padded_tensors
from areal.utils.score_centering import (
    _TopKLogProbsFromLogits,
    gather_train_head_logp,
    score_centering_correction,
)


class TestTopKLogProbsFromLogits:
    """Verify the chunked autograd Function matches log_softmax + gather."""

    def test_forward_matches_log_softmax_gather(self):
        torch.manual_seed(42)
        n, vocab, k = 8, 100, 5
        logits = torch.randn(n, vocab, dtype=torch.float64)
        token_ids = torch.randint(0, vocab, (n, k))

        expected = torch.log_softmax(logits, dim=-1).gather(-1, token_ids)
        result = _TopKLogProbsFromLogits.apply(logits, token_ids, 4)  # chunk_size=4

        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-12)

    def test_backward_gradient_correctness(self):
        torch.manual_seed(42)
        n, vocab, k = 6, 50, 3
        logits = torch.randn(n, vocab, dtype=torch.float64, requires_grad=True)
        token_ids = torch.randint(0, vocab, (n, k))

        # Reference: log_softmax + gather
        logits_ref = logits.detach().clone().requires_grad_(True)
        ref = torch.log_softmax(logits_ref, dim=-1).gather(-1, token_ids)
        ref.sum().backward()

        # Our function
        result = _TopKLogProbsFromLogits.apply(logits, token_ids, 0)
        result.sum().backward()

        torch.testing.assert_close(logits.grad, logits_ref.grad, rtol=1e-10, atol=1e-12)

    def test_bf16_promoted_to_fp32(self):
        n, vocab, k = 4, 30, 3
        logits = torch.randn(n, vocab, dtype=torch.bfloat16)
        token_ids = torch.randint(0, vocab, (n, k))

        result = _TopKLogProbsFromLogits.apply(logits, token_ids, 0)
        assert result.dtype == torch.float32


class TestScoreCenteringCorrection:
    """Verify SC correction math, on-policy no-op, and pad safety."""

    @pytest.fixture
    def on_policy_data(self):
        """Create on-policy data where p == q (trainer == sampler)."""
        torch.manual_seed(42)
        batch, seq, k, vocab = 2, 4, 5, 100
        logits = torch.randn(batch, seq, vocab, dtype=torch.float64)
        log_probs = torch.log_softmax(logits, dim=-1)
        # Top-k tokens and their log-probs
        topk_vals, topk_ids = torch.topk(log_probs, k, dim=-1)
        return {
            "logits": logits,
            "train_head_logp": topk_vals.clone(),  # p == q (on-policy)
            "sampler_topk_logps": topk_vals.clone().detach(),  # q == p
        }

    def test_on_policy_is_exact_noop_value(self, on_policy_data):
        """correction must be exactly 0 when p == q."""
        correction, s_bar, q_mass, p_mass, rho = score_centering_correction(
            train_head_logp=on_policy_data["train_head_logp"],
            sampler_topk_logps=on_policy_data["sampler_topk_logps"],
        )
        assert correction.shape == (2, 4)
        torch.testing.assert_close(correction, torch.zeros_like(correction))
        torch.testing.assert_close(rho, torch.ones_like(rho))

    def test_on_policy_is_exact_noop_gradient(self, on_policy_data):
        """gradient of correction w.r.t. train_head_logp must be 0 when p==q."""
        train_head_logp = on_policy_data["train_head_logp"].clone().requires_grad_(True)
        correction, *_ = score_centering_correction(
            train_head_logp=train_head_logp,
            sampler_topk_logps=on_policy_data["sampler_topk_logps"],
        )
        correction.sum().backward()
        # correction = residual * train_head_logp; residual = 0 on-policy
        torch.testing.assert_close(
            train_head_logp.grad, torch.zeros_like(train_head_logp.grad)
        )

    def test_pad_slots_produce_no_nan(self):
        """-inf sampler logprobs (pad slots) must not produce NaN."""
        batch, seq, k = 2, 3, 4
        train_head_logp = torch.randn(batch, seq, k, dtype=torch.float64)
        # Some slots padded with -inf (as done by pad_topk_logps)
        sampler_topk_logps = torch.randn(batch, seq, k, dtype=torch.float64)
        sampler_topk_logps[0, 2, :] = float("-inf")  # pad entire last position
        sampler_topk_logps[1, 0, 2:] = float("-inf")  # pad 2 slots

        correction, s_bar, q_mass, p_mass, rho = score_centering_correction(
            train_head_logp=train_head_logp,
            sampler_topk_logps=sampler_topk_logps,
        )

        assert not torch.isnan(correction).any(), "correction has NaN"
        assert not torch.isnan(q_mass).any(), "q_mass has NaN"
        assert not torch.isnan(p_mass).any(), "p_mass has NaN"
        assert not torch.isnan(rho).any(), "rho has NaN"

    def test_off_policy_correction_nonzero(self):
        """When p != q, correction should be non-zero."""
        torch.manual_seed(42)
        batch, seq, k = 1, 2, 5
        train_head_logp = torch.randn(batch, seq, k, dtype=torch.float64)
        sampler_topk_logps = torch.randn(batch, seq, k, dtype=torch.float64)

        correction, _, _, _, _ = score_centering_correction(
            train_head_logp=train_head_logp,
            sampler_topk_logps=sampler_topk_logps,
        )
        assert correction.abs().sum() > 0

    def test_mis_mode_no_nan_with_clips(self):
        """mis mode with finite clips should not produce NaN on valid data."""
        torch.manual_seed(42)
        batch, seq, k = 2, 3, 5
        train_head_logp = torch.randn(batch, seq, k, dtype=torch.float64)
        sampler_topk_logps = torch.randn(batch, seq, k, dtype=torch.float64)

        correction, _, _, _, _ = score_centering_correction(
            train_head_logp=train_head_logp,
            sampler_topk_logps=sampler_topk_logps,
            sc_mode="mis",
            sc_clip_low=0.1,
            sc_clip_high=5.0,
        )
        assert not torch.isnan(correction).any()


class TestGatherTrainHeadLogp:
    """Verify gather_train_head_logp matches log_softmax + gather."""

    def test_matches_reference(self):
        torch.manual_seed(42)
        batch, seq, vocab, k = 2, 3, 50, 5
        logits = torch.randn(batch, seq, vocab, dtype=torch.float64)
        token_ids = torch.randint(0, vocab, (batch, seq, k))

        result = gather_train_head_logp(logits, token_ids, temperature=1.0)
        expected = torch.log_softmax(logits, dim=-1).gather(-1, token_ids)

        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-12)

    def test_temperature_scaling(self):
        """Temperature should scale logits before softmax."""
        torch.manual_seed(42)
        batch, seq, vocab, k = 1, 2, 30, 3
        logits = torch.randn(batch, seq, vocab, dtype=torch.float64)
        token_ids = torch.randint(0, vocab, (batch, seq, k))
        temp = 0.7

        result = gather_train_head_logp(logits, token_ids, temperature=temp)
        expected = torch.log_softmax(logits / temp, dim=-1).gather(-1, token_ids)

        torch.testing.assert_close(result, expected, rtol=1e-10, atol=1e-12)


class TestConcatPaddedTensorsPadding:
    """Verify that only sampler_topk_logps gets -inf padding (Finding 1 fix)."""

    def test_logprobs_padded_with_zero(self):
        """logprobs should be 0.0-padded, not -inf (DPO/PPO safety)."""
        d1 = {"logprobs": torch.randn(3)}
        d2 = {"logprobs": torch.randn(5)}
        result = concat_padded_tensors([d1, d2])
        padded = result["logprobs"]
        # The first 3 values from d1, then 2 pad values
        assert padded.shape[0] == 8
        # Pad positions (indices 3-4) should be 0.0, not -inf
        assert torch.all(padded[3:5] == 0.0), "logprobs pad should be 0.0, not -inf"

    def test_ref_logprobs_padded_with_zero(self):
        """ref_logprobs should be 0.0-padded (DPO NaN fix)."""
        d1 = {"ref_logprobs": torch.randn(4)}
        d2 = {"ref_logprobs": torch.randn(6)}
        result = concat_padded_tensors([d1, d2])
        padded = result["ref_logprobs"]
        assert padded.shape[0] == 10
        # Pad positions (indices 4-5) should be 0.0, not -inf
        assert torch.all(padded[4:6] == 0.0), "ref_logprobs pad should be 0.0, not -inf"

    def test_sampler_topk_logps_padded_with_neg_inf(self):
        """sampler_topk_logps should be -inf-padded (SC correctness)."""
        d1 = {"sampler_topk_logps": torch.randn(3)}
        d2 = {"sampler_topk_logps": torch.randn(5)}
        result = concat_padded_tensors([d1, d2])
        padded = result["sampler_topk_logps"]
        assert padded.shape[0] == 8
        # Pad positions should be -inf
        assert torch.all(padded[3:5] == float("-inf")), (
            "sampler_topk_logps pad should be -inf"
        )


class TestSCConfigValidation:
    """Verify PPOActorConfig validation for Score Centering fields."""

    def test_sc_loss_mode_invalid_raises(self):
        """Invalid sc_loss_mode should raise ValueError."""
        from areal.api.cli_args import PPOActorConfig

        with pytest.raises(ValueError, match="sc_loss_mode"):
            PPOActorConfig(path="dummy", sc_loss_mode="invalid")

    def test_sc_loss_mode_reinforce_default(self):
        """Default sc_loss_mode should be 'reinforce'."""
        from areal.api.cli_args import PPOActorConfig

        config = PPOActorConfig(path="dummy")
        assert config.sc_loss_mode == "reinforce"

    def test_sc_loss_mode_ppo_additive_valid(self):
        """sc_loss_mode='ppo_additive' should be accepted."""
        from areal.api.cli_args import PPOActorConfig

        config = PPOActorConfig(path="dummy", sc_loss_mode="ppo_additive")
        assert config.sc_loss_mode == "ppo_additive"

    def test_sc_mode_invalid_raises(self):
        """Invalid sc_mode should raise ValueError."""
        from areal.api.cli_args import PPOActorConfig

        with pytest.raises(ValueError, match="sc_mode"):
            PPOActorConfig(path="dummy", sc_mode="invalid")

    def test_sc_reinforce_baseline_with_sc_on_raises(self):
        """sc_reinforce_baseline=True + use_score_centering=True should raise."""
        from areal.api.cli_args import PPOActorConfig

        with pytest.raises(ValueError, match="sc_reinforce_baseline"):
            PPOActorConfig(
                path="dummy",
                use_score_centering=True,
                sc_reinforce_baseline=True,
                sc_top_k=10,
            )

    def test_tis_mode_requires_finite_clips(self):
        """tis mode with infinite clips should raise (two-sided clamp)."""
        from areal.api.cli_args import PPOActorConfig

        with pytest.raises(ValueError, match="tis.*finite"):
            PPOActorConfig(
                path="dummy",
                use_score_centering=True,
                sc_mode="tis",
                sc_clip_high=float("inf"),
                sc_top_k=10,
            )

    def test_tis_mode_requires_finite_low(self):
        """tis mode with infinite sc_clip_low should raise."""
        from areal.api.cli_args import PPOActorConfig

        with pytest.raises(ValueError, match="tis.*finite"):
            PPOActorConfig(
                path="dummy",
                use_score_centering=True,
                sc_mode="tis",
                sc_clip_low=float("inf"),
                sc_clip_high=5.0,
                sc_top_k=10,
            )

    def test_mis_mode_requires_finite_clips(self):
        """mis mode with infinite clips should raise."""
        from areal.api.cli_args import PPOActorConfig

        with pytest.raises(ValueError, match="mis.*finite"):
            PPOActorConfig(
                path="dummy",
                use_score_centering=True,
                sc_mode="mis",
                sc_clip_high=float("inf"),
                sc_top_k=10,
            )

    def test_sapo_and_reinforce_baseline_mutual_exclusion(self):
        """use_sapo_loss=True + sc_reinforce_baseline=True should raise."""
        from areal.api.cli_args import PPOActorConfig

        with pytest.raises(ValueError, match="sc_reinforce_baseline.*use_sapo_loss"):
            PPOActorConfig(
                path="dummy",
                use_sapo_loss=True,
                sc_reinforce_baseline=True,
            )

    def test_sc_off_does_not_validate_sc_mode(self):
        """SC-off with invalid sc_mode should NOT raise (inert config)."""
        from areal.api.cli_args import PPOActorConfig

        # sc_mode='invalid' but use_score_centering=False — should not raise
        config = PPOActorConfig(
            path="dummy",
            use_score_centering=False,
            sc_mode="invalid",
        )
        assert config.sc_mode == "invalid"
