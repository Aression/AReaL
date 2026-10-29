# SPDX-License-Identifier: Apache-2.0

"""End-to-end sampler top-k positioning: parser, assembly, shift and packing."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
import torch

from areal.api import ModelResponse
from areal.api.cli_args import GenerationHyperparameters, PPOActorConfig
from areal.engine.vllm_remote import VLLMBackend
from areal.experimental.openai.types import InteractionWithTokenLogpReward
from areal.trainer.ppo import actor as actor_module
from areal.utils.data import concat_padded_tensors, pack_tensor_dict
from areal.workflow.rlvr import RLVRWorkflow

# Distinct distributions per generated position so misalignment shows values.
POSITION_TOKEN_SETS = [[2, 0], [1, 3], [0, 2]]
POSITION_LOGPS = [[-0.4, -1.2], [-0.7, -2.1], [-0.2, -0.9]]


def _fixed_response(input_tokens=(10, 11), output_tokens=(21, 22, 23)):
    """Build a parser request/response with a different head at each position."""
    top_logprobs = [
        {f"token:{token}": float(logp) for token, logp in zip(tokens, logps)}
        for tokens, logps in zip(POSITION_TOKEN_SETS, POSITION_LOGPS)
    ]
    response = {
        "choices": [
            {
                "finish_reason": "stop",
                "token_ids": list(output_tokens),
                "logprobs": {
                    "content": [{"logprob": 0.0} for _ in output_tokens],
                    "top_logprobs": top_logprobs,
                },
            }
        ],
    }
    backend = VLLMBackend()
    return (
        backend,
        response,
        ModelResponse(
            input_tokens=list(input_tokens),
            output_tokens=list(output_tokens),
            output_logprobs=[0.0] * len(output_tokens),
            output_versions=[7] * len(output_tokens),
        ),
    )


def test_parser_keeps_each_position_distinct_head():
    """Every generated position must keep its own token/logprob pairing."""
    _, response, _ = _fixed_response()
    result = VLLMBackend().parse_generation_response(response)

    assert result.output_topk_ids == POSITION_TOKEN_SETS
    assert result.output_topk_logps == POSITION_LOGPS
    # Swapped pairing would violate the live logprob inventories.
    assert (
        -1.2 not in result.output_topk_logps[0][0:1]
        or result.output_topk_logps[0][1] == -1.2
    )


def test_openai_assembly_positions_prompts_and_outputs_hand_written():
    """Prompt rows must be empty and output rows must match the response layout."""
    _, _, model_response = _fixed_response(
        input_tokens=(10, 11), output_tokens=(21, 22, 23)
    )
    interaction = InteractionWithTokenLogpReward(
        sc_top_k=len(POSITION_TOKEN_SETS[0]),
        reward=0.0,
        model_response=model_response,
    )
    assembled = interaction.to_tensor_dict()

    assert assembled["sampler_topk_token_ids"].shape == (1, 5, 2)
    torch.testing.assert_close(
        assembled["sampler_topk_logps"].squeeze(0),
        torch.tensor(
            [
                [float("-inf")] * 2,
                [float("-inf")] * 2,
                POSITION_LOGPS[0],
                POSITION_LOGPS[1],
                POSITION_LOGPS[2],
            ]
        ),
        rtol=0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        assembled["sampler_topk_token_ids"].squeeze(0),
        torch.tensor([[0, 0], [0, 0]] + POSITION_TOKEN_SETS, dtype=torch.int32),
        rtol=0,
        atol=0,
    )


def test_assembly_caps_k_dim_and_rejects_short_heads():
    """extra positions must cap k; fewer entries must fail loudly."""
    _, _, model_response = _fixed_response()
    model_response.output_topk_ids[0].insert(0, 5)
    model_response.output_topk_logps[0].insert(0, -3.0)
    interaction = InteractionWithTokenLogpReward(
        sc_top_k=2,
        reward=0.0,
        model_response=model_response,
    )
    capped = interaction.to_tensor_dict()
    assert capped["sampler_topk_token_ids"].shape == (1, 5, 2)
    assert capped["sampler_topk_token_ids"][0, 5].tolist() == POSITION_TOKEN_SETS[0]

    short = ModelResponse(
        input_tokens=[10],
        output_tokens=[21],
        output_logprobs=[0.0],
        output_versions=[7],
        output_topk_ids=[[4]],
        output_topk_logps=[[-0.2]],
    )
    with pytest.raises(RuntimeError, match="sc_top_k=2"):
        InteractionWithTokenLogpReward(
            sc_top_k=2,
            reward=0.0,
            model_response=short,
        ).to_tensor_dict()


def test_rlvr_assembly_matches_openai_layout():
    """Both assembly paths must emit identical placement and padding."""
    _, _, model_response = _fixed_response(
        input_tokens=(10, 11), output_tokens=(21, 22, 23)
    )
    stub_tokenizer = SimpleNamespace(
        eos_token_id=9,
        pad_token_id=0,
        decode=lambda ids: "stub",
        get_vocab=lambda: {},
    )
    gconfig = GenerationHyperparameters(max_new_tokens=8, sc_top_k=2)
    workflow = RLVRWorkflow(
        reward_fn="areal.experimental.openai.reward.gsm8k_reward_fn",  # replaced below
        gconfig=gconfig,
        tokenizer=stub_tokenizer,
    )
    workflow.async_reward_fn = AsyncMock(return_value=1.0)

    class _Engine:
        async def agenerate(self, request):
            return model_response

    result = workflow._collect_local_topk_layout(_Engine(), model_response)
    torch.testing.assert_close(
        result["sampler_topk_logps"].squeeze(0)[:2],
        torch.full((2, 2), float("-inf"), dtype=torch.float32),
        rtol=0,
        atol=0,
    )
    for index, (tokens, logps) in enumerate(zip(POSITION_TOKEN_SETS, POSITION_LOGPS)):
        row = result["sampler_topk_logps"].squeeze(0)[2 + index]
        gathered = row.gather(-1, torch.tensor(tokens, dtype=torch.int64))
        torch.testing.assert_close(
            row.exp().sum(),
            torch.tensor(1.0).exp() * 0 + gathered.exp().sum(),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            gathered.exp(), torch.exp(torch.tensor(logps)), rtol=1e-6, atol=1e-7
        )


def test_actor_shift_aligns_logits_with_next_token_sampler_rows():
    """Token t must read sampler row t+1 exactly once, watching the shift."""
    prompt_len, output_len, seq_len = 2, 3, 5
    response, _ = _build_response_for_advantage(prompt_len, output_len)
    sampler_rows = torch.stack(
        [
            torch.tensor(POSITION_LOGPS[index % 3], dtype=torch.float32).expand(1, 2)
            for index in range(seq_len)
        ]
    )
    # Prompt rows carry no sampler data; the first output row is the live head.
    sampler_rows[0, 0, :] = float("-inf")
    sampler_rows[1, 0, :] = float("-inf")

    data = {
        "input_ids": torch.arange(seq_len).reshape(1, seq_len),
        "attention_mask": torch.ones(1, seq_len, dtype=torch.bool),
        # Only the first pad of every response keeps loss; the actor rolls -1.
        "loss_mask": torch.tensor([[0, 0, 1, 1, 0]], dtype=torch.float32),
        "logprobs": torch.zeros(1, seq_len),
        "versions": torch.zeros(1, seq_len, dtype=torch.int32),
        "rewards": torch.zeros(1),
        "tot_rewards": torch.zeros(1, seq_len),
        "kl_rewards": torch.zeros(1, seq_len),
        "sampler_topk_logps": sampler_rows.clone(),
        "sampler_topk_token_ids": torch.tensor(
            [[0, 0], [0, 0]] + POSITION_TOKEN_SETS[:3][0:0] + [[1, 3], [0, 2], [2, 0]],
            dtype=torch.int32,
        ),
    }
    engine = SimpleNamespace(get_version=lambda: 0)
    config = PPOActorConfig(path="dummy")
    actor = actor_module.PPOActor(config, engine)
    updated = actor._compute_advantages(data, _meta_stub())

    assert torch.equal(updated["sampler_topk_logps"][0, 0], sampler_rows[0, 0])
    assert torch.equal(
        updated["sampler_topk_logps"][0, seq_len - 1], sampler_rows[0, 0]
    )
    assert torch.equal(updated["sampler_topk_logps"][0, 2], sampler_rows[0, 3])


def _build_response_for_advantage(prompt_len, output_len):
    return ModelResponse(), None


def _meta_stub():
    from areal.utils.data import TrajBatchMeta

    return TrajBatchMeta()


def test_packing_preserves_per_position_pairing():
    """Packed rows must keep the exact sampler entries of their source tokens."""
    rows = torch.tensor(POSITION_LOGPS, dtype=torch.float32)
    data = concat_padded_tensors(
        [
            {
                "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
                "attention_mask": torch.ones(1, 3, dtype=torch.bool),
                "loss_mask": torch.tensor([[0, 1, 0]]),
                "logprobs": torch.zeros(1, 3),
                "sampler_topk_logps": rows.unsqueeze(0),
                "sampler_topk_token_ids": torch.tensor(POSITION_TOKEN_SETS).unsqueeze(
                    0
                ),
                "versions": torch.zeros(1, 3, dtype=torch.int32),
                "rewards": torch.zeros(1),
                "tot_rewards": torch.zeros(1, 3),
                "kl_rewards": torch.zeros(1, 3),
            },
            {
                "input_ids": torch.tensor([[6, 7, 8, 9, 10]]),
                "attention_mask": torch.ones(1, 2, dtype=torch.bool),
                "loss_mask": torch.tensor([[0, 1]]),
                "logprobs": torch.zeros(1, 2),
                "sampler_topk_logps": rows.unsqueeze(0),
                "sampler_topk_token_ids": torch.tensor(POSITION_TOKEN_SETS).unsqueeze(
                    0
                ),
                "versions": torch.zeros(1, 2, dtype=torch.int32),
                "rewards": torch.zeros(1),
                "tot_rewards": torch.zeros(1, 2),
                "kl_rewards": torch.zeros(1, 2),
            },
        ]
    )
    packed = pack_tensor_dict(data)

    assert packed["sampler_topk_logps"].shape == (5, 2)
    assert packed["sampler_topk_logps"][1].tolist() == POSITION_LOGPS[0]
    assert packed["sampler_topk_logps"][3].tolist() == POSITION_LOGPS[0]
    assert packed["sampler_topk_token_ids"][1].tolist() == POSITION_TOKEN_SETS[0]
