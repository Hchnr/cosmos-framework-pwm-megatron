# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

import logging
from unittest.mock import Mock

import pytest
import torch

from cosmos_framework.data.generator.processors.qwen3vl_processor import Qwen3VLProcessor

_BOS_ID = 10
_EOS_ID = 11
_ASSISTANT_IDS = [12, 13]
_HEADER_SEPARATOR_ID = 14
_THINK_START_ID = 20
_THINK_END_ID = 21
_ANSWER_ID = 30
_REASONING_ID = 31
_USER_ID = 40


@pytest.mark.parametrize("model_name", ["nvidia/Cosmos3-Nano", "nvidia/Cosmos3-Edge"])
def test_build_processor_routes_nano_and_edge_to_the_same_public_class(
    monkeypatch: pytest.MonkeyPatch,
    model_name: str,
) -> None:
    import cosmos_framework.data.generator.processors as processors

    constructed: list[str] = []

    class UnifiedProcessor:
        def __init__(self, name: str, **_: object) -> None:
            constructed.append(name)

    monkeypatch.setattr(processors, "Qwen3VLProcessor", UnifiedProcessor)

    result = processors.build_processor(model_name, config_variant="hf")

    assert type(result) is UnifiedProcessor
    assert constructed == [model_name]


def _make_processor_stub() -> Qwen3VLProcessor:
    processor = Qwen3VLProcessor.__new__(Qwen3VLProcessor)
    processor.processor = Mock()
    tokenizer = Mock()
    token_ids = {
        "<|im_start|>": _BOS_ID,
        "<|im_end|>": _EOS_ID,
        "<think>": _THINK_START_ID,
        "</think>": _THINK_END_ID,
    }
    tokenizer.convert_tokens_to_ids.side_effect = token_ids.__getitem__
    tokenizer.encode.return_value = [*_ASSISTANT_IDS, _HEADER_SEPARATOR_ID]
    tokenizer.all_special_ids = [_BOS_ID, _EOS_ID, _THINK_START_ID, _THINK_END_ID]
    tokenizer.get_vocab.return_value = token_ids
    processor.processor.tokenizer = tokenizer
    return processor


@pytest.mark.L0
def test_assistant_mask_handles_multi_token_role_and_structure_tokens() -> None:
    processor = _make_processor_stub()
    tokens = [
        _BOS_ID,
        _USER_ID,
        _EOS_ID,
        _BOS_ID,
        *_ASSISTANT_IDS,
        _HEADER_SEPARATOR_ID,
        _THINK_START_ID,
        _REASONING_ID,
        _THINK_END_ID,
        _ANSWER_ID,
        _EOS_ID,
    ]

    mask = processor.add_assistant_tokens_mask(tokens)

    assert mask == [False, False, False, False, False, False, False, False, True, False, True, True]


@pytest.mark.L0
def test_assistant_mask_supports_batches() -> None:
    processor = _make_processor_stub()
    tokens = torch.tensor(
        [
            [_BOS_ID, *_ASSISTANT_IDS, _HEADER_SEPARATOR_ID, _ANSWER_ID, _EOS_ID],
            [_BOS_ID, *_ASSISTANT_IDS, _HEADER_SEPARATOR_ID, _THINK_START_ID, _EOS_ID],
        ]
    )

    mask = processor.add_assistant_tokens_mask(tokens)

    assert torch.equal(
        mask,
        torch.tensor(
            [
                [False, False, False, False, True, True],
                [False, False, False, False, False, True],
            ]
        ),
    )


@pytest.mark.L0
def test_apply_chat_template_forwards_generation_controls_without_false_warning(caplog) -> None:
    processor = Qwen3VLProcessor.__new__(Qwen3VLProcessor)
    processor.processor = Mock()
    processor.processor.image_processor.size = {"shortest_edge": 256, "longest_edge": 1024}

    def _apply_chat_template(*_args, **_kwargs):
        logging.getLogger("transformers.processing_utils").warning(
            "Keyword argument `enable_thinking` is not a valid argument for this processor and will be ignored."
        )
        return {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    processor.processor.apply_chat_template.side_effect = _apply_chat_template

    result = processor.apply_chat_template(
        [{"role": "user", "content": [{"type": "text", "text": "prompt"}]}],
        add_generation_prompt=True,
        enable_thinking=False,
    )

    assert result["input_ids"].tolist() == [1, 2, 3]
    assert processor.processor.apply_chat_template.call_args.kwargs["enable_thinking"] is False
    assert "not a valid argument for this processor" not in caplog.text
