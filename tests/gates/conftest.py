"""Shared fixtures for gate tests. Require GPU and real model."""

import pytest
import torch


def pytest_collection_modifyitems(config, items):  # type: ignore[no-untyped-def]
    """Auto-skip @pytest.mark.gpu tests when no GPU available."""
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="No GPU available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)


@pytest.fixture(scope="session")
def mistral_model_and_tokenizer():
    """Load Mistral-7B-Instruct-v0.2 once per test session.

    Uses fp16 (frozen decision: fp16 only for verification gates).
    Uses eager attention (required for forward-replacement adapter).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer
