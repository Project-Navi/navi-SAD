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
    Sets deterministic CUDA controls so Gate 0 flakes can be
    distinguished from real adapter perturbations.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Deterministic CUDA controls for reproducible gate tests.
    torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
    torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    # Pin revision for reproducibility. Update only after re-validating gates.
    revision = "63a8b081895390a26e140280378bc85ec8bce07a"
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.float16,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer
