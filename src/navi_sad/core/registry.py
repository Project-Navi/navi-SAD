"""Model family registry for navi-SAD.

Maps HuggingFace model architectures to adapter configurations.
Phase 1: Mistral only. Other families earn entries after gates pass.
"""

import logging

from navi_sad.core.types import ModelFamilyConfig

logger = logging.getLogger(__name__)

# Phase 1: Mistral only. Llama and Phi earn their entries after Gate 1.
MODEL_REGISTRY: dict[str, ModelFamilyConfig] = {
    "MistralForCausalLM": ModelFamilyConfig(
        architecture="MistralForCausalLM",
        attn_module_path="model.layers.{}.self_attn",
        capture_tier="A",
        num_kv_heads_attr="num_key_value_heads",
        num_q_heads_attr="num_attention_heads",
        head_dim_attr="head_dim",
        gqa_expansion=True,
        notes="GQA + sliding window (disabled for SAD).",
    ),
}


def get_family_config(model_config) -> ModelFamilyConfig:
    """Look up model family config from a HuggingFace model config.

    Auto-detects from model_config.architectures[0].

    Raises ValueError with diagnostic info if:
    - architectures attribute is missing
    - architectures is empty
    - architecture is not registered
    """
    if not hasattr(model_config, "architectures"):
        available = [a for a in dir(model_config) if not a.startswith("_")]
        raise ValueError(
            f"Model config has no 'architectures' attribute. "
            f"Available attributes: {available}"
        )

    archs = model_config.architectures
    if not archs:
        raise ValueError("Model config.architectures is empty.")

    arch = archs[0]
    if len(archs) > 1:
        logger.warning(
            "Multiple architectures listed: %s. Using first: %s", archs, arch
        )

    if arch not in MODEL_REGISTRY:
        supported = list(MODEL_REGISTRY.keys())
        raise ValueError(
            f"Architecture '{arch}' is not registered. "
            f"Supported: {supported}"
        )

    return MODEL_REGISTRY[arch]
