"""Tests for model family registry: lookup, validation, and error paths."""

from unittest.mock import MagicMock

import pytest

from navi_sad.core.registry import get_family_config
from navi_sad.core.types import ModelFamilyConfig


def _mock_config(architectures: list[str] | None = None, *, has_attr: bool = True) -> MagicMock:
    """Create a mock HuggingFace model config.

    Args:
        architectures: List of architecture strings, or None.
        has_attr: If False, the mock will not have an 'architectures' attribute.
    """
    mock = MagicMock()
    if has_attr:
        mock.architectures = architectures
    else:
        del mock.architectures
    return mock


# ===========================================================================
# TestMistralLookup — happy-path lookups for the Phase 1 model
# ===========================================================================
class TestMistralLookup:
    def test_mistral_lookup(self) -> None:
        """Mock config with architectures=["MistralForCausalLM"] returns correct config."""
        config = _mock_config(["MistralForCausalLM"])
        result = get_family_config(config)
        assert isinstance(result, ModelFamilyConfig)
        assert result.architecture == "MistralForCausalLM"
        assert result.attn_module_path == "model.layers.{}.self_attn"
        assert result.capture_tier == "A"
        assert result.num_kv_heads_attr == "num_key_value_heads"
        assert result.num_q_heads_attr == "num_attention_heads"
        assert result.head_dim_attr == "head_dim"

    def test_mistral_gqa(self) -> None:
        """Verify gqa_expansion is True for Mistral (GQA architecture)."""
        config = _mock_config(["MistralForCausalLM"])
        result = get_family_config(config)
        assert result.gqa_expansion is True

    def test_mistral_has_adapter_factory(self) -> None:
        """Mistral registry entry includes a callable adapter factory."""
        from navi_sad.core.adapter import MistralAdapter

        config = _mock_config(["MistralForCausalLM"])
        result = get_family_config(config)
        assert result.adapter_factory is not None
        instance = result.adapter_factory()
        assert isinstance(instance, MistralAdapter)


# ===========================================================================
# TestNotRegistered — Phase 1 scope enforcement
# ===========================================================================
class TestNotRegistered:
    def test_llama_not_registered(self) -> None:
        """LlamaForCausalLM is not in Phase 1 registry."""
        config = _mock_config(["LlamaForCausalLM"])
        with pytest.raises(ValueError, match="LlamaForCausalLM"):
            get_family_config(config)

    def test_phi_not_registered(self) -> None:
        """Phi3ForCausalLM is not in Phase 1 registry."""
        config = _mock_config(["Phi3ForCausalLM"])
        with pytest.raises(ValueError, match="Phi3ForCausalLM"):
            get_family_config(config)

    def test_unknown_raises(self) -> None:
        """Unknown architecture raises ValueError with name and supported list."""
        config = _mock_config(["UnknownModel"])
        with pytest.raises(ValueError, match="UnknownModel") as exc_info:
            get_family_config(config)
        # Error message should also list what IS supported
        assert "MistralForCausalLM" in str(exc_info.value)


# ===========================================================================
# TestErrorPaths — missing or malformed config attributes
# ===========================================================================
class TestErrorPaths:
    def test_missing_architectures_attr(self) -> None:
        """Config with no architectures attribute raises ValueError."""
        config = _mock_config(has_attr=False)
        with pytest.raises(ValueError, match="architectures"):
            get_family_config(config)

    def test_empty_architectures(self) -> None:
        """Config with empty architectures list raises ValueError."""
        config = _mock_config([])
        with pytest.raises(ValueError, match="empty"):
            get_family_config(config)
