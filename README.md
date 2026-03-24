# navi-SAD (Spectral Attention Divergence)

Research harness for confabulation detection via dual-path attention comparison. SAD compares the attention patterns of a language model processing grounded vs. ungrounded prompts, measuring spectral divergence in the attention heads as a signal for hallucination.

**This is a research harness, not a product.**

## Scope (Phase 1)

- Model: Mistral-7B only
- Benchmark: TruthfulQA
- No baselines

## Installation

```bash
pip install -e ".[dev]"
```

## Testing

```bash
make test
```
