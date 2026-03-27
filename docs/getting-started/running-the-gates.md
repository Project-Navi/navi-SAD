# Running the Gates

The gate tests validate the instrument on a real model. They require a CUDA GPU and Mistral-7B weights.

## Prerequisites

- CUDA-capable GPU with sufficient VRAM for Mistral-7B-Instruct-v0.2 in fp16
- All dependency groups installed: `uv sync --extra dev --extra eval`

## Run

```bash
make test-gpu
```

This runs all tests tagged `@pytest.mark.gpu`, including Gates 0, 1, and 2.

## What the gates test

| Gate | What | Pass criterion |
|------|------|---------------|
| 0 | Non-interference | Bit-identical tokens and logits with/without instrumentation |
| 1 | Parity | Cosine >= 0.999996, relative L2 <= 0.002759 (frozen thresholds) |
| 2 | Stability | Zero VRAM creep over 50 generations (limit 16 MiB) |

See the [Gate Discipline](../instrument/gate-discipline.md) section for the full methodology.
