# Adapter Discipline

*Status: Proven by gates ([Gate 0](gate-0.md) verifies non-interference).*

The patched forward is a verbatim copy of upstream Mistral attention. Three marked insertion points capture tensors; nothing else changes. Any modification to the upstream code requires [Gate 0](gate-0.md) re-verification.

This is not conservative engineering --- it is the [non-interference invariant](gate-discipline.md). If the observer changes the system, the measurement is meaningless.

- `attn_implementation="eager"` required. Non-negotiable.
- Transformers `~=4.57` pinned. Forward-replacement adapter is version-coupled.
- Model revision pinned in gate fixtures.

---

## The three insertion points

The patched forward in `core/adapter.py` is a verbatim copy of `MistralAttention.forward` from transformers 4.57.x. The `self` reference is rebound to `module` via closure; no other changes to upstream code.

### Insertion 1: Tier A capture (post-RoPE Q/K/V)

Located immediately after `apply_rotary_pos_emb`. The `capture_fn` receives live tensors --- not cloned. The callback (`InstrumentManager.capture_fn`) is responsible for `detach().clone().float()` before any computation. This is the primary capture point: post-RoPE Q/K/V at every layer, every generation step.

### Insertion 3: Pre-o_proj diagnostic

Located after `eager_attention_forward` returns but before `reshape` and `o_proj`. Captures native attention output at the newest token position \( [B, 1, H, D] \) *before* head reshape and output projection. Only active when `parity_fn is not None` ([Gate 1](gate-1.md) mode). Exists solely for failure localization.

Note: insertion 3 appears before insertion 2 in the code flow. The numbering reflects development order, not execution order.

### Insertion 2: Parity capture

Located after `o_proj`. Passes post-RoPE Q/K/V, native post-o_proj output, a reference to `o_proj`, and the pre-o_proj diagnostic tensor to the parity validation callback. Hard-fails if `past_key_values is not None` (cache-on parity not validated). Only active during [Gate 1](gate-1.md) testing.

## Why eager-only

The adapter patches `MistralAttention.forward` directly and calls `eager_attention_forward`. SDPA and Flash Attention use fused CUDA kernels that do not expose the intermediate attention output tensor needed for insertion 3. The forward control flow also differs between implementations --- the patched eager forward would produce incorrect results with a different attention backend.

The adapter hard-fails at runtime if `module.config._attn_implementation != "eager"`. This check runs inside every forward call, not just at install time, preventing silent breakage from config mutation.

## Version coupling

The patched forward is copied from transformers `>=4.57.0, <4.58.0`. A runtime version guard runs at `install()` time and raises if the installed transformers version is outside this range. This coupling is intentional --- the forward method's internal structure (argument order, tensor shapes, existence of `eager_attention_forward`) is version-specific. Any upstream change requires re-copying the forward, re-marking insertions, and re-running [Gate 0](gate-0.md).

## Capture tiers

| Tier | Capture Point | Trust Level |
|------|--------------|-------------|
| **A** | Post-RoPE Q/K/V intercepted inside native forward | Full --- strongest claims |
| **B** | Lower-level helper interception | High --- noted in methodology |
| **C** | Hidden-state capture + full Q/K/V recomputation | Degraded --- supplementary only |

Mistral is Tier A: the instrument sees the same tensors the model uses, intercepted inside the forward pass.
