# Gate 1 --- Parity

*Status: Proven by gates. Passes on Mistral-7B-Instruct-v0.2.*

Recomputed fp32 softmax attention, passed through the model's native `o_proj`, matches the native module output for the newest token. Calibrated across 2240 parity records (32 layers, short + medium sequences, 3 prompt shapes).

Frozen thresholds: cosine similarity >= 0.999996 (worst observed: 0.99999869), relative L2 <= 0.002759 (worst observed: 0.00184).

See [Gate Discipline](gate-discipline.md) for the calibration methodology and [Adapter Discipline](adapter-discipline.md) for the forward-replacement design.

---

## The recomputation path

The parity path answers: does our fp32 softmax recomputation, using the same post-RoPE Q/K/V the native model used, produce the same result?

1. **Capture:** The adapter's insertion point 1 captures post-RoPE Q/K/V as live tensors.
2. **fp32 softmax:** The parity callback clones and upcasts to fp32. GQA expansion applied (8 KV heads to 32 Q heads via `repeat_interleave`). Newest-token query slice \( q_{\text{last}} = Q[:, :, -1:, :] \) used for `softmax_attention_last_token`.
3. **Head merge:** Output \( [B, H, 1, D] \) transposed to \( [B, 1, H, D] \), then reshaped to \( [B, 1, H \cdot D] \). The order matters: transpose *then* reshape.
4. **Downcast and project:** Merged tensor downcast to model's native dtype (fp16) and passed through the layer's native `o_proj`.
5. **Compare:** Both recomputed and native outputs upcast to fp32 for metric computation.

### Why newest-token slice

SAD measures divergence at the newest generated token --- the point where the model makes its next-token prediction. Parity validates the same slice the instrument actually uses.

### Precision discipline

Recomputation runs in fp32 to match the instrument's precision. The downcast through `o_proj` is necessary because `o_proj` is a native-dtype linear layer. The \( \epsilon = 10^{-12} \) in the relative L2 denominator prevents division by zero on layers with near-zero native output norms.

## Pre-o_proj diagnostic

Insertion point 3 in the [adapter](adapter-discipline.md) captures the native attention output *before* the output projection: `attn_output[:, -1:, :, :]` with shape \( [B, 1, H, D] \). Cosine similarity between this and the recomputed softmax output isolates failure location.

If the pre-o_proj cosine is high but the post-o_proj cosine (the gate metric) is low, the problem is in the head-merge or projection path. If the pre-o_proj cosine is already low, the attention computation itself diverges --- a more serious problem.

## Layer drift invariant

Parity error should not systematically grow with depth. The test checks early (layer 0), middle (layer 16), and late (layer 31) layer mean cosine across all steps. Each must meet the frozen global minimum (cosine >= 0.999996). No layer showed systematic degradation across the 2240 calibration records.
