# Gate 1 --- Parity

*Status: Proven by gates. Passes on Mistral-7B-Instruct-v0.2.*

Recomputed fp32 softmax attention, passed through the model's native `o_proj`, matches the native module output for the newest token. Calibrated across 2240 parity records (32 layers, short + medium sequences, 3 prompt shapes).

Frozen thresholds: cosine similarity >= 0.999996 (worst observed: 0.99999869), relative L2 <= 0.002759 (worst observed: 0.00184).

See [Gate Discipline](gate-discipline.md) for the calibration methodology and [Adapter Discipline](adapter-discipline.md) for the forward-replacement design.

<!-- Phase 2: Calibration methodology, pre-o_proj diagnostic, precision discipline, layer drift invariant -->
