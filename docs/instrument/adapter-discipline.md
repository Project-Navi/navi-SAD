# Adapter Discipline

*Status: Proven by gates (Gate 0 verifies non-interference).*

The patched forward is a verbatim copy of upstream Mistral attention. Three marked insertion points capture tensors; nothing else changes. Any modification to the upstream code requires Gate 0 re-verification.

This is not conservative engineering --- it is the non-interference invariant. If the observer changes the system, the measurement is meaningless.

- `attn_implementation="eager"` required. Non-negotiable.
- Transformers `~=4.57` pinned. Forward-replacement adapter is version-coupled.
- Model revision pinned in gate fixtures.

<!-- Phase 2: The three insertion points, what they capture, why eager-only, version coupling rationale -->
