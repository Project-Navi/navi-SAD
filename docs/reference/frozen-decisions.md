# Frozen Decisions

*Status: Frozen. These will not change without compelling new evidence.*

| Decision | Choice | Rationale |
|----------|--------|-----------|
| KV cache | **Off** | Method definition --- generalization to cache-on is unverified |
| Quantization | **q8 minimum, fp16 only for gates** | Precision discipline |
| Precision | **Native dtype inference, fp32 instrument branch** | No silent dtype coercion |
| Capture boundary | **Post-RoPE Q/K/V** | Preferred; hidden-state fallback is Tier C |
| Temporal features | **[PE](../theory/takens-embedding.md) per-(layer, head) on first-differenced SAD trajectories (primary) + raw finite differences (supplementary) + [Renyi fingerprint](../research/open-problems.md) (planned)** | PE on pooled grand means is dead; per-head is alive |
| Registry scope | **Mistral only** | Until cross-family gates pass |
| Benchmarks | **Synthetic HMM sequences** | [Gate 3](../instrument/gate-discipline.md); TruthfulQA deferred to post-validation |
| Baselines | **None** | Until signal validated across architectures |
| Package manager | **uv** | No pip fallback; lockfile committed |
| Transformers | **~=4.57 pinned** | Forward-replacement adapter is version-coupled |
| Attention impl | **[Eager only](../instrument/adapter-discipline.md)** | Non-negotiable for instrumented models |
| Model revision | **Pinned** in gate fixtures | Update only after re-validating gates |
| Dataset revision | **Pinned** in pilot script (`741b8276...`) | Reproducibility |
| License | **Apache-2.0** | Copyright Project Navi LLC |
