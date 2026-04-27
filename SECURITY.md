# Security Policy

## Project context

`navi-SAD` is a research harness, not a deployed product. It is intended to
be run locally by researchers against pinned model and dataset revisions
in controlled environments. It does not handle production data, expose
inbound network services, or accept remote-callable surfaces. It does
fetch model weights and datasets from HuggingFace at runtime when caches
are cold; that is an outbound supply-chain dependency rather than an
inbound attack surface, but it is in scope for vulnerability reports.
Threat models that apply to public-facing services are largely out of
scope.

That said, this repo follows responsible security disclosure for any
vulnerability that affects researchers running the code, the integrity
of recorded experimental artifacts, or the gate-parity invariants on
which research claims depend.

## Supported versions

`main` is the only supported branch. Pre-release tags exist for
historical reference but receive no backports.

What is pinned via `uv.lock`: Python dependency versions, the PyTorch
wheel (currently `torch==2.11.0+cu130`), and the CUDA user-space libs
PyTorch ships with (`nvidia-cublas`, `nvidia-cuda-runtime`,
`nvidia-cudnn-cu13`, etc. — pinned as transitive Python deps).

What is pinned in code: the model revision and dataset revision (commit
SHAs in `tests/gates/conftest.py` and `scripts/pilot_gate3.py`).

What is **not** pinned anywhere machine-consumable: GPU compute
capability / model, NVIDIA driver version, host OS / glibc / kernel.
Gate-parity tolerances themselves are frozen in code
(`tests/gates/test_gate1_parity.py` notes the calibration date; Gate 2
uses fixed thresholds not derived from any environment), but the
hardware they were calibrated on is not formally recorded. The
calibration was performed on an RTX 3090 (sm_86) at the dates noted in
the gate test files; numerics on different compute capabilities (e.g.,
A100 sm_80, H100 sm_90) may differ enough to require re-calibration.

This repo does not currently make end-to-end reproducibility claims; it
makes narrower internal claims about gate discipline that hold against
the pinned artifacts on the original calibration hardware.

| Branch | Supported |
|---|---|
| `main` | Yes |
| Tags / historical refs | Reproducibility only — no fixes |

## Reporting a vulnerability

Please use [GitHub's private vulnerability disclosure](https://docs.github.com/en/code-security/security-advisories/guidance-on-reporting-and-writing-information-about-vulnerabilities/privately-reporting-a-security-vulnerability)
on this repository to file a private security advisory.

What to include:

- A description of the vulnerability and the threat model it applies to
- The commit SHA and Python / model / OS environment where you observed it
- A minimal reproduction (test case, command, fixture)
- Whether the issue affects the *instrument* (gate parity, capture
  fidelity, persisted-artifact integrity) or the *infrastructure*
  (dependency CVEs, build pipeline, test harness)

Expected response: acknowledgement within 5 working days, initial triage
within 10 working days. This is a small research project; please be
patient if the response window slips.

## Scope

In scope:

- Vulnerabilities that allow tampering with persisted experimental
  artifacts (`results/**`) without detection
- Vulnerabilities that break the gate-parity invariants
  (Gates 0/1/2 produce false positives or false negatives)
- Vulnerabilities in the analysis pipeline that produce silently incorrect
  statistical results (wrong p-values, wrong effect sizes, wrong null
  distributions)
- Dependency CVEs that affect the runtime path of the instrument or
  pollute the test-harness environment
- Supply-chain risks introduced via build, test, or CI infrastructure

Out of scope:

- Generic Python / CPython vulnerabilities not affecting our usage
- Vulnerabilities only triggered by code paths we explicitly do not
  exercise (see "Accepted risks" below)
- Theoretical risks in transitively-pulled dev tooling (e.g., docs
  generation, linters) that are not present in production runs

## Accepted risks

Some dependency CVEs are tracked but not addressed because the
vulnerable code path is never invoked in our usage:

- **`transformers <5.0.0rc3`, GHSA-69w3-r845-3855 (medium):** Arbitrary
  code execution in HuggingFace's `Trainer` class. `navi-SAD` does not
  use `Trainer`; the codebase performs inference only. The
  forward-replacement adapter at `src/navi_sad/core/adapter.py` patches
  each attention module's `forward` method
  (`attn_module.forward = self._make_capturing_forward(...)` in
  `MistralAdapter.install`); it does not touch `Trainer` at all. The
  fix is in `5.0.0rc3`, which violates the project's frozen-decision
  pin (`transformers ~= 4.57`) — the patched attention forward is a
  verbatim upstream copy from `4.57.x` and any version bump requires
  Gate 0 re-verification before landing. Disposition: tracked,
  dismissed via Dependabot UI as `not_used`. Re-emerges if anyone ever
  invokes `Trainer`, which would itself violate the frozen-decision
  discipline.

If you discover a CVE that is currently classified as "accepted risk"
but believe the disposition is wrong (e.g., we *are* invoking the
vulnerable code path on some new branch), file a private advisory.

## Known security-relevant project decisions

- `transformers ~= 4.57` is pinned and version-coupled to a verbatim
  forward-replacement adapter. Bumps require Gate 0 re-verification.
- `attn_implementation="eager"` is enforced; FlashAttention and
  SDPA paths are explicitly rejected by the instrument because they
  bypass the capture insertion points.
- KV cache is disabled in harness entry points by passing
  `use_cache=False` to `model.generate`; the adapter's patched
  attention forward still supports `past_key_values` mechanically (so
  cache-on calls do not crash) but parity mode hard-fails on
  cache-on usage. Cache-on inference is an unverified scope extension.
- Dependabot has `transformers` ignored (no auto-PRs) but does NOT
  suppress security alerts for it (correct behavior).

See [`CLAUDE.md`](CLAUDE.md) for the full list of frozen decisions.
