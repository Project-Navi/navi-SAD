# Security Policy

## Project context

`navi-SAD` is a research harness, not a deployed product. It is intended to
be run locally by researchers against pinned model and dataset revisions in
controlled environments. It does not handle production data, accept
network input, or expose remote-callable surfaces. Threat models that
apply to public-facing services are largely out of scope.

That said, this repo follows responsible security disclosure for any
vulnerability that affects researchers running the code, the integrity of
recorded experimental artifacts, or the reproducibility of published gate
results.

## Supported versions

`main` is the only supported branch. Pre-release tags exist for
reproducibility but receive no backports. Dependency-pinning happens via
`uv.lock`; the lockfile is the canonical source of truth for what versions
are actually exercised.

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
  use `Trainer`; the codebase performs inference only via the
  forward-replacement adapter at `src/navi_sad/core/adapter.py`, which
  replaces `model.forward` and never invokes any Trainer code path.
  The fix is in `5.0.0rc3`, which violates the project's
  frozen-decision pin (`transformers ~= 4.57`) — the adapter is a
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
- KV cache is disabled by method definition; cache-on inference is an
  unverified scope extension.
- Dependabot has `transformers` ignored (no auto-PRs) but does NOT
  suppress security alerts for it (correct behavior).

See [`CLAUDE.md`](CLAUDE.md) for the full list of frozen decisions.
