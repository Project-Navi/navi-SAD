"""Generate derived analysis records from raw records."""

import gzip
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from navi_sad.core.types import StepRecord
from navi_sad.io.reader import RawRecordReader
from navi_sad.signal.aggregation import aggregate_deltas
from navi_sad.signal.derivatives import compute_derivatives
from navi_sad.signal.ordinal import permutation_entropy, recommended_min_pe_length
from navi_sad.signal.types import DerivedSampleRecord, OrdinalResult
from navi_sad.version import __version__


def derive_from_raw(
    raw_path: Path | str,
    output_path: Path | str,
    aggregation_method: str = "uniform_mean",
) -> int:
    """Generate derived records from raw inference JSONL.

    Args:
        raw_path: path to raw gzipped JSONL
        output_path: path for derived gzipped JSONL
        aggregation_method: how to aggregate per-layer-per-head deltas

    Returns:
        number of records processed
    """
    count = 0
    with gzip.open(Path(output_path), "wt", encoding="utf-8") as out:
        for raw in RawRecordReader(raw_path):
            steps = [StepRecord(**s) for s in raw.get("per_step", [])]

            per_token_delta = aggregate_deltas(steps, method=aggregation_method)
            derivs = compute_derivatives(per_token_delta)

            D, tau = 3, 1
            if len(per_token_delta) >= recommended_min_pe_length(D, tau):
                pe, tie_rate, counts = permutation_entropy(
                    per_token_delta, D=D, tau=tau
                )
                ordinal = OrdinalResult(
                    pe=pe, tie_rate=tie_rate, pattern_counts=counts, D=D, tau=tau
                )
            else:
                ordinal = OrdinalResult(
                    pe=None, tie_rate=0.0, pattern_counts={}, D=D, tau=tau
                )

            dp = derivs["delta_prime"]
            ddp = derivs["delta_double_prime"]
            summary = {
                "delta_mean": float(np.mean(per_token_delta))
                if per_token_delta
                else 0.0,
                "delta_std": float(np.std(per_token_delta))
                if per_token_delta
                else 0.0,
                "delta_prime_mean": float(np.mean(dp)) if dp else 0.0,
                "delta_prime_max": float(np.max(np.abs(dp))) if dp else 0.0,
                "delta_double_prime_max": float(np.max(np.abs(ddp)))
                if ddp
                else 0.0,
            }

            derived = DerivedSampleRecord(
                sample_id=raw.get("sample_id", ""),
                source_run_id=raw.get("metadata", {}).get("run_id", ""),
                per_token_delta=per_token_delta,
                delta_prime=dp,
                delta_double_prime=derivs["delta_double_prime"],
                delta_triple_prime=derivs["delta_triple_prime"],
                ordinal=ordinal,
                summary=summary,
                aggregation_method=aggregation_method,
                analysis_version=__version__,
            )

            out.write(json.dumps(asdict(derived)) + "\n")
            count += 1

    return count
