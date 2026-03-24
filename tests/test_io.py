"""Tests for raw/derived JSONL I/O with gzip compression."""

import math
import random

import pytest

from navi_sad.core.types import RawSampleRecord, StepRecord


# ---------------------------------------------------------------------------
# Helper: build a RawSampleRecord with N steps
# ---------------------------------------------------------------------------
def make_raw_record(
    sample_id: str,
    num_steps: int,
    run_id: str = "run-test-001",
    *,
    seed: int = 42,
) -> RawSampleRecord:
    """Build a RawSampleRecord with *num_steps* StepRecords.

    Each StepRecord has 4 per-head deltas (simulating 4 attention heads)
    with realistic values in the 0.3-0.9 range. Two layers per step.
    """
    rng = random.Random(seed + hash(sample_id))
    steps: list[StepRecord] = []
    for step_idx in range(num_steps):
        for layer_idx in range(2):
            steps.append(
                StepRecord(
                    step_idx=step_idx,
                    layer_idx=layer_idx,
                    per_head_delta=[
                        round(rng.uniform(0.3, 0.9), 6) for _ in range(4)
                    ],
                )
            )
    return RawSampleRecord(
        sample_id=sample_id,
        model="test-model",
        benchmark="test-bench",
        prompt="What is 2+2?",
        generation="4",
        label="correct",
        num_tokens_generated=num_steps,
        layers_hooked=[0, 1],
        per_step=steps,
        metadata={"run_id": run_id, "note": "synthetic test data"},
    )


# ===========================================================================
# TestRawRoundTrip
# ===========================================================================
class TestRawRoundTrip:
    def test_write_read_single(self, tmp_path: pytest.TempPathFactory) -> None:
        """Write one RawSampleRecord, read back, verify all fields match."""
        from navi_sad.io.reader import RawRecordReader
        from navi_sad.io.writer import RawRecordWriter

        path = tmp_path / "single.jsonl.gz"
        original = make_raw_record("sample-001", num_steps=5)

        with RawRecordWriter(path) as w:
            w.write(original)

        records = list(RawRecordReader(path))
        assert len(records) == 1

        rec = records[0]
        assert rec["sample_id"] == "sample-001"
        assert rec["model"] == "test-model"
        assert rec["schema_version"] == 1
        assert rec["record_type"] == "raw"
        assert rec["num_tokens_generated"] == 5
        assert rec["metadata"]["run_id"] == "run-test-001"
        assert len(rec["per_step"]) == 10  # 5 steps x 2 layers
        # Verify a step record's per_head_delta has 4 values
        assert len(rec["per_step"][0]["per_head_delta"]) == 4

    def test_write_read_multiple(self, tmp_path: pytest.TempPathFactory) -> None:
        """Write 10 records, read back, verify count and sample_ids."""
        from navi_sad.io.reader import RawRecordReader
        from navi_sad.io.writer import RawRecordWriter

        path = tmp_path / "multi.jsonl.gz"
        expected_ids = [f"sample-{i:03d}" for i in range(10)]

        with RawRecordWriter(path) as w:
            for sid in expected_ids:
                w.write(make_raw_record(sid, num_steps=8))

        records = list(RawRecordReader(path))
        assert len(records) == 10
        assert [r["sample_id"] for r in records] == expected_ids


# ===========================================================================
# TestDeriveFromRaw
# ===========================================================================
class TestDeriveFromRaw:
    def _write_raw(self, path, records: list[RawSampleRecord]) -> None:
        """Helper: write raw records to a gzipped JSONL file."""
        from navi_sad.io.writer import RawRecordWriter

        with RawRecordWriter(path) as w:
            for rec in records:
                w.write(rec)

    def test_derive_produces_output(self, tmp_path: pytest.TempPathFactory) -> None:
        """Write 3 raw samples (15+ steps each), derive, verify 3 derived records."""
        from navi_sad.io.derived import derive_from_raw
        from navi_sad.io.reader import DerivedRecordReader

        raw_path = tmp_path / "raw.jsonl.gz"
        derived_path = tmp_path / "derived.jsonl.gz"

        raws = [make_raw_record(f"s-{i}", num_steps=20) for i in range(3)]
        self._write_raw(raw_path, raws)

        count = derive_from_raw(raw_path, derived_path)
        assert count == 3

        derived = list(DerivedRecordReader(derived_path))
        assert len(derived) == 3

    def test_derived_has_pe(self, tmp_path: pytest.TempPathFactory) -> None:
        """Derived record has ordinal.pe as a float when enough tokens are present."""
        from navi_sad.io.derived import derive_from_raw
        from navi_sad.io.reader import DerivedRecordReader

        raw_path = tmp_path / "raw_pe.jsonl.gz"
        derived_path = tmp_path / "derived_pe.jsonl.gz"

        # 20 steps -> 20 per-token deltas, well above MIN_PE_LENGTH=10
        self._write_raw(raw_path, [make_raw_record("pe-test", num_steps=20)])

        derive_from_raw(raw_path, derived_path)
        derived = list(DerivedRecordReader(derived_path))
        assert len(derived) == 1

        ordinal = derived[0]["ordinal"]
        assert ordinal is not None
        assert isinstance(ordinal["pe"], float)
        assert 0.0 <= ordinal["pe"] <= 1.0

    def test_short_sequence_pe_none(self, tmp_path: pytest.TempPathFactory) -> None:
        """Derived from a raw record with only 5 steps -> ordinal.pe is None."""
        from navi_sad.io.derived import derive_from_raw
        from navi_sad.io.reader import DerivedRecordReader

        raw_path = tmp_path / "raw_short.jsonl.gz"
        derived_path = tmp_path / "derived_short.jsonl.gz"

        # 5 steps -> 5 per-token deltas, below MIN_PE_LENGTH=10
        self._write_raw(raw_path, [make_raw_record("short-test", num_steps=5)])

        derive_from_raw(raw_path, derived_path)
        derived = list(DerivedRecordReader(derived_path))
        assert len(derived) == 1

        ordinal = derived[0]["ordinal"]
        assert ordinal["pe"] is None

    def test_aggregation_method_stored(self, tmp_path: pytest.TempPathFactory) -> None:
        """Derived record stores aggregation_method='uniform_mean'."""
        from navi_sad.io.derived import derive_from_raw
        from navi_sad.io.reader import DerivedRecordReader

        raw_path = tmp_path / "raw_agg.jsonl.gz"
        derived_path = tmp_path / "derived_agg.jsonl.gz"

        self._write_raw(raw_path, [make_raw_record("agg-test", num_steps=15)])

        derive_from_raw(raw_path, derived_path)
        derived = list(DerivedRecordReader(derived_path))
        assert derived[0]["aggregation_method"] == "uniform_mean"

    def test_round_trip_provenance(self, tmp_path: pytest.TempPathFactory) -> None:
        """source_run_id in derived matches run_id from raw metadata."""
        from navi_sad.io.derived import derive_from_raw
        from navi_sad.io.reader import DerivedRecordReader

        raw_path = tmp_path / "raw_prov.jsonl.gz"
        derived_path = tmp_path / "derived_prov.jsonl.gz"

        run_id = "run-provenance-abc"
        self._write_raw(
            raw_path,
            [make_raw_record("prov-test", num_steps=15, run_id=run_id)],
        )

        derive_from_raw(raw_path, derived_path)
        derived = list(DerivedRecordReader(derived_path))
        assert derived[0]["source_run_id"] == run_id
