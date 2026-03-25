"""Tests for Gate 3 pilot helpers.

Covers: is_word_boundary, extract_leading_span, score_sample,
compute_mean_delta_matrix, find_leading_span_token_count.
"""

from __future__ import annotations

import pytest

from navi_sad.core.types import StepRecord
from navi_sad.pilot.helpers import (
    compute_mean_delta_matrix,
    extract_leading_span,
    find_leading_span_token_count,
    is_word_boundary,
    score_sample,
)

# -------------------------------------------------------------------
# is_word_boundary
# -------------------------------------------------------------------


@pytest.mark.parametrize(
    "char,expected",
    [
        ("", True),  # end-of-string
        (" ", True),  # space
        ("\t", True),  # tab
        ("\n", True),  # newline
        ("\r", True),  # carriage return
        (".", True),  # period (punctuation)
        (",", True),  # comma
        ("!", True),  # exclamation
        (";", True),  # semicolon
        ("(", True),  # paren
        ("-", True),  # hyphen
        ("a", False),  # letter
        ("Z", False),  # uppercase letter
        ("5", False),  # digit
        ("\u2014", False),  # em dash — Unicode, not ASCII
        ("\u201c", False),  # left double quote — Unicode
        ("\u2019", False),  # right single quote — Unicode
    ],
)
def test_is_word_boundary(char: str, expected: bool) -> None:
    assert is_word_boundary(char) == expected


# -------------------------------------------------------------------
# extract_leading_span
# -------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected_span,expected_reason",
    [
        # No break — full text returned
        ("Paris is the capital", "Paris is the capital", "eos"),
        # Period-space
        ("Paris. The city is beautiful", "Paris", "period_space"),
        # Period-newline
        ("Paris.\nThe city is beautiful", "Paris", "period_newline"),
        # Bare newline
        ("Paris\nThe city is beautiful", "Paris", "newline"),
        # Empty input
        ("", "", "eos"),
        # Whitespace-only input
        ("   ", "", "eos"),
        # Strips leading/trailing whitespace, no break inside
        ("  hello  ", "hello", "eos"),
        # Preserves case in span (raw, not lowercased)
        ("PARIS. The city", "PARIS", "period_space"),
        # Mid-word periods — no break (no space or newline after)
        ("U.S.A", "U.S.A", "eos"),
        # Abbreviation: period-space after S
        ("U.S. troops", "U.S", "period_space"),
        # Mixed delimiters: breaks at earliest (period-space before newline)
        ("foo. bar\nbaz", "foo", "period_space"),
        # Mixed delimiters: newline before period-space
        ("foo\nbar. baz", "foo", "newline"),
        # Period-newline takes precedence over bare newline at next position
        ("foo.\nbar", "foo", "period_newline"),
        # Trailing period without space/newline — not a break
        ("Paris.", "Paris.", "eos"),
        # Period at very end of string
        ("Hello world.", "Hello world.", "eos"),
        # Multiple sentences
        ("First. Second. Third.", "First", "period_space"),
    ],
)
def test_extract_leading_span(text: str, expected_span: str, expected_reason: str) -> None:
    span, reason = extract_leading_span(text)
    assert span == expected_span
    assert reason == expected_reason


# -------------------------------------------------------------------
# score_sample
# -------------------------------------------------------------------


class TestScoreSample:
    """Tests for the shadow scorer (truthfulqa_exact_v1)."""

    def test_exact_correct(self) -> None:
        label, mc, mi = score_sample("Paris", ["Paris"], ["London"])
        assert label == "correct"
        assert mc == ["Paris"]
        assert mi == []

    def test_exact_incorrect(self) -> None:
        label, mc, mi = score_sample("London", ["Paris"], ["London"])
        assert label == "incorrect"
        assert mc == []
        assert mi == ["London"]

    def test_both_match_ambiguous(self) -> None:
        label, mc, mi = score_sample("Paris", ["Paris"], ["Paris"])
        assert label == "ambiguous"
        assert mc == ["Paris"]
        assert mi == ["Paris"]

    def test_neither_match_ambiguous(self) -> None:
        label, mc, mi = score_sample("Berlin", ["Paris"], ["London"])
        assert label == "ambiguous"
        assert mc == []
        assert mi == []

    def test_boundary_prefix_comma(self) -> None:
        """Span 'Paris, France' starts with candidate 'Paris' + boundary ','."""
        label, mc, _mi = score_sample("Paris, France", ["Paris"], ["London"])
        assert label == "correct"
        assert mc == ["Paris"]

    def test_boundary_prefix_space(self) -> None:
        """Span 'Paris is great' starts with candidate 'Paris' + boundary ' '."""
        label, mc, _mi = score_sample("Paris is great", ["Paris"], ["London"])
        assert label == "correct"
        assert mc == ["Paris"]

    def test_no_boundary_no_match(self) -> None:
        """'Parisians' starts with 'Paris' but 'i' is not a boundary."""
        label, mc, mi = score_sample("Parisians love art", ["Paris"], ["London"])
        assert label == "ambiguous"
        assert mc == []
        assert mi == []

    def test_case_insensitive(self) -> None:
        label, mc, _mi = score_sample("PARIS", ["paris"], ["london"])
        assert label == "correct"
        assert mc == ["paris"]

    def test_dedup_candidates(self) -> None:
        """Duplicate candidates after normalization don't double-count."""
        label, mc, _mi = score_sample("Paris", ["Paris", "paris", "PARIS"], [])
        assert label == "correct"
        # Only first original kept after dedup
        assert mc == ["Paris"]

    def test_empty_span(self) -> None:
        label, mc, mi = score_sample("", ["Paris"], ["London"])
        assert label == "ambiguous"
        assert mc == []
        assert mi == []

    def test_candidate_longer_than_span(self) -> None:
        """Candidate 'Paris is the capital' does not match span 'Paris'."""
        label, mc, _mi = score_sample("Paris", ["Paris is the capital"], [])
        assert label == "ambiguous"
        assert mc == []

    def test_whitespace_normalization(self) -> None:
        """Leading/trailing whitespace stripped during normalization."""
        label, _mc, _mi = score_sample("  Paris  ", ["  paris  "], [])
        assert label == "correct"

    def test_boundary_prefix_with_space_in_candidate(self) -> None:
        """Span 'Paris Texas' matches candidate 'Paris' at space boundary."""
        label, mc, _mi = score_sample("Paris Texas", ["Paris"], ["London"])
        assert label == "correct"
        assert mc == ["Paris"]


# -------------------------------------------------------------------
# compute_mean_delta_matrix
# -------------------------------------------------------------------


class TestComputeMeanDeltaMatrix:
    """Tests for per-(layer, head) mean delta computation."""

    def test_full_gen(self) -> None:
        records = [
            StepRecord(step_idx=0, layer_idx=0, per_head_delta=[0.1, 0.2]),
            StepRecord(step_idx=1, layer_idx=0, per_head_delta=[0.3, 0.4]),
            StepRecord(step_idx=0, layer_idx=1, per_head_delta=[0.5, 0.6]),
            StepRecord(step_idx=1, layer_idx=1, per_head_delta=[0.7, 0.8]),
        ]
        matrix = compute_mean_delta_matrix(records, num_layers=2, num_heads=2)
        assert matrix is not None
        # Layer 0: mean([0.1, 0.3])=0.2, mean([0.2, 0.4])=0.3
        assert matrix[0] == pytest.approx([0.2, 0.3])
        # Layer 1: mean([0.5, 0.7])=0.6, mean([0.6, 0.8])=0.7
        assert matrix[1] == pytest.approx([0.6, 0.7])

    def test_leading_span(self) -> None:
        records = [
            StepRecord(step_idx=0, layer_idx=0, per_head_delta=[0.1, 0.2]),
            StepRecord(step_idx=1, layer_idx=0, per_head_delta=[0.3, 0.4]),
            StepRecord(step_idx=0, layer_idx=1, per_head_delta=[0.5, 0.6]),
            StepRecord(step_idx=1, layer_idx=1, per_head_delta=[0.7, 0.8]),
        ]
        # Only step 0 included
        matrix = compute_mean_delta_matrix(records, num_layers=2, num_heads=2, max_step=1)
        assert matrix is not None
        assert matrix[0] == pytest.approx([0.1, 0.2])
        assert matrix[1] == pytest.approx([0.5, 0.6])

    def test_empty_records(self) -> None:
        matrix = compute_mean_delta_matrix([], num_layers=2, num_heads=2)
        assert matrix is None

    def test_max_step_zero(self) -> None:
        """max_step=0 filters out all records."""
        records = [
            StepRecord(step_idx=0, layer_idx=0, per_head_delta=[0.1]),
        ]
        matrix = compute_mean_delta_matrix(records, num_layers=1, num_heads=1, max_step=0)
        assert matrix is None

    def test_missing_layer_raises(self) -> None:
        """Missing layer records should raise, not zero-fill."""
        records = [
            StepRecord(step_idx=0, layer_idx=0, per_head_delta=[0.1]),
            # layer 1 has no records
        ]
        with pytest.raises(ValueError, match="Layer 1 has no records"):
            compute_mean_delta_matrix(records, num_layers=2, num_heads=1)

    def test_correct_shape(self) -> None:
        records = [
            StepRecord(step_idx=0, layer_idx=0, per_head_delta=[0.1, 0.2, 0.3]),
            StepRecord(step_idx=0, layer_idx=1, per_head_delta=[0.4, 0.5, 0.6]),
            StepRecord(step_idx=0, layer_idx=2, per_head_delta=[0.7, 0.8, 0.9]),
        ]
        matrix = compute_mean_delta_matrix(records, num_layers=3, num_heads=3)
        assert matrix is not None
        assert len(matrix) == 3
        assert all(len(row) == 3 for row in matrix)


# -------------------------------------------------------------------
# find_leading_span_token_count
# -------------------------------------------------------------------


class MockTokenizer:
    """Mock tokenizer for testing leading-span alignment."""

    def __init__(self, token_strings: dict[int, str]) -> None:
        self._map = token_strings

    def decode(self, token_ids: list[int], **kwargs: object) -> str:
        return "".join(self._map[t] for t in token_ids)


class TestFindLeadingSpanTokenCount:
    """Tests for leading-span token alignment."""

    def test_exact_single_token(self) -> None:
        tok = MockTokenizer({1: "Paris", 2: " is", 3: " great"})
        k, fallback = find_leading_span_token_count([1, 2, 3], "Paris", tok, {})
        assert k == 1
        assert fallback is False

    def test_multi_token_span(self) -> None:
        tok = MockTokenizer({1: "Par", 2: "is"})
        k, fallback = find_leading_span_token_count([1, 2], "Paris", tok, {})
        assert k == 2
        assert fallback is False

    def test_boundary_after_span(self) -> None:
        """Span covered at k=1, word boundary at next char."""
        tok = MockTokenizer({1: "Paris, France", 2: " is", 3: " nice"})
        k, fallback = find_leading_span_token_count([1, 2, 3], "Paris", tok, {})
        assert k == 1
        assert fallback is False

    def test_no_boundary_continues(self) -> None:
        """'Parisians' starts with 'Paris' but no boundary — must continue."""
        tok = MockTokenizer({1: "Parisians", 2: " love"})
        k, fallback = find_leading_span_token_count([1, 2], "Paris", tok, {})
        # Neither decode produces "paris" at a boundary
        assert k == 2
        assert fallback is True

    def test_fallback(self) -> None:
        tok = MockTokenizer({1: "abc", 2: "def"})
        k, fallback = find_leading_span_token_count([1, 2], "xyz", tok, {})
        assert k == 2
        assert fallback is True

    def test_empty_tokens(self) -> None:
        tok = MockTokenizer({})
        k, fallback = find_leading_span_token_count([], "Paris", tok, {})
        assert k == 0
        assert fallback is True

    def test_empty_span(self) -> None:
        tok = MockTokenizer({1: "hello"})
        k, fallback = find_leading_span_token_count([1], "", tok, {})
        assert k == 0
        assert fallback is False

    def test_case_insensitive(self) -> None:
        """Alignment uses lowercase normalization."""
        tok = MockTokenizer({1: "PARIS", 2: " IS"})
        k, fallback = find_leading_span_token_count([1, 2], "paris", tok, {})
        assert k == 1
        assert fallback is False

    def test_decode_kwargs_passed(self) -> None:
        """Verify decode_kwargs are forwarded to tokenizer.decode."""
        received_kwargs: dict[str, object] = {}

        class CapturingTokenizer:
            def decode(self, token_ids: list[int], **kwargs: object) -> str:
                received_kwargs.update(kwargs)
                return "Paris"

        tok = CapturingTokenizer()
        find_leading_span_token_count([1], "Paris", tok, {"skip_special_tokens": True})
        assert received_kwargs["skip_special_tokens"] is True
