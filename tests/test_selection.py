"""Tests for cohort selection (unanimous-only filtering).

Proves: correct filtering, diagnostics, edge cases.
"""

from __future__ import annotations

from navi_sad.analysis.selection import select_unanimous
from navi_sad.analysis.types import SelectionDiagnostics, SubsetSpec


class TestSelectUnanimous:
    def test_all_unanimous(self) -> None:
        """All reviewers agree on every sample."""
        votes = {
            0: ["correct", "correct", "correct"],
            1: ["incorrect", "incorrect", "incorrect"],
        }
        majority_labels = {0: "correct", 1: "incorrect"}
        spec, diag = select_unanimous(votes, majority_labels)
        assert isinstance(spec, SubsetSpec)
        assert isinstance(diag, SelectionDiagnostics)
        assert spec.n_correct == 1
        assert spec.n_incorrect == 1
        assert spec.included_indices == frozenset({0, 1})

    def test_non_unanimous_excluded(self) -> None:
        """Non-unanimous samples are excluded."""
        votes = {
            0: ["correct", "correct", "correct"],
            1: ["incorrect", "incorrect", "correct"],  # non-unanimous
            2: ["incorrect", "incorrect", "incorrect"],
        }
        majority_labels = {0: "correct", 1: "incorrect", 2: "incorrect"}
        spec, diag = select_unanimous(votes, majority_labels)
        assert 1 not in spec.included_indices
        assert spec.n_correct == 1
        assert spec.n_incorrect == 1
        assert diag.n_excluded_non_unanimous == 1

    def test_ambiguous_excluded(self) -> None:
        """Ambiguous majority labels excluded."""
        votes = {
            0: ["correct", "correct", "correct"],
            1: ["ambiguous", "ambiguous", "ambiguous"],
        }
        majority_labels = {0: "correct", 1: "ambiguous"}
        spec, diag = select_unanimous(votes, majority_labels)
        assert 1 not in spec.included_indices
        assert diag.n_excluded_ambiguous == 1

    def test_diagnostics_counts(self) -> None:
        votes = {
            0: ["correct", "correct", "correct"],
            1: ["correct", "correct", "incorrect"],  # non-unanimous
            2: ["incorrect", "incorrect", "incorrect"],
            3: ["ambiguous", "ambiguous", "ambiguous"],
        }
        majority_labels = {0: "correct", 1: "correct", 2: "incorrect", 3: "ambiguous"}
        _spec, diag = select_unanimous(votes, majority_labels)
        assert diag.n_correct_before == 2
        assert diag.n_incorrect_before == 1
        assert diag.n_correct_after == 1
        assert diag.n_incorrect_after == 1
        assert diag.n_excluded_ambiguous == 1
        assert diag.n_excluded_non_unanimous == 1

    def test_provenance_name(self) -> None:
        votes = {0: ["correct", "correct", "correct"]}
        majority_labels = {0: "correct"}
        spec, _diag = select_unanimous(votes, majority_labels)
        assert spec.provenance_name == "unanimous_only"

    def test_all_non_unanimous_returns_empty(self) -> None:
        """No unanimous samples -> empty subset."""
        votes = {
            0: ["correct", "incorrect", "correct"],
            1: ["incorrect", "correct", "incorrect"],
        }
        majority_labels = {0: "correct", 1: "incorrect"}
        spec, _diag = select_unanimous(votes, majority_labels)
        assert spec.n_correct == 0
        assert spec.n_incorrect == 0
        assert len(spec.included_indices) == 0

    def test_selection_name_in_diagnostics(self) -> None:
        votes = {0: ["correct", "correct", "correct"]}
        majority_labels = {0: "correct"}
        _spec, diag = select_unanimous(votes, majority_labels)
        assert diag.selection_name == "unanimous_only"
