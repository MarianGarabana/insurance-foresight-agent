"""
tests/test_impact.py
--------------------
Tests for the v0.3 impact analysis layer.

Covers:
- ImpactResult model creation and validation
- VentureRecord.impact field integration
- VentureRecord.to_flat_dict() includes impact columns
- impact.py validation helpers

No LLM calls are made — all tests are purely local.

Run with:
    pytest tests/test_impact.py -v
"""

import pytest

from src.models import ImpactResult, VentureRecord
from src.impact import _validate_impact_areas, _validate_impact_direction, _validate_time_horizon


# ─────────────────────────────────────────────────────────────────────────────
# ImpactResult model
# ─────────────────────────────────────────────────────────────────────────────

class TestImpactResultModel:

    def test_create_valid_impact_result(self):
        """ImpactResult can be constructed with valid fields."""
        result = ImpactResult(
            drivers=["D1"],
            impact_areas=["I3"],
            impact_direction="positive",
            mechanism="Improves flood prediction reducing property damage.",
            insurance_effect="Reduces claims severity in property insurance.",
            time_horizon="short",
            confidence=0.8,
        )
        assert result.impact_areas == ["I3"]
        assert result.impact_direction == "positive"
        assert result.confidence == 0.8

    def test_confidence_clamped_to_range(self):
        """Pydantic rejects confidence values outside 0.0–1.0."""
        with pytest.raises(Exception):
            ImpactResult(
                drivers=["D1"],
                impact_areas=["I3"],
                impact_direction="positive",
                mechanism="x",
                insurance_effect="x",
                time_horizon="short",
                confidence=1.5,  # out of range
            )

    def test_multiple_impact_areas(self):
        """ImpactResult accepts multiple impact area codes."""
        result = ImpactResult(
            drivers=["D1", "D9"],
            impact_areas=["I3", "I5"],
            impact_direction="mixed",
            mechanism="Affects both claims and capital reserves.",
            insurance_effect="Dual effect on loss ratios and solvency capital.",
            time_horizon="medium",
            confidence=0.65,
        )
        assert "I3" in result.impact_areas
        assert "I5" in result.impact_areas

    def test_default_confidence_is_zero(self):
        """Confidence defaults to 0.0 if not provided."""
        result = ImpactResult(
            drivers=["D2"],
            impact_areas=["I4"],
            impact_direction="positive",
            mechanism="Automates underwriting workflows.",
            insurance_effect="Reduces operational costs.",
            time_horizon="short",
        )
        assert result.confidence == 0.0


# ─────────────────────────────────────────────────────────────────────────────
# VentureRecord — impact field integration
# ─────────────────────────────────────────────────────────────────────────────

def _make_venture(impact=None) -> VentureRecord:
    """Helper: build a minimal VentureRecord for testing."""
    return VentureRecord(
        venture_name="FloodSense",
        website="https://floodsense.io",
        location="Netherlands",
        short_description="Real-time flood prediction for insurers.",
        stage="Series A",
        total_funding="$12M",
        insurance_sector="P&C",
        driver_tags=["D1"],
        category="Indirect",
        confidence=0.85,
        classification_reasoning="Flood analytics drives D1.",
        impact=impact,
    )


class TestVentureRecordImpact:

    def test_impact_defaults_to_none(self):
        """VentureRecord.impact is None by default."""
        venture = _make_venture()
        assert venture.impact is None

    def test_impact_can_be_set(self):
        """VentureRecord.impact accepts a valid ImpactResult."""
        impact = ImpactResult(
            drivers=["D1"],
            impact_areas=["I3"],
            impact_direction="positive",
            mechanism="Flood prediction reduces damage.",
            insurance_effect="Reduces property claims severity.",
            time_horizon="short",
            confidence=0.8,
        )
        venture = _make_venture(impact=impact)
        assert venture.impact is not None
        assert venture.impact.impact_areas == ["I3"]

    def test_to_flat_dict_with_impact(self):
        """to_flat_dict() includes all impact columns when impact is set."""
        impact = ImpactResult(
            drivers=["D1"],
            impact_areas=["I3", "I5"],
            impact_direction="positive",
            mechanism="Reduces flood damage via early warning.",
            insurance_effect="Lower claims and improved solvency buffer.",
            time_horizon="medium",
            confidence=0.75,
        )
        venture = _make_venture(impact=impact)
        flat = venture.to_flat_dict()

        assert flat["impact_areas"] == "I3|I5"
        assert flat["impact_direction"] == "positive"
        assert flat["mechanism"] == "Reduces flood damage via early warning."
        assert flat["time_horizon"] == "medium"
        assert flat["impact_confidence"] == 0.75

    def test_to_flat_dict_without_impact(self):
        """to_flat_dict() fills impact columns with empty/None when impact is None."""
        venture = _make_venture(impact=None)
        flat = venture.to_flat_dict()

        assert flat["impact_areas"] == ""
        assert flat["impact_direction"] == ""
        assert flat["mechanism"] == ""
        assert flat["time_horizon"] == ""
        assert flat["impact_confidence"] is None

    def test_to_flat_dict_no_nested_impact_key(self):
        """to_flat_dict() must not contain a nested 'impact' dict key."""
        venture = _make_venture()
        flat = venture.to_flat_dict()
        assert "impact" not in flat


# ─────────────────────────────────────────────────────────────────────────────
# impact.py — validation helpers
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateImpactAreas:

    def test_valid_single_area(self):
        assert _validate_impact_areas(["I3"], "Test Co") == ["I3"]

    def test_valid_multiple_areas(self):
        assert _validate_impact_areas(["I1", "I3", "I5"], "Test Co") == ["I1", "I3", "I5"]

    def test_lowercase_normalised(self):
        assert _validate_impact_areas(["i3"], "Test Co") == ["I3"]

    def test_invalid_codes_dropped(self):
        result = _validate_impact_areas(["I3", "I9", "X"], "Test Co")
        assert result == ["I3"]

    def test_empty_list_defaults_to_i3(self):
        assert _validate_impact_areas([], "Test Co") == ["I3"]

    def test_non_list_defaults_to_i3(self):
        assert _validate_impact_areas("I3", "Test Co") == ["I3"]  # type: ignore

    def test_all_valid_areas_accepted(self):
        all_areas = ["I1", "I2", "I3", "I4", "I5"]
        assert _validate_impact_areas(all_areas, "Test Co") == all_areas


class TestValidateImpactDirection:

    def test_positive(self):
        assert _validate_impact_direction("positive", "Test Co") == "positive"

    def test_negative(self):
        assert _validate_impact_direction("negative", "Test Co") == "negative"

    def test_mixed(self):
        assert _validate_impact_direction("mixed", "Test Co") == "mixed"

    def test_invalid_defaults_to_mixed(self):
        assert _validate_impact_direction("unknown", "Test Co") == "mixed"

    def test_empty_string_defaults_to_mixed(self):
        assert _validate_impact_direction("", "Test Co") == "mixed"


class TestValidateTimeHorizon:

    def test_short(self):
        assert _validate_time_horizon("short", "Test Co") == "short"

    def test_medium(self):
        assert _validate_time_horizon("medium", "Test Co") == "medium"

    def test_long(self):
        assert _validate_time_horizon("long", "Test Co") == "long"

    def test_invalid_defaults_to_medium(self):
        assert _validate_time_horizon("immediate", "Test Co") == "medium"

    def test_empty_string_defaults_to_medium(self):
        assert _validate_time_horizon("", "Test Co") == "medium"
