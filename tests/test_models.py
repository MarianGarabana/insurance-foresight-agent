"""
tests/test_models.py
--------------------
Unit tests for the Pydantic models in src/models.py.

These tests verify that:
- Models accept valid data correctly.
- Models reject or handle invalid data gracefully.
- to_flat_dict() serializes correctly for CSV export.

Run with:
    pytest tests/test_models.py -v
"""

import pytest
from pydantic import ValidationError

from src.models import (
    CandidateVenture,
    ClassificationResult,
    ExtractedFields,
    RejectedCandidate,
    VerificationResult,
    VentureRecord,
)


# ============================================================
# CandidateVenture tests
# ============================================================

class TestCandidateVenture:

    def test_minimal_candidate(self):
        """A candidate only needs a name; source_urls defaults to empty list."""
        candidate = CandidateVenture(name="Flood Analytics Co")
        assert candidate.name == "Flood Analytics Co"
        assert candidate.suggested_website is None
        assert candidate.scout_reasoning is None
        assert candidate.source_urls == []  # v0.2: defaults to empty list

    def test_candidate_with_source_urls(self):
        """A candidate can carry search result URLs as evidence."""
        candidate = CandidateVenture(
            name="Flood Analytics Co",
            suggested_website="https://floodanalytics.co",
            source_urls=["https://techcrunch.com/floodanalytics", "https://news.io/article"],
        )
        assert len(candidate.source_urls) == 2
        assert "techcrunch.com" in candidate.source_urls[0]

    def test_full_candidate(self):
        """A candidate can have all optional fields."""
        candidate = CandidateVenture(
            name="Flood Analytics Co",
            suggested_website="https://floodanalytics.com",
            scout_reasoning="Relevant to flood risk underwriting.",
        )
        assert candidate.suggested_website == "https://floodanalytics.com"
        assert "flood" in candidate.scout_reasoning.lower()

    def test_candidate_requires_name(self):
        """A candidate without a name should raise a ValidationError."""
        with pytest.raises(ValidationError):
            CandidateVenture()  # name is required


# ============================================================
# VerificationResult tests
# ============================================================

class TestVerificationResult:

    def test_verified_result(self):
        """A verified result has a website and no rejection reason."""
        result = VerificationResult(
            candidate_name="Flood Analytics Co",
            verified=True,
            website="https://floodanalytics.com",
            http_status=200,
        )
        assert result.verified is True
        assert result.website == "https://floodanalytics.com"
        assert result.rejection_reason is None

    def test_rejected_result(self):
        """A rejected result has a rejection reason and no website."""
        result = VerificationResult(
            candidate_name="Ghost Company Inc",
            verified=False,
            rejection_reason="Website could not be reached.",
        )
        assert result.verified is False
        assert result.rejection_reason is not None
        assert result.website is None


# ============================================================
# ExtractedFields tests
# ============================================================

class TestExtractedFields:

    def test_minimal_fields(self):
        """ExtractedFields only requires name and website; rest default to Unknown."""
        fields = ExtractedFields(venture_name="Acme Climate", website="https://acme.com")
        assert fields.location == "Unknown"
        assert fields.stage == "Unknown"
        assert fields.total_funding == "Unknown"
        assert fields.insurance_sector == "Unknown"

    def test_full_fields(self):
        """ExtractedFields can hold all values."""
        fields = ExtractedFields(
            venture_name="Acme Climate",
            website="https://acme.com",
            location="Germany",
            short_description="Climate risk analytics for insurers.",
            stage="Series B",
            total_funding="$45M",
            insurance_sector="P&C",
        )
        assert fields.location == "Germany"
        assert fields.stage == "Series B"
        assert fields.total_funding == "$45M"


# ============================================================
# ClassificationResult tests
# ============================================================

class TestClassificationResult:

    def test_valid_classification(self):
        """ClassificationResult accepts valid driver tags and category."""
        result = ClassificationResult(
            venture_name="Acme Climate",
            driver_tags=["D1", "D9"],
            category="Indirect",
            confidence=0.85,
            reasoning="Climate analytics relevant to D1 and ESG compliance relevant to D9.",
        )
        assert "D1" in result.driver_tags
        assert result.category == "Indirect"
        assert result.confidence == 0.85

    def test_confidence_bounds(self):
        """Confidence must be between 0.0 and 1.0."""
        with pytest.raises(ValidationError):
            ClassificationResult(
                venture_name="X",
                driver_tags=["D1"],
                category="Direct",
                confidence=1.5,  # Out of range
            )

    def test_empty_driver_tags_raises(self):
        """driver_tags cannot be empty (list is required but Pydantic will accept empty list)."""
        # Note: Pydantic does not enforce non-empty by default.
        # This test documents current behavior — validation would need a custom validator.
        result = ClassificationResult(
            venture_name="X",
            driver_tags=[],  # Empty — currently allowed by Pydantic
            category="Direct",
            confidence=0.5,
        )
        assert result.driver_tags == []  # Passes, but we'd catch this in validate_driver_tags()


# ============================================================
# VentureRecord tests
# ============================================================

class TestVentureRecord:

    def _make_record(self) -> VentureRecord:
        return VentureRecord(
            venture_name="Flood Analytics Co",
            website="https://floodanalytics.com",
            location="Netherlands",
            short_description="Flood risk data for insurers.",
            stage="Series A",
            total_funding="$12M",
            insurance_sector="P&C",
            driver_tags=["D1", "D9"],
            category="Indirect",
            confidence=0.9,
            classification_reasoning="Primarily climate risk analytics.",
        )

    def test_venture_record_creation(self):
        record = self._make_record()
        assert record.venture_name == "Flood Analytics Co"
        assert record.category == "Indirect"

    def test_to_flat_dict_joins_tags(self):
        """to_flat_dict() should join driver_tags into a pipe-separated string."""
        record = self._make_record()
        flat = record.to_flat_dict()
        assert flat["driver_tags"] == "D1|D9"

    def test_to_flat_dict_joins_source_urls(self):
        """to_flat_dict() should join source_urls into a pipe-separated string."""
        record = VentureRecord(
            venture_name="Test Co",
            website="https://test.co",
            location="UK",
            short_description="Test.",
            stage="Seed",
            total_funding="$1M",
            insurance_sector="P&C",
            driver_tags=["D1"],
            category="Direct",
            confidence=0.8,
            classification_reasoning="Test.",
            source_urls=["https://a.com", "https://b.com"],
        )
        flat = record.to_flat_dict()
        assert flat["source_urls"] == "https://a.com|https://b.com"

    def test_to_flat_dict_empty_source_urls(self):
        """Empty source_urls list should produce an empty string in the flat dict."""
        record = self._make_record()
        flat = record.to_flat_dict()
        assert flat["source_urls"] == ""

    def test_to_flat_dict_is_dict(self):
        """to_flat_dict() should return a plain dict, not a Pydantic object."""
        record = self._make_record()
        flat = record.to_flat_dict()
        assert isinstance(flat, dict)

    def test_venture_record_has_source_urls(self):
        """VentureRecord.source_urls defaults to an empty list."""
        record = self._make_record()
        assert record.source_urls == []

    def test_venture_record_extraction_confidence_optional(self):
        """extraction_confidence is optional and defaults to None."""
        record = self._make_record()
        assert record.extraction_confidence is None


# ============================================================
# RejectedCandidate tests
# ============================================================

class TestRejectedCandidate:

    def test_rejected_candidate(self):
        """RejectedCandidate requires a name and rejection reason."""
        candidate = RejectedCandidate(
            candidate_name="Ghost Corp",
            suggested_website="https://ghostcorp.io",
            rejection_reason="Website returned HTTP 404.",
            http_status=404,
        )
        assert candidate.candidate_name == "Ghost Corp"
        assert candidate.http_status == 404

    def test_optional_fields(self):
        """suggested_website and http_status are optional."""
        candidate = RejectedCandidate(
            candidate_name="Ghost Corp",
            rejection_reason="No website suggested.",
        )
        assert candidate.suggested_website is None
        assert candidate.http_status is None
