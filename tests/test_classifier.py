"""
tests/test_classifier.py
------------------------
Unit tests for the validation and helper functions in src/classifier.py.

These tests do NOT call the OpenAI API — they test only the local
validation logic (_validate_driver_tags, _validate_category, etc.).

Run with:
    pytest tests/test_classifier.py -v
"""

import pytest

from src.classifier import _validate_driver_tags, _validate_category, _fallback_classification


class TestValidateDriverTags:

    def test_valid_single_tag(self):
        """A single valid tag is returned unchanged."""
        result = _validate_driver_tags(["D1"], venture_name="Test Co")
        assert result == ["D1"]

    def test_valid_multiple_tags(self):
        """Multiple valid tags are all returned."""
        result = _validate_driver_tags(["D1", "D4", "D9"], venture_name="Test Co")
        assert result == ["D1", "D4", "D9"]

    def test_lowercase_tags_normalized(self):
        """Tags like 'd1' should be normalized to uppercase 'D1'."""
        result = _validate_driver_tags(["d1", "d9"], venture_name="Test Co")
        assert result == ["D1", "D9"]

    def test_invalid_tags_are_dropped(self):
        """Tags not in D1–D11 are silently dropped."""
        result = _validate_driver_tags(["D1", "D99", "NotATag"], venture_name="Test Co")
        assert result == ["D1"]

    def test_all_invalid_returns_fallback(self):
        """If all tags are invalid, return the fallback ['D1']."""
        result = _validate_driver_tags(["D99", "X"], venture_name="Test Co")
        assert result == ["D1"]

    def test_empty_list_returns_fallback(self):
        """An empty list should return the fallback ['D1']."""
        result = _validate_driver_tags([], venture_name="Test Co")
        assert result == ["D1"]

    def test_non_list_input_returns_fallback(self):
        """If the input is not a list, return the fallback ['D1']."""
        result = _validate_driver_tags("D1", venture_name="Test Co")  # type: ignore
        assert result == ["D1"]

    def test_all_valid_driver_tags(self):
        """All 11 driver tags should be accepted."""
        all_tags = ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11"]
        result = _validate_driver_tags(all_tags, venture_name="Test Co")
        assert result == all_tags


class TestValidateCategory:

    def test_direct(self):
        """'Direct' is valid."""
        result = _validate_category("Direct", venture_name="Test Co")
        assert result == "Direct"

    def test_indirect(self):
        """'Indirect' is valid."""
        result = _validate_category("Indirect", venture_name="Test Co")
        assert result == "Indirect"

    def test_case_insensitive(self):
        """Category matching should be case-insensitive (we normalize with capitalize())."""
        result = _validate_category("direct", venture_name="Test Co")
        assert result == "Direct"

        result = _validate_category("INDIRECT", venture_name="Test Co")
        assert result == "Indirect"

    def test_invalid_category_defaults_to_indirect(self):
        """Any invalid category string should fall back to 'Indirect'."""
        result = _validate_category("Maybe", venture_name="Test Co")
        assert result == "Indirect"

    def test_empty_string_defaults_to_indirect(self):
        """Empty string should fall back to 'Indirect'."""
        result = _validate_category("", venture_name="Test Co")
        assert result == "Indirect"

    def test_none_defaults_to_indirect(self):
        """None input should fall back to 'Indirect'."""
        result = _validate_category(None, venture_name="Test Co")  # type: ignore
        assert result == "Indirect"


class TestFallbackClassification:

    def test_fallback_returns_classification_result(self):
        """_fallback_classification returns a valid ClassificationResult."""
        from src.models import ClassificationResult
        result = _fallback_classification("Test Co")
        assert isinstance(result, ClassificationResult)

    def test_fallback_has_zero_confidence(self):
        """Fallback classification should have 0.0 confidence."""
        result = _fallback_classification("Test Co")
        assert result.confidence == 0.0

    def test_fallback_has_d1_tag(self):
        """Fallback classification defaults to D1 tag."""
        result = _fallback_classification("Test Co")
        assert "D1" in result.driver_tags

    def test_fallback_is_indirect(self):
        """Fallback classification defaults to Indirect category."""
        result = _fallback_classification("Test Co")
        assert result.category == "Indirect"

    def test_fallback_includes_venture_name(self):
        """Fallback classification should preserve the venture name."""
        result = _fallback_classification("MyVenture Corp")
        assert result.venture_name == "MyVenture Corp"
