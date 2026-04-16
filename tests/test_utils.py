"""
tests/test_utils.py
-------------------
Unit tests for the helper functions in src/utils.py.

These tests have no external dependencies (no API calls, no files).

Run with:
    pytest tests/test_utils.py -v
"""

import pytest

from src.utils import clean_url, parse_json_response, safe_get, truncate


class TestCleanUrl:

    def test_adds_https_when_missing(self):
        """URLs without a scheme should get https:// prepended."""
        assert clean_url("example.com") == "https://example.com"

    def test_preserves_https(self):
        """URLs already starting with https:// should not be changed."""
        assert clean_url("https://example.com") == "https://example.com"

    def test_preserves_http(self):
        """URLs already starting with http:// should not be changed."""
        assert clean_url("http://example.com") == "http://example.com"

    def test_strips_trailing_slash(self):
        """Trailing slashes should be stripped."""
        assert clean_url("https://example.com/") == "https://example.com"

    def test_strips_multiple_trailing_slashes(self):
        """Multiple trailing slashes should all be stripped."""
        assert clean_url("https://example.com///") == "https://example.com"

    def test_strips_whitespace(self):
        """Leading and trailing whitespace should be stripped."""
        assert clean_url("  https://example.com  ") == "https://example.com"

    def test_preserves_path(self):
        """URL paths (not trailing /) should be preserved."""
        assert clean_url("https://example.com/about") == "https://example.com/about"


class TestParseJsonResponse:

    def test_parses_valid_json(self):
        """Valid JSON string should parse correctly."""
        result = parse_json_response('{"key": "value"}')
        assert result == {"key": "value"}

    def test_parses_json_with_code_fences(self):
        """JSON wrapped in markdown code fences should still parse."""
        raw = "```json\n{\"key\": \"value\"}\n```"
        result = parse_json_response(raw)
        assert result == {"key": "value"}

    def test_parses_json_with_plain_code_fences(self):
        """JSON wrapped in plain ``` fences should parse."""
        raw = "```\n{\"key\": \"value\"}\n```"
        result = parse_json_response(raw)
        assert result == {"key": "value"}

    def test_returns_none_for_invalid_json(self):
        """Malformed JSON should return None without raising an exception."""
        result = parse_json_response("this is not json")
        assert result is None

    def test_returns_none_for_empty_string(self):
        """Empty string should return None."""
        result = parse_json_response("")
        assert result is None

    def test_parses_nested_json(self):
        """Nested JSON objects should parse correctly."""
        raw = '{"outer": {"inner": [1, 2, 3]}}'
        result = parse_json_response(raw)
        assert result["outer"]["inner"] == [1, 2, 3]


class TestSafeGet:

    def test_returns_value_for_existing_key(self):
        """Returns the value when the key exists."""
        assert safe_get({"a": "hello"}, "a") == "hello"

    def test_returns_default_for_missing_key(self):
        """Returns the default when the key is missing."""
        assert safe_get({}, "a") == "Unknown"

    def test_returns_default_for_none_value(self):
        """Returns the default when the value is None."""
        assert safe_get({"a": None}, "a") == "Unknown"

    def test_returns_default_for_empty_string(self):
        """Returns the default when the value is an empty string."""
        assert safe_get({"a": ""}, "a") == "Unknown"

    def test_custom_default(self):
        """Custom default value is returned when key is missing."""
        assert safe_get({}, "missing", default="N/A") == "N/A"

    def test_zero_is_not_replaced(self):
        """A value of 0 (integer zero) is truthy enough — but empty string is not."""
        # Note: 0 is falsy in Python, but we check for None and ""
        # Current implementation: 0 will NOT be treated as empty (it's not None or "")
        # This tests the current behavior as documentation.
        result = safe_get({"count": 0}, "count")
        # 0 is falsy, but our check is `if value is None or value == ""`
        # so 0 should pass through
        assert result == 0


class TestTruncate:

    def test_short_string_unchanged(self):
        """A string shorter than max_length is returned as-is."""
        assert truncate("hello", max_length=100) == "hello"

    def test_exact_length_unchanged(self):
        """A string exactly at max_length is returned as-is."""
        assert truncate("hello", max_length=5) == "hello"

    def test_long_string_truncated(self):
        """A string longer than max_length is truncated with '...'."""
        result = truncate("hello world", max_length=5)
        assert result == "hello..."
        assert len(result) == 8  # 5 chars + "..."

    def test_default_max_length(self):
        """Default max_length is 200."""
        long_string = "x" * 300
        result = truncate(long_string)
        assert len(result) == 203  # 200 + "..."
        assert result.endswith("...")
