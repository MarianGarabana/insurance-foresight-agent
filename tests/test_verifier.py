"""
tests/test_verifier.py
----------------------
Unit tests for src/verifier.py.

Tests the three verification layers:
  1. Domain sanity check (_is directory domain rejected?)
  2. HTTP reachability (mocked)
  3. Content matching (is_content_matching_company)

No real HTTP calls are made — we mock the requests library.

Run with:
    pytest tests/test_verifier.py -v
"""

from unittest.mock import MagicMock, patch

import pytest

from src.models import CandidateVenture
from src.verifier import (
    verify_venture,
    is_content_matching_company,
    _extract_domain,
    _build_url_variants,
    _DIRECTORY_DOMAINS,
)


class TestIsContentMatchingCompany:
    """Tests for the content matching helper."""

    def test_exact_name_match(self):
        """Company name appearing exactly in text should match."""
        assert is_content_matching_company("FloodSense", "FloodSense is a climate analytics startup.")

    def test_case_insensitive(self):
        """Matching should be case-insensitive."""
        assert is_content_matching_company("FloodSense", "floodsense helps insurers model flood risk.")

    def test_no_match_on_unrelated_page(self):
        """Unrelated page content should not match."""
        assert not is_content_matching_company("FloodSense", "Page not found. 404 error. Go back home.")

    def test_multi_word_name_partial_match(self):
        """At least 50% of meaningful words must match."""
        # "Flood Analytics" → meaningful words: ["flood", "analytics"]
        # "flood risk platform" contains "flood" but not "analytics" → 50% match → PASS
        assert is_content_matching_company(
            "Flood Analytics",
            "A flood risk platform for the insurance industry."
        )

    def test_stop_words_are_ignored(self):
        """Stop words like 'Inc', 'Ltd', 'Group' should not count against matching."""
        # "Acme Inc Ltd Group" → meaningful words: ["acme"] (others are stop words)
        # Only need "acme" to appear
        assert is_content_matching_company("Acme Inc Ltd Group", "Acme provides AI tools.")

    def test_empty_text_returns_false(self):
        """Empty page text should not match."""
        assert not is_content_matching_company("FloodSense", "")

    def test_empty_name_returns_true(self):
        """Empty company name (edge case) should return True to not block the pipeline."""
        assert is_content_matching_company("", "Some page text.")

    def test_all_stop_words_name(self):
        """If company name is all stop words, return True (avoid false rejection)."""
        assert is_content_matching_company("The Group Inc Ltd", "Any text here.")


class TestExtractDomain:
    """Tests for the domain extraction helper."""

    def test_extracts_domain_from_https(self):
        assert _extract_domain("https://example.com/page") == "example.com"

    def test_extracts_domain_with_www(self):
        assert _extract_domain("https://www.example.com") == "www.example.com"

    def test_handles_plain_domain(self):
        # clean_url adds https://, so this should work
        result = _extract_domain("example.com")
        assert "example.com" in result

    def test_returns_lowercase(self):
        assert _extract_domain("https://EXAMPLE.COM") == "example.com"


class TestBuildUrlVariants:
    """Tests for URL variant generation."""

    def test_adds_www_variant(self):
        """Non-www URL should produce two variants."""
        variants = _build_url_variants("https://example.com")
        assert "https://example.com" in variants
        assert "https://www.example.com" in variants

    def test_no_duplicate_www(self):
        """URL that already has www should not duplicate."""
        variants = _build_url_variants("https://www.example.com")
        # Should still work and not have double www
        assert not any("www.www." in v for v in variants)

    def test_normalizes_scheme(self):
        """Bare domain should get https:// prepended."""
        variants = _build_url_variants("example.com")
        assert any(v.startswith("https://") for v in variants)


class TestDirectoryDomainRejection:
    """Tests that directory/aggregator domains are correctly rejected."""

    def test_linkedin_rejected(self):
        """LinkedIn URLs should be rejected."""
        candidate = CandidateVenture(
            name="FloodSense",
            suggested_website="https://linkedin.com/company/floodsense",
        )
        result = verify_venture(candidate)
        assert not result.verified
        assert "directory" in result.rejection_reason.lower() or "aggregator" in result.rejection_reason.lower()

    def test_crunchbase_rejected(self):
        """Crunchbase URLs should be rejected."""
        candidate = CandidateVenture(
            name="FloodSense",
            suggested_website="https://crunchbase.com/organization/floodsense",
        )
        result = verify_venture(candidate)
        assert not result.verified

    def test_wikipedia_rejected(self):
        """Wikipedia URLs should be rejected."""
        candidate = CandidateVenture(
            name="FloodSense",
            suggested_website="https://en.wikipedia.org/wiki/FloodSense",
        )
        result = verify_venture(candidate)
        assert not result.verified

    def test_known_directories_are_in_set(self):
        """Our directory domain set should contain the expected domains."""
        assert "linkedin.com" in _DIRECTORY_DOMAINS
        assert "crunchbase.com" in _DIRECTORY_DOMAINS
        assert "wikipedia.org" in _DIRECTORY_DOMAINS
        assert "techcrunch.com" in _DIRECTORY_DOMAINS


class TestVerifyVentureWithMockedHTTP:
    """Tests for the full verify_venture() function with mocked HTTP."""

    def _make_mock_response(self, status_code: int, url: str, text: str = ""):
        """Helper to build a mock requests.Response object."""
        mock_response = MagicMock()
        mock_response.status_code = status_code
        mock_response.url = url
        mock_response.raw = MagicMock()
        mock_response.raw.read.return_value = text.encode("utf-8")
        return mock_response

    def test_verified_when_http_200_and_name_in_content(self):
        """A candidate should be verified when HTTP 200 and name found in page."""
        candidate = CandidateVenture(
            name="FloodSense",
            suggested_website="https://floodsense.io",
        )
        mock_response = self._make_mock_response(
            status_code=200,
            url="https://floodsense.io",
            text="<html><title>FloodSense - Flood Risk Analytics</title></html>",
        )

        with patch("src.verifier.requests.get", return_value=mock_response):
            result = verify_venture(candidate)

        assert result.verified
        assert result.website is not None
        assert result.http_status == 200

    def test_rejected_when_no_website(self):
        """A candidate with no website should be rejected immediately."""
        candidate = CandidateVenture(name="Ghost Corp", suggested_website=None)
        result = verify_venture(candidate)
        assert not result.verified
        assert "no official website" in result.rejection_reason.lower()

    def test_rejected_when_http_fails(self):
        """A candidate should be rejected when HTTP request fails."""
        import requests as req
        candidate = CandidateVenture(
            name="Unreachable Corp",
            suggested_website="https://unreachable-abc123.com",
        )

        with patch("src.verifier.requests.get", side_effect=req.exceptions.ConnectionError):
            result = verify_venture(candidate)

        assert not result.verified

    def test_rejected_when_content_mismatch(self):
        """A candidate should be rejected when page content doesn't mention the company."""
        candidate = CandidateVenture(
            name="FloodSense",
            suggested_website="https://some-wrong-site.com",
        )
        # Page that is reachable but mentions a completely different company
        mock_response = self._make_mock_response(
            status_code=200,
            url="https://some-wrong-site.com",
            text="<html><body>Welcome to CyberShield Corporation.</body></html>",
        )

        with patch("src.verifier.requests.get", return_value=mock_response):
            result = verify_venture(candidate)

        assert not result.verified
        assert "content" in result.rejection_reason.lower()


class TestSaver:
    """Tests for timestamped output filenames."""

    def test_make_run_id_format(self):
        """make_run_id() should return a string in YYYYMMDD_HHMMSS format."""
        import re
        from src.saver import make_run_id

        run_id = make_run_id()
        assert re.match(r"^\d{8}_\d{6}$", run_id), f"Unexpected format: {run_id}"

    def test_make_run_id_is_unique(self):
        """Two successive calls should produce different run IDs (at 1-second resolution)."""
        import time
        from src.saver import make_run_id

        id1 = make_run_id()
        time.sleep(1.1)
        id2 = make_run_id()
        assert id1 != id2, "Two run IDs generated 1 second apart should differ"

    def test_output_filenames_include_run_id(self, tmp_path):
        """Saved files should include the run_id in their names."""
        from src.saver import save_verified_ventures, save_rejected_candidates
        from src.models import VentureRecord, RejectedCandidate

        run_id = "20260101_120000"

        # Temporarily redirect output to tmp_path
        with patch("src.saver.settings") as mock_settings:
            from pathlib import Path
            mock_settings.output_dir = tmp_path

            save_verified_ventures([], run_id=run_id)
            save_rejected_candidates([], run_id=run_id)

        saved_files = [f.name for f in tmp_path.iterdir()]
        assert f"verified_ventures_{run_id}.csv" in saved_files
        assert f"rejected_candidates_{run_id}.csv" in saved_files

    def test_different_runs_dont_overwrite(self, tmp_path):
        """Each run should produce uniquely named files, not overwrite previous runs."""
        from src.saver import save_verified_ventures

        with patch("src.saver.settings") as mock_settings:
            from pathlib import Path
            mock_settings.output_dir = tmp_path

            save_verified_ventures([], run_id="20260101_120000")
            save_verified_ventures([], run_id="20260101_130000")

        csv_files = [f.name for f in tmp_path.glob("verified_ventures_*.csv")]
        assert len(csv_files) == 2, "Two runs should produce two separate files"
