"""
models.py
---------
Pydantic data models for the Venture Scout workflow.

These models define the shape of data at each stage of the pipeline.
Pydantic validates the data automatically — if a required field is missing
or has the wrong type, you get a clear error message immediately.

Pipeline stages and their models:
  1. Scout        → CandidateVenture
  2. Verifier     → VerificationResult
  3. Extractor    → ExtractedFields
  4. Classifier   → ClassificationResult
  5. Saver        → VentureRecord (verified), RejectedCandidate (rejected)
"""

from typing import Optional
from pydantic import BaseModel, Field


# ============================================================
# Stage 1 — Scout output
# ============================================================

class CandidateVenture(BaseModel):
    """
    A raw candidate venture name produced by the Scout.

    At this point we only have a name and maybe a suggested website.
    Nothing has been verified yet.

    v0.2 addition: source_urls tracks which search result URLs mentioned this
    company. This provides evidence that the company is real and grounded in
    actual web results, not just LLM memory.
    """

    name: str = Field(description="Company or venture name")
    suggested_website: Optional[str] = Field(
        default=None,
        description=(
            "Website URL the LLM suggested for this company. "
            "This is a GUESS — it must still be verified."
        ),
    )
    scout_reasoning: Optional[str] = Field(
        default=None,
        description="Why the LLM thought this company was relevant to the query.",
    )
    source_urls: list[str] = Field(
        default_factory=list,
        description=(
            "URLs from web search results that mentioned or described this company. "
            "Empty list if the Scout ran in LLM-only mode (no search API configured)."
        ),
    )


# ============================================================
# Stage 2 — Verifier output
# ============================================================

class VerificationResult(BaseModel):
    """
    Result of attempting to verify a candidate venture.

    If verified=True, the website field contains a confirmed URL.
    If verified=False, the rejection_reason explains why.
    """

    candidate_name: str
    verified: bool = Field(description="True if the venture was confirmed to exist.")
    website: Optional[str] = Field(
        default=None,
        description="Confirmed official website URL. Only set if verified=True.",
    )
    http_status: Optional[int] = Field(
        default=None,
        description="HTTP status code returned when we tried to reach the website.",
    )
    rejection_reason: Optional[str] = Field(
        default=None,
        description="Why the venture was rejected. Only set if verified=False.",
    )


# ============================================================
# Stage 3 — Extractor output
# ============================================================

class ExtractedFields(BaseModel):
    """
    Structured fields extracted for a verified venture.

    These map directly to the columns in the final output CSV.
    Fields that cannot be determined are left as "Unknown".
    """

    venture_name: str
    website: str
    location: str = Field(
        default="Unknown",
        description="Country or city where the venture is headquartered.",
    )
    short_description: str = Field(
        default="",
        description="One or two sentence description of what the venture does.",
    )
    stage: str = Field(
        default="Unknown",
        description=(
            "Funding stage: Seed, Series A, Series B, Series C, Growth, "
            "Public, Unknown, etc."
        ),
    )
    total_funding: str = Field(
        default="Unknown",
        description="Total funding raised, e.g. '$12M', '$450M', 'Unknown'.",
    )
    insurance_sector: str = Field(
        default="Unknown",
        description=(
            "Which insurance sector this venture is most relevant to, "
            "e.g. P&C, Life, Health, Cyber, Reinsurance, Multi-line, etc."
        ),
    )


# ============================================================
# Stage 5 — Impact Analysis output
# ============================================================

# Valid impact area codes
VALID_IMPACT_AREAS = ("I1", "I2", "I3", "I4", "I5")

# Valid impact directions
VALID_IMPACT_DIRECTIONS = ("positive", "negative", "mixed")

# Valid time horizons
VALID_TIME_HORIZONS = ("short", "medium", "long")


class ImpactResult(BaseModel):
    """
    Insurance impact analysis for a verified venture.

    Maps the venture's effect onto the five insurance outcome areas (I1–I5):
      I1 — Premium Revenue
      I2 — Investment Revenue
      I3 — Claims Costs
      I4 — Operational Costs
      I5 — Capital & Reserves

    Added in v0.3.
    """

    drivers: list[str] = Field(
        description="D1–D11 driver tags that are the root cause of the impact."
    )
    impact_areas: list[str] = Field(
        description="Which insurance outcome areas are affected (e.g. ['I3', 'I5'])."
    )
    impact_direction: str = Field(
        description="'positive' = reduces risk/cost, 'negative' = increases risk/cost, 'mixed'."
    )
    mechanism: str = Field(
        description="Short technical explanation of how the venture produces the impact."
    )
    insurance_effect: str = Field(
        description="Business-level explanation of what this means for insurers."
    )
    time_horizon: str = Field(
        description="'short' (<2 years), 'medium' (2–5 years), or 'long' (>5 years)."
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Analyst confidence in this impact assessment (0.0–1.0).",
    )


# ============================================================
# Stage 4 — Classifier output
# ============================================================

# Valid driver tags. We define them as a tuple for easy iteration and validation.
VALID_DRIVERS = (
    "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11"
)

# Valid category values
VALID_CATEGORIES = ("Direct", "Indirect")


class ClassificationResult(BaseModel):
    """
    Classification of a verified venture into the research framework.

    A venture can have multiple driver tags (e.g. ["D1", "D9"]).
    The category is either "Direct" (clearly insurtech) or "Indirect"
    (affects insurance but is not primarily an insurance company).
    """

    venture_name: str
    driver_tags: list[str] = Field(
        description=(
            "List of D1–D11 tags that apply to this venture. "
            "At least one tag is required."
        )
    )
    category: str = Field(
        description="'Direct' if clearly insurtech, 'Indirect' if adjacent."
    )
    confidence: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description=(
            "Classifier's confidence in this classification, 0.0–1.0. "
            "1.0 = very confident, 0.5 = uncertain."
        ),
    )
    reasoning: str = Field(
        default="",
        description=(
            "Short explanation of why these driver tags and category were assigned. "
            "Helps the researcher audit the classification."
        ),
    )


# ============================================================
# Final output models
# ============================================================

class VentureRecord(BaseModel):
    """
    A fully verified, extracted, and classified venture.

    This is what gets written to output/verified_ventures.csv and .json.

    v0.2 additions:
    - source_urls: evidence trail showing where this company was found
    - extraction_confidence: how confident the LLM was about the extracted fields
    """

    # From extraction
    venture_name: str
    website: str
    location: str
    short_description: str
    stage: str
    total_funding: str
    insurance_sector: str

    # From classification
    driver_tags: list[str]
    category: str
    confidence: float
    classification_reasoning: str

    # Evidence / provenance (v0.2)
    source_urls: list[str] = Field(
        default_factory=list,
        description=(
            "URLs from search results or other sources that provided evidence for this venture. "
            "Allows researchers to trace exactly where each entry came from."
        ),
    )
    extraction_confidence: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description=(
            "How confident the extractor was in the extracted fields (0.0–1.0). "
            "None if confidence was not reported."
        ),
    )

    # Impact analysis (v0.3)
    impact: Optional[ImpactResult] = Field(
        default=None,
        description="Insurance impact analysis produced in Step 5. None if not yet analysed.",
    )

    def to_flat_dict(self) -> dict:
        """
        Convert to a flat dictionary suitable for a CSV row.

        Lists are joined into pipe-separated strings for CSV compatibility:
        - driver_tags: ["D1", "D9"]  → "D1|D9"
        - source_urls: [...]          → pipe-separated URLs

        Impact fields are flattened from the nested ImpactResult:
        - impact_areas: ["I3", "I5"] → "I3|I5"
        - impact_direction, mechanism, time_horizon, impact_confidence are scalars.
        """
        d = self.model_dump()
        d["driver_tags"] = "|".join(self.driver_tags)
        d["source_urls"] = "|".join(self.source_urls)

        # Remove nested impact dict; replace with flat columns
        d.pop("impact", None)
        if self.impact:
            d["impact_areas"] = "|".join(self.impact.impact_areas)
            d["impact_direction"] = self.impact.impact_direction
            d["mechanism"] = self.impact.mechanism
            d["time_horizon"] = self.impact.time_horizon
            d["impact_confidence"] = self.impact.confidence
        else:
            d["impact_areas"] = ""
            d["impact_direction"] = ""
            d["mechanism"] = ""
            d["time_horizon"] = ""
            d["impact_confidence"] = None

        return d


class RejectedCandidate(BaseModel):
    """
    A candidate venture that was rejected during verification.

    This is what gets written to output/rejected_candidates.csv.
    Keeping rejected candidates is important — it lets researchers audit
    what the system tried and why it was discarded.
    """

    candidate_name: str
    suggested_website: Optional[str] = None
    rejection_reason: str
    http_status: Optional[int] = None
