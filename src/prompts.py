"""
prompts.py
----------
All LLM prompts live here — none are hardcoded in the workflow modules.

Why keep prompts in one place?
- Easier to read, tweak, and improve
- Keeps the workflow modules focused on logic, not text
- Easier to version-control prompt changes

Each section has:
- A SYSTEM prompt (tells the model its role and rules)
- A builder function that creates the USER prompt for a specific input

The LLM is always asked to return valid JSON.
We pass response_format={"type": "json_object"} to enforce this.
"""


# ============================================================
# STEP 1 — Candidate Discovery
# ============================================================

# ──────────────────────────────────────────────────────────────────────────────
# Scout — LLM-only mode (v0.1 fallback, used when no search API is configured)
# ──────────────────────────────────────────────────────────────────────────────

SCOUT_SYSTEM_PROMPT = """You are a research assistant helping a foresight research team identify \
real companies and ventures relevant to the insurance industry.

Your task is to generate a list of real, existing companies that are relevant to \
a research query about insurance-related trends.

IMPORTANT RULES:
- Only suggest companies that you are highly confident actually exist.
- Do NOT invent or hallucinate company names.
- For each company, provide your best guess for its official website.
- Prefer companies that are active and relatively well-known within their space.
- Companies can be insurtechs OR adjacent companies (climate tech, cyber, health tech, etc.) \
  that materially affect insurance outcomes.

Respond ONLY with valid JSON in this exact format:
{
  "candidates": [
    {
      "name": "Company Name",
      "suggested_website": "https://example.com",
      "source_urls": [],
      "reasoning": "One sentence explaining why this company is relevant."
    }
  ]
}
"""


def build_scout_prompt(query: str, max_candidates: int) -> str:
    """
    Build the user prompt for the Scout step (LLM-only mode).

    Used when no search API is configured. The LLM generates candidate names
    from its training knowledge only.

    Args:
        query: The researcher's natural language query.
        max_candidates: How many candidates to request.

    Returns:
        A string to send as the user message to the LLM.
    """
    return (
        f"Research query: {query}\n\n"
        f"Please generate up to {max_candidates} real companies that are relevant "
        f"to this query. Follow all the rules in your system instructions.\n\n"
        f"Return your answer as JSON."
    )


# ──────────────────────────────────────────────────────────────────────────────
# Scout — Search-grounded mode (v0.2, used when search results are available)
# ──────────────────────────────────────────────────────────────────────────────

SCOUT_FROM_SEARCH_SYSTEM_PROMPT = """You are a research assistant helping a foresight research \
team identify real companies and ventures relevant to the insurance industry.

You will be given real web search results for a research query. Your job is to:
1. Read the search results carefully.
2. Identify real company or venture names mentioned in the results.
3. Extract their official website (the company's own domain — NOT the search result URL).
4. Record which search result URLs mentioned each company.
5. Explain why each company is relevant to the research query.

IMPORTANT RULES:
- Only extract companies that are clearly mentioned in the search results.
- Do NOT invent companies not mentioned in the results.
- Do NOT use the search result URL as the company website — find the company's own domain.
  Example: if the result URL is "techcrunch.com/2023/flood-sense-raises-12m",
  the company website is "floodsense.io" (or similar), NOT the TechCrunch URL.
- If you cannot determine the official website with confidence, leave it empty ("").
- Companies can be insurtechs OR adjacent companies (climate tech, cyber, health tech, etc.)
  that materially affect insurance outcomes.
- Include the URLs of search results that mentioned each company in "source_urls".

Respond ONLY with valid JSON in this exact format:
{
  "candidates": [
    {
      "name": "Company Name",
      "suggested_website": "https://company-own-domain.com",
      "source_urls": ["https://search-result-url-that-mentioned-it.com"],
      "reasoning": "One sentence explaining why this company is relevant to the query."
    }
  ]
}
"""


def build_scout_from_search_prompt(
    query: str,
    max_candidates: int,
    formatted_search_results: str,
) -> str:
    """
    Build the user prompt for the Scout step (search-grounded mode).

    Used when Tavily search results are available. The LLM acts as a
    parser/extractor of real search results, dramatically reducing hallucination.

    Args:
        query: The researcher's original natural language query.
        max_candidates: How many candidates to extract (maximum).
        formatted_search_results: Pre-formatted string from search.format_results_for_prompt().

    Returns:
        A string to send as the user message to the LLM.
    """
    return (
        f"Research query: {query}\n\n"
        f"Web search results:\n"
        f"{formatted_search_results}\n\n"
        f"Please extract up to {max_candidates} real company names from these search results "
        f"that are relevant to the research query. "
        f"Follow all the rules in your system instructions.\n\n"
        f"Return your answer as JSON."
    )


# ============================================================
# STEP 3 — Field Extraction
# ============================================================

EXTRACTOR_SYSTEM_PROMPT = """You are a structured data extraction assistant for a foresight \
research project about the insurance industry.

You will be given the name and confirmed website of a company. Your job is to extract \
structured information about that company from your knowledge.

IMPORTANT RULES:
- If you do not know the value of a field, use the string "Unknown".
- Do not guess or invent funding amounts or locations — use "Unknown" if unsure.
- Keep descriptions concise (1–2 sentences maximum).
- For insurance_sector, choose the most relevant from:
  P&C, Life, Health, Cyber, Reinsurance, Multi-line, Climate/CAT, Unknown

Respond ONLY with valid JSON in this exact format:
{
  "venture_name": "...",
  "website": "...",
  "location": "...",
  "short_description": "...",
  "stage": "...",
  "total_funding": "...",
  "insurance_sector": "..."
}
"""


def build_extractor_prompt(
    venture_name: str,
    website: str,
    source_urls: list[str] | None = None,
) -> str:
    """
    Build the user prompt for the Extraction step.

    If source_urls are provided (from the Scout's search results), they are
    included as context to help the LLM extract more accurate information.

    Args:
        venture_name: The verified company name.
        website: The confirmed official website URL.
        source_urls: Optional list of URLs from search results that mentioned
                     this company. Provides extra context for the LLM.

    Returns:
        A string to send as the user message to the LLM.
    """
    lines = [
        f"Company name: {venture_name}",
        f"Official website: {website}",
    ]

    if source_urls:
        lines.append(f"Also mentioned at:")
        for url in source_urls[:3]:  # Include up to 3 source URLs
            lines.append(f"  - {url}")

    lines.append(
        "\nPlease extract the structured fields for this company. "
        "Use 'Unknown' for any field you are not confident about.\n"
        "\nReturn your answer as JSON."
    )

    return "\n".join(lines)


# ============================================================
# STEP 4 — Classification
# ============================================================

# The full driver framework definition, included in the system prompt.
# This is long but necessary — the model needs to see the exact definitions.
_DRIVER_DEFINITIONS = """
DRIVER FRAMEWORK (D1–D11):

D1 — Climate Change & Catastrophe Losses
  Relevant: climate analytics, flood risk, wildfire, natural catastrophe modelling,
  weather intelligence, parametric climate products, resilience tech, adaptation technology.

D2 — AI & Digital Transformation
  Relevant: AI underwriting, claims automation, fraud detection, insurance workflow AI,
  digital insurance infrastructure, InsurTech platforms.

D3 — Investment Returns & Financial Markets
  Relevant: insurance capital markets, risk transfer infrastructure, underwriting market
  platforms, investment-linked insurance, ILS (insurance-linked securities).

D4 — Systemic Cyber Risk
  Relevant: cyber insurance, cyber risk analytics, cyber resilience platforms, cyber
  underwriting, cyber protection, incident response.

D5 — Demographic Shifts & Longevity Risk
  Relevant: life insurance innovation, health insurance, longevity modelling, aging
  population solutions, mortality analytics, benefits innovation.

D6 — Geopolitical Fragmentation & Trade Disruption
  Relevant: trade credit insurance, supply chain intelligence, political risk analytics,
  satellite intelligence for geopolitical exposure, sanctions monitoring.

D7 — Social Inflation & Litigation Cost Escalation
  Relevant: claims intelligence, litigation analytics, legal cost prediction, fraud
  detection, casualty claims automation, reserving support.

D8 — Protection Gap & Underinsurance
  Relevant: expanding insurance access, SME insurance, microinsurance, parametric
  insurance for underserved segments, embedded insurance for financial inclusion.

D9 — Regulatory Evolution & ESG Compliance
  Relevant: climate compliance tools, ESG reporting, solvency analytics, regulatory
  technology for insurers, carbon accounting with insurance relevance.

D10 — Distribution Disruption & Embedded Insurance
  Relevant: embedded insurance platforms, API-based insurance distribution, digital
  brokers, affinity insurance platforms, insurance-as-a-service.

D11 — Sovereign & Corporate Debt Crisis
  Relevant: insurance-linked capital solutions, balance-sheet stress transfer, private
  market insurance infrastructure, systemic exposure of insurer investment portfolios.
"""

CLASSIFIER_SYSTEM_PROMPT = f"""You are a classification assistant for a foresight research \
project about the insurance industry.

You will be given information about a company (name, description, website, sector).
Your task is to classify the company into the research framework below.

{_DRIVER_DEFINITIONS}

CATEGORY DEFINITIONS:
- "Direct"   = The company is clearly insurance-specific or an insurtech.
               Examples: an insurtech startup, a cyber insurance platform, a parametric
               insurance provider, a claims automation tool for insurers.
- "Indirect" = The company is NOT primarily in insurance, but its technology, data,
               or service materially affects insurance outcomes.
               Examples: a flood sensor company, a wildfire analytics firm, an ESG
               reporting platform that insurers use, a climate risk data provider.

IMPORTANT RULES:
- Assign at least one driver tag. You may assign multiple if genuinely relevant.
- Do NOT assign a tag just because there is a remote connection — be conservative.
- confidence should reflect how certain you are (0.0 = very uncertain, 1.0 = very certain).
- Keep reasoning short (2–3 sentences).

Respond ONLY with valid JSON in this exact format:
{{
  "venture_name": "...",
  "driver_tags": ["D1", "D9"],
  "category": "Direct",
  "confidence": 0.85,
  "reasoning": "..."
}}
"""


def build_classifier_prompt(
    venture_name: str,
    website: str,
    short_description: str,
    insurance_sector: str,
) -> str:
    """
    Build the user prompt for the Classification step.

    Args:
        venture_name: Company name.
        website: Confirmed website URL.
        short_description: One-sentence description from extraction step.
        insurance_sector: Insurance sector from extraction step.

    Returns:
        A string to send as the user message to the LLM.
    """
    return (
        f"Company: {venture_name}\n"
        f"Website: {website}\n"
        f"Description: {short_description}\n"
        f"Insurance sector: {insurance_sector}\n\n"
        f"Please classify this company using the driver framework (D1–D11) "
        f"and assign a category (Direct or Indirect).\n\n"
        f"Return your answer as JSON."
    )
