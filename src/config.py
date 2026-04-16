"""
config.py
---------
Loads all configuration from environment variables.

We use python-dotenv to automatically read a .env file in the project root.
All settings live here — no scattered os.environ calls throughout the code.

Usage:
    from src.config import settings
    print(settings.openai_model)
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file: src/ -> project/)
_project_root = Path(__file__).parent.parent
load_dotenv(_project_root / ".env")


class Settings:
    """
    Central settings object.

    All values come from environment variables. Defaults are provided
    for optional settings so the project works with minimal configuration.

    Note: OPENAI_API_KEY is read lazily — a missing key will only raise an
    error when the first API call is actually made. This allows tests that
    do not call the LLM to import and use this module without an API key set.
    """

    def __init__(self) -> None:
        # ------------------------------------------------------------------ #
        # Required settings (stored as-is; validated lazily)
        # ------------------------------------------------------------------ #
        # We do NOT raise here so tests can import without an API key.
        # The error will be raised by OpenAI SDK when an actual call is made.
        self.openai_api_key: str = os.getenv("OPENAI_API_KEY", "")

        # Mock mode (offline development)
        self.use_mock_llm: bool = os.getenv("USE_MOCK_LLM", "false").lower() == "true"

        # ------------------------------------------------------------------ #
        # Optional settings (with sensible defaults)
        # ------------------------------------------------------------------ #
        self.openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")

        # Where to write output files (relative to project root)
        self.output_dir: Path = Path(os.getenv("OUTPUT_DIR", "output"))

        # Where to write log files
        self.logs_dir: Path = Path("logs")

        # How many candidate ventures to generate per query
        self.max_candidates: int = int(os.getenv("MAX_CANDIDATES", "10"))

        # Timeout (in seconds) for HTTP requests when verifying websites
        self.request_timeout: int = 10

        # Optional: Tavily API key for web search (grounded candidate discovery)
        # Get one free at https://tavily.com
        # If not set, the Scout falls back to LLM-only mode.
        self.tavily_api_key: str = os.getenv("TAVILY_API_KEY", "")

        # Derived: True if at least one search API is configured.
        # Modules can check this to decide between search-grounded vs LLM-only mode.
        self.search_enabled: bool = bool(self.tavily_api_key)

    def validate_api_key(self) -> None:
        """
        Skip validation if mock mode is enabled.
        """
        if self.use_mock_llm:
            return

        if not self.openai_api_key:
            raise EnvironmentError(
                "Required environment variable 'OPENAI_API_KEY' is not set.\n"
                "Copy .env.example to .env and fill in your API key."
            )

    def ensure_dirs(self) -> None:
        """Create output and logs directories if they do not already exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


# A single shared instance — import this everywhere instead of creating new ones.
settings = Settings()
