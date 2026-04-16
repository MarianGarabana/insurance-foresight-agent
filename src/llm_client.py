"""
llm_client.py
-------------
Shared OpenAI API client for all LLM calls in the pipeline.

Previously, every module (scout.py, extractor.py, classifier.py) had its own
copy of a _call_llm() function. That was repetitive and hard to maintain.

This module provides a single `call_llm()` function that all modules import.

Benefits:
- One place to change the model, temperature defaults, retry logic
- One place to handle all OpenAI error types
- Easy to swap to a different LLM provider in the future

Usage:
    from src.llm_client import call_llm

    response = call_llm(
        system_prompt="You are ...",
        user_prompt="Classify this company ...",
        context="classifier",
        temperature=0.1,
    )
"""

import time

import openai

from src.config import settings
from src.logger import get_logger

log = get_logger(__name__)

# Default number of times to retry a failed LLM call before giving up.
# This handles transient network errors and temporary rate limits.
_DEFAULT_MAX_RETRIES = 2

# How long to wait between retries (in seconds).
# Simple fixed delay — not exponential backoff, but good enough for a research tool.
_RETRY_DELAY_SECONDS = 3


def call_llm(
    system_prompt: str,
    user_prompt: str,
    context: str = "",
    temperature: float = 0.1,
    max_retries: int = _DEFAULT_MAX_RETRIES,
) -> str | None:
    """
    Call the OpenAI chat completion API and return the raw response text.

    All LLM calls in the pipeline go through this function.
    It handles errors, retries, and logging in one place.

    The model is forced to return JSON (response_format=json_object).
    Every prompt in prompts.py already asks for JSON, so this enforces it.

    Args:
        system_prompt: The system message — role definition and rules for the model.
        user_prompt: The user message — the specific task for this call.
        context: A short label for log messages (e.g. "scout", "classifier").
                 Helps you trace which module made which call in the log file.
        temperature: Controls randomness. Lower = more deterministic.
                     Use ~0.2 for creative tasks (scout), ~0.1 for factual tasks.
        max_retries: How many times to retry on transient failures (rate limits,
                     network errors). Does NOT retry on auth errors.

    Returns:
        The raw text content of the LLM response, or None if all attempts failed.
    """
    client = openai.OpenAI(api_key=settings.openai_api_key)

    for attempt in range(1, max_retries + 2):  # +2 because range is exclusive
        try:
            response = client.chat.completions.create(
                model=settings.openai_model,
                response_format={"type": "json_object"},  # Always return JSON
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
            )
            content = response.choices[0].message.content
            log.debug(
                f"[{context}] LLM call succeeded (attempt {attempt}). "
                f"Response preview: {content[:150] if content else 'empty'}"
            )
            return content

        except openai.AuthenticationError:
            # This will never succeed on retry — bad API key. Stop immediately.
            log.error(
                "OpenAI authentication failed. "
                "Check that OPENAI_API_KEY in your .env file is correct."
            )
            return None

        except openai.RateLimitError:
            if attempt <= max_retries:
                log.warning(
                    f"[{context}] OpenAI rate limit hit (attempt {attempt}/{max_retries + 1}). "
                    f"Waiting {_RETRY_DELAY_SECONDS}s before retry..."
                )
                time.sleep(_RETRY_DELAY_SECONDS)
            else:
                log.error(
                    f"[{context}] OpenAI rate limit hit on all {max_retries + 1} attempts. Giving up."
                )
                return None

        except openai.APIConnectionError:
            if attempt <= max_retries:
                log.warning(
                    f"[{context}] OpenAI connection error (attempt {attempt}). Retrying..."
                )
                time.sleep(_RETRY_DELAY_SECONDS)
            else:
                log.error(f"[{context}] OpenAI connection failed on all attempts.")
                return None

        except openai.APIError as e:
            # Catch-all for other OpenAI errors (server errors, etc.)
            log.error(f"[{context}] OpenAI API error: {e}")
            return None

        except Exception as e:
            # Unexpected errors — don't retry, just log and return None.
            log.error(f"[{context}] Unexpected error calling LLM: {e}")
            return None

    # Should not reach here, but just in case
    return None
