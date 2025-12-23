"""
Configuration management for LLM Deliberate.

Loads environment variables and provides model configurations for OpenRouter.
Supports dynamic model fetching from OpenRouter's public API.
"""
import os
import asyncio
import aiohttp
from pathlib import Path
from dotenv import load_dotenv
from datetime import datetime, timedelta, timezone

# Load .env from project root
ENV_FILE = Path(__file__).parent.parent / ".env"
load_dotenv(ENV_FILE)

# OpenRouter API Configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Timeout and retry configuration
LLM_TIMEOUT = int(os.getenv("LLM_TIMEOUT", "60"))
LLM_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
LLM_DEFAULT_MAX_ROUNDS = int(os.getenv("LLM_DEFAULT_MAX_ROUNDS", "3"))

# Cost warning threshold (USD)
COST_WARNING_THRESHOLD = 5.0

# Cache for dynamically fetched models
_models_cache = None
_models_cache_time = None
MODELS_CACHE_TTL = 3600  # Cache for 1 hour


def _parse_model_from_openrouter(model_data: dict) -> dict:
    """Parse OpenRouter model data into our format."""
    model_id = model_data.get("id", "")
    pricing = model_data.get("pricing", {})

    # Convert pricing strings to floats.
    # OpenRouter pricing fields are expressed in USD per token.
    prompt_price = float(pricing.get("prompt", 0) or 0)
    completion_price = float(pricing.get("completion", 0) or 0)

    return {
        "id": model_id,
        "name": model_data.get("name", model_id),
        "provider": model_data.get("id", "").split("/")[0] if "/" in model_data.get("id", "") else "unknown",
        "pricing": {"prompt": prompt_price, "completion": completion_price},
        "context_window": model_data.get("context_length", 4096),
    }


async def fetch_models_from_openrouter() -> list[dict]:
    """Fetch all available models from OpenRouter API.

    Returns:
        List of model configurations from OpenRouter, or empty list if fetch fails.
    """
    global _models_cache, _models_cache_time

    # Return cached models if still fresh
    if _models_cache is not None and _models_cache_time is not None:
        if datetime.now(timezone.utc) - _models_cache_time < timedelta(seconds=MODELS_CACHE_TTL):
            return _models_cache

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{OPENROUTER_BASE_URL}/models",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("data", [])

                    # Parse and cache the models
                    parsed_models = [_parse_model_from_openrouter(m) for m in models]
                    _models_cache = parsed_models
                    _models_cache_time = datetime.now(timezone.utc)

                    return parsed_models
    except Exception as e:
        print(f"Error fetching models from OpenRouter: {e}")

    return []


def get_models_sync() -> list[dict]:
    """Synchronously get models, using cached data if available.

    This is a blocking wrapper for fetch_models_from_openrouter.
    """
    global _models_cache

    # Return cached models if available
    if _models_cache is not None:
        return _models_cache

    # Try to fetch asynchronously
    try:
        # Create new event loop if one doesn't exist in this thread
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        models = loop.run_until_complete(fetch_models_from_openrouter())
        if models:
            return models
    except Exception as e:
        print(f"Error getting models synchronously: {e}")

    # Fallback to default models if fetch fails
    return list(DEFAULT_MODELS.values())


# Default models as fallback (these are OpenRouter model IDs)
DEFAULT_MODELS = {
    "gpt-4o": {
        "id": "openai/gpt-4o",
        "name": "GPT-4o",
        "provider": "openai",
        # Stored as USD per token to match OpenRouter's pricing object.
        "pricing": {"prompt": 0.0025 / 1000, "completion": 0.01 / 1000},
        "context_window": 128000,
    },
    "gpt-4-turbo": {
        "id": "openai/gpt-4-turbo",
        "name": "GPT-4 Turbo",
        "provider": "openai",
        "pricing": {"prompt": 0.01 / 1000, "completion": 0.03 / 1000},
        "context_window": 128000,
    },
    "claude-3.5-sonnet": {
        "id": "anthropic/claude-3.5-sonnet",
        "name": "Claude 3.5 Sonnet",
        "provider": "anthropic",
        "pricing": {"prompt": 0.003 / 1000, "completion": 0.015 / 1000},
        "context_window": 200000,
    },
    "claude-3-opus": {
        "id": "anthropic/claude-3-opus",
        "name": "Claude 3 Opus",
        "provider": "anthropic",
        "pricing": {"prompt": 0.015 / 1000, "completion": 0.075 / 1000},
        "context_window": 200000,
    },
}


def is_automation_available() -> bool:
    """Check if API automation is available (OPENROUTER_API_KEY configured)."""
    return bool(OPENROUTER_API_KEY)


def get_api_key() -> str:
    """Get OpenRouter API key, raise error if not configured."""
    if not OPENROUTER_API_KEY:
        raise ValueError(
            "OPENROUTER_API_KEY not configured. "
            "Set it in .env file or environment variables to enable automation."
        )
    return OPENROUTER_API_KEY


def get_model_config(model_id: str) -> dict:
    """Get configuration for a model by ID from OpenRouter.

    Args:
        model_id: The model ID (e.g., "openai/gpt-4o")

    Returns:
        Model configuration dict

    Raises:
        ValueError: If model not found
    """
    models = get_models_sync()

    for model in models:
        if model["id"] == model_id:
            return model

    # Check fallback models
    for model in DEFAULT_MODELS.values():
        if model["id"] == model_id:
            return model

    raise ValueError(f"Model {model_id} not found in OpenRouter")


def estimate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost for a model API call.

    Args:
        model_id: The OpenRouter model ID
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Estimated cost in USD
    """
    model_config = get_model_config(model_id)

    # Preferred: normalized OpenRouter-style pricing (USD per token)
    pricing = model_config.get("pricing") or {}
    prompt_price = pricing.get("prompt")
    completion_price = pricing.get("completion")
    if prompt_price is not None or completion_price is not None:
        prompt = float(prompt_price or 0)
        completion = float(completion_price or 0)
        return (input_tokens * prompt) + (output_tokens * completion)

    # Backward compatibility: some callers/configs still provide USD per 1k tokens
    input_per_1k = model_config.get("cost_per_1k_input")
    output_per_1k = model_config.get("cost_per_1k_output")
    if input_per_1k is not None or output_per_1k is not None:
        return (input_tokens / 1000) * float(input_per_1k or 0) + (output_tokens / 1000) * float(output_per_1k or 0)

    return 0.0


def get_available_models() -> list[dict]:
    """Get list of available models from OpenRouter with pricing.

    Returns fresh data from OpenRouter if available, falls back to defaults.
    """
    models = get_models_sync()

    if not models:
        models = list(DEFAULT_MODELS.values())

    normalized = []
    for idx, model in enumerate(models):
        pricing = model.get("pricing") or {}
        prompt_price = float(pricing.get("prompt", 0) or 0)
        completion_price = float(pricing.get("completion", 0) or 0)
        normalized.append(
            {
                "id": model.get("id"),
                "name": model.get("name") or model.get("id"),
                "provider": model.get("provider") or (model.get("id", "").split("/")[0] if "/" in (model.get("id") or "") else "unknown"),
                "available": True,
                "pricing": {"prompt": prompt_price, "completion": completion_price},
                "context_window": model.get("context_window", 4096),
                # Preserve OpenRouter's ordering so the UI can show "top ranked" first.
                "openrouter_order": idx,
            }
        )

    return normalized
