"""
LLM API client for OpenRouter integration.

Handles API calls with retry logic, token counting, and cost tracking.
"""
import httpx
import time
from typing import Any, Optional
from pydantic import BaseModel
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    RetryError,
)
from . import config


def _extract_text_from_dict(value: Any) -> str | None:
    if not isinstance(value, dict):
        return None
    text = value.get("text")
    if isinstance(text, str):
        return text
    content = value.get("content")
    if isinstance(content, str):
        return content
    return None


def _extract_text_from_part(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    return _extract_text_from_dict(value)


def extract_text_content(raw_content: Any) -> str:
    """Normalize OpenAI/OpenRouter message content into plain text."""
    if raw_content is None:
        return ""

    if isinstance(raw_content, str):
        return raw_content

    dict_text = _extract_text_from_dict(raw_content)
    if dict_text is not None:
        return dict_text

    if isinstance(raw_content, list):
        parts: list[str] = []
        for part in raw_content:
            part_text = _extract_text_from_part(part)
            if part_text:
                parts.append(part_text)
        return "".join(parts)

    return str(raw_content)


def extract_openrouter_choice0(response_json: dict[str, Any], *, model_name: str) -> dict[str, Any]:
    """Return choices[0] from OpenRouter payload or raise a structured error."""
    try:
        return (response_json.get("choices") or [])[0]
    except Exception as e:
        raise LLMClientError(
            f"Malformed OpenRouter response for {model_name}",
            model=model_name,
            cause=e,
        ) from e


def extract_openrouter_message_content(choice0: dict[str, Any], *, model_name: str) -> Any:
    """Return choice0.message.content (raw) or raise a structured error."""
    try:
        message0 = choice0.get("message") or {}
        return message0.get("content")
    except Exception as e:
        raise LLMClientError(
            f"Malformed OpenRouter response for {model_name}",
            model=model_name,
            cause=e,
        ) from e


def extract_openrouter_usage_tokens(response_json: dict[str, Any]) -> tuple[int, int]:
    usage = response_json.get("usage") or {}
    tokens_input = int(usage.get("prompt_tokens") or 0)
    tokens_output = int(usage.get("completion_tokens") or 0)
    return tokens_input, tokens_output


class LLMClientError(Exception):
    """Exception raised when LLM API calls fail."""

    def __init__(self, message: str, model: str | None = None, cause: Exception | None = None):
        self.message = message
        self.model = model
        self.cause = cause
        super().__init__(message)


class LLMResponse(BaseModel):
    """Response from LLM API call."""

    content: str
    tokens_input: int
    tokens_output: int
    latency_ms: int
    cost_usd: float
    model_id: str
    provider: str


class OpenRouterClient:
    """Client for OpenRouter API with retry logic and cost tracking."""

    def __init__(self, api_key: str):
        """Initialize the OpenRouter client."""
        self.api_key = api_key
        self.base_url = config.OPENROUTER_BASE_URL
        self.timeout = config.LLM_TIMEOUT

    @retry(
        stop=stop_after_attempt(config.LLM_MAX_RETRIES),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type((httpx.TimeoutException, httpx.HTTPError)),
        reraise=True,
    )
    async def _call_api(
        self, model_id: str, messages: list[dict[str, Any]], temperature: float = 0.7
    ) -> tuple[dict[str, Any], int]:
        """Make API call with retry logic."""
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            start_time = time.time()

            response = await client.post(
                f"{self.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "HTTP-Referer": "https://llm-deliberate.research",
                    "X-Title": "LLM Deliberate",
                },
                json={
                    "model": model_id,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": 2000,
                },
            )

            response.raise_for_status()
            latency_ms = int((time.time() - start_time) * 1000)
            return response.json(), latency_ms

    async def generate(
        self,
        prompt: str,
        model_name: str,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> LLMResponse:
        """Generate a response from the model."""
        model_config = config.get_model_config(model_name)
        model_id = model_config["id"]

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        try:
            response_json, latency_ms = await self._call_api(
                model_id, messages, temperature
            )
        except RetryError as e:
            raise LLMClientError(
                f"Failed to get response from {model_name} after retries",
                model=model_name,
                cause=e,
            ) from e

        choice0 = extract_openrouter_choice0(response_json, model_name=model_name)
        raw_content = extract_openrouter_message_content(choice0, model_name=model_name)
        content = extract_text_content(raw_content)
        tokens_input, tokens_output = extract_openrouter_usage_tokens(response_json)

        # Treat empty completions as an error so we don't persist blank responses.
        # This commonly occurs when providers return an empty message with 0 output tokens.
        if not content.strip():
            finish_reason = choice0.get("finish_reason")
            raise LLMClientError(
                f"Empty completion returned (finish_reason={finish_reason}, tokens_output={tokens_output})",
                model=model_name,
            )

        # Calculate cost
        cost = config.estimate_cost(model_name, tokens_input, tokens_output)

        return LLMResponse(
            content=content,
            tokens_input=tokens_input,
            tokens_output=tokens_output,
            latency_ms=latency_ms,
            cost_usd=cost,
            model_id=model_id,
            provider=model_config["provider"],
        )

    async def generate_batch(
        self,
        prompts: list[str],
        model_name: str,
        temperature: float = 0.7,
        system_prompt: Optional[str] = None,
    ) -> list[LLMResponse]:
        """Generate responses for multiple prompts concurrently."""
        import asyncio

        tasks = [
            self.generate(prompt, model_name, temperature, system_prompt)
            for prompt in prompts
        ]
        return await asyncio.gather(*tasks)


def create_client() -> OpenRouterClient:
    """Factory function to create an OpenRouter client."""
    api_key = config.get_api_key()
    return OpenRouterClient(api_key)
