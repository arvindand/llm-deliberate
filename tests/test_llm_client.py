from backend.llm_client import EmptyCompletionError, OpenRouterClient, extract_text_content


def test_empty_completion_error_is_retriable_by_openrouter_client():
    retry_predicate = OpenRouterClient._call_api.retry.retry

    assert EmptyCompletionError in retry_predicate.exception_types


def test_extract_text_content_supports_openrouter_content_parts():
    assert extract_text_content([{"type": "text", "text": "Hello"}, {"text": " world"}]) == (
        "Hello world"
    )
