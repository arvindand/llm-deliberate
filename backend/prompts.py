"""
Prompt templates for LLM Deliberate.

Includes prompts for response generation and ranking evaluation.
"""
import json
from typing import Optional

RESPONSE_PROMPT = """You are tasked with answering a question thoughtfully and thoroughly.

Question: {question}

Please provide a clear, well-reasoned response."""

RANKING_PROMPT = """You are an expert evaluator tasked with ranking responses to a question.

Question: {question}

Below are {num_responses} responses from different AI models:

{responses_formatted}

Please evaluate each response based on:
1. **Accuracy**: How correct and factually sound is the response?
2. **Completeness**: Does it fully address the question?
3. **Clarity**: Is the explanation clear and well-organized?
4. **Depth**: Does it show genuine insight and reasoning?

Provide your ranking from best to worst. Respond in valid JSON format (no markdown):
{{
  "rankings": ["Response A", "Response B", "Response C"],
  "confidence": 0.85,
  "reasoning": "Brief explanation of your ranking decisions"
}}

Important: The "rankings" array should contain the response letters (A, B, C, etc.) in order from best to worst.
Make sure the JSON is valid and can be parsed."""

DELIBERATION_PROMPT = """You previously answered the following question:

Question: {question}

Your previous response: {previous_response}

You now see responses from other models:

{other_responses_formatted}

Given these other perspectives, would you like to refine or modify your answer?
Consider if other models have made valid points you hadn't considered, identified errors in your reasoning, or provided complementary insights.

Please provide your updated response (or confirm your previous response if you still think it's best):"""


def format_response_prompt(question: str) -> str:
    """Format the response generation prompt."""
    return RESPONSE_PROMPT.format(question=question)


def format_ranking_prompt(question: str, responses: list[dict]) -> str:
    """Format the ranking evaluation prompt.

    Args:
        question: The question being evaluated
        responses: List of dicts with 'model' and 'content' keys
    """
    responses_formatted = "\n\n".join(
        f"Response {chr(65 + i)}: {resp['content']}"
        for i, resp in enumerate(responses)
    )

    return RANKING_PROMPT.format(
        question=question,
        num_responses=len(responses),
        responses_formatted=responses_formatted,
    )


def format_deliberation_prompt(
    question: str, previous_response: str, other_responses: list[dict]
) -> str:
    """Format the deliberation prompt for multi-round refinement.

    Args:
        question: The original question
        previous_response: The model's previous response
        other_responses: List of dicts with 'model' and 'content' keys
    """
    other_responses_formatted = "\n\n".join(
        f"**{resp['model']}**: {resp['content']}" for resp in other_responses
    )

    return DELIBERATION_PROMPT.format(
        question=question,
        previous_response=previous_response,
        other_responses_formatted=other_responses_formatted,
    )


def parse_ranking_response(response_text: str) -> tuple[list[str], float, str]:
    """Parse ranking response from model.

    Returns:
        Tuple of (rankings, confidence, reasoning)
    """
    try:
        # Try to extract JSON from the response
        data = json.loads(response_text)
        rankings = data.get("rankings", [])
        confidence = float(data.get("confidence", 0.5))
        reasoning = data.get("reasoning", "")

        # Validate confidence is in [0, 1]
        confidence = max(0.0, min(1.0, confidence))

        return rankings, confidence, reasoning
    except json.JSONDecodeError:
        # Fallback: try to extract rankings from text
        # This is a simple heuristic - in production you might want more robust parsing
        raise ValueError(f"Could not parse ranking response as JSON: {response_text}")
