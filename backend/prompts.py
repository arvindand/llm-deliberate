"""
Prompt templates for LLM Deliberate.

Includes prompts for response generation and ranking evaluation.
"""

import json
import re
from typing import Any

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


def index_to_label(index: int) -> str:
    """Convert a zero-based index to spreadsheet-style labels: A..Z, AA..AZ."""
    if index < 0:
        raise ValueError("index must be non-negative")

    label = ""
    value = index + 1
    while value:
        value, remainder = divmod(value - 1, 26)
        label = chr(ord("A") + remainder) + label
    return label


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
        f"Response {index_to_label(i)}: {resp['content']}" for i, resp in enumerate(responses)
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


def _parse_json_object(response_text: str) -> dict[str, Any]:
    """Parse a JSON object, tolerating markdown fences and surrounding text."""
    text = response_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            raise
        data = json.loads(text[start : end + 1])

    if not isinstance(data, dict):
        raise ValueError("Expected a JSON object")
    return data


def parse_ranking_response(response_text: str) -> tuple[list[str], float, str]:
    """Parse ranking response from model.

    Returns:
        Tuple of (rankings, confidence, reasoning)
    """
    try:
        data = _parse_json_object(response_text)
        rankings = data.get("rankings", [])
        if not isinstance(rankings, list) or not all(isinstance(item, str) for item in rankings):
            raise ValueError("Ranking response must contain a string array named 'rankings'")
        confidence = float(data.get("confidence", 0.5))
        reasoning = str(data.get("reasoning", ""))

        # Validate confidence is in [0, 1]
        confidence = max(0.0, min(1.0, confidence))

        return rankings, confidence, reasoning
    except (json.JSONDecodeError, TypeError, ValueError):
        # Fallback: try to extract rankings from text using regex

        # Look for numbered patterns like "1. Response A", "2. Response B"
        numbered_matches = re.findall(
            r"\d+[.)]\s*Response\s+([A-Z]+)\b", response_text, flags=re.IGNORECASE
        )

        if numbered_matches:
            rankings = [match.upper() for match in numbered_matches]
            confidence = 0.5
            reasoning = "Extracted from numbered list format"
            return rankings, confidence, reasoning

        # Try another pattern: "Response A, Response B, Response C"
        text_matches = re.findall(r"Response\s+([A-Z]+)\b", response_text, flags=re.IGNORECASE)

        if text_matches:
            rankings = [match.upper() for match in text_matches]
            confidence = 0.5
            reasoning = "Extracted from text format"
            return rankings, confidence, reasoning

        raise ValueError(f"Could not parse ranking response: {response_text}") from None


def create_anonymized_labels(count: int) -> list[str]:
    """Create anonymized labels (Response A, Response B, etc.) for responses.

    Args:
        count: Number of responses to label

    Returns:
        List of anonymized labels like ["Response A", "Response B"]
    """
    return [f"Response {index_to_label(i)}" for i in range(count)]


def format_ranking_prompt_anonymized(question: str, responses: list[dict]) -> str:
    """Format ranking evaluation prompt with anonymized responses.

    Args:
        question: The question being evaluated
        responses: List of dicts with 'content' keys only (model names anonymized)

    Returns:
        Formatted prompt string
    """
    responses_formatted = "\n\n".join(
        f"Response {index_to_label(i)}: {resp['content']}" for i, resp in enumerate(responses)
    )

    return RANKING_PROMPT.format(
        question=question,
        num_responses=len(responses),
        responses_formatted=responses_formatted,
    )


def de_anonymize_rankings(rankings: list[str], model_names: list[str]) -> list[str]:
    """Convert anonymized labels (Response A, Response B) back to model names.

    Args:
        rankings: List of anonymized labels from judge response
        model_names: List of actual model names in original order

    Returns:
        List of model names (e.g., ["openai/gpt-4o", "anthropic/claude-3.5-sonnet"])
    """
    result = []

    for ranking in rankings:
        # Accept either "Response A" or the compact "A" form.
        match = re.fullmatch(r"(?:Response\s+)?([A-Z]+)", ranking.strip(), flags=re.IGNORECASE)

        if match:
            label = match.group(1).upper()
            index = 0
            for character in label:
                index = index * 26 + (ord(character) - ord("A") + 1)
            index -= 1

            if 0 <= index < len(model_names):
                result.append(model_names[index])

    return result


CHAIRMAN_PROMPT = """You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and other AI models have ranked those responses.

Your task is to synthesize ALL of this information into a single, comprehensive, accurate answer to the user's original question.

**Original Question:**
{question}

**Individual Model Responses:**
{responses_formatted}

**Peer Rankings and Evaluations:**
{rankings_formatted}

**Your Analysis Guidelines:**
1. Consider ALL model responses, not just the top-ranked ones
2. Use peer evaluations to identify strengths and weaknesses
3. Look for patterns of agreement and disagreement
4. Synthesize a unified answer that incorporates:
   - Points of consensus (what most models agree on)
   - Minority viewpoints that have merit
   - Nuance and qualifications from different perspectives
5. Do NOT simply copy-paste or average responses
6. Provide a clear, well-structured final answer

Please provide your synthesized final answer in plain text."""


def format_chairman_prompt(
    question: str,
    responses: list[dict],
    rankings: list[dict],
) -> str:
    """Format chairman synthesis prompt.

    Args:
        question: The original question
        responses: List of response dicts with 'model' and 'content'
        rankings: List of ranking dicts with 'judge', 'rankings', 'confidence', 'reasoning'

    Returns:
        Formatted chairman prompt
    """
    responses_formatted = "\n\n".join(f"**{r['model']}**: {r['content']}" for r in responses)

    rankings_formatted = "\n\n".join(
        f"Ranking by {r['judge']}:\n"
        f"  {r.get('rankings', [])}\n"
        f"  Confidence: {r.get('confidence', 0.5):.2f}\n"
        f"  Reasoning: {r.get('reasoning', 'N/A')}"
        for r in rankings
    )

    return CHAIRMAN_PROMPT.format(
        question=question,
        responses_formatted=responses_formatted,
        rankings_formatted=rankings_formatted,
    )


DEBATE_INITIAL_PROMPT = """You are participating in a debate about the following question:

{question}

You are Debater {position} (out of {total_debaters} debaters).
Your assigned stance is: {stance}

Present the strongest accurate case for that stance. This is the first round, so state your position clearly, support it with reasoning, and anticipate the strongest objection.

Provide your argument:"""


DEBATE_REBUTTAL_PROMPT = """You are participating in a debate about the following question:

{question}

Your previous argument was:
{own_previous}

Other debaters have argued:
{other_arguments}

Now provide a rebuttal that addresses the opposing arguments while strengthening your position:"""


DEBATE_JUDGE_PROMPT = """You are an impartial judge evaluating a debate. The question being debated is:

{question}

The final arguments from each debater are:

{arguments}

Evaluate which debater presented the most convincing, accurate, and well-reasoned argument. Consider:
- Logical soundness
- Evidence and reasoning quality
- Addressing counterarguments
- Factual accuracy

Respond in JSON format:
{{
  "winner": "debater_model_name",
  "confidence": 0.0-1.0,
  "reasoning": "explanation of your decision"
}}"""


def format_debate_initial_prompt(question: str, debater_idx: int, total_debaters: int) -> str:
    if total_debaters == 2:
        stance = (
            "Argue in favor of the proposition or answer implied by the question."
            if debater_idx == 0
            else "Argue against the proposition or present the strongest competing answer."
        )
    else:
        stance = (
            "Develop a distinct position from the other debaters. Focus on a different "
            "interpretation, trade-off, or solution path."
        )

    return DEBATE_INITIAL_PROMPT.format(
        question=question,
        position=debater_idx + 1,
        total_debaters=total_debaters,
        stance=stance,
    )


def format_debate_rebuttal_prompt(
    question: str, own_previous: str, other_arguments: list[dict]
) -> str:
    other_formatted = "\n\n".join(
        f"**{arg['model']}:**\n{arg['content']}" for arg in other_arguments
    )

    return DEBATE_REBUTTAL_PROMPT.format(
        question=question,
        own_previous=own_previous,
        other_arguments=other_formatted,
    )


def format_debate_judge_prompt(question: str, final_arguments: list[dict]) -> str:
    arguments_formatted = "\n\n".join(
        f"**{arg['model']}:**\n{arg['content']}" for arg in final_arguments
    )

    return DEBATE_JUDGE_PROMPT.format(
        question=question,
        arguments=arguments_formatted,
    )


def parse_debate_judgment(response_text: str) -> tuple[str | None, float, str]:
    """Parse judge's verdict from debate.

    Returns:
        Tuple of (winner_model, confidence, reasoning)
    """
    try:
        data = _parse_json_object(response_text)
        winner = data.get("winner")
        if winner is not None and not isinstance(winner, str):
            raise ValueError("winner must be a string or null")
        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))
        reasoning = str(data.get("reasoning", ""))
        return winner, confidence, reasoning
    except (json.JSONDecodeError, TypeError, ValueError, KeyError):
        return None, 0.0, f"Failed to parse judgment: {response_text[:200]}"
