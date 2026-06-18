from backend.prompts import (
    create_anonymized_labels,
    de_anonymize_rankings,
    parse_debate_judgment,
    parse_ranking_response,
)


def test_parse_ranking_response_accepts_fenced_json():
    rankings, confidence, reasoning = parse_ranking_response(
        """```json
        {"rankings": ["Response B", "Response A"], "confidence": 0.9, "reasoning": "B wins"}
        ```"""
    )

    assert rankings == ["Response B", "Response A"]
    assert confidence == 0.9
    assert reasoning == "B wins"


def test_parse_ranking_response_numbered_fallback_returns_full_labels():
    rankings, confidence, _ = parse_ranking_response("1. Response B\n2. Response A\n3. Response C")

    assert rankings == ["B", "A", "C"]
    assert confidence == 0.5


def test_de_anonymize_rankings_accepts_compact_and_long_labels():
    model_names = [f"model-{index}" for index in range(27)]

    assert de_anonymize_rankings(["A", "Response B", "AA"], model_names) == [
        "model-0",
        "model-1",
        "model-26",
    ]


def test_anonymized_labels_scale_past_z():
    assert create_anonymized_labels(27)[-2:] == ["Response Z", "Response AA"]


def test_parse_debate_judgment_clamps_confidence():
    winner, confidence, reasoning = parse_debate_judgment(
        '{"winner": "model-a", "confidence": 4, "reasoning": "Strongest case"}'
    )

    assert winner == "model-a"
    assert confidence == 1.0
    assert reasoning == "Strongest case"
