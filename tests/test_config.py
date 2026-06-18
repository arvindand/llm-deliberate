from backend.config import _normalize_available_models, _parse_model_from_openrouter


def test_openrouter_variable_pricing_is_not_exposed_as_negative_cost():
    model = _parse_model_from_openrouter(
        {
            "id": "openrouter/auto",
            "pricing": {"prompt": "-1", "completion": "-1"},
            "architecture": {"output_modalities": ["text"]},
        }
    )

    assert model["pricing"] == {"prompt": 0.0, "completion": 0.0}
    assert model["pricing_unknown"] is True


def test_non_text_output_models_are_not_offered_for_deliberation():
    models = _normalize_available_models(
        [
            {
                "id": "provider/audio-only",
                "pricing": {"prompt": 0, "completion": 0},
                "output_modalities": ["audio"],
            },
            {
                "id": "provider/text",
                "pricing": {"prompt": 0, "completion": 0},
                "output_modalities": ["text"],
            },
        ]
    )

    assert [model["id"] for model in models] == ["provider/text"]
