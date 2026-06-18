import asyncio
from types import SimpleNamespace

from backend import deliberation
from backend.models import Response


class FakeEmbeddingClient:
    async def get_embedding(self, text):
        vectors = {
            "answer one": [1.0, 0.0],
            "answer two": [0.99, 0.01],
            "opposite": [0.0, 1.0],
        }
        return SimpleNamespace(embedding=vectors[text])


def test_convergence_compares_models_with_each_other(monkeypatch):
    monkeypatch.setattr(deliberation, "create_client", lambda: FakeEmbeddingClient())
    current = [
        Response(model="model-a", content="answer one", round=2),
        Response(model="model-b", content="answer two", round=2),
    ]
    previous = [
        Response(model="model-a", content="opposite", round=1),
        Response(model="model-b", content="opposite", round=1),
    ]

    assert asyncio.run(deliberation.check_convergence(current, previous, threshold=0.95))


def test_convergence_rejects_cross_model_disagreement(monkeypatch):
    monkeypatch.setattr(deliberation, "create_client", lambda: FakeEmbeddingClient())
    current = [
        Response(model="model-a", content="answer one", round=2),
        Response(model="model-b", content="opposite", round=2),
    ]

    assert not asyncio.run(deliberation.check_convergence(current, current, threshold=0.95))
