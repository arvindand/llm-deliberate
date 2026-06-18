import asyncio

from backend import automation, main
from backend.llm_client import LLMResponse
from backend.models import Experiment, Question, QuestionType, Ranking, Response


class FakeRankingClient:
    async def generate(self, prompt, model_name):
        return LLMResponse(
            content=(
                '{"rankings": ["Response B", "Response A"], '
                '"confidence": 0.8, "reasoning": "B is clearer"}'
            ),
            tokens_input=100,
            tokens_output=20,
            latency_ms=10,
            cost_usd=0.001,
            model_id=model_name,
            provider="test",
        )


def test_automated_anonymized_rankings_map_to_response_ids(monkeypatch):
    responses = [
        Response(id="response-a", model="model-a", content="Answer A"),
        Response(id="response-b", model="model-b", content="Answer B"),
    ]
    monkeypatch.setattr(automation, "create_client", lambda: FakeRankingClient())

    rankings, errors = asyncio.run(
        automation.collect_rankings_automated(
            "Question?",
            responses,
            ["judge-model"],
            anonymize=True,
        )
    )

    assert errors == []
    assert len(rankings) == 1
    assert rankings[0].rankings == ["response-b", "response-a"]


def test_rank_labels_support_more_than_26_responses():
    assert automation.rank_letter_to_index("A") == 0
    assert automation.rank_letter_to_index("Response Z") == 25
    assert automation.rank_letter_to_index("AA") == 26


def test_chairman_job_persists_synthesis(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DATA_DIR", tmp_path)
    main._load_experiment_cached.cache_clear()

    question = Question(
        id="question-1",
        text="Question?",
        question_type=QuestionType.SUBJECTIVE,
        responses=[Response(id="response-1", model="model-a", content="Answer")],
        rankings=[
            Ranking(id="ranking-1", judge="judge-a", rankings=["response-1"])
        ],
    )
    experiment = Experiment(id="experiment-1", name="Test", questions=[question])
    main.save_experiment(experiment)

    async def fake_synthesis(*_args, **_kwargs):
        return "Persisted final answer"

    monkeypatch.setattr(automation.deliberation, "chairman_synthesis", fake_synthesis)

    async def run_job():
        job_id = automation.start_chairman_job(
            question_text=question.text,
            responses=question.responses,
            rankings=[],
            chairman_model="model-a",
            experiment_id=experiment.id,
            question_id=question.id,
        )
        await asyncio.gather(*list(automation._background_tasks))
        return job_id

    job_id = asyncio.run(run_job())
    saved_question = main.load_experiment(experiment.id).get_question_by_id(question.id)

    assert saved_question is not None
    assert saved_question.chairman_synthesis is not None
    assert saved_question.chairman_synthesis.content == "Persisted final answer"
    assert saved_question.chairman_synthesis.chairman_model == "model-a"
    assert saved_question.chairman_synthesis.job_id == job_id
    assert automation.jobs[job_id].progress == {"completed": 1, "total": 1}

    automation.jobs.pop(job_id, None)
    main._load_experiment_cached.cache_clear()
