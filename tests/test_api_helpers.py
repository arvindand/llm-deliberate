import asyncio

import httpx
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from backend import automation, main
from backend.aggregation import agreement_matrix
from backend.models import Experiment, Question, QuestionType, Ranking, Response


def test_job_status_includes_results_and_is_scoped_to_experiment():
    job = automation.JobProgress(
        job_id="job-1",
        experiment_id="experiment-a",
        status=automation.JobStatus.COMPLETED,
        results=[{"content": "Synthesized answer"}],
    )
    automation.jobs[job.job_id] = job
    try:
        status = main.get_job_status("experiment-a", job.job_id)
        assert status["results"] == [{"content": "Synthesized answer"}]

        with pytest.raises(HTTPException) as exc_info:
            main.get_job_status("experiment-b", job.job_id)
        assert exc_info.value.status_code == 404
    finally:
        automation.jobs.pop(job.job_id, None)


def test_deliberation_request_rejects_single_model_and_zero_rounds():
    with pytest.raises(ValidationError):
        main.DeliberateRequest(question_id="question", models=["model-a"], max_rounds=0)


def test_agreement_matrix_preserves_duplicate_judge_ballots():
    rankings = [
        Ranking(id="rank-1", judge="same-judge", rankings=["a", "b"]),
        Ranking(id="rank-2", judge="same-judge", rankings=["b", "a"]),
    ]

    matrix = agreement_matrix(rankings, ["a", "b"])

    assert list(matrix) == ["same-judge #1", "same-judge #2"]
    assert matrix["same-judge #1"]["same-judge #1"] == 1.0
    assert matrix["same-judge #1"]["same-judge #2"] == 0.0


def test_experiment_api_flow_computes_results(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DATA_DIR", tmp_path)
    main._load_experiment_cached.cache_clear()

    async def run_flow():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            experiment_id = (
                await client.post(
                    "/experiments",
                    json={"name": "Integration test"},
                )
            ).json()["id"]
            question_id = (
                await client.post(
                    f"/experiments/{experiment_id}/questions",
                    json={"text": "Pick the better answer", "question_type": "subjective"},
                )
            ).json()["id"]
            response_ids = [
                (
                    await client.post(
                        f"/experiments/{experiment_id}/responses",
                        json={
                            "question_id": question_id,
                            "model": model,
                            "content": content,
                        },
                    )
                ).json()["id"]
                for model, content in [
                    ("model-a", "Answer A"),
                    ("model-b", "Answer B"),
                ]
            ]
            ranking_response = await client.post(
                f"/experiments/{experiment_id}/rankings",
                json={
                    "question_id": question_id,
                    "judge": "judge",
                    "rankings": list(reversed(response_ids)),
                    "confidence": 0.9,
                },
            )
            comparison = await client.get(
                f"/experiments/{experiment_id}/compare",
                params={"question_id": question_id},
            )
            return ranking_response, comparison, response_ids

    ranking_response, comparison, response_ids = asyncio.run(run_flow())

    assert ranking_response.status_code == 200
    assert comparison.status_code == 200
    assert comparison.json()["methods"]["borda"]["winner"] == response_ids[1]
    assert comparison.json()["response_labels"][response_ids[1]].startswith("model-b")
    main._load_experiment_cached.cache_clear()


def test_compute_and_compare_preserve_same_model_samples_and_rounds(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DATA_DIR", tmp_path)
    main._load_experiment_cached.cache_clear()
    question = Question(
        id="q",
        text="Which answer?",
        question_type=QuestionType.SUBJECTIVE,
        responses=[
            Response(id="sample-a", model="same-model", content="A", round=1),
            Response(id="sample-b", model="same-model", content="B", round=1),
            Response(id="round-2", model="same-model", content="C", round=2),
        ],
        rankings=[Ranking(judge="judge", rankings=["round-2", "sample-b", "sample-a"])],
    )
    experiment = Experiment(id="exp", name="Test", questions=[question])
    main.save_experiment(experiment)

    computation = main.compute_results("exp", main.ComputeResultsRequest(question_id="q"))
    comparison = main.compare_all_methods("exp", "q")

    assert computation["scores"] == {"sample-a": 0.0, "sample-b": 1.0, "round-2": 2.0}
    assert computation["raw_scores"] == computation["scores"]
    assert computation["winner"]["id"] == "round-2"
    assert computation["winner_ids"] == ["round-2"]
    assert "round 2" in computation["response_labels"]["round-2"]
    assert len(set(computation["response_labels"].values())) == 3
    assert all(len(result["scores"]) == 3 for result in comparison["methods"].values())
    # Top-2 approval ties two different responses from the same model.
    assert comparison["methods"]["approval"]["winner"] is None
    assert comparison["methods"]["approval"]["winner_ids"] == ["sample-b", "round-2"]
    assert comparison["unanimous"] is False
    main._load_experiment_cached.cache_clear()


def test_symmetric_comparison_and_compute_expose_ties_without_unanimity(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DATA_DIR", tmp_path)
    main._load_experiment_cached.cache_clear()
    question = Question(
        id="q",
        text="Which answer?",
        question_type=QuestionType.SUBJECTIVE,
        responses=[Response(id=id, model="same-model", content=id) for id in ["a", "b"]],
        rankings=[
            Ranking(judge="j1", rankings=["a", "b"]),
            Ranking(judge="j2", rankings=["b", "a"]),
        ],
    )
    main.save_experiment(Experiment(id="exp", name="Test", questions=[question]))

    computation = main.compute_results("exp", main.ComputeResultsRequest(question_id="q"))
    comparison = main.compare_all_methods("exp", "q")

    assert computation["winner"] is None
    assert computation["winner_id"] is None
    assert computation["winner_ids"] == ["a", "b"]
    assert comparison["unanimous"] is False
    assert all(result["winner"] is None for result in comparison["methods"].values())
    assert comparison["methods"]["stv"]["status"] == "unresolved"
    main._load_experiment_cached.cache_clear()


def test_ranked_pairs_api_keeps_unranked_new_response_out_of_winners(tmp_path, monkeypatch):
    monkeypatch.setattr(main, "DATA_DIR", tmp_path)
    main._load_experiment_cached.cache_clear()

    async def run_flow():
        transport = httpx.ASGITransport(app=main.app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            experiment = await client.post(
                "/experiments", json={"name": "New response after ballots"}
            )
            experiment_id = experiment.json()["id"]
            question = await client.post(
                f"/experiments/{experiment_id}/questions",
                json={"text": "Which answer?", "question_type": "subjective"},
            )
            question_id = question.json()["id"]
            response_ids = []
            for content in ["A", "B", "C"]:
                response = await client.post(
                    f"/experiments/{experiment_id}/responses",
                    json={"question_id": question_id, "model": "same-model", "content": content},
                )
                assert response.status_code == 200
                response_ids.append(response.json()["id"])
            ranking = await client.post(
                f"/experiments/{experiment_id}/rankings",
                json={"question_id": question_id, "judge": "judge", "rankings": response_ids},
            )
            assert ranking.status_code == 200
            added_response = await client.post(
                f"/experiments/{experiment_id}/responses",
                json={"question_id": question_id, "model": "same-model", "content": "D"},
            )
            assert added_response.status_code == 200
            unranked_id = added_response.json()["id"]
            computation = await client.post(
                f"/experiments/{experiment_id}/compute",
                json={"question_id": question_id, "method": "ranked_pairs"},
            )
            comparison = await client.get(
                f"/experiments/{experiment_id}/compare", params={"question_id": question_id}
            )
            assert computation.status_code == comparison.status_code == 200
            return computation.json(), comparison.json(), response_ids[0], unranked_id

    computation, comparison, original_winner, unranked_id = asyncio.run(run_flow())

    assert computation["winner_ids"] == [original_winner]
    assert computation["scores"][unranked_id] == 0
    assert computation["status"] == "winner"
    ranked_pairs_result = comparison["methods"]["ranked_pairs"]
    assert ranked_pairs_result["winner_ids"] == [original_winner]
    assert ranked_pairs_result["winner"] == original_winner
    assert ranked_pairs_result["scores"][unranked_id] == 0
    assert len(ranked_pairs_result["scores"]) == 4
    main._load_experiment_cached.cache_clear()
