import asyncio

import httpx
import pytest
from fastapi import HTTPException
from pydantic import ValidationError

from backend import automation, main
from backend.aggregation import agreement_matrix
from backend.models import Ranking


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
            return ranking_response, comparison

    ranking_response, comparison = asyncio.run(run_flow())

    assert ranking_response.status_code == 200
    assert comparison.status_code == 200
    assert comparison.json()["methods"]["borda"]["winner"] == "model-b"
    main._load_experiment_cached.cache_clear()
