import json
from types import SimpleNamespace

import pytest

from backend import cli
from backend.export_utils import export_experiment_to_json
from backend.models import Experiment, Question, QuestionType, Ranking, Response


def test_export_import_preserves_experiment_and_does_not_overwrite(tmp_path, monkeypatch):
    question = Question(
        id="q",
        text="Which answer?",
        question_type=QuestionType.REASONING,
        responses=[
            Response(
                id="a",
                model="provider/model-a",
                content="Answer A",
                round=2,
                metadata={"cost_usd": 0.002, "tokens_input": 10},
            ),
            Response(id="b", model="provider/model-b", content="Answer B", round=2),
        ],
        rankings=[Ranking(id="rank", judge="judge", rankings=["b", "a"], confidence=0.8)],
    )
    experiment = Experiment(id="round-trip", name="Reproducible", questions=[question])
    export = tmp_path / "export.json"
    export.write_text(json.dumps(export_experiment_to_json(experiment), default=str))
    destination = tmp_path / "restored"
    destination.mkdir()
    monkeypatch.setattr(cli, "DATA_DIR", destination)

    cli.cmd_import(SimpleNamespace(path=str(export)))
    restored = cli.load_experiment(experiment.id)
    assert restored.model_dump() == experiment.model_dump()
    original_bytes = (destination / "round-trip.json").read_bytes()
    with pytest.raises(SystemExit) as error:
        cli.cmd_import(SimpleNamespace(path=str(export)))
    assert error.value.code == 1
    assert (destination / "round-trip.json").read_bytes() == original_bytes


@pytest.mark.parametrize("content", ["invalid JSON", '{"name":"bad","id":"../outside"}'])
def test_invalid_import_does_not_write_files(tmp_path, monkeypatch, content):
    export = tmp_path / "export.json"
    export.write_text(content)
    destination = tmp_path / "restored"
    destination.mkdir()
    monkeypatch.setattr(cli, "DATA_DIR", destination)
    with pytest.raises(SystemExit) as error:
        cli.cmd_import(SimpleNamespace(path=str(export)))
    assert error.value.code == 1
    assert list(destination.iterdir()) == []


def test_bundled_demo_can_be_restored_without_api_access(tmp_path, monkeypatch):
    from pathlib import Path

    demo = Path(__file__).resolve().parents[1] / "data/experiments/demo_showcase.json"
    monkeypatch.setattr(cli, "DATA_DIR", tmp_path)
    cli.cmd_import(SimpleNamespace(path=str(demo)))
    experiment = Experiment.model_validate_json(demo.read_text())
    assert cli.load_experiment(experiment.id).model_dump() == experiment.model_dump()
