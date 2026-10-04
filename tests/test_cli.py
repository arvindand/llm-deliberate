from types import SimpleNamespace

from backend import cli
from backend.models import Experiment, Question, QuestionType, Ranking, Response


def test_cli_displays_same_model_ties_without_unanimity(monkeypatch, capsys):
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
    monkeypatch.setattr(
        cli, "load_experiment", lambda _id: Experiment(name="Test", questions=[question])
    )

    cli.cmd_compare(SimpleNamespace(exp_id="exp", question_id="q"))
    output = capsys.readouterr().out

    assert "same-model · round 1 · a" in output
    assert "same-model · round 1 · b" in output
    assert "Tie:" in output
    assert "Unresolved: elimination tie" in output
    assert "UNANIMOUS" not in output


def test_cli_recognizes_unanimous_unique_response_id(monkeypatch, capsys):
    question = Question(
        id="q",
        text="Which answer?",
        question_type=QuestionType.SUBJECTIVE,
        responses=[Response(id=id, model="same-model", content=id) for id in ["a", "b", "c"]],
        rankings=[
            Ranking(judge=f"j{i}", rankings=list(order))
            for i, order in enumerate(["abc", "abc", "acb", "bac", "cab"])
        ],
    )
    monkeypatch.setattr(
        cli, "load_experiment", lambda _id: Experiment(name="Test", questions=[question])
    )

    cli.cmd_compare(SimpleNamespace(exp_id="exp", question_id="q"))

    assert "UNANIMOUS: All methods agree on same-model · round 1 · a" in capsys.readouterr().out
