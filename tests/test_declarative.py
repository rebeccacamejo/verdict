"""Tests for the declarative path: verdict.yaml validation, loading, execution."""

import pytest

from verdict.declarative import DecisionPlan, load_file, validate_data

VALID = {
    "version": 1,
    "backends": {"kev": {"base_url": "http://localhost:8000/v1/systemone"}},
    "decisions": [
        {
            "name": "reason",
            "kind": "choice",
            "instructions": "Pick one.",
            "options": {"a": "Option A", "b": "Option B"},
            "policy": {"backend": "kev", "auto_threshold": 0.9,
                       "review_threshold": 0.5, "on_backend_error": "human_review"},
        },
        {
            "name": "quality",
            "kind": "score",
            "instructions": "Rate it.",
            "levels": ["bad", "ok", "good"],
        },
        {
            "name": "flagged",
            "kind": "boolean",
            "instructions": "Is it flagged?",
            "policy": {"backend": "mock"},
        },
    ],
}


def test_validate_accepts_good_file():
    assert validate_data(VALID) == []


def _bad(mutate):
    import copy

    data = copy.deepcopy(VALID)
    mutate(data)
    return data


def test_validate_rejects_bad_version():
    assert validate_data(_bad(lambda d: d.update(version=2)))


def test_validate_rejects_unknown_kind():
    errs = validate_data(_bad(lambda d: d["decisions"][0].update(kind="vibes")))
    assert any("kind" in e for e in errs)


def test_validate_rejects_choice_without_options():
    errs = validate_data(_bad(lambda d: d["decisions"][0].pop("options")))
    assert any("options" in e for e in errs)


def test_validate_rejects_duplicate_names():
    errs = validate_data(
        _bad(lambda d: d["decisions"].append(dict(d["decisions"][0])))
    )
    assert any("duplicate" in e for e in errs)


def test_validate_rejects_inverted_thresholds():
    def mutate(d):
        d["decisions"][0]["policy"]["review_threshold"] = 0.95

    errs = validate_data(_bad(mutate))
    assert any("review_threshold" in e for e in errs)


def test_validate_rejects_unknown_backend():
    def mutate(d):
        d["decisions"][0]["policy"]["backend"] = "nope"

    errs = validate_data(_bad(mutate))
    assert any("unknown" in e for e in errs)


def test_load_file_and_decide(tmp_path):
    import yaml

    path = tmp_path / "verdict.yaml"
    data = {
        "version": 1,
        "decisions": [
            {
                "name": "reason",
                "kind": "choice",
                "instructions": "Pick one.",
                "options": {"a": "Option A", "b": "Option B"},
                "policy": {"backend": "mock"},
            }
        ],
    }
    path.write_text(yaml.safe_dump(data))
    plan = load_file(path)
    assert plan.decision_names() == ["reason"]
    outcome = plan.decide("reason", "some state")
    assert outcome.value == "a"  # mock default: first option, neutral
    assert outcome.confidence == 0.5


def test_decide_unknown_name_raises(tmp_path):
    import yaml

    path = tmp_path / "v.yaml"
    path.write_text(yaml.safe_dump({"version": 1, "decisions": [
        {"name": "x", "kind": "boolean", "instructions": "yes?"}]}))
    plan = load_file(path)
    with pytest.raises(KeyError):
        plan.decide("nope", "state")


def test_load_file_rejects_invalid(tmp_path):
    path = tmp_path / "v.yaml"
    path.write_text("version: 1\ndecisions: []\n")
    with pytest.raises(ValueError, match="non-empty"):
        load_file(path)


def test_from_data_compiles_all_kinds():
    plan = DecisionPlan.from_data(VALID)
    assert plan.decision_names() == ["flagged", "quality", "reason"]
    assert plan.decisions["reason"].backend_kwargs == {
        "base_url": "http://localhost:8000/v1/systemone"
    }
