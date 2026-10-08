"""Tests for the Kev, SemIf, and Laya backends + shared wire logic.

No network, no weights, no API keys: everything here exercises config
defaults, wire-shape mapping, graceful-unavailable paths, and the
temperature plumbing. Live behavior is marked UNCERTAIN in the source.
"""

import shutil

import pytest

from verdict.backends.base import BackendAnswer, BackendQuestion, QuestionKind, TokenUsage
from verdict.backends.calibration import apply_temperature, apply_temperature_binary
from verdict.backends.jev import API_ENDPOINT, JevBackend, SystemOneHttpBackend
from verdict.backends.kev import KEV_LOCAL_DEFAULT, KevBackend
from verdict.backends.laya import LayaBackend
from verdict.backends.semif import SemIfBackend
from verdict.backends import get_backend, list_backends
from verdict.eval.harness import make_backend, run_comparison
from verdict.eval.presets import reason_code_step
from verdict.runtime.errors import BackendError


def _choice_q():
    return BackendQuestion(
        kind=QuestionKind.CHOICE,
        instructions="pick one",
        options={"a": "Option A", "b": "Option B"},
    )


# -- shared System One wire logic ----------------------------------------


def test_http_payload_choice_shape():
    b = SystemOneHttpBackend()
    payload = b._http_payload("state", {"q": _choice_q()})
    assert payload["state"] == "state"
    assert payload["questions"]["q"]["type"] == "choice"
    assert payload["questions"]["q"]["criteria"] == {"a": "Option A", "b": "Option B"}


def test_http_payload_score_shape():
    b = SystemOneHttpBackend()
    q = BackendQuestion(kind=QuestionKind.SCORE, instructions="rate",
                        levels=("bad", "ok", "good"))
    payload = b._http_payload("state", {"q": q})
    assert payload["questions"]["q"]["criteria"] == ["bad", "ok", "good"]


def test_http_payload_noul_has_no_criteria():
    b = SystemOneHttpBackend()
    q = BackendQuestion(kind=QuestionKind.NOUL, instructions="is it?")
    payload = b._http_payload("state", {"q": q})
    assert "criteria" not in payload["questions"]["q"]


def test_jev_backend_defaults_unchanged():
    # Existing public contract preserved by the refactor.
    b = JevBackend(api_key="k")
    assert b.base_url == API_ENDPOINT
    assert b.model == "jev-latest"
    assert b.api_key == "k"


def test_jev_backend_still_requires_key(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(BackendError):
        JevBackend()


# -- Kev -------------------------------------------------------------------


def test_kev_default_base_url_is_typesafe():
    assert KevBackend.default_base_url == API_ENDPOINT


def test_kev_local_override_needs_no_key():
    b = KevBackend(base_url=KEV_LOCAL_DEFAULT)
    assert b.base_url == KEV_LOCAL_DEFAULT
    assert b.api_key is None


def test_kev_requires_key_for_typesafe_default(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    with pytest.raises(BackendError):
        KevBackend()  # default base_url is TypeSafe's -> key required


def test_kev_explicit_key_for_typesafe_default():
    b = KevBackend(api_key="k")
    assert b.api_key == "k"
    assert b.name == "kev"


def test_kev_self_hosted_is_unmetered():
    b = KevBackend(base_url=KEV_LOCAL_DEFAULT)
    assert b.input_usd_per_mtok == 0.0


# -- Laya --------------------------------------------------------------------


def test_laya_defaults():
    b = LayaBackend()
    assert b.name == "laya"
    assert b.temperatures == {}
    assert b.api_key is None  # keyless self-hosted
    assert b.input_usd_per_mtok == 0.0


def test_laya_temperature_config_plumbing():
    b = LayaBackend(temperatures={"choice": 1.5, "noul": 2.0})
    assert b.temperatures == {"choice": 1.5, "noul": 2.0}


def _laya_choice_answer():
    return BackendAnswer(
        question_name="q", kind=QuestionKind.CHOICE, value="a",
        probabilities={"a": 0.7, "b": 0.3}, confidence=0.7,
        model_version="laya", usage=TokenUsage(),
        cost_usd=0.0, latency_ms=1.0,
    )


def test_laya_no_temperature_is_identity():
    b = LayaBackend()
    ans = _laya_choice_answer()
    assert b._rescale_answer(ans) is ans


def test_laya_temperature_sharpens_and_recomputes_confidence():
    b = LayaBackend(temperatures={"choice": 0.5})
    out = b._rescale_answer(_laya_choice_answer())
    assert out.probabilities["a"] > 0.7  # sharpened
    assert out.confidence == pytest.approx(max(out.probabilities.values()))
    assert sum(out.probabilities.values()) == pytest.approx(1.0)


def test_laya_temperature_flattens():
    b = LayaBackend(temperatures={"choice": 5.0})
    out = b._rescale_answer(_laya_choice_answer())
    assert out.probabilities["a"] < 0.7  # flattened toward uniform
    assert out.value == "a"  # argmax never changes


def test_laya_noul_temperature_scales_value():
    b = LayaBackend(temperatures={"noul": 2.0})
    ans = BackendAnswer(
        question_name="n", kind=QuestionKind.NOUL, value=0.9,
        probabilities=None, confidence=None,
        model_version="laya", usage=TokenUsage(),
        cost_usd=0.0, latency_ms=1.0,
    )
    out = b._rescale_answer(ans)
    assert 0.5 < out.value < 0.9  # pulled toward 0.5


# -- calibration helper ------------------------------------------------------


def test_apply_temperature_identity():
    probs = {"a": 0.7, "b": 0.3}
    assert apply_temperature(probs, 1.0) == probs


def test_apply_temperature_sums_to_one():
    out = apply_temperature({"a": 0.7, "b": 0.2, "c": 0.1}, 2.5)
    assert sum(out.values()) == pytest.approx(1.0)


def test_apply_temperature_rejects_nonpositive():
    with pytest.raises(ValueError):
        apply_temperature({"a": 1.0}, 0.0)


def test_apply_temperature_binary_midpoint_fixed():
    assert apply_temperature_binary(0.5, 3.0) == pytest.approx(0.5)


# -- SemIf ---------------------------------------------------------------------


def test_semif_missing_cli_is_backend_error(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    b = SemIfBackend()
    with pytest.raises(BackendError) as excinfo:
        b.evaluate("state", {"q": _choice_q()})
    assert "semif-score" in str(excinfo.value)


def test_semif_build_rows_verified_shape():
    b = SemIfBackend()
    rows = b._build_rows("some state", {"q": _choice_q()})
    assert rows == [{
        "id": "q",
        "state": "some state",
        "question": "pick one",
        "options": [
            {"id": "a", "description": "Option A"},
            {"id": "b", "description": "Option B"},
        ],
    }]


def test_semif_build_rows_score_and_noul_mappings():
    b = SemIfBackend()
    score_q = BackendQuestion(kind=QuestionKind.SCORE, instructions="rate",
                              levels=("bad", "ok"))
    noul_q = BackendQuestion(kind=QuestionKind.NOUL, instructions="is it?")
    rows = b._build_rows("s", {"sc": score_q, "n": noul_q})
    by_id = {r["id"]: r for r in rows}
    assert [o["id"] for o in by_id["sc"]["options"]] == ["0", "1"]
    assert {o["id"] for o in by_id["n"]["options"]} == {"true", "false"}


def test_semif_extract_probabilities_known_key():
    probs = SemIfBackend._extract_probabilities(
        {"probabilities": {"a": 0.8, "b": 0.2}}, "q")
    assert probs == {"a": 0.8, "b": 0.2}


def test_semif_extract_probabilities_unknown_keys_fail_loudly():
    with pytest.raises(BackendError) as excinfo:
        SemIfBackend._extract_probabilities({"weird": 1}, "q")
    assert "tighten" in str(excinfo.value)


def test_semif_temperature_plumbing():
    b = SemIfBackend(temperature=2.0)
    assert b.temperature == 2.0


# -- harness factory / comparison ----------------------------------------------


def test_make_backend_unknown_raises():
    with pytest.raises(BackendError):
        make_backend("nope")


def test_make_backend_mock_ok():
    assert make_backend("mock").name == "mock"


def test_registry_covers_all_backends():
    assert set(list_backends()) == {
        "mock", "jev", "kev", "semif", "laya",
        "clef", "clef-flash", "strands", "opendecider", "wald",
        "openjev", "von", "mapika", "clm",
    }


def test_new_backends_construct_with_placeholder_url():
    # Self-hosted adapters must construct with no args (no network on init).
    from verdict.backends import get_backend

    for name in ("clef", "clef-flash", "strands", "opendecider", "wald",
                 "openjev", "von", "mapika", "clm"):
        b = get_backend(name)
        assert b.name == name
        assert b.input_usd_per_mtok == 0.0


def test_get_backend_unknown_raises_backend_error():
    with pytest.raises(BackendError):
        get_backend("nope")


def test_run_comparison_marks_unavailable_not_crash(monkeypatch):
    # jev without a key, semif without its CLI: both must degrade gracefully.
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.setattr(shutil, "which", lambda _: None)
    dataset = [{"id": "r1", "state": "charged twice", "label": "duplicate"}]
    comps = run_comparison(reason_code_step(), dataset, ["mock", "jev", "semif"])
    by_name = {c.backend: c for c in comps}
    assert by_name["mock"].available
    assert not by_name["jev"].available and by_name["jev"].note
    assert not by_name["semif"].available and by_name["semif"].note
