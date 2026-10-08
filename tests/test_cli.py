"""CLI smoke tests: init -> validate -> decide -> eval, every command exits 0."""

from verdict.cli import main


def test_backends_lists_all_registered(capsys):
    assert main(["backends"]) == 0
    out = capsys.readouterr().out
    for name in ("jev", "kev", "semif", "laya", "mock", "clef-flash",
                 "strands", "opendecider", "von"):
        assert name in out
    assert "→ next:" in out


def test_init_validate_decide_eval(tmp_path, capsys, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["init"]) == 0
    assert (tmp_path / "verdict.yaml").exists()
    assert (tmp_path / "eval" / "labels.jsonl").exists()

    assert main(["validate", "-f", "verdict.yaml"]) == 0
    assert "valid" in capsys.readouterr().out

    rc = main(["decide", "-f", "verdict.yaml",
               "--decision", "dispute_reason_code",
               "--state", "I was charged twice."])
    assert rc == 0
    out = capsys.readouterr().out
    assert "value:" in out and "→ next:" in out

    rc = main(["eval", "-f", "verdict.yaml", "--decision", "dispute_reason_code",
               "--dataset", "eval/labels.jsonl"])
    assert rc == 0
    assert "→ next:" in capsys.readouterr().out


def test_decide_inline_choice(capsys):
    rc = main(["decide", "--kind", "choice",
               "--instructions", "Pick one.",
               "--options", '{"a": "A", "b": "B"}',
               "--state", "hello"])
    assert rc == 0
    assert "value:" in capsys.readouterr().out


def test_decide_file_needs_decision():
    import pytest

    with pytest.raises(SystemExit):
        main(["decide", "-f", "verdict.yaml", "--state", "x"])


def test_validate_rejects_bad_file(tmp_path, capsys):
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: 1\ndecisions: []\n")
    assert main(["validate", "-f", str(bad)]) == 1
    assert "INVALID" in capsys.readouterr().out
