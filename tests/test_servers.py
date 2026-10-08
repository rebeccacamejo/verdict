"""Tests for the server lifecycle layer: recipes, probing, YAML wiring."""

import json
import os
import socket
import threading

import pytest

from verdict.backends import get_backend, list_backends
from verdict.servers import (
    RECIPES,
    ServerManager,
    managed_url,
    probe,
    resolve_url,
    set_policy_backend,
    wire_yaml,
)


# -- recipes ---------------------------------------------------------------


def test_every_backend_has_a_recipe_or_manual_path():
    for name in list_backends():
        assert name in RECIPES, f"{name} missing from RECIPES"
        r = RECIPES[name]
        if r.auto:
            assert r.serve, f"{name}: auto recipe needs a serve command"
            assert r.port > 0
        else:
            assert r.manual, f"{name}: manual recipe needs guidance text"


def test_recipe_urls_are_well_formed():
    for name, r in RECIPES.items():
        if r.auto:
            assert r.default_url.startswith("http://")
            assert r.default_url.endswith("/v1/systemone")


# -- probing -----------------------------------------------------------------


def test_probe_finds_open_port():
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    srv.listen(1)
    ready = threading.Event()

    def _accept():
        ready.set()
        try:
            conn, _ = srv.accept()
            conn.close()
        except OSError:
            pass

    t = threading.Thread(target=_accept, daemon=True)
    t.start()
    assert ready.wait(timeout=5), "test server did not start"
    try:
        assert probe(f"http://127.0.0.1:{port}/v1/systemone")
    finally:
        srv.close()


def test_probe_misses_closed_port():
    # Port 1 is (almost) certainly closed.
    assert not probe("http://127.0.0.1:1/v1/systemone", timeout=0.5)


# -- managed server discovery --------------------------------------------------


def test_get_backend_prefers_managed_server(tmp_path, monkeypatch):
    # Fake a managed server using our own (alive) pid.
    monkeypatch.setattr("verdict.servers.home",
                        lambda: tmp_path / ".verdict" / "servers")
    d = tmp_path / ".verdict" / "servers" / "von"
    d.mkdir(parents=True)
    url = "http://127.0.0.1:9999/v1/systemone"
    (d / "server.json").write_text(json.dumps(
        {"pid": os.getpid(), "url": url, "port": 9999}))
    assert managed_url("von") == url
    b = get_backend("von")
    assert b.base_url == url


def test_get_backend_explicit_wins_over_managed(tmp_path, monkeypatch):
    monkeypatch.setattr("verdict.servers.home",
                        lambda: tmp_path / ".verdict" / "servers")
    d = tmp_path / ".verdict" / "servers" / "von"
    d.mkdir(parents=True)
    (d / "server.json").write_text(json.dumps(
        {"pid": os.getpid(), "url": "http://127.0.0.1:9999/v1/systemone",
         "port": 9999}))
    b = get_backend("von", base_url="http://example.com/v1/systemone")
    assert b.base_url == "http://example.com/v1/systemone"


def test_managed_url_ignores_dead_pid(tmp_path, monkeypatch):
    monkeypatch.setattr("verdict.servers.home",
                        lambda: tmp_path / ".verdict" / "servers")
    d = tmp_path / ".verdict" / "servers" / "von"
    d.mkdir(parents=True)
    (d / "server.json").write_text(json.dumps(
        {"pid": 2**30, "url": "http://127.0.0.1:9999/v1/systemone", "port": 9999}))
    assert managed_url("von") is None


def test_resolve_url_prefers_env(tmp_path, monkeypatch):
    monkeypatch.setenv("VON_BASE_URL", "http://env-host:1234/v1/systemone")
    assert resolve_url("von") == "http://env-host:1234/v1/systemone"


# -- verdict.yaml surgery ------------------------------------------------------


def test_wire_yaml_creates_backends_section(tmp_path):
    p = tmp_path / "v.yaml"
    p.write_text("version: 1\n\ndecisions: []\n")
    wire_yaml(p, "von", "http://127.0.0.1:8000/v1/systemone")
    text = p.read_text()
    assert "backends:\n  von:\n    base_url: http://127.0.0.1:8000/v1/systemone" in text
    assert "decisions: []" in text  # rest preserved


def test_wire_yaml_updates_existing_url(tmp_path):
    p = tmp_path / "v.yaml"
    p.write_text(
        "version: 1\n\n"
        "backends:\n"
        "  von:\n"
        "    base_url: http://old:8000/v1/systemone\n"
        "  kev:\n"
        "    base_url: http://kev:8009/v1/systemone\n"
        "\ndecisions: []\n"
    )
    wire_yaml(p, "von", "http://new:8000/v1/systemone")
    text = p.read_text()
    assert "base_url: http://new:8000/v1/systemone" in text
    assert "base_url: http://kev:8009/v1/systemone" in text  # untouched


def test_wire_yaml_adds_backend_to_existing_section(tmp_path):
    p = tmp_path / "v.yaml"
    p.write_text("version: 1\nbackends:\n  kev:\n    base_url: http://x\ndecisions: []\n")
    wire_yaml(p, "von", "http://127.0.0.1:8000/v1/systemone")
    text = p.read_text()
    assert "  von:\n    base_url: http://127.0.0.1:8000/v1/systemone" in text
    assert "  kev:\n    base_url: http://x" in text


def test_wire_yaml_preserves_comments(tmp_path):
    p = tmp_path / "v.yaml"
    p.write_text("# a comment\nversion: 1\ndecisions: []\n")
    wire_yaml(p, "von", "http://127.0.0.1:8000/v1/systemone")
    assert p.read_text().startswith("# a comment\n")


def test_set_policy_backend(tmp_path):
    p = tmp_path / "v.yaml"
    p.write_text(
        "decisions:\n"
        "  - name: a\n"
        "    policy:\n"
        "      backend: mock\n"
        "  - name: b\n"
        "    policy:\n"
        "      backend: mock\n"
    )
    assert set_policy_backend(p, "von") == 2
    assert p.read_text().count("      backend: von") == 2


# -- manager -----------------------------------------------------------------


def test_manager_unknown_backend():
    with pytest.raises(KeyError):
        ServerManager("nope")


def test_manager_stop_when_nothing_running(tmp_path, monkeypatch):
    monkeypatch.setattr("verdict.servers.home",
                        lambda: tmp_path / ".verdict" / "servers")
    assert ServerManager("von").stop() is False
    assert not ServerManager("von").is_running()


def test_manager_start_manual_backend_raises(tmp_path, monkeypatch):
    monkeypatch.setattr("verdict.servers.home",
                        lambda: tmp_path / ".verdict" / "servers")
    with pytest.raises(RuntimeError, match="no automatic setup"):
        ServerManager("clef").start()
