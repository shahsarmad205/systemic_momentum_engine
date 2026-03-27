from __future__ import annotations

import os

from utils import run_context


def test_get_run_id_respects_env(monkeypatch):
    monkeypatch.setenv("RUN_ID", "fixed-id")
    assert run_context.get_run_id() == "fixed-id"


def test_context_dict_has_keys():
    d = run_context.context_dict(dry_run=True)
    assert "run_id" in d
    assert "git_sha" in d
    assert d["dry_run"] is True


def test_ensure_run_id_in_env_sets_stable(monkeypatch):
    monkeypatch.delenv("RUN_ID", raising=False)
    monkeypatch.delenv("CORRELATION_ID", raising=False)
    rid = run_context.ensure_run_id_in_env()
    assert rid
    assert os.environ.get("RUN_ID") == rid
    assert run_context.ensure_run_id_in_env() == rid
