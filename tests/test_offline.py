"""The package must run with no network access: audits, reports and the CLI open no sockets (1.0 offline guarantee)."""
import socket

import pytest

from baseline_increment.cli import main


@pytest.fixture
def no_network(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("baseline-increment attempted a network connection")
    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def test_cli_audit_and_report_offline(no_network, tmp_path):
    out = tmp_path / "audit.html"
    rc = main(["examples/synthetic_ragstyle.csv", "--score", "detector_score", "--label", "hallucinated",
               "--group", "response_id", "--baselines", "task_type,generator,log_words", "--fast",
               "--report", str(out), "--json", str(tmp_path / "audit.json")])
    assert rc == 0 and out.stat().st_size > 0
