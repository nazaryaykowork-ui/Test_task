"""Clean exits instead of tracebacks, and PII-free library logs."""

import logging
from types import SimpleNamespace

import pytest
from google.auth.exceptions import DefaultCredentialsError

from agent import cli


def run_cli(monkeypatch, *args: str) -> str:
    monkeypatch.setattr("sys.argv", ["agent.cli", *args])
    with pytest.raises(SystemExit) as exit_:
        cli.main()
    return str(exit_.value.code)


def test_missing_env_settings_exit_with_a_clear_message(monkeypatch):
    monkeypatch.setattr("agent.config.load_dotenv", lambda *_: None)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert run_cli(monkeypatch, "--user", "alice") == (
        "Cannot start: GEMINI_API_KEY is not set. Copy .env.example to .env and fill it in."
    )


def test_missing_google_credentials_exit_with_a_clear_message(monkeypatch, cfg):
    def no_credentials(_):
        raise DefaultCredentialsError("File not found")

    monkeypatch.setattr(cli, "load_config", lambda: cfg)
    monkeypatch.setattr(cli, "setup_logging", lambda _: None)
    monkeypatch.setattr(cli, "open_graph", no_credentials)
    assert run_cli(monkeypatch, "--user", "alice") == cli.NO_CREDENTIALS


def test_ctrl_c_during_a_turn_ends_the_chat_cleanly(monkeypatch, capsys):
    def interrupted(_):
        raise KeyboardInterrupt

    monkeypatch.setattr("builtins.input", lambda _: "Total revenue?")
    cli.chat_loop(SimpleNamespace(user_id="alice", thread_id="t1", send=interrupted))
    assert "Bye. Resume this conversation with --thread t1" in capsys.readouterr().out


def test_library_logs_are_masked():
    record = logging.LogRecord("google", logging.ERROR, __file__, 1, "Query by jane@example.com failed", None, None)
    assert cli.MaskingFormatter("%(message)s").format(record) == "Query by [EMAIL] failed"
