"""Launcher of the lab: argument parsing, start-up messages, clean shutdown."""

from __future__ import annotations

import re
import socket
import webbrowser
from pathlib import Path
from unittest import mock

import pytest

from rl_fun.lab import cli

CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)
ADDRESS = re.compile(r"http://127\.0\.0\.1:(\d+)")


def _run_quick(tmp_path: Path, *extra: str) -> list[str]:
    return ["--port", "0", "--max-seconds", "1", "--runs-dir", str(tmp_path / "runs"), *extra]


# ---- arguments -------------------------------------------------------------------------------


def test_defaults() -> None:
    args = cli.build_parser().parse_args([])

    assert args.port == 8765
    assert args.runs_dir == "runs/lab"
    assert args.no_browser is False
    assert args.max_seconds is None


def test_all_options_are_parsed() -> None:
    args = cli.build_parser().parse_args(
        ["--port", "0", "--runs-dir", "x/y", "--no-browser", "--max-seconds", "2.5"]
    )

    assert args.port == 0
    assert args.runs_dir == "x/y"
    assert args.no_browser is True
    assert args.max_seconds == 2.5


@pytest.mark.parametrize("bad", [["--port", "-1"], ["--port", "70000"], ["--port", "abc"]])
def test_invalid_port_is_rejected_with_exit_code_2(bad: list[str]) -> None:
    with pytest.raises(SystemExit) as info:
        cli.build_parser().parse_args(bad)

    assert info.value.code == 2


# ---- running ---------------------------------------------------------------------------------


def test_runs_and_exits_with_zero_printing_the_address_in_russian(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with mock.patch.object(webbrowser, "open") as opened:
        code = cli.main(_run_quick(tmp_path, "--no-browser"))

    out = capsys.readouterr().out
    assert code == 0
    assert ADDRESS.search(out)
    assert int(ADDRESS.search(out).group(1)) > 0  # type: ignore[union-attr]
    assert CYRILLIC.search(out)
    opened.assert_not_called()


def test_browser_is_opened_with_the_printed_address_without_no_browser(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with mock.patch.object(webbrowser, "open") as opened:
        code = cli.main(_run_quick(tmp_path))

    match = ADDRESS.search(capsys.readouterr().out)
    assert code == 0
    assert match is not None
    opened.assert_called_once_with(match.group(0))


def test_keyboard_interrupt_closes_the_server_and_returns_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with mock.patch.object(webbrowser, "open", side_effect=KeyboardInterrupt):
        code = cli.main(["--port", "0", "--runs-dir", str(tmp_path / "runs")])

    match = ADDRESS.search(capsys.readouterr().out)
    assert code == 0
    assert match is not None
    with pytest.raises(OSError):  # the port was released
        socket.create_connection(("127.0.0.1", int(match.group(1))), timeout=2.0).close()


def test_busy_port_gives_russian_error_and_exit_code_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    blocker = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        blocker.bind(("127.0.0.1", 0))
        blocker.listen()
        port = blocker.getsockname()[1]
        with mock.patch.object(webbrowser, "open") as opened:
            code = cli.main(
                ["--port", str(port), "--runs-dir", str(tmp_path / "runs"), "--max-seconds", "1"]
            )
    finally:
        blocker.close()

    captured = capsys.readouterr()
    assert code == 2
    assert str(port) in captured.err
    assert CYRILLIC.search(captured.err)
    assert "Traceback" not in captured.err
    opened.assert_not_called()
