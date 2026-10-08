"""Command line launcher of the lab: starts the local server and waits for Ctrl+C."""

from __future__ import annotations

import argparse
import sys
import threading
import time
import webbrowser
from collections.abc import Sequence

from rl_fun.lab.server import make_server

DEFAULT_PORT = 8765
DEFAULT_RUNS_DIR = "runs/lab"
_WAIT_SLICE = 0.2


def _port(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        message = f"порт должен быть целым числом, получено {text!r}"
        raise argparse.ArgumentTypeError(message) from None
    if not 0 <= value <= 65535:
        raise argparse.ArgumentTypeError(f"порт должен быть от 0 до 65535, получено {value}")
    return value


def _seconds(text: str) -> float:
    try:
        value = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"нужно число секунд, получено {text!r}") from None
    if not value > 0:
        raise argparse.ArgumentTypeError("число секунд должно быть больше нуля")
    return value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="lab.py", description="Лаборатория машинок: локальный сервер и веб-интерфейс."
    )
    parser.add_argument(
        "--port",
        type=_port,
        default=DEFAULT_PORT,
        help=f"порт на 127.0.0.1 (по умолчанию {DEFAULT_PORT}; 0 — любой свободный)",
    )
    parser.add_argument(
        "--runs-dir",
        default=DEFAULT_RUNS_DIR,
        help=f"каталог для артефактов запусков (по умолчанию {DEFAULT_RUNS_DIR})",
    )
    parser.add_argument(
        "--no-browser", action="store_true", help="не открывать браузер автоматически"
    )
    parser.add_argument(
        "--max-seconds",
        type=_seconds,
        default=None,
        help="служебный параметр: остановиться через указанное число секунд",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the lab until Ctrl+C (or `--max-seconds`). Returns the process exit code."""
    args = build_parser().parse_args(argv)
    try:
        server = make_server(port=args.port, runs_dir=args.runs_dir)
    except OSError as error:
        print(
            f"Не удалось запустить лабораторию на порту {args.port}: порт занят или недоступен "
            f"({error.strerror or error}). Закройте другую копию лаборатории или выберите "
            "другой порт, например: --port 0",
            file=sys.stderr,
        )
        return 2
    try:
        server.serve_in_thread()
        print(f"Лаборатория запущена: {server.url}")
        print(f"Артефакты запусков: {args.runs_dir}")
        print("Остановить: Ctrl+C")
        if not args.no_browser:
            webbrowser.open(server.url)
        _wait(args.max_seconds)
    except KeyboardInterrupt:
        print("\nОстановка…")
    finally:
        server.close()
    return 0


def _wait(max_seconds: float | None) -> None:
    deadline = None if max_seconds is None else time.monotonic() + max_seconds
    stop = threading.Event()
    while deadline is None or time.monotonic() < deadline:
        stop.wait(_WAIT_SLICE)  # short slices keep Ctrl+C responsive on Windows
