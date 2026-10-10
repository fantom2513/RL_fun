"""HTTP server of the lab: static files, JSON API and SSE on the loopback interface only.

Security: binds 127.0.0.1, accepts only `Host: localhost|127.0.0.1:<server port>` (guards against
DNS rebinding), serves files only from `web/`, caps request bodies at 1 MiB (checked from
Content-Length before anything is read) and never puts exception details in a response.
"""

from __future__ import annotations

import json
import logging
import queue
import select
import socket
import threading
import time
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

from rl_fun.lab.catalog import build_catalog, build_track
from rl_fun.lab.manager import RunManager
from rl_fun.lab.tracks import TrackExistsError, delete_track, list_tracks, save_track
from rl_fun.racing.track import available_tracks

LOGGER = logging.getLogger(__name__)

WEB_DIR = Path(__file__).resolve().parent / "web"
MAX_BODY_BYTES = 1024 * 1024
KEEPALIVE_SECONDS = 15.0
DRAIN_LIMIT_BYTES = 4 * 1024 * 1024
DRAIN_SECONDS = 2.0
REQUEST_TIMEOUT = 30.0
_STREAM_QUEUE_SIZE = 8
_PUMP_IDLE_SECONDS = 0.5
_SSE_KINDS = ("frame", "gen", "status", "notice")
_END = object()

CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".mjs": "text/javascript; charset=utf-8",
    ".svg": "image/svg+xml",
    ".json": "application/json; charset=utf-8",
    ".woff2": "font/woff2",
}
_STATUS_TEXT = {
    400: "некорректный запрос",
    403: "доступ запрещён",
    404: "не найдено",
    405: "метод не поддерживается",
    408: "время ожидания запроса истекло",
    409: "конфликт: такое имя уже занято",
    411: "нужен заголовок Content-Length",
    413: "тело запроса слишком большое (не более 1 МБ)",
    414: "слишком длинный адрес",
    431: "слишком большие заголовки",
    500: "внутренняя ошибка сервера",
    501: "метод не поддерживается",
    503: "сервер останавливается",
}


class _HttpError(Exception):
    """An error with a status code and a Russian message that is safe to show to the client."""

    def __init__(self, status: int, message: str | None = None) -> None:
        super().__init__(message or _STATUS_TEXT.get(status, "ошибка запроса"))
        self.status = status
        self.message = message or _STATUS_TEXT.get(status, "ошибка запроса")


class LabServer(ThreadingHTTPServer):
    """Threading HTTP server bound to the loopback interface, owning a `RunManager`."""

    daemon_threads = True  # a dead client or a stuck stream must never block shutdown
    allow_reuse_address = False  # on Windows SO_REUSEADDR would let another process share the port
    request_queue_size = 32

    def __init__(self, port: int, manager: RunManager, tracks_dir: Path | None = None) -> None:
        self.manager = manager
        self.tracks_dir = tracks_dir if tracks_dir is not None else manager.tracks_dir
        self.track_names = available_tracks(self.tracks_dir)
        self._catalog = build_catalog(self.track_names)
        self._serving = False
        self._closed = False
        self._close_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        super().__init__(("127.0.0.1", port), _Handler)

    @property
    def port(self) -> int:
        return int(self.server_address[1])

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def catalog(self) -> dict[str, Any]:
        return self._catalog

    def refresh_tracks(self) -> None:
        """Rebuild the catalog after a track was saved or deleted."""
        self.track_names = available_tracks(self.tracks_dir)
        self._catalog = build_catalog(self.track_names)

    def serve_forever(self, poll_interval: float = 0.1) -> None:
        self._serving = True
        super().serve_forever(poll_interval)

    def serve_in_thread(self) -> threading.Thread:
        """Start serving in a daemon thread and return it."""
        self._serving = True
        thread = threading.Thread(target=self.serve_forever, name="lab-server", daemon=True)
        self._thread = thread
        thread.start()
        return thread

    def close(self) -> None:
        """Stop serving, release the port and shut the run manager (and its workers) down."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        try:
            if self._serving:
                self.shutdown()
            if self._thread is not None:
                self._thread.join(5.0)
            self.server_close()
        finally:
            self.manager.shutdown()


def make_server(
    port: int = 0,
    runs_dir: str | Path = "runs/lab",
    manager: RunManager | None = None,
) -> LabServer:
    """Create a server on 127.0.0.1:`port` (0 = any free port).

    Without a `manager` a new `RunManager(runs_dir)` is created. Either way `close()` shuts the
    manager down.
    """
    manager = manager if manager is not None else RunManager(runs_dir)
    return LabServer(port, manager, getattr(manager, "tracks_dir", Path(runs_dir) / "tracks"))


class _Handler(BaseHTTPRequestHandler):
    server: LabServer
    server_version = "RLFunLab"
    sys_version = ""
    timeout = REQUEST_TIMEOUT

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._body_length = 0
        self._body_read = False
        super().__init__(*args, **kwargs)

    # ---- plumbing ----------------------------------------------------------------------------

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        LOGGER.debug("%s - %s", self.address_string(), format % args)

    def send_error(  # type: ignore[override]
        self, code: int, message: str | None = None, explain: str | None = None
    ) -> None:
        """Errors raised by the base class (bad request line, unknown method) are JSON too."""
        self.close_connection = True
        try:
            self._send_json(code, {"error": _STATUS_TEXT.get(code, "ошибка запроса")})
        except OSError:
            pass

    def do_GET(self) -> None:  # noqa: N802
        self._dispatch("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._dispatch("POST")

    def do_DELETE(self) -> None:  # noqa: N802
        self._dispatch("DELETE")

    def do_PUT(self) -> None:  # noqa: N802
        self._dispatch("PUT")

    do_PATCH = do_PUT  # noqa: N815
    do_OPTIONS = do_PUT  # noqa: N815
    do_HEAD = do_PUT  # noqa: N815

    def _dispatch(self, method: str) -> None:
        self.close_connection = True
        try:
            try:
                self._check_host()
                self._route(method)
            except _HttpError as error:
                self._send_json(error.status, {"error": error.message})
            except (BrokenPipeError, ConnectionError, TimeoutError):
                pass
            except Exception:
                LOGGER.exception("lab request failed: %s %s", method, self.path)
                try:
                    self._send_json(500, {"error": _STATUS_TEXT[500]})
                except OSError:
                    pass
        finally:
            self._drain_unread_body()

    def _check_host(self) -> None:
        port = self.server.port
        allowed = {f"localhost:{port}", f"127.0.0.1:{port}"}
        host = (self.headers.get("Host") or "").strip().lower()
        if host not in allowed:
            raise _HttpError(403)

    def _route(self, method: str) -> None:
        path = urlsplit(self.path).path
        if path.startswith("/api/"):
            self._route_api(method, path[len("/api/") :], urlsplit(self.path).query)
        elif method == "GET" and (path in ("/", "/index.html") or path.startswith("/static/")):
            self._serve_static(path)
        elif path in ("/", "/index.html") or path.startswith("/static/"):
            raise _HttpError(405)
        else:
            raise _HttpError(404)

    # ---- API ---------------------------------------------------------------------------------

    def _route_api(self, method: str, path: str, query: str) -> None:
        manager = self.server.manager
        parts = [unquote(part) for part in path.split("/")]
        if parts == ["catalog"]:
            self._expect(method, "GET")
            self._send_json(200, self.server.catalog)
        elif parts == ["tracks"]:
            if method == "GET":
                self._send_json(200, list_tracks(self.server.tracks_dir))
            elif method == "POST":
                self._save_track()
            else:
                raise _HttpError(405)
        elif len(parts) == 2 and parts[0] == "tracks":
            if method == "DELETE":
                self._delete_track(parts[1])
                return
            self._expect(method, "GET")
            try:
                data = build_track(parts[1], self.server.tracks_dir)
            except ValueError:
                raise _HttpError(404, "трасса не найдена") from None
            self._send_json(200, data)
        elif parts == ["garage"]:
            self._expect(method, "GET")
            self._send_json(200, manager.garage())
        elif len(parts) == 3 and parts[0] == "garage" and parts[2] == "export":
            self._expect(method, "GET")
            self._send_json(200, self._found(lambda: manager.export(parts[1])))
        elif parts == ["demos"]:
            self._expect(method, "POST")
            self._create_demo()
        elif parts == ["runs"]:
            if method == "GET":
                self._send_json(200, manager.list())
            elif method == "POST":
                self._create_run()
            else:
                raise _HttpError(405)
        elif len(parts) == 2 and parts[0] == "runs":
            if method == "GET":
                self._send_json(200, self._found(lambda: manager.get(parts[1])))
            elif method == "DELETE":
                self._found(lambda: manager.delete(parts[1]))
                self._send_json(200, {"ok": True})
            else:
                raise _HttpError(405)
        elif len(parts) == 3 and parts[0] == "runs" and parts[2] == "command":
            self._expect(method, "POST")
            self._command(parts[1])
        elif len(parts) == 3 and parts[0] == "runs" and parts[2] == "stream":
            self._expect(method, "GET")
            self._stream(parts[1], query)
        else:
            raise _HttpError(404)

    @staticmethod
    def _expect(method: str, expected: str) -> None:
        if method != expected:
            raise _HttpError(405)

    @staticmethod
    def _found(action: Any) -> Any:
        try:
            return action()
        except KeyError:
            raise _HttpError(404, "запуск не найден") from None

    def _save_track(self) -> None:
        data = self._read_json_object()
        overwrite = data.pop("overwrite", False) is True
        try:
            row = save_track(self.server.tracks_dir, data, overwrite=overwrite)
        except TrackExistsError as error:
            raise _HttpError(409, str(error)) from None
        except ValueError as error:
            raise _HttpError(400, str(error)) from None
        except OSError:
            raise _HttpError(500, "не удалось сохранить трассу на диск") from None
        self.server.refresh_tracks()
        self._send_json(201, {"name": row["name"]})

    def _delete_track(self, name: str) -> None:
        try:
            delete_track(self.server.tracks_dir, name)
        except ValueError as error:
            raise _HttpError(400, str(error)) from None
        except KeyError:
            raise _HttpError(404, "трасса не найдена") from None
        self.server.refresh_tracks()
        self._send_json(200, {"ok": True})

    def _create_demo(self) -> None:
        data = self._read_json_object()
        run_id, track = data.get("run"), data.get("track")
        if not isinstance(run_id, str) or not isinstance(track, str):
            raise _HttpError(400, "нужны поля run (запуск) и track (трасса)")
        options = {key: data[key] for key in ("speed", "laps", "max_steps") if key in data}
        try:
            demo_id = self.server.manager.create_demo(run_id, track, **options)
        except KeyError:
            raise _HttpError(404, "запуск или его сеть не найдены") from None
        except ValueError as error:
            raise _HttpError(400, str(error)) from None
        except RuntimeError:
            raise _HttpError(503) from None
        self._send_json(201, {"id": demo_id})

    def _create_run(self) -> None:
        config = self._read_json_object()
        try:
            run_id = self.server.manager.create(config)
        except ValueError as error:
            raise _HttpError(400, str(error)) from None
        except RuntimeError:
            raise _HttpError(503) from None
        self._send_json(201, {"id": run_id})

    def _command(self, run_id: str) -> None:
        command = self._read_json_object()
        try:
            self.server.manager.command(run_id, command)
        except KeyError:
            raise _HttpError(404, "запуск не найден") from None
        except ValueError as error:
            raise _HttpError(400, str(error)) from None
        self._send_json(200, {"ok": True})

    # ---- request body ------------------------------------------------------------------------

    def _read_json_object(self) -> dict[str, Any]:
        if self.headers.get("Transfer-Encoding"):
            raise _HttpError(400, "передача частями не поддерживается")
        raw_length = (self.headers.get("Content-Length") or "0").strip()
        if not raw_length.isascii() or not raw_length.isdigit():
            raise _HttpError(400, "некорректный заголовок Content-Length")
        length = int(raw_length)
        self._body_length = length
        if length > MAX_BODY_BYTES:
            raise _HttpError(413)
        data = self.rfile.read(length) if length else b""
        self._body_read = True
        if len(data) != length:
            raise _HttpError(400, "тело запроса получено не полностью")
        try:
            value = json.loads(data.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise _HttpError(400, "тело запроса должно быть корректным JSON в UTF-8") from None
        if not isinstance(value, dict):
            raise _HttpError(400, "тело запроса должно быть JSON-объектом")
        return value

    def _drain_unread_body(self) -> None:
        """After an early error, swallow (a bounded amount of) the unread body.

        Closing a socket with unread incoming data resets the connection on some systems, which
        could destroy the response we just sent.
        """
        if self._body_read:
            return
        if not self._body_length:
            raw = (self.headers.get("Content-Length") or "") if hasattr(self, "headers") else ""
            self._body_length = int(raw) if raw.isascii() and raw.isdigit() else 0
        remaining = min(self._body_length, DRAIN_LIMIT_BYTES)
        deadline = time.monotonic() + DRAIN_SECONDS
        try:
            self.connection.settimeout(0.5)
            while remaining > 0 and time.monotonic() < deadline:
                chunk = self.connection.recv(min(65536, remaining))
                if not chunk:
                    break
                remaining -= len(chunk)
        except (OSError, ValueError):
            pass

    # ---- responses ---------------------------------------------------------------------------

    def _send_bytes(self, status: int, content_type: str, data: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, status: int, payload: Any) -> None:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self._send_bytes(status, "application/json; charset=utf-8", data)

    # ---- static files ------------------------------------------------------------------------

    def _serve_static(self, path: str) -> None:
        relative = "index.html" if path in ("/", "/index.html") else path[len("/static/") :]
        file = _resolve_static(relative)
        content_type = CONTENT_TYPES.get(file.suffix.lower()) if file is not None else None
        if file is None or content_type is None:
            raise _HttpError(404)
        try:
            data = file.read_bytes()
        except OSError:
            raise _HttpError(404) from None
        self._send_bytes(200, content_type, data)

    # ---- SSE ---------------------------------------------------------------------------------

    def _stream(self, run_id: str, query: str) -> None:
        from_index = 0
        for pair in query.split("&"):
            name, _, value = pair.partition("=")
            if name == "from" and value.isascii() and value.isdigit():
                from_index = int(value)
        events = self._found(
            lambda: self.server.manager.subscribe(
                run_id, from_index, idle_timeout=_PUMP_IDLE_SECONDS
            )
        )

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Accel-Buffering", "no")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.flush()

        feed: queue.Queue[Any] = queue.Queue(maxsize=_STREAM_QUEUE_SIZE)
        stop = threading.Event()
        pump = threading.Thread(
            target=_pump, args=(events, feed, stop), name=f"lab-sse-{run_id}", daemon=True
        )
        pump.start()
        last_write = time.monotonic()
        try:
            while True:
                try:
                    item = feed.get(timeout=1.0)
                except queue.Empty:
                    if _client_gone(self.connection):
                        return
                    if time.monotonic() - last_write >= KEEPALIVE_SECONDS:
                        self.wfile.write(b": keep-alive\n\n")
                        last_write = time.monotonic()
                    continue
                if item is _END:
                    return
                kind, payload = item
                text = json.dumps(payload, separators=(",", ":"))
                self.wfile.write(f"event: {kind}\ndata: {text}\n\n".encode())
                last_write = time.monotonic()
        finally:
            stop.set()


def _pump(events: Iterator[tuple[str, dict[str, Any]]], feed: queue.Queue[Any], stop: Any) -> None:
    """Move events from the manager into a small queue; stops when the handler is gone.

    The manager yields an `("idle", None)` item at least every `_PUMP_IDLE_SECONDS` on a quiet
    run; it is never forwarded, it only lets this loop notice `stop` and exit.
    """

    def put(item: Any) -> bool:
        while not stop.is_set():
            try:
                feed.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    try:
        for kind, payload in events:
            if stop.is_set():
                return
            if kind in _SSE_KINDS and not put((kind, payload)):
                return
        put(_END)
    except Exception:
        LOGGER.exception("lab event stream failed")
        put(_END)
    finally:
        close = getattr(events, "close", None)
        if close is not None:
            try:
                close()
            except Exception:  # noqa: BLE001
                pass


def _client_gone(connection: socket.socket) -> bool:
    """True if the peer closed its end (readable socket that yields EOF) or the socket broke."""
    try:
        readable, _, _ = select.select([connection], [], [], 0)
        if not readable:
            return False
        return connection.recv(1, socket.MSG_PEEK) == b""
    except (OSError, ValueError):
        return True


def _resolve_static(relative: str) -> Path | None:
    """Map a request path below /static/ to a file inside `web/`, or None if it is not allowed."""
    text = unquote(relative)
    if not text or "\x00" in text or "\\" in text or ":" in text:
        return None
    segments = text.split("/")
    if any(segment in ("..", ".") for segment in segments):
        return None
    root = WEB_DIR.resolve()
    try:
        candidate = (root / text.lstrip("/")).resolve()
    except (OSError, ValueError):
        return None
    if not candidate.is_relative_to(root) or not candidate.is_file():
        return None
    return candidate
