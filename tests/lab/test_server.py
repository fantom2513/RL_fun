"""HTTP server of the lab: static files, JSON API, SSE and the security checks around them."""

from __future__ import annotations

import http.client
import json
import re
import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pytest

from rl_fun.lab.manager import RunManager
from rl_fun.lab.server import LabServer, make_server

from . import manager_targets as targets

TIMEOUT = 90.0
MAX_BODY = 1024 * 1024
CYRILLIC = re.compile(r"[а-яё]", re.IGNORECASE)


def _config(**overrides: Any) -> dict[str, Any]:
    config: dict[str, Any] = {
        "name": "t",
        "track": "oval",
        "population": 8,
        "elite": 2,
        "max_steps": 60,
        "generations": 1,
    }
    config.update(overrides)
    return config


def wait_for(condition: Callable[[], Any], timeout: float = TIMEOUT) -> Any:
    deadline = time.monotonic() + timeout
    while True:
        value = condition()
        if value:
            return value
        if time.monotonic() > deadline:
            pytest.fail("condition was not met in time")
        time.sleep(0.05)


def request(
    server: LabServer,
    method: str,
    path: str,
    body: bytes | dict[str, Any] | list[Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, dict[str, str], bytes]:
    """One request; returns status, lower-cased headers and the whole body."""
    if isinstance(body, dict | list):
        body = json.dumps(body).encode("utf-8")
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=TIMEOUT)
    try:
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        data = response.read()
        return response.status, {k.lower(): v for k, v in response.getheaders()}, data
    finally:
        connection.close()


def call_json(
    server: LabServer, method: str, path: str, body: Any = None
) -> tuple[int, dict[str, Any]]:
    status, headers, data = request(server, method, path, body)
    assert headers["content-type"].startswith("application/json")
    return status, json.loads(data.decode("utf-8"))


def read_events(
    server: LabServer, path: str, stop_after: str | None = None
) -> tuple[http.client.HTTPResponse, list[tuple[str, dict[str, Any]]], bool]:
    """Read an SSE stream until `stop_after` arrived or the server closed it.

    Returns the response, the events read and whether the stream was closed by the server.
    """
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=TIMEOUT)
    connection.request("GET", path)
    response = connection.getresponse()
    events: list[tuple[str, dict[str, Any]]] = []
    kind = ""
    while True:
        line = response.fp.readline()
        if not line:
            connection.close()
            return response, events, True
        text = line.decode("utf-8").rstrip("\r\n")
        if text.startswith("event: "):
            kind = text[len("event: ") :]
        elif text.startswith("data: "):
            events.append((kind, json.loads(text[len("data: ") :])))
            if kind == stop_after:
                connection.close()
                return response, events, False


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[LabServer]:
    runs = tmp_path_factory.mktemp("runs")
    lab = make_server(port=0, runs_dir=runs, manager=RunManager(runs))
    lab.serve_in_thread()
    try:
        yield lab
    finally:
        lab.close()


@contextmanager
def started_run(server: LabServer, **overrides: Any) -> Iterator[str]:
    status, data = call_json(server, "POST", "/api/runs", _config(**overrides))
    assert status == 201
    run_id = data["id"]
    try:
        yield run_id
    finally:
        request(server, "DELETE", f"/api/runs/{run_id}")


def handler_threads() -> int:
    names = [t.name for t in threading.enumerate()]
    return sum(1 for name in names if "process_request_thread" in name or "lab-sse" in name)


# ---- binding and basics ----------------------------------------------------------------------


def test_server_binds_loopback_only(server: LabServer) -> None:
    assert server.server_address[0] == "127.0.0.1"
    assert server.url == f"http://127.0.0.1:{server.port}"
    assert server.port > 0


def test_handler_threads_are_daemon(server: LabServer) -> None:
    assert server.daemon_threads is True


def test_index_is_html(server: LabServer) -> None:
    status, headers, body = request(server, "GET", "/")

    assert status == 200
    assert headers["content-type"].startswith("text/html")
    assert b"/static/app.js" in body


def test_static_script_is_javascript(server: LabServer) -> None:
    status, headers, body = request(server, "GET", "/static/app.js")

    assert status == 200
    assert headers["content-type"].startswith("text/javascript")
    assert body


def test_missing_static_file_is_404(server: LabServer) -> None:
    assert request(server, "GET", "/static/nothing.js")[0] == 404


@pytest.mark.parametrize(
    "path",
    [
        "/static/../config.py",
        "/static/%2e%2e/config.py",
        "/static/%2E%2E/config.py",
        "/static/..%5cconfig.py",
        "/static/..\\config.py",
        "/static/%2e%2e%5cconfig.py",
        "/static/%252e%252e/config.py",
        "/static/../../../../pyproject.toml",
        "/static/%2e%2e/%2e%2e/%2e%2e/pyproject.toml",
        "/static/C:/Windows/win.ini",
        "/static/C:\\Windows\\win.ini",
        "/static/app.js%00.html",
        "/static//config.py",
        "/..%2fconfig.py",
        "/static/",
    ],
)
def test_static_path_traversal_is_blocked(server: LabServer, path: str) -> None:
    status, _, body = request(server, "GET", path)

    assert status == 404
    assert b"RunConfig" not in body
    assert b"[project]" not in body


def test_static_content_types(server: LabServer) -> None:
    web = Path(__file__).resolve().parents[2] / "src" / "rl_fun" / "lab" / "web"
    probe = web / "_probe_test.svg"
    probe.write_text("<svg xmlns='http://www.w3.org/2000/svg'/>", encoding="utf-8")
    try:
        status, headers, _ = request(server, "GET", "/static/_probe_test.svg")
    finally:
        probe.unlink()

    assert status == 200
    assert headers["content-type"].startswith("image/svg+xml")


def _probe(server: LabServer, relative: str, data: bytes) -> tuple[int, dict[str, str], bytes]:
    """Serve a temporary file placed at `relative` below web/ and remove it (and new dirs) after."""
    web = Path(__file__).resolve().parents[2] / "src" / "rl_fun" / "lab" / "web"
    probe = web / relative
    created = [d for d in reversed(probe.relative_to(web).parents) if not (web / d).exists()]
    probe.parent.mkdir(parents=True, exist_ok=True)
    probe.write_bytes(data)
    try:
        return request(server, "GET", f"/static/{relative}")
    finally:
        probe.unlink()
        for directory in reversed(created):
            (web / directory).rmdir()


def test_static_font_is_served_as_woff2(server: LabServer) -> None:
    status, headers, body = _probe(server, "_probe_test.woff2", b"wOF2probe")

    assert (status, headers["content-type"], body) == (200, "font/woff2", b"wOF2probe")


def test_static_files_in_nested_subdirectories_are_served(server: LabServer) -> None:
    status, headers, body = _probe(server, "_probe_dir/inner/probe.css", b"a{}")

    assert (status, headers["content-type"].split(";")[0], body) == (200, "text/css", b"a{}")


@pytest.mark.parametrize(
    ("path", "content_type"),
    [
        ("/static/css/tokens.css", "text/css"),
        ("/static/css/base.css", "text/css"),
        ("/static/icons.svg", "image/svg+xml"),
        ("/static/fonts/onest-cyrillic-wght-normal.woff2", "font/woff2"),
    ],
)
def test_design_system_assets_are_served(server: LabServer, path: str, content_type: str) -> None:
    status, headers, body = request(server, "GET", path)

    assert (status, headers["content-type"].split(";")[0]) == (200, content_type)
    assert body


@pytest.mark.parametrize("name", ["_probe_test.md","_probe_test.txt", "_probe_test.py"])
def test_static_files_of_unlisted_types_are_404(server: LabServer, name: str) -> None:
    assert _probe(server, name, b"secret")[0] == 404


# ---- host header -----------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/", "/static/app.js", "/api/catalog", "/api/runs"])
def test_foreign_host_is_403(server: LabServer, path: str) -> None:
    status, _, body = request(server, "GET", path, headers={"Host": "evil.example"})

    assert status == 403
    assert b"Traceback" not in body


@pytest.mark.parametrize(
    "host",
    ["localhost", "127.0.0.1", "127.0.0.1:1", "evil.example:{port}", "localhost:1", "[::1]:{port}"],
)
def test_wrong_host_or_port_is_403(server: LabServer, host: str) -> None:
    status, _, _ = request(
        server, "GET", "/api/catalog", headers={"Host": host.format(port=server.port)}
    )

    assert status == 403


@pytest.mark.parametrize("host", ["localhost:{port}", "127.0.0.1:{port}"])
def test_allowed_hosts_pass(server: LabServer, host: str) -> None:
    status, _, _ = request(
        server, "GET", "/api/catalog", headers={"Host": host.format(port=server.port)}
    )

    assert status == 200


# ---- read-only API ---------------------------------------------------------------------------


def test_catalog_endpoint(server: LabServer) -> None:
    status, data = call_json(server, "GET", "/api/catalog")

    assert status == 200
    assert {"inputs", "outputs", "activations", "fitness", "defaults", "tracks", "style"} <= set(
        data
    )
    assert "oval" in data["tracks"]


def test_catalog_endpoint_lists_the_learners_with_their_parameter_groups(
    server: LabServer,
) -> None:
    status, data = call_json(server, "GET", "/api/catalog")

    assert status == 200
    assert [learner["id"] for learner in data["learners"]] == ["evolution", "ppo"]
    ppo_keys = [
        param["key"] for group in data["learners"][1]["groups"] for param in group["params"]
    ]
    assert "ppo.learning_rate" in ppo_keys


def test_track_endpoint(server: LabServer) -> None:
    status, data = call_json(server, "GET", "/api/tracks/oval")

    assert status == 200
    assert data["name"] == "oval"
    assert data["centerline"] and data["left"] and data["right"]


@pytest.mark.parametrize("name", ["nope", "..%2Fracing", "C:%5Cx", "oval.json"])
def test_unknown_track_is_404(server: LabServer, name: str) -> None:
    status, data = call_json(server, "GET", f"/api/tracks/{name}")

    assert status == 404
    assert CYRILLIC.search(data["error"])


def test_unknown_route_is_404(server: LabServer) -> None:
    status, data = call_json(server, "GET", "/api/nothing")

    assert status == 404
    assert "error" in data


def test_unsupported_method_is_405(server: LabServer) -> None:
    status, data = call_json(server, "PUT", "/api/runs", {})

    assert status == 405
    assert "error" in data


# ---- creating runs and bad input -------------------------------------------------------------


def test_invalid_config_is_400_with_russian_error(server: LabServer) -> None:
    status, data = call_json(server, "POST", "/api/runs", _config(population=1))

    assert status == 400
    assert "population" in data["error"]
    assert CYRILLIC.search(data["error"])
    assert "Traceback" not in data["error"]


def test_unknown_config_key_is_400(server: LabServer) -> None:
    status, data = call_json(server, "POST", "/api/runs", _config(bogus=1))

    assert status == 400
    assert "bogus" in data["error"]


def test_broken_json_is_400(server: LabServer) -> None:
    status, data = call_json(server, "POST", "/api/runs", b'{"name": ')

    assert status == 400
    assert CYRILLIC.search(data["error"])


@pytest.mark.parametrize("body", [b"", b"[1, 2]", b'"text"', b"null", b"\xff\xfe\x00"])
def test_non_object_or_undecodable_body_is_400(server: LabServer, body: bytes) -> None:
    status, data = call_json(server, "POST", "/api/runs", body)

    assert status == 400
    assert CYRILLIC.search(data["error"])


def test_invalid_content_length_is_400(server: LabServer) -> None:
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=TIMEOUT)
    try:
        connection.putrequest("POST", "/api/runs")
        connection.putheader("Content-Length", "abc")
        connection.endheaders()
        response = connection.getresponse()
        data = json.loads(response.read())
    finally:
        connection.close()

    assert response.status == 400
    assert CYRILLIC.search(data["error"])


def test_body_over_limit_is_413_by_content_length_alone(server: LabServer) -> None:
    connection = http.client.HTTPConnection("127.0.0.1", server.port, timeout=TIMEOUT)
    try:
        connection.putrequest("POST", "/api/runs")
        connection.putheader("Content-Length", str(MAX_BODY + 1))
        connection.endheaders()  # the body is never sent: the server must not wait for it
        response = connection.getresponse()
        data = json.loads(response.read())
    finally:
        connection.close()

    assert response.status == 413
    assert CYRILLIC.search(data["error"])


def test_oversized_body_is_413(server: LabServer) -> None:
    status, data = call_json(server, "POST", "/api/runs", b" " * (MAX_BODY + 1))

    assert status == 413
    assert "error" in data


def test_body_at_the_limit_is_read_and_rejected_as_json(server: LabServer) -> None:
    status, data = call_json(server, "POST", "/api/runs", b" " * MAX_BODY)

    assert status == 400
    assert "error" in data


def test_create_list_and_get_run(server: LabServer) -> None:
    with started_run(server, name="alpha", generations=None) as run_id:
        rows = call_json(server, "GET", "/api/runs")[1]
        status, info = call_json(server, "GET", f"/api/runs/{run_id}")

        assert any(row["id"] == run_id and row["name"] == "alpha" for row in rows)
        assert status == 200
        assert info["id"] == run_id
        assert info["config"]["name"] == "alpha"
        assert info["config"]["track"] == "oval"
        assert "history" in info


def test_unknown_run_is_404(server: LabServer) -> None:
    status, data = call_json(server, "GET", "/api/runs/nope")

    assert status == 404
    assert CYRILLIC.search(data["error"])


# ---- commands and delete ---------------------------------------------------------------------


def test_commands(server: LabServer) -> None:
    with started_run(server, generations=None) as run_id:
        path = f"/api/runs/{run_id}/command"

        assert call_json(server, "POST", path, {"cmd": "pause"})[0] == 200
        assert call_json(server, "POST", path, {"cmd": "resume"})[0] == 200
        assert call_json(server, "POST", path, {"cmd": "speed", "value": "max"})[0] == 200
        status, data = call_json(server, "POST", path, {"cmd": "explode"})
        assert status == 400
        assert CYRILLIC.search(data["error"])
        assert call_json(server, "POST", path, {"cmd": "speed", "value": 3})[0] == 400
        assert call_json(server, "POST", path, [1])[0] == 400
        assert call_json(server, "POST", path, b"{")[0] == 400


def test_command_for_unknown_run_is_404(server: LabServer) -> None:
    status, data = call_json(server, "POST", "/api/runs/nope/command", {"cmd": "pause"})

    assert status == 404
    assert "error" in data


def test_delete_removes_run(server: LabServer) -> None:
    status, data = call_json(server, "POST", "/api/runs", _config(generations=None))
    run_id = data["id"]

    status, _ = call_json(server, "DELETE", f"/api/runs/{run_id}")

    assert status == 200
    assert all(row["id"] != run_id for row in call_json(server, "GET", "/api/runs")[1])
    assert call_json(server, "GET", f"/api/runs/{run_id}")[0] == 404


def test_delete_unknown_run_is_404(server: LabServer) -> None:
    assert call_json(server, "DELETE", "/api/runs/nope")[0] == 404


# ---- SSE -------------------------------------------------------------------------------------


def test_stream_delivers_events_and_closes_after_finish(server: LabServer) -> None:
    with started_run(server) as run_id:
        response, events, closed = read_events(server, f"/api/runs/{run_id}/stream")

    assert response.status == 200
    assert response.getheader("Content-Type").startswith("text/event-stream")
    assert response.getheader("Cache-Control") == "no-cache"
    assert closed is True
    kinds = [kind for kind, _ in events]
    assert "gen" in kinds
    assert set(kinds[: kinds.index("gen")]) & {"frame", "status"}
    assert set(kinds) <= {"frame", "gen", "status", "notice"}
    assert events[-1][0] == "status"
    assert events[-1][1]["status"] == "finished"
    for kind, payload in events:
        if kind in ("frame", "gen", "status"):
            assert payload["t"] == kind


def test_stream_stops_reading_at_requested_event(server: LabServer) -> None:
    with started_run(server, generations=None) as run_id:
        _, events, closed = read_events(server, f"/api/runs/{run_id}/stream", stop_after="gen")

    assert closed is False
    assert events[-1][0] == "gen"


def test_stream_of_unknown_run_is_404(server: LabServer) -> None:
    status, data = call_json(server, "GET", "/api/runs/nope/stream")

    assert status == 404
    assert "error" in data


def test_stream_client_disconnect_frees_threads(server: LabServer) -> None:
    wait_for(lambda: handler_threads() == 0)
    with started_run(server, generations=None) as run_id:
        sock = socket.create_connection(("127.0.0.1", server.port), timeout=TIMEOUT)
        try:
            sock.sendall(
                f"GET /api/runs/{run_id}/stream HTTP/1.1\r\nHost: 127.0.0.1:{server.port}\r\n\r\n"
                .encode()
            )
            assert b"event:" in _recv_until(sock, b"event:")
        finally:
            sock.close()

        wait_for(lambda: handler_threads() == 0, timeout=30.0)
        assert call_json(server, "GET", f"/api/runs/{run_id}")[1]["status"] in (
            "running",
            "paused",
        )


def sse_pump_threads() -> int:
    return sum(1 for t in threading.enumerate() if t.name.startswith("lab-sse-"))


def test_stream_disconnect_from_idle_run_stops_the_pump_thread(tmp_path: Path) -> None:
    manager = RunManager(tmp_path, worker_target=targets.idle_target)
    lab = make_server(port=0, runs_dir=tmp_path, manager=manager)
    lab.serve_in_thread()
    try:
        run_id = call_json(lab, "POST", "/api/runs", _config(generations=None))[1]["id"]
        for _ in range(3):  # repeated reconnects must not accumulate parked threads
            sock = socket.create_connection(("127.0.0.1", lab.port), timeout=TIMEOUT)
            try:
                sock.sendall(
                    f"GET /api/runs/{run_id}/stream HTTP/1.1\r\nHost: 127.0.0.1:{lab.port}\r\n\r\n"
                    .encode()
                )
                assert b"event:" in _recv_until(sock, b"event:")
                assert sse_pump_threads() >= 1
            finally:
                sock.close()

        wait_for(lambda: sse_pump_threads() == 0, timeout=5.0)
        assert call_json(lab, "GET", f"/api/runs/{run_id}")[1]["status"] == "running"
        assert manager.process(run_id).is_alive()
    finally:
        lab.close()


def _recv_until(sock: socket.socket, marker: bytes) -> bytes:
    data = b""
    while marker not in data:
        chunk = sock.recv(4096)
        if not chunk:
            break
        data += chunk
    return data


# ---- errors and shutdown ---------------------------------------------------------------------


class _BrokenManager:
    def __init__(self) -> None:
        self.shutdowns = 0

    def list(self) -> list[dict[str, Any]]:
        raise RuntimeError("secret detail C:\\internal\\path.py")

    def shutdown(self) -> None:
        self.shutdowns += 1


def test_internal_error_is_500_without_details(tmp_path: Path) -> None:
    manager = _BrokenManager()
    lab = make_server(port=0, runs_dir=tmp_path, manager=manager)  # type: ignore[arg-type]
    lab.serve_in_thread()
    try:
        status, _, body = request(lab, "GET", "/api/runs")
    finally:
        lab.close()

    assert status == 500
    text = body.decode("utf-8")
    assert "secret detail" not in text
    assert "Traceback" not in text
    assert ".py" not in text
    assert CYRILLIC.search(json.loads(text)["error"])


def test_close_shuts_down_manager_once_and_releases_port(tmp_path: Path) -> None:
    manager = _BrokenManager()
    lab = make_server(port=0, runs_dir=tmp_path, manager=manager)  # type: ignore[arg-type]
    lab.serve_in_thread()
    port = lab.port

    lab.close()
    lab.close()

    assert manager.shutdowns == 1
    with pytest.raises(OSError):
        socket.create_connection(("127.0.0.1", port), timeout=2.0).close()


def test_close_without_serving(tmp_path: Path) -> None:
    manager = _BrokenManager()
    lab = make_server(port=0, runs_dir=tmp_path, manager=manager)  # type: ignore[arg-type]

    lab.close()

    assert manager.shutdowns == 1


def test_close_stops_worker_processes(tmp_path: Path) -> None:
    manager = RunManager(tmp_path)
    lab = make_server(port=0, runs_dir=tmp_path, manager=manager)
    lab.serve_in_thread()
    try:
        status, data = call_json(lab, "POST", "/api/runs", _config(generations=None))
        assert status == 201
        process = manager.process(data["id"])
        assert process.is_alive()
    finally:
        lab.close()

    assert not process.is_alive()


def test_make_server_creates_default_manager(tmp_path: Path) -> None:
    lab = make_server(port=0, runs_dir=tmp_path)
    lab.serve_in_thread()
    try:
        status, rows = call_json(lab, "GET", "/api/runs")
    finally:
        lab.close()

    assert status == 200
    assert rows == []


# ---- custom tracks over HTTP -------------------------------------------------------------------


def _loop(name: str) -> dict[str, Any]:
    from tests.lab.test_tracks import ellipse

    return {"name": name, "width": 10.0, "centerline": ellipse()}


def test_tracks_endpoint_lists_built_in_and_saved_tracks(server: LabServer) -> None:
    status, data = call_json(server, "POST", "/api/tracks", _loop("Список"))
    assert status == 201 and data["name"] == "Список"

    status, rows = call_json(server, "GET", "/api/tracks")

    assert status == 200
    assert [row["name"] for row in rows[:3]] == ["circuit", "oval", "wavy"]
    mine = next(row for row in rows if row["name"] == "Список")
    assert mine["builtin"] is False and len(mine["centerline"]) == 24
    call_json(server, "DELETE", "/api/tracks/%D0%A1%D0%BF%D0%B8%D1%81%D0%BE%D0%BA")


def test_a_saved_track_reaches_the_catalog_and_serves_its_geometry(server: LabServer) -> None:
    call_json(server, "POST", "/api/tracks", _loop("Каталог"))

    _, catalog = call_json(server, "GET", "/api/catalog")
    status, geometry = call_json(server, "GET", "/api/tracks/%D0%9A%D0%B0%D1%82%D0%B0%D0%BB%D0%BE%D0%B3")

    assert "Каталог" in catalog["tracks"]
    assert status == 200 and geometry["name"] == "Каталог" and geometry["left"]
    assert call_json(server, "DELETE", "/api/tracks/%D0%9A%D0%B0%D1%82%D0%B0%D0%BB%D0%BE%D0%B3")[0] == 200
    assert "Каталог" not in call_json(server, "GET", "/api/catalog")[1]["tracks"]


def test_invalid_and_duplicate_tracks_get_clear_errors(server: LabServer) -> None:
    bad = _loop("Плохая")
    bad["centerline"] = [[0, 0], [50, 50], [50, 0], [0, 50]]
    status, body = call_json(server, "POST", "/api/tracks", bad)
    assert status == 400 and CYRILLIC.search(body["error"])

    call_json(server, "POST", "/api/tracks", _loop("Дубль"))
    status, body = call_json(server, "POST", "/api/tracks", _loop("Дубль"))
    assert status == 409 and "уже есть" in body["error"]
    again = {**_loop("Дубль"), "overwrite": True}
    assert call_json(server, "POST", "/api/tracks", again)[0] == 201
    call_json(server, "DELETE", "/api/tracks/%D0%94%D1%83%D0%B1%D0%BB%D1%8C")


def test_built_in_tracks_cannot_be_deleted_and_unknown_ones_are_not_found(
    server: LabServer,
) -> None:
    assert call_json(server, "DELETE", "/api/tracks/oval")[0] == 400
    assert call_json(server, "DELETE", "/api/tracks/nothing")[0] == 404
    assert call_json(server, "PUT", "/api/tracks")[0] == 405


def test_a_run_can_use_a_saved_track(server: LabServer) -> None:
    call_json(server, "POST", "/api/tracks", _loop("Гонка"))

    status, data = call_json(server, "POST", "/api/runs", _config(track="Гонка", name="own"))
    assert status == 201

    info = wait_for(
        lambda: (lambda row: row if row["status"] in ("finished", "error") else None)(
            call_json(server, "GET", f"/api/runs/{data['id']}")[1]
        )
    )
    assert info["status"] == "finished"
    call_json(server, "DELETE", "/api/tracks/%D0%93%D0%BE%D0%BD%D0%BA%D0%B0")
