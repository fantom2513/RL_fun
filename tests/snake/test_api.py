import http.client
import json

import pytest

from rl_fun.lab.server import make_server


def request(server, method, path, body=None):
    # Windows spawn imports Torch before it can produce the first live frame.
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=35)
    try:
        conn.request(method, path, body=None if body is None else json.dumps(body))
        response = conn.getresponse()
        return response.status, json.loads(response.read())
    finally:
        conn.close()


@pytest.fixture
def server(tmp_path):
    server = make_server(runs_dir=tmp_path)
    server.serve_in_thread()
    yield server
    server.close()


def test_game_lifecycle_and_bad_inputs(server):
    status, game = request(server, "POST", "/api/snake/games", {"mode": "manual", "size": 8})
    assert status == 201
    status, next_game = request(
        server, "POST", f"/api/snake/games/{game['id']}/step", {"direction": 1}
    )
    assert status == 200 and next_game["human"]["direction"] == 1
    assert (
        request(server, "POST", f"/api/snake/games/{game['id']}/step", {"direction": "up"})[0]
        == 400
    )
    assert request(server, "POST", "/api/snake/train", {"config": {"size": 3}})[0] == 400
    assert request(server, "POST", "/api/snake/games", [])[0] == 400
    assert request(server, "DELETE", f"/api/snake/games/{game['id']}")[0] == 200
    assert request(server, "POST", f"/api/snake/games/{game['id']}/step", {})[0] == 404
    assert request(server, "GET", "/api/snake/state")[0] == 200
    assert request(server, "PUT", "/api/snake/state")[0] == 405


@pytest.mark.parametrize(
    "data",
    [{}, {"enabled": True, "fps": 0}, {"enabled": True, "fps": True}, {"enabled": "yes", "fps": 8}],
)
def test_bad_animation_settings_rejected(server, data):
    assert request(server, "POST", "/api/snake/visualization", data)[0] == 400
    assert request(server, "POST", "/api/snake/train", {"visualization": data})[0] == 400


def test_sse_streams_actual_training_frames(server):
    status, run = request(
        server,
        "POST",
        "/api/snake/train",
        {
            "config": {"size": 8, "episodes": 10, "max_steps": 10},
            "visualization": {"enabled": True, "fps": 20},
        },
    )
    assert status == 201
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
    try:
        conn.request("GET", "/api/snake/events")
        response = conn.getresponse()
        assert response.status == 200
        event = ""
        while True:
            line = response.readline().decode().strip()
            if line.startswith("event: "):
                event = line[7:]
            elif event == "frame" and line.startswith("data: "):
                frame = json.loads(line[6:])
                if frame and frame["frame"]["steps"] > 0:
                    assert frame["run_id"] == run["id"]
                    assert frame["episode"] >= 1 and "epsilon" in frame
                    assert len(line) < 3000
                    break
    finally:
        conn.close()
        server.snake.control("stop")
