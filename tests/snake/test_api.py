import http.client
import json

import pytest

from rl_fun.lab.server import make_server


def request(server, method, path, body=None):
    conn = http.client.HTTPConnection("127.0.0.1", server.port, timeout=10)
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
