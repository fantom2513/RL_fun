"""Snake HTTP endpoints sharing the laboratory's request protections."""

import json

from rl_fun.snake.config import SnakeConfig


def handle_snake_request(handler, method, parts):
    from rl_fun.lab.server import _HttpError

    manager = handler.server.snake

    def expect(wanted):
        if method != wanted:
            raise _HttpError(405)

    try:
        if parts == ["catalog"]:
            expect("GET")
            handler._send_json(
                200, {"defaults": SnakeConfig().to_dict(), "algorithms": ["dqn", "evolution"]}
            )
        elif parts == ["state"]:
            expect("GET")
            handler._send_json(200, manager.state())
        elif parts == ["models"]:
            expect("GET")
            handler._send_json(200, manager.models())
        elif parts == ["train"]:
            expect("POST")
            data = handler._read_json_object()
            run_id = manager.train(data.get("config", {}), data.get("resume"))
            handler._send_json(201, {"id": run_id})
        elif parts == ["control"]:
            expect("POST")
            manager.control(handler._read_json_object().get("command"))
            handler._send_json(200, {"ok": True})
        elif parts == ["events"]:
            expect("GET")
            stream_state(handler, manager)
        elif parts == ["games"]:
            expect("POST")
            handler._send_json(201, manager.create_game(handler._read_json_object()))
        elif len(parts) == 2 and parts[0] == "games":
            expect("DELETE")
            manager.delete_game(parts[1])
            handler._send_json(200, {"ok": True})
        elif len(parts) == 3 and parts[0] == "games" and parts[2] == "step":
            expect("POST")
            handler._send_json(200, manager.step_game(parts[1], handler._read_json_object()))
        else:
            raise _HttpError(404)
    except KeyError:
        raise _HttpError(404, "игра или модель не найдена") from None
    except ValueError as error:
        raise _HttpError(400, str(error)) from None
    except RuntimeError as error:
        raise _HttpError(409, str(error)) from None


def stream_state(handler, manager):
    from rl_fun.lab.server import _client_gone

    handler.send_response(200)
    handler.send_header("Content-Type", "text/event-stream; charset=utf-8")
    handler.send_header("Cache-Control", "no-cache")
    handler.send_header("X-Content-Type-Options", "nosniff")
    handler.send_header("Connection", "close")
    handler.end_headers()
    revision = -1
    while not manager.closed and not _client_gone(handler.connection):
        with manager.condition:
            if manager.revision == revision:
                manager.condition.wait(0.5)
                continue
            state = manager.state()
            revision = state["revision"]
        handler.wfile.write(("event: state\ndata: " + json.dumps(state) + "\n\n").encode())
        handler.wfile.flush()
