import json
import shutil
import subprocess

import pytest

from rl_fun.lab.server import WEB_DIR


def node(module, expression):
    executable = shutil.which("node")
    if not executable:
        pytest.skip("Node unavailable")
    url = (WEB_DIR / module).as_uri()
    result = subprocess.run(
        [
            executable,
            "--input-type=module",
            "-e",
            f"import * as m from {json.dumps(url)};console.log(JSON.stringify({expression}));",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def test_snake_route_is_recognized():
    assert node("core/router.js", "m.parseRoute('#/snake')") == {
        "name": "snake",
        "rest": [],
        "valid": True,
    }


def test_turn_queue_prevents_same_tick_reversal():
    result = node(
        "snake/input.js",
        "(()=>{const q=new m.TurnQueue();q.push(3,0);"
        "q.push(2,0);q.push(0,0);return [q.take(),q.take(),q.take()];})()",
    )
    assert result == [3, 2, None]


def test_turn_after_inflight_request_uses_pending_direction():
    result = node(
        "snake/input.js",
        "(()=>{const q=new m.TurnQueue();q.push(3,0);q.take();q.push(2,0);return q.take();})()",
    )
    assert result == 2


def test_game_keys_require_canvas_focus_and_ignore_modifiers():
    result = node(
        "snake/input.js",
        "[m.keyDirection({code:'ArrowUp',target:{tagName:'INPUT'}}),"
        "m.keyDirection({code:'KeyW',target:{tagName:'CANVAS'}}),"
        "m.keyDirection({code:'KeyW',ctrlKey:true,target:{tagName:'CANVAS'}})]",
    )
    assert result == [None, 3, None]


def test_chart_bounds_history_and_preserves_last_point():
    assert node(
        "snake/chart.js",
        "m.sampleHistory(Array.from({length:2000},"
        "(_,i)=>({iteration:i}))).map(x=>x.iteration).slice(-1)",
    ) == [1999]
    assert (
        node(
            "snake/chart.js",
            "m.sampleHistory(Array.from({length:2000},(_,i)=>({iteration:i}))).length",
        )
        <= 500
    )


@pytest.mark.parametrize(
    "key,label,low,high,step,value",
    [
        ("learning_rate", "lr", 0.000001, 0.1, 0.0001, 0.001),
        ("mutation_scale", "mutation", 0.001, 2, 0.01, 0.1),
    ],
)
def test_default_numeric_training_fields_pass_native_step_constraints(
    key, label, low, high, step, value
):
    markup = node(
        "snake/screen.js",
        f"m.numericField({json.dumps(key)}, {json.dumps(label)}, {low}, {high}, {step}, {value})",
    )
    import re

    attributes = dict(re.findall(r'([a-z]+)="([^\"]*)"', markup))
    assert float(attributes["min"]) <= float(attributes["value"]) <= float(attributes["max"])
    if attributes["step"] != "any":
        offset = (float(attributes["value"]) - float(attributes["min"])) / float(attributes["step"])
        assert offset == pytest.approx(round(offset))
