import json

from rl_fun.tracking.metrics import JsonlMetricSink, MemoryMetricSink


def test_memory_sink_keeps_metric_event():
    # Arrange
    sink = MemoryMetricSink()

    # Act
    sink.log(3, {"reward": 1.25})

    # Assert
    assert sink.events[0].metrics == {"reward": 1.25}


def test_jsonl_sink_writes_one_json_object_per_event(tmp_path):
    # Arrange
    path = tmp_path / "metrics.jsonl"

    # Act
    with JsonlMetricSink(path) as sink:
        sink.log(7, {"regret": 0.5})

    # Assert
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "step": 7,
        "metrics": {"regret": 0.5},
    }
