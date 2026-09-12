from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType
from typing import Protocol, Self


@dataclass(frozen=True, slots=True)
class MetricEvent:
    step: int
    metrics: dict[str, float]


class MetricSink(Protocol):
    def log(self, step: int, metrics: Mapping[str, float]) -> None: ...


class MemoryMetricSink:
    def __init__(self) -> None:
        self.events: list[MetricEvent] = []

    def log(self, step: int, metrics: Mapping[str, float]) -> None:
        self.events.append(MetricEvent(step, {key: float(value) for key, value in metrics.items()}))


class JsonlMetricSink:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = path.open("w", encoding="utf-8")

    def log(self, step: int, metrics: Mapping[str, float]) -> None:
        event = MetricEvent(step, {key: float(value) for key, value in metrics.items()})
        self._handle.write(json.dumps(asdict(event), sort_keys=True) + "\n")

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
