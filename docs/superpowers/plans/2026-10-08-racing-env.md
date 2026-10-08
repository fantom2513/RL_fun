# Racing Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Добавить 2D-среду гонок `RLFun/Racing-v0` с лучевыми сенсорами, отрисовкой на pygame и ручным управлением.

**Architecture:** Новый пакет `src/rl_fun/racing/`: трасса (`track.py`), сменная динамика (`dynamics.py`), лучи (`sensors.py`), среда (`env.py`), рендерер (`render.py`) и логика ручной игры (`play.py`). Среда регистрируется в `LOCAL_ENVIRONMENTS`, поэтому существующие runtime, `random` и `scripts/train.py` работают без изменений. Спека: `docs/superpowers/specs/2026-10-08-racing-env-design.md`.

**Tech Stack:** Python 3.12, gymnasium, numpy, pygame (новая зависимость, одобрена пользователем), pytest, ruff, uv.

**Conventions (CLAUDE.md пользователя):** TDD — красные тесты коммитятся отдельно от реализации; Conventional Commits; AAA в тестах; аннотации типов; ветка `feat/racing-env` (уже создана). Каждое сообщение коммита заканчивается пустой строкой и `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Команды запускаются из `D:\projects\RL_fun`.

**Системы координат:** мир — метры, ось y вверх, курс (`heading`) в радианах против часовой. Положительный руль = поворот налево. Рендерер переворачивает y при отрисовке.

---

## File Structure

| Файл | Ответственность |
|---|---|
| `pyproject.toml`, `uv.lock` | зависимость `pygame` |
| `src/rl_fun/racing/__init__.py` | пакет |
| `src/rl_fun/racing/geometry.py` | `cross` для 2D-векторов |
| `src/rl_fun/racing/track.py` | `Track`, `load_track`, `available_tracks` |
| `src/rl_fun/racing/tracks/oval.json`, `wavy.json` | готовые трассы |
| `src/rl_fun/racing/dynamics.py` | `VehicleState`, `KinematicBicycle`, `make_dynamics` |
| `src/rl_fun/racing/sensors.py` | `ray_angles`, `cast_rays` |
| `src/rl_fun/racing/env.py` | `RacingEnv` |
| `src/rl_fun/racing/render.py` | `Renderer`, протокол `SidePanel` |
| `src/rl_fun/racing/play.py` | `action_from_keys`, `parse_args`, `main` |
| `src/rl_fun/environments/__init__.py` | регистрация `RLFun/Racing-v0` |
| `scripts/play.py` | тонкая обёртка над `rl_fun.racing.play.main` |
| `configs/racing-random.json` | пример конфигурации для runtime |
| `tests/racing/conftest.py` | `SDL_VIDEODRIVER=dummy` |
| `tests/racing/test_*.py` | тесты по модулям |
| `tests/integration/test_racing_runtime.py` | запуск среды через `scripts/train.py` |
| `README.md` | русский раздел про гонки |

---

### Task 1: Зависимость pygame и каркас пакета

**Files:**
- Modify: `pyproject.toml`, `uv.lock` (через `uv add`)
- Create: `src/rl_fun/racing/__init__.py`, `src/rl_fun/racing/geometry.py`, `tests/racing/conftest.py`

- [ ] **Step 1: Добавить зависимость**

Run: `uv add pygame`
Expected: `pyproject.toml` получает `"pygame>=…"` в `dependencies`, `uv.lock` обновлён.

- [ ] **Step 2: Создать каркас**

`src/rl_fun/racing/__init__.py`:

```python
"""2D racing environment: tracks, vehicle dynamics, sensors and rendering."""
```

`src/rl_fun/racing/geometry.py`:

```python
"""Small 2D vector helpers shared by track and sensor code."""

import numpy as np


def cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Return the z component of the 2D cross product along the last axis."""
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
```

`tests/racing/conftest.py`:

```python
"""Run pygame without opening real windows or audio devices in tests."""

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
```

- [ ] **Step 3: Проверить**

Run: `uv run --all-groups python -c "import pygame, rl_fun.racing.geometry"` и `uv run --all-groups ruff check .`
Expected: без ошибок (приветствие pygame допустимо).

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock src/rl_fun/racing tests/racing/conftest.py
git commit -m "chore: add pygame and racing package skeleton"
```

---

### Task 2: Трасса

**Files:**
- Create: `tests/racing/test_track.py`, `src/rl_fun/racing/track.py`, `src/rl_fun/racing/tracks/oval.json`, `src/rl_fun/racing/tracks/wavy.json`

- [ ] **Step 1: Написать падающие тесты** — `tests/racing/test_track.py`

```python
import json

import pytest

from rl_fun.racing.track import Track, available_tracks, load_track

SQUARE = [[0, 0], [100, 0], [100, 100], [0, 100]]


@pytest.fixture
def square() -> Track:
    return Track("square", SQUARE, width=10.0)


def test_length_is_perimeter(square: Track):
    assert square.length == pytest.approx(400.0)


def test_project_returns_arclength_and_distance(square: Track):
    # Act
    progress, distance = square.project((50, 3))

    # Assert
    assert progress == pytest.approx(50.0)
    assert distance == pytest.approx(3.0)


def test_project_on_second_edge(square: Track):
    progress, distance = square.project((100, 50))

    assert progress == pytest.approx(150.0)
    assert distance == pytest.approx(0.0)


def test_contains_uses_half_width(square: Track):
    assert square.contains((50, 4.9))
    assert not square.contains((50, 5.1))


def test_start_pose_points_along_first_segment(square: Track):
    position, heading = square.start_pose()

    assert tuple(position) == (0.0, 0.0)
    assert heading == pytest.approx(0.0)


@pytest.mark.parametrize(
    ("progress_from", "progress_to", "expected"),
    [(10.0, 20.0, 10.0), (395.0, 5.0, 10.0), (5.0, 395.0, -10.0)],
)
def test_progress_delta_wraps_around_start(
    square: Track, progress_from: float, progress_to: float, expected: float
):
    assert square.progress_delta(progress_from, progress_to) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("centerline", "width"),
    [
        ([[0, 0], [1, 0]], 10.0),
        (SQUARE, 0.0),
        (SQUARE, -1.0),
        ([[0, 0], [10, 10], [10, 0], [0, 10]], 5.0),
        ([[0, 0], [0, 0], [10, 0], [10, 10]], 5.0),
    ],
)
def test_invalid_track_raises(centerline: list[list[float]], width: float):
    with pytest.raises(ValueError):
        Track("bad", centerline, width)


def test_boundary_segments_cover_both_sides(square: Track):
    assert square.boundary_segments.shape == (8, 2, 2)


def test_available_tracks_lists_defaults():
    assert {"oval", "wavy"} <= set(available_tracks())


def test_builtin_tracks_load_and_start_inside():
    for name in available_tracks():
        track = load_track(name)
        position, _ = track.start_pose()
        assert track.contains(position)


def test_builtin_boundaries_are_half_width_from_centerline():
    track = load_track("oval")

    for point in track.left:
        _, distance = track.project(point)
        assert distance == pytest.approx(track.width / 2, abs=0.5)


def test_load_track_from_json_path(tmp_path):
    path = tmp_path / "custom.json"
    path.write_text(
        json.dumps({"name": "custom", "width": 8, "centerline": SQUARE}), encoding="utf-8"
    )

    track = load_track(str(path))

    assert track.name == "custom"
    assert track.width == 8.0


def test_unknown_track_raises():
    with pytest.raises(ValueError, match="unknown track"):
        load_track("does-not-exist")


def test_from_dict_requires_all_keys():
    with pytest.raises(ValueError, match="centerline"):
        Track.from_dict({"name": "x", "width": 5})
```

- [ ] **Step 2: Запустить и убедиться, что падает**

Run: `uv run --all-groups pytest tests/racing/test_track.py -q`
Expected: ошибка импорта `rl_fun.racing.track`.

- [ ] **Step 3: Закоммитить красные тесты**

```bash
git add tests/racing/test_track.py
git commit -m "test: specify racing track geometry"
```

- [ ] **Step 4: Сгенерировать файлы трасс** (однократно)

Run:

```bash
uv run --all-groups python -c "
import json, math
from pathlib import Path
out = Path('src/rl_fun/racing/tracks'); out.mkdir(parents=True, exist_ok=True)
def write(name, width, pts):
    data = {'name': name, 'width': width, 'centerline': [[round(x, 3), round(y, 3)] for x, y in pts]}
    (out / f'{name}.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
n = 36
write('oval', 10.0, [(60 * math.cos(2 * math.pi * k / n), 35 * math.sin(2 * math.pi * k / n)) for k in range(n)])
m = 72
def wavy(k):
    t = 2 * math.pi * k / m
    r = 55 * (1 + 0.25 * math.cos(3 * t))
    return r * math.cos(t), r * math.sin(t)
write('wavy', 10.0, [wavy(k) for k in range(m)])
"
```

Expected: созданы `oval.json` и `wavy.json`.

- [ ] **Step 5: Реализация** — `src/rl_fun/racing/track.py`

```python
"""Closed 2D racing track defined by a centerline and a width."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from rl_fun.racing.geometry import cross

TRACKS_DIR = Path(__file__).parent / "tracks"


def _segments_intersect(p: np.ndarray, q: np.ndarray, r: np.ndarray, s: np.ndarray) -> bool:
    d1 = cross(q - p, r - p)
    d2 = cross(q - p, s - p)
    d3 = cross(s - r, p - r)
    d4 = cross(s - r, q - r)
    return bool(d1 * d2 < 0 and d3 * d4 < 0)


def _self_intersects(starts: np.ndarray, ends: np.ndarray) -> bool:
    count = len(starts)
    for i in range(count):
        for j in range(i + 2, count):
            if i == 0 and j == count - 1:
                continue
            if _segments_intersect(starts[i], ends[i], starts[j], ends[j]):
                return True
    return False


class Track:
    """Closed track: a centerline polyline and a constant width in meters."""

    def __init__(self, name: str, centerline: Sequence[Sequence[float]], width: float) -> None:
        points = np.asarray(centerline, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 2 or len(points) < 3:
            raise ValueError("centerline must contain at least 3 points of [x, y]")
        if not width > 0:
            raise ValueError("width must be positive")
        ends = np.roll(points, -1, axis=0)
        vectors = ends - points
        lengths = np.linalg.norm(vectors, axis=1)
        if np.any(lengths == 0):
            raise ValueError("centerline must not repeat consecutive points")
        if _self_intersects(points, ends):
            raise ValueError("centerline must not intersect itself")

        self.name = name
        self.width = float(width)
        self.centerline = points
        self._vectors = vectors
        self._lengths = lengths
        self._offsets = np.concatenate(([0.0], np.cumsum(lengths)[:-1]))
        self.length = float(lengths.sum())
        self.left, self.right = self._boundaries()
        self.boundary_segments = np.concatenate(
            [np.stack([ring, np.roll(ring, -1, axis=0)], axis=1) for ring in (self.left, self.right)]
        )

    @classmethod
    def from_dict(cls, values: Mapping[str, object]) -> Track:
        """Build a track from the JSON structure `{name, width, centerline}`."""
        for key in ("name", "width", "centerline"):
            if key not in values:
                raise ValueError(f"track definition is missing {key!r}")
        return cls(str(values["name"]), values["centerline"], float(values["width"]))  # type: ignore[arg-type]

    @classmethod
    def from_json(cls, path: Path) -> Track:
        """Load a track from a JSON file."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def project(self, point: Sequence[float]) -> tuple[float, float]:
        """Return arclength along the centerline and distance to the nearest centerline point."""
        position = np.asarray(point, dtype=np.float64)
        relative = position - self.centerline
        fractions = np.clip(
            np.einsum("ij,ij->i", relative, self._vectors) / self._lengths**2, 0.0, 1.0
        )
        closest = self.centerline + fractions[:, None] * self._vectors
        distances = np.linalg.norm(position - closest, axis=1)
        index = int(np.argmin(distances))
        progress = float(self._offsets[index] + fractions[index] * self._lengths[index])
        return progress, float(distances[index])

    def contains(self, point: Sequence[float]) -> bool:
        """Return whether a point lies on the road."""
        return self.project(point)[1] <= self.width / 2

    def start_pose(self) -> tuple[np.ndarray, float]:
        """Return the start position and heading along the first centerline segment."""
        direction = self._vectors[0]
        return self.centerline[0].copy(), float(np.arctan2(direction[1], direction[0]))

    def progress_delta(self, previous: float, current: float) -> float:
        """Return the signed arclength change, wrapping around the start line."""
        delta = current - previous
        half = self.length / 2
        if delta > half:
            delta -= self.length
        elif delta < -half:
            delta += self.length
        return delta

    def _boundaries(self) -> tuple[np.ndarray, np.ndarray]:
        directions = self._vectors / self._lengths[:, None]
        normals = np.stack([-directions[:, 1], directions[:, 0]], axis=1)
        mean = normals + np.roll(normals, 1, axis=0)
        mean /= np.linalg.norm(mean, axis=1, keepdims=True)
        scale = np.clip(np.einsum("ij,ij->i", mean, normals), 0.5, 1.0)
        offset = mean * (self.width / 2 / scale)[:, None]
        return self.centerline + offset, self.centerline - offset


def available_tracks() -> list[str]:
    """Return the names of the built-in tracks."""
    return sorted(path.stem for path in TRACKS_DIR.glob("*.json"))


def load_track(name_or_path: str | Path) -> Track:
    """Load a built-in track by name or a track JSON file by path."""
    text = str(name_or_path)
    builtin = TRACKS_DIR / f"{text}.json"
    path = builtin if builtin.is_file() else Path(text)
    if not path.is_file():
        raise ValueError(
            f"unknown track {text!r}; built-in tracks: {', '.join(available_tracks())}"
        )
    return Track.from_json(path)
```

- [ ] **Step 6: Тесты зелёные и линтер**

Run: `uv run --all-groups pytest tests/racing/test_track.py -q` и `uv run --all-groups ruff check src tests`
Expected: все тесты PASS, ruff чистый. Если ruff жалуется на длину строки (100) или импорты — исправить форматирование, не поведение.

- [ ] **Step 7: Commit**

```bash
git add src/rl_fun/racing/track.py src/rl_fun/racing/tracks
git commit -m "feat: add racing track geometry and built-in tracks"
```

---

### Task 3: Динамика

**Files:**
- Create: `tests/racing/test_dynamics.py`, `src/rl_fun/racing/dynamics.py`

- [ ] **Step 1: Написать падающие тесты** — `tests/racing/test_dynamics.py`

```python
import math

import pytest

from rl_fun.racing.dynamics import KinematicBicycle, VehicleState, make_dynamics


def test_throttle_accelerates_along_heading():
    # Arrange
    model = KinematicBicycle()
    state = VehicleState(0.0, 0.0, 0.0)

    # Act
    after = model.step(state, steer=0.0, throttle=1.0, dt=0.1)

    # Assert
    assert after.v_long == pytest.approx(model.acceleration * 0.1)
    assert after.x == pytest.approx(after.v_long * 0.1)
    assert after.y == pytest.approx(0.0)


def test_speed_is_capped():
    model = KinematicBicycle(max_speed=5.0)
    state = VehicleState(0.0, 0.0, 0.0, v_long=4.99)

    after = model.step(state, 0.0, 1.0, dt=1.0)

    assert after.v_long == pytest.approx(5.0)


def test_braking_stops_without_reversing():
    model = KinematicBicycle()
    state = VehicleState(0.0, 0.0, 0.0, v_long=1.0)

    after = model.step(state, 0.0, -1.0, dt=1.0)

    assert after.v_long == 0.0


def test_positive_steer_turns_left():
    model = KinematicBicycle()
    state = VehicleState(0.0, 0.0, 0.0, v_long=10.0)

    after = model.step(state, steer=1.0, throttle=0.0, dt=0.1)

    expected_rate = 10.0 / model.wheelbase * math.tan(model.max_steer)
    assert after.yaw_rate == pytest.approx(expected_rate)
    assert after.heading == pytest.approx(expected_rate * 0.1)
    assert after.y > 0.0


def test_kinematic_model_has_no_lateral_velocity():
    model = KinematicBicycle()
    state = VehicleState(0.0, 0.0, 0.0, v_long=10.0)

    assert model.step(state, 1.0, 1.0, dt=0.1).v_lat == 0.0


def test_out_of_range_inputs_are_clipped():
    model = KinematicBicycle()
    state = VehicleState(0.0, 0.0, 0.0, v_long=10.0)

    clipped = model.step(state, 5.0, 5.0, dt=0.1)
    limit = model.step(state, 1.0, 1.0, dt=0.1)

    assert clipped == limit


def test_max_yaw_rate_matches_full_lock_at_top_speed():
    model = KinematicBicycle()

    assert model.max_yaw_rate == pytest.approx(
        model.max_speed / model.wheelbase * math.tan(model.max_steer)
    )


def test_make_dynamics_creates_kinematic_model():
    assert isinstance(make_dynamics("kinematic"), KinematicBicycle)


def test_make_dynamics_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown dynamics"):
        make_dynamics("hovercraft")
```

- [ ] **Step 2: Убедиться, что падает**

Run: `uv run --all-groups pytest tests/racing/test_dynamics.py -q`
Expected: ошибка импорта `rl_fun.racing.dynamics`.

- [ ] **Step 3: Закоммитить красные тесты**

```bash
git add tests/racing/test_dynamics.py
git commit -m "test: specify kinematic vehicle dynamics"
```

- [ ] **Step 4: Реализация** — `src/rl_fun/racing/dynamics.py`

```python
"""Vehicle dynamics models. A model maps (state, steer, throttle, dt) to the next state."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class VehicleState:
    """Pose in world coordinates (meters, radians) and body-frame velocities."""

    x: float
    y: float
    heading: float
    v_long: float = 0.0
    v_lat: float = 0.0
    yaw_rate: float = 0.0


class VehicleDynamics(Protocol):
    """Interchangeable vehicle model. Steer and throttle are in [-1, 1]."""

    max_speed: float
    max_yaw_rate: float

    def step(
        self, state: VehicleState, steer: float, throttle: float, dt: float
    ) -> VehicleState:
        """Advance the vehicle by dt seconds."""


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


@dataclass(frozen=True, slots=True)
class KinematicBicycle:
    """Kinematic bicycle model: the car moves exactly where it points, with no sliding."""

    wheelbase: float = 2.5
    max_steer: float = 0.4
    max_speed: float = 20.0
    acceleration: float = 8.0
    braking: float = 14.0

    @property
    def max_yaw_rate(self) -> float:
        """Yaw rate at full steering lock and top speed."""
        return self.max_speed / self.wheelbase * math.tan(self.max_steer)

    def step(
        self, state: VehicleState, steer: float, throttle: float, dt: float
    ) -> VehicleState:
        """Advance the vehicle by dt seconds."""
        steer = _clip(steer, -1.0, 1.0)
        throttle = _clip(throttle, -1.0, 1.0)
        rate = throttle * self.acceleration if throttle >= 0 else throttle * self.braking
        speed = _clip(state.v_long + rate * dt, 0.0, self.max_speed)
        yaw_rate = speed / self.wheelbase * math.tan(steer * self.max_steer)
        heading = state.heading + yaw_rate * dt
        return VehicleState(
            x=state.x + speed * math.cos(heading) * dt,
            y=state.y + speed * math.sin(heading) * dt,
            heading=heading,
            v_long=speed,
            v_lat=0.0,
            yaw_rate=yaw_rate,
        )


_DYNAMICS = {"kinematic": KinematicBicycle}


def make_dynamics(name: str) -> VehicleDynamics:
    """Create a dynamics model by name."""
    try:
        return _DYNAMICS[name]()
    except KeyError as error:
        supported = ", ".join(sorted(_DYNAMICS))
        raise ValueError(f"unknown dynamics {name!r}; supported: {supported}") from error
```

- [ ] **Step 5: Зелёные тесты и линтер**

Run: `uv run --all-groups pytest tests/racing/test_dynamics.py -q` и `uv run --all-groups ruff check src tests`
Expected: PASS, ruff чистый.

- [ ] **Step 6: Commit**

```bash
git add src/rl_fun/racing/dynamics.py
git commit -m "feat: add kinematic vehicle dynamics"
```

---

### Task 4: Лучевые сенсоры

**Files:**
- Create: `tests/racing/test_sensors.py`, `src/rl_fun/racing/sensors.py`

- [ ] **Step 1: Написать падающие тесты** — `tests/racing/test_sensors.py`

```python
import math

import numpy as np
import pytest

from rl_fun.racing.sensors import cast_rays, ray_angles
from rl_fun.racing.track import Track

WALL = np.array([[[10.0, -5.0], [10.0, 5.0]]])


def test_ray_angles_span_field_of_view():
    angles = ray_angles(3, fov_degrees=180.0)

    assert angles == pytest.approx([-math.pi / 2, 0.0, math.pi / 2])


def test_single_ray_points_forward():
    assert ray_angles(1) == pytest.approx([0.0])


def test_ray_angles_reject_empty_count():
    with pytest.raises(ValueError):
        ray_angles(0)


def test_ray_hits_wall_ahead():
    distances = cast_rays(WALL, np.array([0.0, 0.0]), 0.0, np.array([0.0]), max_range=50.0)

    assert distances == pytest.approx([10.0])


def test_ray_is_clipped_to_max_range():
    distances = cast_rays(WALL, np.array([0.0, 0.0]), 0.0, np.array([0.0]), max_range=6.0)

    assert distances == pytest.approx([6.0])


def test_parallel_and_backward_rays_miss():
    angles = np.array([math.pi / 2, math.pi])

    distances = cast_rays(WALL, np.array([0.0, 0.0]), 0.0, angles, max_range=50.0)

    assert distances == pytest.approx([50.0, 50.0])


def test_angle_is_relative_to_heading():
    distances = cast_rays(
        WALL, np.array([0.0, 0.0]), math.pi / 2, np.array([-math.pi / 2]), max_range=50.0
    )

    assert distances == pytest.approx([10.0])


def test_side_rays_see_track_edges_at_half_width():
    track = Track("square", [[0, 0], [100, 0], [100, 100], [0, 100]], width=10.0)
    angles = np.array([-math.pi / 2, math.pi / 2])

    distances = cast_rays(
        track.boundary_segments, np.array([50.0, 0.0]), 0.0, angles, max_range=40.0
    )

    assert distances == pytest.approx([5.0, 5.0], abs=1e-6)
```

- [ ] **Step 2: Убедиться, что падает**

Run: `uv run --all-groups pytest tests/racing/test_sensors.py -q`
Expected: ошибка импорта `rl_fun.racing.sensors`.

- [ ] **Step 3: Закоммитить красные тесты**

```bash
git add tests/racing/test_sensors.py
git commit -m "test: specify racing ray sensors"
```

- [ ] **Step 4: Реализация** — `src/rl_fun/racing/sensors.py`

```python
"""Ray sensors that measure the distance from the car to the track boundaries."""

from __future__ import annotations

import math

import numpy as np

from rl_fun.racing.geometry import cross


def ray_angles(count: int, fov_degrees: float = 180.0) -> np.ndarray:
    """Return ray angles relative to the heading, spread evenly over the field of view."""
    if count < 1:
        raise ValueError("at least one ray is required")
    if count == 1:
        return np.zeros(1)
    half = math.radians(fov_degrees) / 2
    return np.linspace(-half, half, count)


def cast_rays(
    segments: np.ndarray,
    origin: np.ndarray,
    heading: float,
    angles: np.ndarray,
    max_range: float,
) -> np.ndarray:
    """Return the distance to the nearest segment for every ray, clipped to max_range.

    `segments` has shape (S, 2, 2): start and end point of each wall segment.
    """
    absolute = heading + angles
    directions = np.stack([np.cos(absolute), np.sin(absolute)], axis=1)[:, None, :]
    starts = segments[:, 0, :]
    edges = segments[:, 1, :] - starts
    offsets = (starts - origin)[None, :, :]
    denominator = cross(directions, edges[None])
    with np.errstate(divide="ignore", invalid="ignore"):
        along_ray = cross(offsets, edges[None]) / denominator
        along_edge = cross(offsets, directions) / denominator
    hit = (np.abs(denominator) > 1e-12) & (along_ray >= 0) & (along_edge >= 0) & (along_edge <= 1)
    nearest = np.where(hit, along_ray, np.inf).min(axis=1)
    return np.minimum(nearest, max_range)
```

- [ ] **Step 5: Зелёные тесты и линтер**

Run: `uv run --all-groups pytest tests/racing/test_sensors.py -q` и `uv run --all-groups ruff check src tests`
Expected: PASS, ruff чистый.

- [ ] **Step 6: Commit**

```bash
git add src/rl_fun/racing/sensors.py
git commit -m "feat: add ray sensors for racing"
```

---

### Task 5: Среда `RacingEnv` и регистрация

**Files:**
- Create: `tests/racing/test_env.py`, `src/rl_fun/racing/env.py`
- Modify: `src/rl_fun/environments/__init__.py`

- [ ] **Step 1: Написать падающие тесты** — `tests/racing/test_env.py`

```python
import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from rl_fun.environments import register_environments
from rl_fun.racing.env import RacingEnv

FORWARD = np.array([0.0, 1.0], dtype=np.float32)
IDLE = np.array([0.0, 0.0], dtype=np.float32)


@pytest.fixture
def env():
    environment = RacingEnv()
    yield environment
    environment.close()


def test_spaces_match_contract(env: RacingEnv):
    assert env.action_space.shape == (2,)
    assert env.observation_space.shape == (5 + 3,)


def test_environment_is_registered_in_gymnasium():
    register_environments()

    created = gym.make("RLFun/Racing-v0", track="oval")

    try:
        assert isinstance(created.unwrapped, RacingEnv)
    finally:
        created.close()


def test_passes_gymnasium_env_checker():
    environment = RacingEnv()
    try:
        check_env(environment, skip_render_check=True)
    finally:
        environment.close()


def test_reset_returns_valid_observation(env: RacingEnv):
    observation, info = env.reset(seed=1)

    assert env.observation_space.contains(observation)
    assert info["progress"] == 0.0
    assert info["collided"] is False


def test_initial_side_rays_see_edges_and_speeds_are_zero(env: RacingEnv):
    observation, _ = env.reset(seed=1)

    assert observation[0] == pytest.approx(5.0 / env.ray_range, abs=0.02)
    assert observation[4] == pytest.approx(5.0 / env.ray_range, abs=0.02)
    assert observation[-3:] == pytest.approx([0.0, 0.0, 0.0])


def test_reset_is_deterministic_for_same_seed():
    first, second = RacingEnv(), RacingEnv()
    try:
        a, _ = first.reset(seed=3)
        b, _ = second.reset(seed=3)
    finally:
        first.close()
        second.close()

    assert np.array_equal(a, b)


def test_driving_forward_earns_reward_equal_to_progress(env: RacingEnv):
    env.reset(seed=1)
    total = 0.0

    for _ in range(30):
        _, reward, terminated, truncated, info = env.step(FORWARD)
        total += reward

    assert total > 0.0
    assert total == pytest.approx(info["progress"])
    assert not (terminated or truncated)


def test_driving_straight_ends_with_collision(env: RacingEnv):
    env.reset(seed=1)

    terminated = False
    info: dict = {}
    for _ in range(600):
        _, _, terminated, _, info = env.step(FORWARD)
        if terminated:
            break

    assert terminated
    assert info["collided"] is True


def test_finishing_the_lap_terminates_without_collision(env: RacingEnv):
    env.reset(seed=1)
    env.unwrapped._progress = env.track.length - 0.001

    _, _, terminated, truncated, info = env.step(FORWARD)

    assert terminated and not truncated
    assert info["lap"] == 1
    assert info["collided"] is False


def test_truncates_at_max_steps():
    environment = RacingEnv(max_steps=3)
    try:
        environment.reset(seed=1)
        results = [environment.step(IDLE) for _ in range(3)]
    finally:
        environment.close()

    assert [result[2] for result in results] == [False, False, False]
    assert [result[3] for result in results] == [False, False, True]


def test_step_after_episode_end_requires_reset():
    environment = RacingEnv(max_steps=1)
    try:
        environment.reset(seed=1)
        environment.step(IDLE)
        with pytest.raises(RuntimeError, match="reset"):
            environment.step(IDLE)
    finally:
        environment.close()


def test_step_before_reset_raises(env: RacingEnv):
    with pytest.raises(RuntimeError, match="reset"):
        env.step(IDLE)


@pytest.mark.parametrize("action", [np.array([2.0, 0.0], dtype=np.float32), np.zeros(3)])
def test_invalid_action_raises(env: RacingEnv, action: np.ndarray):
    env.reset(seed=1)

    with pytest.raises(ValueError):
        env.step(action)


@pytest.mark.parametrize(
    "kwargs",
    [{"ray_angles_deg": []}, {"ray_range": 0.0}, {"dt": 0.0}, {"max_steps": 0}, {"laps": 0},
     {"track": "missing"}, {"dynamics": "missing"}, {"render_mode": "ansi"}],
)
def test_invalid_constructor_arguments_raise(kwargs: dict):
    with pytest.raises(ValueError):
        RacingEnv(**kwargs)


def test_rgb_array_render_returns_a_frame():
    environment = RacingEnv(render_mode="rgb_array")
    try:
        environment.reset(seed=1)
        frame = environment.render()
    finally:
        environment.close()

    assert frame.dtype == np.uint8
    assert frame.ndim == 3 and frame.shape[2] == 3
    assert frame.std() > 0


def test_human_render_mode_steps_without_error():
    environment = RacingEnv(render_mode="human")
    try:
        environment.reset(seed=1)
        environment.step(IDLE)
    finally:
        environment.close()
```

- [ ] **Step 2: Убедиться, что падает**

Run: `uv run --all-groups pytest tests/racing/test_env.py -q`
Expected: ошибка импорта `rl_fun.racing.env`.

- [ ] **Step 3: Закоммитить красные тесты**

```bash
git add tests/racing/test_env.py
git commit -m "test: specify racing environment contract"
```

- [ ] **Step 4: Реализация** — `src/rl_fun/racing/env.py`

```python
"""Gymnasium environment: a car with ray sensors driving around a closed 2D track."""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from rl_fun.racing.dynamics import VehicleState, make_dynamics
from rl_fun.racing.sensors import cast_rays
from rl_fun.racing.track import load_track


class RacingEnv(gym.Env[np.ndarray, np.ndarray]):
    """Drive one or more laps without touching the track boundary.

    Action: [steer, throttle] in [-1, 1]; positive steer turns left, negative throttle brakes.
    Observation: normalized ray distances, then longitudinal speed, lateral speed, yaw rate.
    Reward: lap progress gained this step as a fraction of the lap length.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        track: str = "oval",
        dynamics: str = "kinematic",
        ray_angles_deg: Sequence[float] = (-90.0, -30.0, 0.0, 30.0, 90.0),
        ray_range: float = 40.0,
        dt: float = 1 / 30,
        max_steps: int = 1500,
        laps: int = 1,
        render_mode: str | None = None,
    ) -> None:
        if len(ray_angles_deg) < 1:
            raise ValueError("ray_angles_deg must contain at least one angle")
        if ray_range <= 0 or dt <= 0:
            raise ValueError("ray_range and dt must be positive")
        if max_steps < 1 or laps < 1:
            raise ValueError("max_steps and laps must be at least 1")
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"unsupported render_mode {render_mode!r}")

        self.track = load_track(track)
        self.dynamics = make_dynamics(dynamics)
        self.ray_range = float(ray_range)
        self.dt = float(dt)
        self.max_steps = int(max_steps)
        self.laps = int(laps)
        self.render_mode = render_mode
        self.metadata = {**RacingEnv.metadata, "render_fps": round(1 / self.dt)}
        self.side_panel = None  # optional rl_fun.racing.render.SidePanel, set before first render

        self._angles = np.radians(np.asarray(ray_angles_deg, dtype=np.float64))
        n_rays = len(self._angles)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
        low = np.concatenate([np.zeros(n_rays), [0.0, -1.0, -1.0]]).astype(np.float32)
        self.observation_space = spaces.Box(low, np.ones(n_rays + 3, np.float32), dtype=np.float32)

        self._state: VehicleState | None = None
        self._distances = np.zeros(n_rays)
        self._previous_arclength = 0.0
        self._progress = 0.0
        self._steps = 0
        self._done = True
        self._collided = False
        self._renderer = None

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        position, heading = self.track.start_pose()
        self._state = VehicleState(float(position[0]), float(position[1]), heading)
        self._previous_arclength = self.track.project(position)[0]
        self._progress = 0.0
        self._steps = 0
        self._done = False
        self._collided = False
        observation = self._observe()
        if self.render_mode == "human":
            self.render()
        return observation, self._info()

    def step(self, action: np.ndarray):
        if self._state is None or self._done:
            raise RuntimeError("call reset() before step()")
        action = np.asarray(action, dtype=np.float32)
        if not self.action_space.contains(action):
            raise ValueError(f"invalid action {action!r}")

        self._state = self.dynamics.step(
            self._state, float(action[0]), float(action[1]), self.dt
        )
        self._steps += 1
        arclength, _ = self.track.project((self._state.x, self._state.y))
        delta = self.track.progress_delta(self._previous_arclength, arclength)
        self._previous_arclength = arclength
        self._progress += delta
        self._collided = not self.track.contains((self._state.x, self._state.y))

        finished = self._progress >= self.laps * self.track.length
        terminated = self._collided or finished
        truncated = self._steps >= self.max_steps and not terminated
        self._done = terminated or truncated

        observation = self._observe()
        if self.render_mode == "human":
            self.render()
        return observation, delta / self.track.length, terminated, truncated, self._info()

    def render(self):
        if self.render_mode is None:
            return None
        if self._state is None:
            raise RuntimeError("call reset() before render()")
        if self._renderer is None:
            from rl_fun.racing.render import Renderer

            self._renderer = Renderer(
                self.track, self.render_mode, self.metadata["render_fps"], self.side_panel
            )
        state = self._state
        directions = np.stack(
            [np.cos(state.heading + self._angles), np.sin(state.heading + self._angles)], axis=1
        )
        ray_points = np.array([state.x, state.y]) + directions * self._distances[:, None]
        lines = [
            f"Скорость: {state.v_long:4.1f} м/с",
            f"Прогресс: {self._progress / self.track.length:6.1%}",
            f"Шаг: {self._steps}",
        ]
        return self._renderer.draw(state.x, state.y, state.heading, ray_points, lines)

    def close(self) -> None:
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def _observe(self) -> np.ndarray:
        state = self._state
        assert state is not None
        self._distances = cast_rays(
            self.track.boundary_segments,
            np.array([state.x, state.y]),
            state.heading,
            self._angles,
            self.ray_range,
        )
        motion = np.array(
            [
                state.v_long / self.dynamics.max_speed,
                state.v_lat / self.dynamics.max_speed,
                state.yaw_rate / self.dynamics.max_yaw_rate,
            ]
        )
        observation = np.concatenate([self._distances / self.ray_range, motion])
        return np.clip(observation, self.observation_space.low, 1.0).astype(np.float32)

    def _info(self) -> dict:
        return {
            "progress": self._progress / self.track.length,
            "lap": int(max(self._progress, 0.0) // self.track.length),
            "collided": self._collided,
            "speed": 0.0 if self._state is None else self._state.v_long,
            "steps": self._steps,
        }
```

- [ ] **Step 5: Зарегистрировать среду** — в `src/rl_fun/environments/__init__.py` дополнить словарь:

```python
LOCAL_ENVIRONMENTS = {
    "RLFun/StationaryBandit-v0": "rl_fun.environments.bandit:StationaryBanditEnv",
    "RLFun/Racing-v0": "rl_fun.racing.env:RacingEnv",
}
```

- [ ] **Step 6: Запустить тесты, кроме отрисовки**

Run: `uv run --all-groups pytest tests/racing/test_env.py -q -k "not render"`
Expected: PASS. Тесты с `render` упадут до Task 6 (нет `render.py`) — это ожидаемо.

- [ ] **Step 7: Полный набор и линтер**

Run: `uv run --all-groups pytest -q --deselect tests/racing/test_env.py::test_rgb_array_render_returns_a_frame --deselect tests/racing/test_env.py::test_human_render_mode_steps_without_error` и `uv run --all-groups ruff check src tests`
Expected: все PASS (включая существующие тесты регистрации и factory), ruff чистый.

- [ ] **Step 8: Commit**

```bash
git add src/rl_fun/racing/env.py src/rl_fun/environments/__init__.py
git commit -m "feat: add racing Gymnasium environment"
```

---

### Task 6: Отрисовка pygame

**Files:**
- Create: `tests/racing/test_render.py`, `src/rl_fun/racing/render.py`

- [ ] **Step 1: Написать падающие тесты** — `tests/racing/test_render.py`

```python
import numpy as np
import pygame
import pytest

from rl_fun.racing.render import GAME_HEIGHT, GAME_WIDTH, Renderer
from rl_fun.racing.track import load_track


class StubPanel:
    width = 200

    def __init__(self) -> None:
        self.calls: list[tuple[int, int]] = []

    def draw(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        self.calls.append((rect.width, rect.height))
        surface.fill((255, 0, 255))


def draw_once(renderer: Renderer) -> np.ndarray | None:
    return renderer.draw(60.0, 0.0, np.pi / 2, np.array([[60.0, 10.0]]), ["Скорость: 0"])


def test_rgb_array_frame_has_game_size():
    renderer = Renderer(load_track("oval"), "rgb_array", fps=30)
    try:
        frame = draw_once(renderer)
    finally:
        renderer.close()

    assert frame.shape == (GAME_HEIGHT, GAME_WIDTH, 3)
    assert frame.dtype == np.uint8


def test_frame_shows_road_and_grass():
    renderer = Renderer(load_track("oval"), "rgb_array", fps=30)
    try:
        frame = draw_once(renderer)
    finally:
        renderer.close()

    assert len({tuple(pixel) for pixel in frame.reshape(-1, 3)}) > 3


def test_side_panel_extends_frame_and_receives_its_area():
    panel = StubPanel()
    renderer = Renderer(load_track("oval"), "rgb_array", fps=30, side_panel=panel)
    try:
        frame = draw_once(renderer)
    finally:
        renderer.close()

    assert frame.shape == (GAME_HEIGHT, GAME_WIDTH + 200, 3)
    assert panel.calls == [(200, GAME_HEIGHT)]
    assert tuple(frame[10, GAME_WIDTH + 10]) == (255, 0, 255)


def test_human_mode_draw_returns_none():
    renderer = Renderer(load_track("oval"), "human", fps=1000)
    try:
        assert draw_once(renderer) is None
    finally:
        renderer.close()


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        Renderer(load_track("oval"), "ansi", fps=30)
```

- [ ] **Step 2: Убедиться, что падает**

Run: `uv run --all-groups pytest tests/racing/test_render.py -q`
Expected: ошибка импорта `rl_fun.racing.render`.

- [ ] **Step 3: Закоммитить красные тесты**

```bash
git add tests/racing/test_render.py
git commit -m "test: specify racing renderer"
```

- [ ] **Step 4: Реализация** — `src/rl_fun/racing/render.py`

```python
"""pygame renderer for the racing environment."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Protocol

import numpy as np
import pygame

from rl_fun.racing.track import Track

GAME_WIDTH = 900
GAME_HEIGHT = 640
MARGIN = 40
CAR_LENGTH = 4.5
CAR_WIDTH = 2.0

GRASS = (34, 110, 50)
ROAD = (70, 70, 75)
CURB_RED = (200, 40, 40)
CURB_WHITE = (235, 235, 235)
START_LINE = (240, 240, 240)
CAR = (240, 200, 40)
CAR_NOSE = (200, 30, 30)
RAY = (255, 255, 0)
TEXT = (255, 255, 255)
PANEL_BACKGROUND = (18, 18, 24)


class SidePanel(Protocol):
    """Optional widget drawn to the right of the track, for example a neural network view."""

    width: int

    def draw(self, surface: pygame.Surface, rect: pygame.Rect) -> None:
        """Draw into `surface`; `rect` is the panel area in the surface's own coordinates."""


class Renderer:
    """Draw the track, car, rays and HUD; show a window or return RGB frames."""

    def __init__(
        self, track: Track, mode: str, fps: int, side_panel: SidePanel | None = None
    ) -> None:
        if mode not in ("human", "rgb_array"):
            raise ValueError(f"unsupported render mode {mode!r}")
        self._track = track
        self._mode = mode
        self._fps = fps
        self._panel = side_panel
        panel_width = side_panel.width if side_panel is not None else 0
        self._size = (GAME_WIDTH + panel_width, GAME_HEIGHT)
        self._canvas = pygame.Surface(self._size)
        pygame.font.init()
        self._font = pygame.font.Font(None, 26)
        if mode == "human":
            pygame.display.init()
            self._window = pygame.display.set_mode(self._size)
            pygame.display.set_caption("RL Fun — гонки")
            self._clock = pygame.time.Clock()
        self._fit_track()
        self._background = self._draw_background()

    def draw(
        self,
        x: float,
        y: float,
        heading: float,
        ray_points: np.ndarray,
        lines: Sequence[str],
    ) -> np.ndarray | None:
        """Draw one frame. Returns an RGB array in `rgb_array` mode, otherwise None."""
        self._canvas.fill(PANEL_BACKGROUND)
        self._canvas.blit(self._background, (0, 0))
        origin = self._to_screen((x, y))
        for point in ray_points:
            end = self._to_screen(point)
            pygame.draw.line(self._canvas, RAY, origin, end, 1)
            pygame.draw.circle(self._canvas, RAY, end, 3)
        self._draw_car(x, y, heading)
        for index, text in enumerate(lines):
            self._canvas.blit(self._font.render(text, True, TEXT), (12, 10 + index * 24))
        if self._panel is not None:
            area = pygame.Rect(GAME_WIDTH, 0, self._panel.width, GAME_HEIGHT)
            self._panel.draw(
                self._canvas.subsurface(area), pygame.Rect(0, 0, area.width, area.height)
            )
        if self._mode == "human":
            self._window.blit(self._canvas, (0, 0))
            pygame.event.pump()
            pygame.display.flip()
            self._clock.tick(self._fps)
            return None
        return np.transpose(pygame.surfarray.array3d(self._canvas), (1, 0, 2))

    def close(self) -> None:
        """Release pygame resources."""
        pygame.display.quit()
        pygame.font.quit()

    def _fit_track(self) -> None:
        points = np.concatenate([self._track.left, self._track.right])
        low, high = points.min(axis=0), points.max(axis=0)
        span = np.maximum(high - low, 1e-9)
        self._center = (low + high) / 2
        self._scale = min(
            (GAME_WIDTH - 2 * MARGIN) / span[0], (GAME_HEIGHT - 2 * MARGIN) / span[1]
        )

    def _to_screen(self, point: Sequence[float]) -> tuple[float, float]:
        return (
            (point[0] - self._center[0]) * self._scale + GAME_WIDTH / 2,
            GAME_HEIGHT / 2 - (point[1] - self._center[1]) * self._scale,
        )

    def _draw_background(self) -> pygame.Surface:
        surface = pygame.Surface((GAME_WIDTH, GAME_HEIGHT))
        surface.fill(GRASS)
        left, right = self._track.left, self._track.right
        count = len(left)
        for i in range(count):
            j = (i + 1) % count
            quad = [
                self._to_screen(left[i]),
                self._to_screen(left[j]),
                self._to_screen(right[j]),
                self._to_screen(right[i]),
            ]
            pygame.draw.polygon(surface, ROAD, quad)
            pygame.draw.polygon(surface, ROAD, quad, 1)
        for i in range(count):
            j = (i + 1) % count
            color = CURB_RED if i % 2 == 0 else CURB_WHITE
            for ring in (left, right):
                pygame.draw.line(
                    surface, color, self._to_screen(ring[i]), self._to_screen(ring[j]), 5
                )
        pygame.draw.line(
            surface, START_LINE, self._to_screen(left[0]), self._to_screen(right[0]), 4
        )
        return surface

    def _draw_car(self, x: float, y: float, heading: float) -> None:
        cos, sin = math.cos(heading), math.sin(heading)
        half_length, half_width = CAR_LENGTH / 2, CAR_WIDTH / 2
        corners = [
            (half_length, half_width),
            (half_length, -half_width),
            (-half_length, -half_width),
            (-half_length, half_width),
        ]
        body = [
            self._to_screen((x + cx * cos - cy * sin, y + cx * sin + cy * cos))
            for cx, cy in corners
        ]
        pygame.draw.polygon(self._canvas, CAR, body)
        pygame.draw.line(self._canvas, CAR_NOSE, body[0], body[1], 3)
```

- [ ] **Step 5: Зелёные тесты отрисовки и среды**

Run: `uv run --all-groups pytest tests/racing -q`
Expected: все PASS, включая два теста отрисовки в `test_env.py`.

- [ ] **Step 6: Линтер**

Run: `uv run --all-groups ruff check src tests`
Expected: чистый.

- [ ] **Step 7: Commit**

```bash
git add src/rl_fun/racing/render.py
git commit -m "feat: add pygame racing renderer"
```

---

### Task 7: Ручное управление `scripts/play.py`

**Files:**
- Create: `tests/racing/test_play.py`, `src/rl_fun/racing/play.py`, `scripts/play.py`

- [ ] **Step 1: Написать падающие тесты** — `tests/racing/test_play.py`

```python
import os
import subprocess
import sys

import numpy as np
import pytest

from rl_fun.racing.play import action_from_keys, main, parse_args


def test_left_key_steers_left_and_up_key_accelerates():
    action = action_from_keys(left=True, right=False, up=True, down=False)

    assert action.dtype == np.float32
    assert action.tolist() == [1.0, 1.0]


def test_opposite_keys_cancel_out():
    action = action_from_keys(left=True, right=True, up=True, down=True)

    assert action.tolist() == [0.0, 0.0]


def test_right_and_down_keys_steer_right_and_brake():
    action = action_from_keys(left=False, right=True, up=False, down=True)

    assert action.tolist() == [-1.0, -1.0]


def test_parse_args_defaults():
    arguments = parse_args([])

    assert arguments.track == "oval"
    assert arguments.max_steps is None
    assert arguments.max_frames is None


def test_parse_args_reads_options():
    arguments = parse_args(["wavy", "--max-steps", "50", "--max-frames", "2"])

    assert (arguments.track, arguments.max_steps, arguments.max_frames) == ("wavy", 50, 2)


def test_main_runs_a_few_frames_in_a_dummy_window():
    assert main(["--max-frames", "3"]) == 0


def test_main_rejects_unknown_track():
    with pytest.raises(SystemExit) as error:
        main(["no-such-track", "--max-frames", "1"])

    assert error.value.code == 2


def test_script_help_is_in_russian():
    result = subprocess.run(
        [sys.executable, "scripts/play.py", "--help"],
        capture_output=True, text=True, encoding="utf-8",
        env={**os.environ, "PYTHONIOENCODING": "utf-8"}, timeout=60, check=False,
    )

    assert result.returncode == 0
    assert "трасс" in result.stdout.lower()
```

- [ ] **Step 2: Убедиться, что падает**

Run: `uv run --all-groups pytest tests/racing/test_play.py -q`
Expected: ошибка импорта `rl_fun.racing.play`.

- [ ] **Step 3: Закоммитить красные тесты**

```bash
git add tests/racing/test_play.py
git commit -m "test: specify interactive racing play script"
```

- [ ] **Step 4: Реализация** — `src/rl_fun/racing/play.py`

```python
"""Interactive keyboard driving for the racing environment."""

from __future__ import annotations

import argparse
from collections.abc import Sequence

import gymnasium as gym
import numpy as np
import pygame

from rl_fun.environments import register_environments


def action_from_keys(left: bool, right: bool, up: bool, down: bool) -> np.ndarray:
    """Map pressed keys to a [steer, throttle] action; positive steer turns left."""
    return np.array([float(left) - float(right), float(up) - float(down)], dtype=np.float32)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Ручное вождение по гоночной трассе. Стрелки или WASD — управление, "
        "R — перезапуск, Esc — выход."
    )
    parser.add_argument(
        "track", nargs="?", default="oval", help="Имя встроенной трассы или путь к JSON-файлу"
    )
    parser.add_argument("--max-steps", type=int, default=None, help="Лимит шагов эпизода")
    parser.add_argument(
        "--max-frames", type=int, default=None, help="Остановиться после N кадров (для проверок)"
    )
    return parser.parse_args(argv)


def _summary(info: dict) -> str:
    if info["collided"]:
        outcome = "авария"
    elif info["lap"] >= 1:
        outcome = "круг пройден"
    else:
        outcome = "время вышло"
    return f"Эпизод завершён: {outcome}, прогресс {info['progress']:.0%}"


def main(argv: Sequence[str] | None = None) -> int:
    """Run the interactive game loop."""
    parser = argparse.ArgumentParser()  # only used to report errors in the same style
    arguments = parse_args(argv)
    register_environments()
    options = {"track": arguments.track, "render_mode": "human"}
    if arguments.max_steps is not None:
        options["max_steps"] = arguments.max_steps
    try:
        env = gym.make("RLFun/Racing-v0", **options)
    except (OSError, ValueError) as error:
        parser.error(f"Не удалось создать среду: {error}")

    frames = 0
    try:
        env.reset()
        while arguments.max_frames is None or frames < arguments.max_frames:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return 0
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return 0
                if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                    env.reset()
            keys = pygame.key.get_pressed()
            action = action_from_keys(
                left=keys[pygame.K_LEFT] or keys[pygame.K_a],
                right=keys[pygame.K_RIGHT] or keys[pygame.K_d],
                up=keys[pygame.K_UP] or keys[pygame.K_w],
                down=keys[pygame.K_DOWN] or keys[pygame.K_s],
            )
            _, _, terminated, truncated, info = env.step(action)
            frames += 1
            if terminated or truncated:
                print(_summary(info))
                env.reset()
        return 0
    finally:
        env.close()
```

`scripts/play.py`:

```python
from rl_fun.racing.play import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Зелёные тесты и линтер**

Run: `uv run --all-groups pytest tests/racing/test_play.py -q` и `uv run --all-groups ruff check src tests scripts`
Expected: PASS, ruff чистый.

- [ ] **Step 6: Commit**

```bash
git add src/rl_fun/racing/play.py scripts/play.py
git commit -m "feat: add interactive racing play script"
```

---

### Task 8: Интеграция с runtime, README и финальная проверка

**Files:**
- Create: `configs/racing-random.json`, `tests/integration/test_racing_runtime.py`
- Modify: `README.md`

- [ ] **Step 1: Конфигурация** — `configs/racing-random.json`

```json
{
  "name": "racing-random",
  "environment": {"id": "RLFun/Racing-v0", "kwargs": {"track": "oval"}},
  "algorithm": {"id": "random", "kwargs": {}},
  "run": {"seeds": [11, 22], "total_steps": 300, "workers": 1, "output_root": "runs"},
  "evaluation": {"episodes": 2, "max_episode_steps": 300}
}
```

- [ ] **Step 2: Написать тест интеграции** — `tests/integration/test_racing_runtime.py`

```python
import json
import os
import subprocess
import sys
from pathlib import Path


def run_script(script: str, *args: object) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, script, *map(str, args)], capture_output=True, text=True,
        encoding="utf-8", env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        timeout=120, check=False,
    )


def racing_config(tmp_path: Path) -> Path:
    values = json.loads(Path("configs/racing-random.json").read_text(encoding="utf-8"))
    values["run"]["output_root"] = str(tmp_path / "runs")
    values["run"]["total_steps"] = 50
    values["evaluation"]["max_episode_steps"] = 50
    path = tmp_path / "config.json"
    path.write_text(json.dumps(values), encoding="utf-8")
    return path


def test_train_runs_random_policy_on_racing_environment(tmp_path: Path):
    result = run_script("scripts/train.py", racing_config(tmp_path))

    assert result.returncode == 0, result.stderr
    assert isinstance(json.loads(result.stdout), dict)
    summaries = list((tmp_path / "runs").glob("racing-random/seed-*/summary.json"))
    assert len(summaries) == 2
    assert all(
        json.loads(path.read_text(encoding="utf-8"))["status"] == "success" for path in summaries
    )


def test_evaluate_runs_random_policy_on_racing_environment(tmp_path: Path):
    result = run_script("scripts/evaluate.py", racing_config(tmp_path), "--seed", 11)

    assert result.returncode == 0, result.stderr
```

- [ ] **Step 3: Запустить**

Run: `uv run --all-groups pytest tests/integration/test_racing_runtime.py -v`
Expected: PASS. Если тест падает из-за несовместимости `random`-алгоритма или метрик с непрерывным action space, **остановиться и сообщить** (это находка в runtime, а не повод менять тест молча).

- [ ] **Step 4: README** — добавить в `README.md` перед разделом «Структура проекта» раздел и обновить список структуры (`src/rl_fun/racing/` — гоночная среда, `scripts/play.py` — ручное вождение):

````markdown
## Гоночная среда

`RLFun/Racing-v0` — машинка с лучевыми сенсорами на замкнутой 2D-трассе. Наблюдение: расстояния лучей, продольная и боковая скорость, угловая скорость. Действие: `[руль, газ]` в диапазоне `[-1, 1]`; положительный руль — поворот налево, отрицательный газ — торможение. Награда за шаг — прирост прогресса вдоль трассы в долях круга; эпизод завершается аварией (касание границы), полным кругом или лимитом шагов.

Ручное вождение (нужно окно):

```powershell
uv run --all-groups python scripts/play.py
uv run --all-groups python scripts/play.py wavy
```

Стрелки или WASD — управление, `R` — перезапуск, `Esc` — выход. Трасса — имя встроенной (`oval`, `wavy`) или путь к JSON: `{"name": "...", "width": 10, "centerline": [[x, y], ...]}`; точки осевой линии идут по замкнутому контуру без самопересечений.

Через runtime среда запускается как любая другая, например `uv run --all-groups python scripts/train.py configs/racing-random.json`. Параметры среды (`track`, `ray_angles_deg` — по умолчанию `[-90, -30, 0, 30, 90]`, `ray_range`, `dt`, `max_steps`, `laps`, `dynamics`) задаются в `environment.kwargs`. Сейчас доступна кинематическая модель `kinematic` без заноса; другие модели динамики подключаются через `make_dynamics`. В рендерер заложено место под боковую панель (например, схему нейросети), но самой панели пока нет.
````

- [ ] **Step 5: Финальная проверка**

Run: `uv run --all-groups pytest -q` и `uv run --all-groups ruff check .`
Expected: весь набор PASS (старые тесты включены), ruff чистый.

- [ ] **Step 6: Commit**

```bash
git add configs/racing-random.json tests/integration/test_racing_runtime.py README.md
git commit -m "docs: document racing environment and runtime integration"
```

- [ ] **Step 7: Ручная проверка пользователем** — запустить `uv run --all-groups python scripts/play.py` и проехать круг; убедиться, что машинка управляема, лучи видны, авария сбрасывает эпизод. (Выполняет пользователь: нужен реальный дисплей.)

---

## Self-Review

- **Покрытие спеки:** трасса/JSON/готовые трассы — Task 2; сменная динамика — Task 3; лучи — Task 4; среда, контракт, `info`, регистрация, ошибки — Task 5; pygame (`human`/`rgb_array`), место под панель — Task 6; `play.py` — Task 7; README, `pygame` в `pyproject.toml`, runtime-интеграция — Tasks 1 и 8. Модели, панель нейросети, редактор, трава, занос — не входят (спека §3).
- **Плейсхолдеры:** нет.
- **Согласованность типов:** `Track.project/contains/start_pose/progress_delta/boundary_segments/left/right/length/width`, `VehicleState`, `KinematicBicycle.max_speed/max_yaw_rate`, `cast_rays(segments, origin, heading, angles, max_range)`, `Renderer(track, mode, fps, side_panel).draw(x, y, heading, ray_points, lines)` и `GAME_WIDTH/GAME_HEIGHT` используются одинаково во всех задачах. Тест круга обращается к приватному `_progress` среды намеренно (единственное место).
