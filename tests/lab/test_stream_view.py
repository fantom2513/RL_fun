import numpy as np
import pytest
from rl_fun.lab.stream_view import StopRun, StreamView

from rl_fun.racing.fleet import RacingFleet

DT = 1 / 30


class FakeClock:
    """Monotonic time that moves only when the test or the fake sleep advances it."""

    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class FakeSleep:
    """Records every requested pause and moves the fake clock forward by it."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.calls: list[float] = []

    def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        self.clock.advance(seconds)


class FakePoll:
    """Returns the scripted batches one per call; empty lists once the script runs out."""

    def __init__(self, *batches: list[dict]) -> None:
        self.batches = list(batches)
        self.calls = 0

    def __call__(self) -> list[dict]:
        self.calls += 1
        return self.batches.pop(0) if self.batches else []


def _fleet() -> RacingFleet:
    fleet = RacingFleet(3, track="oval", max_steps=50)
    fleet.reset()
    return fleet


def _frames(sent: list[dict]) -> list[dict]:
    return [message for message in sent if message["t"] == "frame"]


def _make_view(
    clock: FakeClock,
    sleep: FakeSleep,
    poll: FakePoll,
    sent: list[dict],
    **kwargs: object,
) -> StreamView:
    return StreamView(
        sent.append,
        poll,
        clock=clock,
        sleep=sleep,
        **kwargs,  # type: ignore[arg-type]
    )


def test_frames_are_sent_at_about_fps():
    # Arrange
    clock = FakeClock()
    sleep = FakeSleep(clock)
    sent: list[dict] = []
    view = _make_view(clock, sleep, FakePoll(), sent, fps=10.0, dt=DT)
    fleet = _fleet()

    # Act
    for _ in range(30):
        view.draw(fleet, ["x"], None)
        clock.advance(DT)

    # Assert
    assert len(_frames(sent)) == 10


def test_max_speed_never_sleeps():
    # Arrange
    clock = FakeClock()
    sleep = FakeSleep(clock)
    sent: list[dict] = []
    view = _make_view(clock, sleep, FakePoll(), sent, speed="max")
    fleet = _fleet()

    # Act
    for _ in range(10):
        view.draw(fleet, ["x"], None)

    # Assert
    assert sleep.calls == []


def test_finite_speed_sleeps_one_step_interval_per_step():
    # Arrange
    clock = FakeClock()
    sleep = FakeSleep(clock)
    sent: list[dict] = []
    view = _make_view(clock, sleep, FakePoll(), sent, speed=1, dt=DT)
    fleet = _fleet()

    # Act
    for _ in range(30):
        view.draw(fleet, ["x"], None)

    # Assert
    assert sum(sleep.calls) == pytest.approx(30 * DT)


def test_speed_command_changes_pace():
    # Arrange
    clock = FakeClock()
    sleep = FakeSleep(clock)
    sent: list[dict] = []
    poll = FakePoll([{"cmd": "speed", "value": 4}])
    view = _make_view(clock, sleep, poll, sent, speed=1, dt=DT)
    fleet = _fleet()

    # Act
    for _ in range(8):
        view.draw(fleet, ["x"], None)

    # Assert
    assert sum(sleep.calls) == pytest.approx(8 * DT / 4)


def test_pause_blocks_until_resume():
    # Arrange
    clock = FakeClock()
    sleep = FakeSleep(clock)
    sent: list[dict] = []
    poll = FakePoll(
        [{"cmd": "pause"}],
        [],
        [{"cmd": "resume"}],
    )
    view = _make_view(clock, sleep, poll, sent, fps=30.0, dt=DT)
    fleet = _fleet()

    # Act
    view.draw(fleet, ["x"], None)

    # Assert
    assert poll.calls == 3
    assert sleep.calls == [pytest.approx(1 / 30), pytest.approx(1 / 30)]
    statuses = [message["status"] for message in sent if message["t"] == "status"]
    assert statuses == ["paused", "running"]
    assert len(_frames(sent)) >= 1


def test_stop_while_running_raises_stop_run():
    # Arrange
    clock = FakeClock()
    sleep = FakeSleep(clock)
    sent: list[dict] = []
    view = _make_view(clock, sleep, FakePoll([{"cmd": "stop"}]), sent)
    fleet = _fleet()

    # Act
    with pytest.raises(StopRun):
        view.draw(fleet, ["x"], None)

    # Assert
    assert view.pending_updates == []


def test_stop_while_paused_raises_stop_run():
    # Arrange
    clock = FakeClock()
    sleep = FakeSleep(clock)
    sent: list[dict] = []
    poll = FakePoll([{"cmd": "pause"}], [], [{"cmd": "stop"}])
    view = _make_view(clock, sleep, poll, sent)
    fleet = _fleet()

    # Act
    with pytest.raises(StopRun):
        view.draw(fleet, ["x"], None)

    # Assert
    assert poll.calls == 3


def test_update_command_is_queued_for_worker():
    # Arrange
    clock = FakeClock()
    sleep = FakeSleep(clock)
    sent: list[dict] = []
    poll = FakePoll(
        [{"cmd": "update", "params": {"mutation_rate": 0.5}}],
        [{"cmd": "update", "params": {"elite": 3}}],
    )
    view = _make_view(clock, sleep, poll, sent)
    fleet = _fleet()

    # Act
    view.draw(fleet, ["x"], None)
    view.draw(fleet, ["x"], None)

    # Assert
    assert view.pending_updates == [{"mutation_rate": 0.5}, {"elite": 3}]


def test_frames_carry_network_state_when_given():
    # Arrange
    clock = FakeClock()
    sleep = FakeSleep(clock)
    sent: list[dict] = []
    view = _make_view(clock, sleep, FakePoll(), sent)
    fleet = _fleet()
    network = ([np.ones((2, 3))], [np.zeros(3), np.zeros(2)])

    # Act
    view.draw(fleet, ["x"], network)
    frame = _frames(sent)[-1]

    # Assert
    assert frame["net"] is not None
    assert frame["net"]["matrices"] == [[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]
    assert frame["net"]["activations"] == [[0.0, 0.0, 0.0], [0.0, 0.0]]


def test_frames_without_network_have_null_net():
    # Arrange
    clock = FakeClock()
    sleep = FakeSleep(clock)
    sent: list[dict] = []
    view = _make_view(clock, sleep, FakePoll(), sent)
    fleet = _fleet()

    # Act
    view.draw(fleet, ["x"], None)
    frame = _frames(sent)[-1]

    # Assert
    assert frame["net"] is None


def test_frames_use_current_generation_number():
    # Arrange
    clock = FakeClock()
    sleep = FakeSleep(clock)
    sent: list[dict] = []
    view = _make_view(clock, sleep, FakePoll(), sent)
    view.current_generation = 7
    fleet = _fleet()

    # Act
    view.draw(fleet, ["x"], None)
    frame = _frames(sent)[-1]

    # Assert
    assert frame["gen"] == 7
