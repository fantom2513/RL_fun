import numpy as np
import pygame
import pytest

from rl_fun.racing.fleet import RacingFleet
from rl_fun.racing.fleet_view import FleetView, NetworkPanel, ViewClosed
from rl_fun.racing.generation import run_generation
from rl_fun.racing.render import GAME_HEIGHT, GAME_WIDTH


def network_state() -> tuple[list[np.ndarray], list[np.ndarray]]:
    matrices = [np.ones((3, 2)), -np.ones((2, 3))]
    activations = [np.array([0.5, -0.5]), np.zeros(3), np.array([0.1, 0.9])]
    return matrices, activations


def test_network_panel_draws_something_once_updated():
    panel = NetworkPanel(["a", "b"], ["x", "y"], width=300)
    panel.update(*network_state())
    surface = pygame.Surface((300, GAME_HEIGHT))
    surface.fill((0, 0, 0))

    panel.draw(surface, surface.get_rect())

    assert pygame.surfarray.array3d(surface).any()


def test_network_panel_without_state_does_not_crash():
    panel = NetworkPanel(["a"], ["x"], width=200)
    surface = pygame.Surface((200, GAME_HEIGHT))

    panel.draw(surface, surface.get_rect())


def test_fleet_view_frame_includes_network_panel():
    fleet = RacingFleet(5)
    fleet.reset()
    view = FleetView(fleet, mode="rgb_array", show_network=True)
    try:
        frame = view.draw(fleet, ["Поколение 1"], None)
    finally:
        view.close()

    assert frame.shape == (GAME_HEIGHT, GAME_WIDTH + view.panel.width, 3)


def test_fleet_view_without_network_has_game_size():
    fleet = RacingFleet(5)
    fleet.reset()
    view = FleetView(fleet, mode="rgb_array", show_network=False)
    try:
        frame = view.draw(fleet, [], None)
    finally:
        view.close()

    assert frame.shape == (GAME_HEIGHT, GAME_WIDTH, 3)


def test_dead_cars_are_drawn_differently_from_living_ones():
    fleet = RacingFleet(3)
    fleet.reset()
    view = FleetView(fleet, mode="rgb_array", show_network=False)
    try:
        alive_frame = view.draw(fleet, [], None)
        fleet.alive[1:] = False
        dead_frame = view.draw(fleet, [], None)
    finally:
        view.close()

    assert not np.array_equal(alive_frame, dead_frame)


def test_run_generation_draws_every_step_and_passes_network_state():
    fleet = RacingFleet(3, max_steps=5)
    view = FleetView(fleet, mode="rgb_array", show_network=True)
    networks: list[object] = []
    original = view.draw

    def spy(fleet_arg, lines, network):
        networks.append(network)
        return original(fleet_arg, lines, network)

    view.draw = spy
    try:
        run_generation(
            fleet, np.zeros((3, 2)), lambda w, o: np.array([0.0, 1.0]),
            view=view, inspect=lambda w, o: network_state(), label="Поколение 0",
        )
    finally:
        view.close()

    assert len(networks) == 5
    assert all(network is not None for network in networks)


def test_closed_window_error_propagates_from_run_generation():
    fleet = RacingFleet(2, max_steps=5)

    class ClosingView:
        def draw(self, fleet, lines, network):
            raise ViewClosed

    with pytest.raises(ViewClosed):
        run_generation(
            fleet, np.zeros((2, 1)), lambda w, o: np.zeros(2), view=ClosingView()
        )
