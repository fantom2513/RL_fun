import numpy as np
import pytest

from rl_fun.snake.gpu.cycle import hamiltonian_cycle

SIZES = (4, 8, 12, 16, 20)


@pytest.mark.parametrize("size", SIZES)
def test_every_cell_gets_a_unique_index(size):
    order = hamiltonian_cycle(size)

    assert sorted(order.ravel().tolist()) == list(range(size * size))


@pytest.mark.parametrize("size", SIZES)
def test_consecutive_indices_are_neighbours_including_the_closing_edge(size):
    order = hamiltonian_cycle(size)
    cells = np.zeros((size * size, 2), dtype=int)
    for y in range(size):
        for x in range(size):
            cells[order[y, x]] = (x, y)

    steps = np.abs(cells - np.roll(cells, -1, axis=0)).sum(axis=1)

    assert (steps == 1).all()


def test_order_grid_is_indexed_by_row_then_column():
    order = hamiltonian_cycle(8)

    assert order.shape == (8, 8)


@pytest.mark.parametrize("size", (3, 5, 7, 15))
def test_odd_sizes_are_rejected(size):
    with pytest.raises(ValueError, match="чётн"):
        hamiltonian_cycle(size)


@pytest.mark.parametrize("size", (0, 2, -4))
def test_too_small_sizes_are_rejected(size):
    with pytest.raises(ValueError):
        hamiltonian_cycle(size)


def test_non_integer_size_is_rejected():
    with pytest.raises(ValueError):
        hamiltonian_cycle(8.0)
