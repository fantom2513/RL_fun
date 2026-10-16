"""Hamiltonian cycle of a square grid, the backbone of the teacher and its observation."""

import numpy as np


def hamiltonian_cycle(size: int) -> np.ndarray:
    """Cycle index of every cell, indexed ``[y, x]``; only even sides admit a cycle."""
    if type(size) is not int:
        raise ValueError("размер поля должен быть целым числом")
    if size % 2:
        raise ValueError("цикл по клеткам существует только для чётной стороны поля")
    if size < 4:
        raise ValueError("размер поля должен быть не меньше 4")
    path = [(x, 0) for x in range(size)]
    for y in range(1, size):
        columns = range(size - 1, 0, -1) if y % 2 else range(1, size)
        path += [(x, y) for x in columns]
    path += [(0, y) for y in range(size - 1, 0, -1)]
    order = np.empty((size, size), dtype=np.int64)
    for index, (x, y) in enumerate(path):
        order[y, x] = index
    if (size // 2) % 2:
        # Run the cycle backwards so the standard start (middle row, heading right) follows it.
        order = size * size - 1 - order
    return order
