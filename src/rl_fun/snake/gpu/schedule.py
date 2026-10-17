"""Learning rate schedule: constant, then a linear fall to a fraction of the base rate."""


def learning_rate_at(
    step: int, total: int, base: float, decay_start: float = 0.7, floor: float = 0.02
) -> float:
    """Rate at ``step`` of ``total``: ``base`` until ``decay_start * total``, then linear to
    ``floor * base`` at ``total`` (and after)."""
    if total < 1:
        raise ValueError("число шагов должно быть положительным")
    if not 0 <= decay_start <= 1:
        raise ValueError("начало спада: число от 0 до 1")
    if not 0 <= floor <= 1:
        raise ValueError("конечная доля скорости обучения: число от 0 до 1")
    start = decay_start * total
    if step <= start:
        return base
    progress = 1.0 if total <= start else min(1.0, (step - start) / (total - start))
    return base * (1 - (1 - floor) * progress)
