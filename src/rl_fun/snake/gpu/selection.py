"""Keep track of the best checkpoint by greedy win rate, then occupancy."""


class BestTracker:
    def __init__(self):
        self.rank: tuple[float, float] | None = None
        self.iteration: int | None = None

    def update(self, report: dict, iteration: int) -> bool:
        """Record the report if it beats the best so far; return whether it did."""
        rank = (report["win_rate"], report["occupancy"])
        if self.rank is not None and rank <= self.rank:
            return False
        self.rank, self.iteration = rank, iteration
        return True
