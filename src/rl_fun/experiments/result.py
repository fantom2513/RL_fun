from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class RunSummary:
    status: Literal["success", "failure", "cancelled"]
    seed: int
    run_dir: Path
    metrics: dict[str, float]
    elapsed_seconds: float
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["run_dir"] = str(self.run_dir)
        return result
