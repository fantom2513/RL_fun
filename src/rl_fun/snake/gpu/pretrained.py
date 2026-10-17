"""Ready-made GPU Snake models shipped with the package, for replay in the lab."""

from __future__ import annotations

import json
from pathlib import Path

from rl_fun.snake.config import SnakeConfig
from rl_fun.snake.presets import verification_status

ASSETS = Path(__file__).resolve().parent / "assets"
PRETRAINED = {
    "snake16-cycle-clone": {
        "size": 16,
        "file": "snake16-cycle-clone.pt",
        "evidence": "snake16-cycle-clone.json",
    }
}


class GpuPolicy:
    """Adapter that lets a lab game session drive a GPU-trained network."""

    observation_mode = "egocentric_cycle"
    action_adapter = None

    def __init__(self, model):
        self.model = model.eval()
        self.size = model.size
        self._window = 2 * model.size - 1

    def act(self, observation) -> int:
        import torch

        planes = torch.as_tensor(observation, dtype=torch.float32)
        planes = planes.view(1, self.model.channels, self._window, self._window)
        return int(self.model.greedy(planes)[0])


def load_pretrained(name: str, directory: Path = ASSETS) -> GpuPolicy:
    """Load a shipped model by name; the size is checked against the registry."""
    if not isinstance(name, str) or name not in PRETRAINED:
        raise ValueError("неизвестная готовая модель")
    from rl_fun.snake.gpu.checkpoint import load_model

    entry = PRETRAINED[name]
    model, _ = load_model(Path(directory) / entry["file"], size=entry["size"])
    return GpuPolicy(model)


def pretrained_rows(registry: dict = PRETRAINED, directory: Path = ASSETS) -> list[dict]:
    """Lab model-list rows of the shipped models whose evidence file can be read."""
    rows = []
    for name, entry in registry.items():
        try:
            evidence = json.loads((Path(directory) / entry["evidence"]).read_text(encoding="utf-8"))
            games, wins = int(evidence["games"]), int(evidence["wins"])
        except (OSError, ValueError, KeyError, TypeError):
            continue
        rows.append(
            {
                "id": name,
                "algorithm": "pretrained",
                "config": SnakeConfig(size=entry["size"]).to_dict(),
                "size": entry["size"],
                "created": 0,
                "status": "finished",
                "iteration": 0,
                "history": [],
                "builtin": True,
                "verification": {
                    "status": verification_status(entry["size"], games, wins),
                    "wins": wins,
                    "episodes": games,
                },
            }
        )
    return rows
