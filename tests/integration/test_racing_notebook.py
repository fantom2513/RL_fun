import json
import os
import re
import subprocess
import sys
from pathlib import Path

PATH = Path("notebooks/02_racing/01_evolution.ipynb")
STUBS = ("init_population", "forward", "next_generation", "inspect")
REFERENCE_CELLS = {
    "init_population": (
        "from rl_fun.racing import reference as ref\n"
        "def init_population(rng, n, n_weights):\n"
        "    return ref.init_population(rng, n, SIZES, 1.0)\n"
    ),
    "forward": (
        "def forward(weights, observation):\n"
        "    return ref.forward(weights, observation, SIZES)\n"
    ),
    "next_generation": (
        "def next_generation(population, fitness, rng):\n"
        "    params = ref.EvolutionParams(population=len(population), elite=2)\n"
        "    return ref.next_generation(population, fitness, rng, params)\n"
    ),
    "inspect": (
        "def inspect(weights, observation):\n"
        "    return ref.inspect(weights, observation, SIZES)\n"
    ),
}


def load() -> dict:
    return json.loads(PATH.read_text(encoding="utf-8"))


def source(cell: dict) -> str:
    return "".join(cell["source"])


def stub_cell(notebook: dict, name: str) -> dict:
    matches = [
        cell for cell in notebook["cells"]
        if cell["cell_type"] == "code" and re.search(rf"def {name}\(", source(cell))
    ]
    assert len(matches) == 1, f"expected exactly one cell defining {name}"
    return matches[0]


def test_every_stub_cell_is_a_stub():
    notebook = load()

    for name in STUBS:
        assert "NotImplementedError" in source(stub_cell(notebook, name))


def test_stub_cells_do_not_contain_a_solution():
    notebook = load()

    for name in STUBS:
        text = source(stub_cell(notebook, name))
        for giveaway in ("tanh", "argsort", "rng.normal", "np.dot", "@ "):
            assert giveaway not in text


def test_every_code_cell_follows_a_russian_explanation():
    cells = load()["cells"]
    markdown = [source(c) for c in cells if c["cell_type"] == "markdown"]

    assert len(markdown) >= 8
    assert any(re.search("[А-Яа-я]", text) for text in markdown)
    for index, cell in enumerate(cells):
        if cell["cell_type"] == "code":
            assert index > 0 and cells[index - 1]["cell_type"] == "markdown", index


def test_reference_is_used_only_in_comparison_sections():
    heading = ""
    for cell in load()["cells"]:
        text = source(cell)
        if cell["cell_type"] == "markdown":
            headings = [line for line in text.splitlines() if line.startswith("#")]
            heading = headings[0].lower() if headings else heading
        elif "reference" in text or "benchmark" in text:
            assert "эталон" in heading, heading


def test_notebook_runs_end_to_end_with_reference_solution(tmp_path: Path):
    notebook = load()
    for name in STUBS:
        stub_cell(notebook, name)["source"] = REFERENCE_CELLS[name].splitlines(keepends=True)
    for cell in notebook["cells"]:
        text = source(cell)
        if "GENERATIONS =" in text:
            text = re.sub(r"GENERATIONS = \d+", "GENERATIONS = 2", text)
            text = re.sub(r"POPULATION = \d+", "POPULATION = 8", text)
            text = re.sub(r"MAX_STEPS = \d+", "MAX_STEPS = 100", text)
            text = re.sub(r"SHOW_WINDOW = \w+", "SHOW_WINDOW = False", text)
            cell["source"] = text.splitlines(keepends=True)
    prepared = tmp_path / "prepared.ipynb"
    prepared.write_text(json.dumps(notebook), encoding="utf-8")

    completed = subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
         "--ExecutePreprocessor.timeout=300", "--output", str(tmp_path / "executed.ipynb"),
         str(prepared)],
        capture_output=True, text=True, timeout=360, check=False,
        env={**os.environ, "SDL_VIDEODRIVER": "dummy", "SDL_AUDIODRIVER": "dummy"},
    )

    assert completed.returncode == 0, completed.stderr
