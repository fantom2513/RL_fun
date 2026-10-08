import json
from pathlib import Path


def test_readme_uses_russian_experiment_instructions():
    # Arrange
    path = Path("README.md")

    # Act
    readme = path.read_text(encoding="utf-8")

    # Assert
    assert "Запуск эксперимента" in readme


def test_authoring_guide_explains_new_environments_in_russian():
    # Arrange
    path = Path("docs/environment-authoring.md")

    # Act
    guide = path.read_text(encoding="utf-8") if path.exists() else ""

    # Assert
    assert "Как добавить новую среду" in guide


def test_notebook_markdown_contains_russian_learning_sections():
    # Arrange
    path = Path("notebooks/01_bandits/01_epsilon_greedy.ipynb")

    # Act
    notebook = json.loads(path.read_text(encoding="utf-8"))
    markdown = "\n".join(
        "".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "markdown"
    )

    # Assert
    assert {"Жадный алгоритм", "ε-жадный алгоритм"} <= {
        title for title in ("Жадный алгоритм", "ε-жадный алгоритм") if title in markdown
    }
