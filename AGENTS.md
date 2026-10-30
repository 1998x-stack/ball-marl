# Repository Guidelines

## Project Structure & Module Organization

This is a Python 3.10+ multi-agent reinforcement learning sandbox. `specifications/*.yaml` defines the proposed rules and contracts; `engine/` implements simulation, while `observations/`, `actions/`, and `rewards/` define agent-facing behavior. `environments/` adapts the engine to the training API. Learning and experiments live in `algorithms/`, `training/`, `evaluation/`, `credit_assignment/`, and `league/`. The local browser game is in `game_server/`, with HTML, CSS, and JavaScript under `game_server/static/`. Put regression tests in `tests/test_*.py`, supporting scripts in `scripts/`, and design notes in `docs/`. Local checkpoints and reports belong under `runs/`.

## Build, Test, and Development Commands

- `python -m pip install -e '.[test]'` installs the core package and pytest.
- `python -m pytest -q` runs the repository test suite.
- `python -m training.cli train --episodes 5 --steps 40 --out runs/demo` runs a small training experiment.
- `python -m scripts.release_gate --out runs/gate-local-001` checks compilation, tests, training, inference, evaluation, credit assignment, and benchmarking. Use a fresh output directory.
- `python -m pip install -e '.[game,game-test,test]'` and `python -m game_server --host 127.0.0.1 --port 8000` install and run the local web game.

## Coding Style & Naming Conventions

Use four spaces for Python indentation and follow the style of the module you edit. Name modules and functions in `snake_case`, classes in `PascalCase`, and tests `test_<behavior>.py` with `test_` functions. Keep the YAML contracts and Python implementations aligned when changing rules or observation/action shapes. No formatter or linter is configured in `pyproject.toml`; avoid unrelated formatting changes.

## Testing Guidelines

Pytest discovers tests in `tests/`. Add focused regression coverage for changes to deterministic snapshots, mass accounting, rewards, training artifacts, or server behavior. Run `python -m pytest -q` before opening a pull request. The optional PettingZoo compatibility checks require `python -m pip install -e '.[pettingzoo,test]'`.

## Commit & Pull Request Guidelines

Recent commits use Conventional Commit-style subjects such as `feat(arcade): ...`, `fix(pettingzoo): ...`, and `test(perf): ...`; use a short imperative subject with a relevant scope. In pull requests, describe the behavior change, note affected contracts or rules, list the commands run, and link related issues. Include screenshots for browser UI changes and keep generated `runs/` artifacts out of source changes.
