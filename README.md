# MasterThesis

Fleet allocation AI repo for master thesis 2025/26

# Dev setup

brew install uv
uv venv
uv sync --locked
code . # then select ./.venv/bin/python in VS Code

# Common tasks

uv run python -m pytest -q
uv run ruff check .
uv run python main.py
