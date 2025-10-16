# MasterThesis

Fleet allocation AI repo for master thesis 2025/26

## Tooling
- Managed with [uv](https://astral.sh/uv) for Python packaging and virtualenvs
- Python 3.12 is pinned by `.python-version`
- VS Code is the primary editor used in this project

## Install uv
- macOS: `brew install uv` or `curl -Ls https://astral.sh/uv/install.sh | sh`
- Windows (PowerShell): `winget install Astral.Uv` or `powershell -ExecutionPolicy Bypass -Command "irm https://astral.sh/uv/install.ps1 | iex"`

## Environment setup

### macOS / Linux
1. `uv python install` (ensures Python 3.12 is available)
2. `uv venv`
3. `uv sync --locked`
4. `code .` then pick `./.venv/bin/python` in VS Code

### Windows (PowerShell)
1. `uv python install`
2. `uv venv`
3. `uv sync --locked`
4. `code .` then select `./.venv/Scripts/python.exe` as the interpreter

If PowerShell blocks the install script, run PowerShell as Administrator once and execute `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

## Common tasks
- `uv run python -m pytest`
- `uv run ruff check .`
- `uv run python main.py`
- `uv run black .` (format)

## Data / models
- `DataSets/` and `Models/` are git-ignored stash directories for local assets
- Keep raw data out of version control unless explicitly scrubbed
- Customer-aware pipelines (Northern Lights by default):
  - `uv run python Models/build_port_turnaround_dataset.py [--customer stena]`
  - `uv run python QA/port_turnaround_dataset_qa.py [--customer stena]`
  - `uv run python Models/port_turnaround_lookup.py [--customer stena]`
  - `uv run python QA/port_turnaround_lookup_qa.py [--customer stena]`
  - Use `--list-customers` with any command to see available identifiers and their directory layout.

## Tips
- `uv sync --locked` reads the existing `uv.lock` to keep dependency resolution deterministic
- Use `uv run <cmd>` so dependencies from the virtualenv are always active (this works cross-platform)
- For Jupyter, launch with `uv run --group ml jupyter lab`
