# MasterThesis

Fleet allocation repo for master thesis 2025/26
Python code for cleaning datasets, machine learning models, optimiser and outputs

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

- `uv run pytest` (project is configured for a `tests/` folder; this may currently report no tests collected)
- `uv run ruff check .`
- `uv run python main.py` (sanity check; currently prints a placeholder message)
- `uv run black .` (format)

## Data / models

- This public repository is intended to contain source code only.
- Private datasets, trained model artifacts, generated optimiser outputs, and local notes are git-ignored and should remain local.
- If you reproduce the pipeline locally, keep raw customer data and derived scenario files out of version control unless they have been explicitly approved for publication.
- Customer-aware pipelines (`customer2` by default):
  - `uv run python Models/build_port_turnaround_dataset.py [--customer customer1]`
  - `uv run python QA/port_turnaround_dataset_qa.py [--customer customer1]`
  - `uv run python Models/port_turnaround_lookup.py [--customer customer1]`
  - `uv run python QA/port_turnaround_lookup_qa.py [--customer customer1]`
  - `uv run python Models/build_sailing_time_dataset.py [--customer customer1]`
  - `uv run python QA/sailing_time_dataset_qa.py [--customer customer1]`
  - `uv run python Models/sailing_time_lookup.py [--customer customer1]`
  - `uv run python Models/fit_port_turnaround_model.py [--customer customer1]`
  - `uv run python Models/fit_sailing_time_model.py [--customer customer1]`
  - Use `--list-customers` with any command to see available identifiers and their directory layout.

## Thesis demo

- The locked thesis demo scenario is currently the anonymised `customer1` sample bundle under `DataSets/Derived/Customer1/SampleScenario`
- Reproduce the thesis-style optimiser run with:
  - `uv run python Models/run_thesis_optimizer_demo.py --skip-gantt`
- Run the sample optimiser entry point with:
  - `uv run python Models/run_optimizer_sample.py`

## Citation / reproducibility

- In the thesis appendix or any external write-up, cite an exact tagged release or commit hash rather than only a moving GitHub branch URL.
- Retrieve the current commit hash locally with `git rev-parse HEAD`.

## Tips

- `uv sync --locked` reads the existing `uv.lock` to keep dependency resolution deterministic
- Use `uv run <cmd>` so dependencies from the virtualenv are always active (this works cross-platform)
- For Jupyter, launch with `uv run --group ml jupyter lab`
