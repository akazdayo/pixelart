# AGENTS.md

Guidance for agentic coding assistants working in this repository.
Read this file before editing code.

## Rule Files (Cursor/Copilot)

As of 2026-02-08, no additional Cursor or Copilot instruction files are present:
- `.cursorrules` (not found)
- `.cursor/rules/` (not found)
- `.github/copilot-instructions.md` (not found)

If any of these are added later, treat them as high-priority local rules.
Update this file to reflect those rules before making broad changes.

## Repository Quick Map

```text
pixelart/
|- main.py                 # Streamlit entrypoint
|- src/run.py              # main pipeline orchestration
|- src/draw.py             # Streamlit UI and controls
|- src/convert.py          # Rust wrapper + color conversion helpers
|- src/filters.py          # filters, mosaic, grid mask, dithering
|- src/ai.py               # KMeans palette generation
|- pages/                  # Streamlit subpages
|- color/*.csv             # RGB palette files
|- tests/test_convert.py   # pytest coverage for Convert class
|- pyproject.toml          # deps, lint/test/build config
```

## Environment and Setup

Preferred toolchain is `uv`.

```bash
# install project + dev dependencies
uv sync --dev

# run any tool in the project environment
uv run <command>
```

Optional Nix workflow (repo includes `flake.nix`):

```bash
nix develop
```

Python requirement: `>=3.12`.

## Build / Lint / Test Commands

### Run app

```bash
uv run streamlit run main.py
```

### Build distributions

```bash
uv build
uv build --wheel
```

### Lint and format

```bash
uv run ruff check src/ pages/ tests/ main.py
uv run ruff check src/convert.py
uv run ruff check --fix src/ pages/ tests/ main.py
uv run ruff format src/ pages/ tests/ main.py
uv run ruff format src/filters.py
```

### Type checking

`pyright` is configured in `pyproject.toml` but may not be installed.

```bash
uv run pyright src/
uvx pyright src/
```

### Tests (pytest)

Pytest defaults from `pyproject.toml`:
- `testpaths = ["tests"]`
- `addopts = "-v --cov=src --cov-report=term-missing"`

```bash
# run all tests
uv run pytest

# run a single test file
uv run pytest tests/test_convert.py -v

# run a single test class
uv run pytest tests/test_convert.py::TestResizeImage -v

# run a single test function (key pattern)
uv run pytest tests/test_convert.py::TestResizeImage::test_resize_large_image -v

# run a focused subset by keyword
uv run pytest -k "resize or alpha" -v

# skip coverage locally for faster iteration
uv run pytest tests/test_convert.py::TestResizeImage::test_resize_large_image -v --no-cov
```

## Code Style Guidelines

### Imports

Use three import groups in this order:
1. standard library
2. third-party packages
3. local project imports (`src.*`)

Keep import order changes minimal in untouched files.
Common aliases in this repo: `np`, `pd`, `st`; keep `cv2` unaliased.

### Formatting

- Indentation: 4 spaces
- Prefer double quotes
- Keep lines around 88-100 chars
- Use trailing commas in multiline literals/calls
- Use `ruff format` for wrapping and whitespace

### Type hints

Type hints are partial in this codebase.

- Always annotate `__init__` with `-> None`
- Add hints for complex return types (especially NumPy arrays)
- Use `numpy.typing.NDArray` and `typing.cast` where useful
- Do not mass-annotate legacy code unless needed for your change

### Naming conventions

- Functions/variables: `snake_case`
- Classes: `PascalCase`
- Modules/files: `snake_case.py`
- Constants are often `snake_case` in this repository

Common abbreviations: `img`, `conv`, `fdir`.

### Error handling

- Prefer explicit checks over broad `try/except`
- Raise `ValueError` for invalid user input (e.g., empty custom palette)
- Preserve RGB/RGBA behavior; do not silently drop alpha channels

### Comments and docstrings

- Keep comments brief; explain only non-obvious logic
- English and Japanese comments both exist; follow local style in touched files
- Use triple double-quoted docstrings when adding function docs

### Structure patterns

- Classes group related behavior (`Convert`, `EdgeFilter`, `ImageEnhancer`)
- Empty `__init__` is acceptable when consistent with nearby code
- Prefer small helper methods over deep inheritance

## Image Processing Conventions

- OpenCV arrays are BGR (not RGB)
- Palette CSV files store RGB values
- Handle alpha channels explicitly (`shape[2] == 4`)
- Preserve transparent-pixel semantics used in `src/convert.py`
- Resize very large images toward FullHD budget before heavy processing

## Testing Expectations

- If you touch `src/convert.py`, run targeted tests in `tests/test_convert.py`
- If you touch filters or color logic, run all tests
- Keep tests deterministic when practical (small synthetic NumPy fixtures)
- Add or update tests for bug fixes and behavior changes

## Agent Checklist Before Finishing

1. Run `uv run ruff check src/ pages/ tests/ main.py`.
2. Run relevant pytest command(s), ideally including a single-test node.
3. If tests cannot run, report exact failure and missing dependency.
4. Keep edits minimal and aligned with existing architecture.
5. Mention changed files and any follow-up verification steps.
