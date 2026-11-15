# Contributing

1. Fork the repository and create a feature branch.
2. Install the dev and semantic extras: `pip install -e ".[dev,semantic]"` (the full suite needs the `semantic` extra, as CI does).
3. Run `ruff check src/ tests/` and fix any lint errors.
4. Run `pytest` and ensure all tests pass.
5. Open a pull request with a clear description of your changes.
