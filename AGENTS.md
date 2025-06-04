# AGENTS

This repository contains a Python synthesizer and its accompanying tests.

## Guidelines

- Use **Python 3.10+** when running or modifying the code.
- Keep line length under **100 characters**.
- Use four spaces for indentation.
- Provide a module level docstring at the top of each new Python file. Document
  all public classes and functions with docstrings that start with a one line
  summary.
- Use `snake_case` for variables and functions and `PascalCase` for classes.

## Programmatic Checks

1. Install dependencies with `uv sync`.
2. Run the linter using `uv run ruff .` and fix issues when possible.
3. Execute the tests with `uv run pytest` before committing.

## Pull Requests

Summaries in pull request bodies should briefly describe the implemented changes
and mention the result of running the test suite.
