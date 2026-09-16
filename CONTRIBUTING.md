# Contributing

Thanks for contributing to tinyrom.

## Scope

tinyrom is the core ROM math and definition-import library. Please keep contributions aligned with that boundary.

Good fits:

- ROM read/write primitives
- TOML schema helpers
- XDF XML to TOML conversion
- Tests
- Documentation
- Synthetic fixtures

Not good fits for this repo:

- Application workflows
- User interfaces
- Project-specific tuning policy
- Flashing workflows

## Local Setup

Use Python 3.11 or newer:

```bash
python3 --version
python3 -m pip install -e .
```

Useful checks:

```bash
python3 -m unittest tests.test_core
```

CI runs Ubuntu 3.11–3.14 and Windows 3.11.

## Fixtures

Do not commit proprietary ROMs, XDFs, installers, passwords, or vendor binaries.

Only commit:

- synthetic ROM examples
- redistributable XML fixtures
- small TOML samples

## Change Guidelines

- Keep `src/tinyrom/core.py` small and focused.
- Prefer deterministic math over clever abstractions.
- Add tests for behavior changes.
- Update docs when public behavior changes.
