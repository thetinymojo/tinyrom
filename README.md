# tinyrom

A ROM map library in under 60 lines.

Load the bin, slice tables as NumPy views, do `real = raw * factor + offset`, save. That's `src/tinyrom/core.py`. The rest is import.

TOML is the working schema. XDF XML converts in. No UI, no flash, no tuner policy.

## Quickstart

Use Python 3.11 or newer (`python3` on Unix, `py` on Windows):

```bash
python3 --version
```

Install tinyrom for local development:

```bash
python3 -m pip install -e .
```

Run the tests:

```bash
python3 -m unittest tests.test_core
```

Read sample maps:

```bash
python3 examples/read_maps.py
```

Patch a copied map and write a new ROM:

```bash
python3 examples/patch_copy.py
```

Convert a TunerPro XDF XML export into tinyrom TOML (synthetic fixture matches `definitions.toml`):

```bash
tinyrom-xdf2toml examples/definitions.xdf definitions.out.toml
```

## Open Source Notes

Do not commit proprietary ROMs, XDFs, passwords, vendor installers, or TunerPro artifacts. Source, docs, tests, and clean fixtures only.
