from __future__ import annotations

import tomllib

import numpy as np


def _addr(value: str | int) -> int:
    return int(value, 16) if isinstance(value, str) else int(value)


class TinyRom:
    def __init__(self, bin_path: str, toml_path: str):
        with open(bin_path, "rb") as f:
            self.rom = bytearray(f.read())
        with open(toml_path, "rb") as f:
            self.definition = tomllib.load(f)
        self.tables = self.definition.get("tables", {})

    def raw(self, name: str) -> np.ndarray:
        table = self.tables[name]
        shape = rows, cols = tuple(table["shape"])
        dtype = np.dtype(table.get("datatype", "u1"))
        addr = _addr(table["address"])
        elem = int(dtype.itemsize)
        major = int(table.get("major_stride_bits") or 0)
        minor = int(table.get("minor_stride_bits") or 0)
        col = minor // 8 if minor > 0 else elem
        if rows == 1 and major > 0:  # TunerPro 1xN: major = cell step
            col = major // 8
        row = major // 8 if major > 0 and rows > 1 else col * cols
        last = addr + (rows - 1) * row + (cols - 1) * col + elem
        if last > len(self.rom):
            raise ValueError(f"{name} stride view ends at {last}, ROM size {len(self.rom)}")
        return np.ndarray(shape, dtype, self.rom, addr, (row, col))

    def get_map(self, name: str) -> np.ndarray:
        table = self.tables[name]
        return (self.raw(name) * table.get("factor", 1.0)) + table.get("offset", 0.0)

    def patch_map(self, name: str, matrix: np.ndarray) -> None:
        table = self.tables[name]
        target = self.raw(name)
        values = np.asarray(matrix)
        if values.shape != target.shape:
            raise ValueError(f"{name} expects shape {target.shape}, got {values.shape}")
        raw = np.rint((values - table.get("offset", 0.0)) / table.get("factor", 1.0))
        target[:] = raw.astype(target.dtype)

    def save(self, out_path: str) -> None:
        with open(out_path, "wb") as f:
            f.write(self.rom)
