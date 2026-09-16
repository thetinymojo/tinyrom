from __future__ import annotations

import tempfile
from pathlib import Path

from tinyrom import TinyRom


HERE = Path(__file__).resolve().parent
OUT = Path(tempfile.gettempdir()) / "tinyrom-patched.bin"


def main() -> None:
    ecu = TinyRom(str(HERE / "factory_rom.bin"), str(HERE / "definitions.toml"))

    original_fuel = ecu.get_map("primary_fuel")
    edited_fuel = original_fuel.copy()
    edited_fuel *= 1.03

    ecu.patch_map("primary_fuel", edited_fuel)
    ecu.save(str(OUT))

    original = TinyRom(str(HERE / "factory_rom.bin"), str(HERE / "definitions.toml"))
    patched = TinyRom(str(OUT), str(HERE / "definitions.toml"))

    print("primary_fuel[0, 0]")
    print(f"  original: {original.get_map('primary_fuel')[0, 0]:.6g}")
    print(f"  patched:  {patched.get_map('primary_fuel')[0, 0]:.6g}")
    print(f"  output:   {OUT}")


if __name__ == "__main__":
    main()
