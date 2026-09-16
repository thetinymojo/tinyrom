from __future__ import annotations

import tempfile
import tomllib
import unittest
from pathlib import Path

import numpy as np

from tinyrom import TinyRom
from tinyrom.xdf import convert_xdf


ROOT = Path(__file__).resolve().parents[1]
ROM = str(ROOT / "examples" / "factory_rom.bin")
TOML = str(ROOT / "examples" / "definitions.toml")

FUEL_FACTOR = 1 / 3.641
IGN_FACTOR = 1 / 3.6408
IGN_OFFSET = 7282 / 3.6408

STRIDE_XDF = """\
<XDFFORMAT version="1.60">
  <XDFHEADER>
    <deftitle>stride-emit</deftitle>
  </XDFHEADER>
  <XDFTABLE uniqueid="0x1">
    <title>Fuel Rev Limiter</title>
    <XDFAXIS id="x">
      <EMBEDDEDDATA mmedelementsizebits="8" mmedmajorstridebits="-32" mmedminorstridebits="0" />
      <indexcount>2</indexcount>
      <LABEL index="0" value="Low" />
      <LABEL index="1" value="High" />
      <MATH equation="X"><VAR id="X" /></MATH>
    </XDFAXIS>
    <XDFAXIS id="z">
      <EMBEDDEDDATA mmedtypeflags="0x02" mmedaddress="0xAC93" mmedelementsizebits="16" mmedrowcount="1" mmedcolcount="2" mmedmajorstridebits="80" mmedminorstridebits="0" />
      <MATH equation="X/3.641"><VAR id="X" /></MATH>
    </XDFAXIS>
  </XDFTABLE>
  <XDFTABLE uniqueid="0x2">
    <title>Packed Table</title>
    <XDFAXIS id="z">
      <EMBEDDEDDATA mmedtypeflags="0x02" mmedaddress="0x10" mmedelementsizebits="8" mmedrowcount="2" mmedcolcount="2" mmedmajorstridebits="0" mmedminorstridebits="0" />
      <MATH equation="X"><VAR id="X" /></MATH>
    </XDFAXIS>
  </XDFTABLE>
</XDFFORMAT>
"""


def _write_pair(tmp: Path, rom: bytes, toml: str) -> tuple[str, str]:
    bin_path = tmp / "rom.bin"
    toml_path = tmp / "def.toml"
    bin_path.write_bytes(rom)
    toml_path.write_text(toml, encoding="utf-8")
    return str(bin_path), str(toml_path)


def _fuel_rom() -> bytearray:
    rom = bytearray(64)
    rom[0x10:0x12] = b"\x89\x88"
    rom[0x12:0x14] = b"\x7e\xbf"
    rom[0x1A:0x1C] = b"\x61\x8b"
    return rom


def _fuel_toml() -> str:
    return (
        "[tables.fuel_rev_limiter]\n"
        'address = "0x10"\n'
        "shape = [1, 2]\n"
        'datatype = "u2"\n'
        f"factor = {FUEL_FACTOR!r}\n"
        "offset = 0.0\n"
        "major_stride_bits = 80\n"
    )


class EmitTests(unittest.TestCase):
    def test_convert_emits_positive_stride_once(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            xdf_path = root / "stride.xdf"
            toml_path = root / "stride.toml"
            xdf_path.write_text(STRIDE_XDF, encoding="utf-8")
            convert_xdf(str(xdf_path), str(toml_path))
            text = toml_path.read_text(encoding="utf-8")
            data = tomllib.loads(text)

        self.assertIn("fuel_rev_limiter", data["tables"])
        self.assertNotIn("fuel_rev_limiter_low", data["tables"])
        self.assertNotIn("fuel_rev_limiter_high", data["tables"])
        self.assertEqual(data["tables"]["fuel_rev_limiter"]["major_stride_bits"], 80)
        self.assertEqual(data["tables"]["fuel_rev_limiter"]["z_axis"]["major_stride_bits"], 80)
        self.assertNotIn("major_stride_bits", data["tables"]["fuel_rev_limiter"].get("x_axis", {}))
        self.assertNotIn("major_stride_bits", data["tables"]["packed_table"])
        self.assertNotIn("major_stride_bits = 0", text)
        self.assertNotIn("major_stride_bits = -32", text)


class StridedMapTests(unittest.TestCase):
    def test_fuel_shaped_reads_strided_cells(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bin_path, toml_path = _write_pair(Path(tmp), bytes(_fuel_rom()), _fuel_toml())
            ecu = TinyRom(bin_path, toml_path)
            packed = np.frombuffer(ecu.rom, dtype=np.dtype("u2"), count=2, offset=0x10)

        self.assertTrue(np.allclose(ecu.get_map("fuel_rev_limiter"), [[9600.0, 9800.0]], atol=0.5))
        self.assertEqual(int(ecu.raw("fuel_rev_limiter")[0, 1]), 35681)
        self.assertNotEqual(int(packed[1]), 35681)

    def test_ignition_shaped_high_then_low(self) -> None:
        rom = bytearray(64)
        rom[0x10:0x12] = b"\x61\x8b"
        rom[0x12:0x14] = b"\x7e\xbf"
        rom[0x1A:0x1C] = b"\x89\x88"
        toml = (
            "[tables.ignition_rev_limiter]\n"
            'address = "0x10"\n'
            "shape = [1, 2]\n"
            'datatype = "u2"\n'
            f"factor = {IGN_FACTOR!r}\n"
            f"offset = {IGN_OFFSET!r}\n"
            "major_stride_bits = 80\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            bin_path, toml_path = _write_pair(Path(tmp), bytes(rom), toml)
            values = TinyRom(bin_path, toml_path).get_map("ignition_rev_limiter")
        self.assertTrue(np.allclose(values, [[11800.0, 11600.0]], atol=0.5))

    def test_patch_high_skips_gap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            bin_path, toml_path = _write_pair(root, bytes(_fuel_rom()), _fuel_toml())
            ecu = TinyRom(bin_path, toml_path)
            ecu.patch_map("fuel_rev_limiter", np.array([[9600.0, 10000.0]]))
            self.assertEqual(bytes(ecu.rom[0x12:0x14]), b"\x7e\xbf")
            self.assertNotEqual(bytes(ecu.rom[0x1A:0x1C]), b"\x61\x8b")
            out = str(root / "patched.bin")
            ecu.save(out)
            reloaded = TinyRom(out, toml_path)
            self.assertTrue(np.allclose(reloaded.get_map("fuel_rev_limiter"), [[9600.0, 10000.0]], atol=0.5))
            self.assertEqual(bytes(reloaded.rom[0x12:0x14]), b"\x7e\xbf")

    def test_sensor_shaped_steps_two_bytes(self) -> None:
        rom = bytearray(32)
        expected = np.arange(10, dtype=np.uint8).reshape(1, 10) * 10
        for i, value in enumerate(expected.flat):
            rom[i * 2] = int(value)
            rom[i * 2 + 1] = 0xFF
        toml = (
            "[tables.sensor_curve]\n"
            'address = "0x0"\n'
            "shape = [1, 10]\n"
            'datatype = "u1"\n'
            "factor = 1.0\n"
            "offset = 0.0\n"
            "major_stride_bits = 16\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            bin_path, toml_path = _write_pair(Path(tmp), bytes(rom), toml)
            values = TinyRom(bin_path, toml_path).get_map("sensor_curve")
            packed = np.frombuffer(rom, dtype=np.dtype("u1"), count=10, offset=0)
        self.assertTrue(np.array_equal(values, expected))
        self.assertFalse(np.array_equal(packed, expected.reshape(10)))

    def test_packed_fallback_matches_core_fuel(self) -> None:
        fuel = TinyRom(ROM, TOML).get_map("primary_fuel")
        self.assertEqual(fuel.shape, (16, 16))
        self.assertTrue(np.isclose(fuel[0, 0], 0.625))
        self.assertTrue(np.isclose(fuel[-1, -1], 1.796875))

    def test_oob_stride_raises(self) -> None:
        rom = bytearray(20)
        toml = (
            "[tables.fuel_rev_limiter]\n"
            'address = "0x10"\n'
            "shape = [1, 2]\n"
            'datatype = "u2"\n'
            "factor = 1.0\n"
            "offset = 0.0\n"
            "major_stride_bits = 80\n"
        )
        with tempfile.TemporaryDirectory() as tmp:
            bin_path, toml_path = _write_pair(Path(tmp), bytes(rom), toml)
            ecu = TinyRom(bin_path, toml_path)
            with self.assertRaises(ValueError):
                ecu.raw("fuel_rev_limiter")


if __name__ == "__main__":
    unittest.main()
