from __future__ import annotations

import tempfile
import tomllib
import unittest
from pathlib import Path

import numpy as np

from tinyrom import TinyRom
from tinyrom.xdf import convert_xdf


ROOT = Path(__file__).resolve().parents[1]

STRIDED_XDF = """\
<XDFFORMAT version="1.60">
  <XDFHEADER>
    <flags>0x0</flags>
    <deftitle>SYNTHETIC-STRIDE</deftitle>
    <CATEGORY index="0x0" name="Limiter" />
    <CATEGORY index="0x1" name="Sensor" />
  </XDFHEADER>
  <XDFTABLE uniqueid="0x1" flags="0x30">
    <title>Fuel Rev Limiter</title>
    <CATEGORYMEM index="0" category="0" />
    <XDFAXIS id="x">
      <EMBEDDEDDATA mmedelementsizebits="8" mmedmajorstridebits="-32" mmedminorstridebits="0" />
      <indexcount>2</indexcount>
      <LABEL index="0" value="Low" />
      <LABEL index="1" value="High" />
      <MATH equation="X"><VAR id="X" /></MATH>
    </XDFAXIS>
    <XDFAXIS id="z">
      <EMBEDDEDDATA mmedtypeflags="0x02" mmedaddress="0x10" mmedelementsizebits="16" mmedrowcount="1" mmedcolcount="2" mmedmajorstridebits="80" mmedminorstridebits="0" />
      <MATH equation="X/3.641"><VAR id="X" /></MATH>
    </XDFAXIS>
  </XDFTABLE>
  <XDFTABLE uniqueid="0x2" flags="0x30">
    <title>Ignition Rev Limiter</title>
    <CATEGORYMEM index="0" category="0" />
    <XDFAXIS id="x">
      <EMBEDDEDDATA mmedelementsizebits="8" mmedmajorstridebits="-32" mmedminorstridebits="0" />
      <indexcount>2</indexcount>
      <LABEL index="0" value="High" />
      <LABEL index="1" value="Low" />
      <MATH equation="X"><VAR id="X" /></MATH>
    </XDFAXIS>
    <XDFAXIS id="z">
      <EMBEDDEDDATA mmedtypeflags="0x02" mmedaddress="0x20" mmedelementsizebits="16" mmedrowcount="1" mmedcolcount="2" mmedmajorstridebits="80" mmedminorstridebits="0" />
      <MATH equation="(X+7282)/3.6408"><VAR id="X" /></MATH>
    </XDFAXIS>
  </XDFTABLE>
  <XDFTABLE uniqueid="0x3" flags="0x30">
    <title>Sensor Curve</title>
    <CATEGORYMEM index="0" category="1" />
    <XDFAXIS id="x">
      <EMBEDDEDDATA mmedtypeflags="0x02" mmedaddress="0x40" mmedelementsizebits="8" mmedcolcount="10" mmedmajorstridebits="16" mmedminorstridebits="0" />
      <indexcount>10</indexcount>
      <MATH equation="X"><VAR id="X" /></MATH>
    </XDFAXIS>
    <XDFAXIS id="z">
      <EMBEDDEDDATA mmedtypeflags="0x02" mmedaddress="0x30" mmedelementsizebits="8" mmedrowcount="1" mmedcolcount="10" mmedmajorstridebits="16" mmedminorstridebits="0" />
      <MATH equation="X"><VAR id="X" /></MATH>
    </XDFAXIS>
  </XDFTABLE>
</XDFFORMAT>
"""


def _synthetic_rom() -> bytearray:
    rom = bytearray(80)
    rom[0x10:0x12] = b"\x89\x88"
    rom[0x12:0x14] = b"\x7e\xbf"
    rom[0x1A:0x1C] = b"\x61\x8b"
    rom[0x20:0x22] = b"\x61\x8b"
    rom[0x22:0x24] = b"\x7e\xbf"
    rom[0x2A:0x2C] = b"\x89\x88"
    for i in range(10):
        rom[0x30 + i * 2] = (i + 1) * 10
        rom[0x30 + i * 2 + 1] = 0xFF
    return rom


class StridedXdfTests(unittest.TestCase):
    def test_convert_round_trip_reads_stride(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            xdf_path = root / "strided.xdf"
            toml_path = root / "strided.toml"
            bin_path = root / "synthetic.bin"
            xdf_path.write_text(STRIDED_XDF, encoding="utf-8")
            bin_path.write_bytes(_synthetic_rom())
            convert_xdf(str(xdf_path), str(toml_path))

            text = toml_path.read_text(encoding="utf-8")
            data = tomllib.loads(text)
            tables = data["tables"]
            self.assertIn("fuel_rev_limiter", tables)
            self.assertIn("ignition_rev_limiter", tables)
            self.assertIn("sensor_curve", tables)
            self.assertNotIn("fuel_rev_limiter_low", tables)
            self.assertNotIn("ignition_rev_limiter_high", tables)
            self.assertEqual(tables["fuel_rev_limiter"]["major_stride_bits"], 80)
            self.assertEqual(tables["ignition_rev_limiter"]["major_stride_bits"], 80)
            self.assertEqual(tables["sensor_curve"]["major_stride_bits"], 16)
            self.assertEqual(tables["sensor_curve"]["x_axis"]["major_stride_bits"], 16)
            self.assertNotIn("major_stride_bits", tables["fuel_rev_limiter"].get("x_axis", {}))
            self.assertNotIn("major_stride_bits = -32", text)

            ecu = TinyRom(str(bin_path), str(toml_path))
            fuel = ecu.get_map("fuel_rev_limiter")
            ign = ecu.get_map("ignition_rev_limiter")
            self.assertTrue(np.allclose(fuel, [[9600.0, 9800.0]], atol=0.5))
            self.assertTrue(np.allclose(ign, [[11800.0, 11600.0]], atol=0.5))
            self.assertTrue(
                np.array_equal(ecu.get_map("sensor_curve"), np.arange(10, 110, 10, dtype=np.float64).reshape(1, 10))
            )
            packed = np.frombuffer(ecu.rom, dtype=np.dtype("u2"), count=2, offset=0x10)
            self.assertNotEqual(int(packed[1]), 35681)

            ecu.patch_map("fuel_rev_limiter", np.array([[9600.0, 10000.0]]))
            out = str(root / "patched.bin")
            ecu.save(out)
            reloaded = TinyRom(out, str(toml_path))
            self.assertTrue(np.allclose(reloaded.get_map("fuel_rev_limiter"), [[9600.0, 10000.0]], atol=0.5))
            self.assertEqual(bytes(reloaded.rom[0x12:0x14]), b"\x7e\xbf")

    def test_sample_definitions_xdf_emits_no_stride(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            toml_path = Path(tmp) / "definitions.toml"
            convert_xdf(str(ROOT / "examples" / "definitions.xdf"), str(toml_path))
            text = toml_path.read_text(encoding="utf-8")
        self.assertNotIn("major_stride_bits", text)
        self.assertNotIn("minor_stride_bits", text)


if __name__ == "__main__":
    unittest.main()
