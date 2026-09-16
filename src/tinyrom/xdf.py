from __future__ import annotations

import argparse
import ast
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


def clean_name(name: str) -> str:
    clean = re.sub(r"[^a-zA-Z0-9]+", "_", name.lower())
    return re.sub(r"_+", "_", clean).strip("_") or "unnamed"


def read_xml_root(path: Path) -> ET.Element:
    text = path.read_text(encoding="utf-8", errors="ignore").lstrip("\ufeff")
    if "<XDFFORMAT" not in text:
        raise ValueError(
            f"{path} does not look like an XML XDF export. "
            "Unlock binary .xdf files first, then export them from TunerPro as XDF XML Definition."
        )
    return ET.fromstring(text)


def parse_int(value: str | None, default: int = 0) -> int:
    if value is None:
        return default
    try:
        return int(value, 0)
    except ValueError:
        return default


def parse_float(value: str | None, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def infer_dtype(bits: int, signed: bool = False, floating: bool = False) -> str:
    if floating:
        return {16: "f2", 32: "f4", 64: "f8"}.get(bits, "f4")
    prefix = "i" if signed else "u"
    return {8: f"{prefix}1", 16: f"{prefix}2", 32: f"{prefix}4", 64: f"{prefix}8"}.get(bits, "u1")


class LinearExprEvaluator(ast.NodeVisitor):
    def __init__(self, variable_name: str) -> None:
        self.variable_name = variable_name.lower()
        self.value = 0.0

    def evaluate(self, expression: str, variable_value: float) -> float:
        self.value = variable_value
        tree = ast.parse(expression, mode="eval")
        return float(self.visit(tree.body))

    def visit_BinOp(self, node: ast.BinOp) -> float:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        raise ValueError("unsupported operator")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> float:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.UAdd):
            return operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError("unsupported unary operator")

    def visit_Name(self, node: ast.Name) -> float:
        if node.id.lower() != self.variable_name:
            raise ValueError(f"unknown variable: {node.id}")
        return self.value

    def visit_Constant(self, node: ast.Constant) -> float:
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError("unsupported constant")

    def generic_visit(self, node: ast.AST) -> float:
        raise ValueError(f"unsupported expression node: {type(node).__name__}")


def derive_linear_math(equation: str | None, variable_name: str | None) -> tuple[float, float]:
    if not equation or not variable_name:
        return 1.0, 0.0

    evaluator = LinearExprEvaluator(variable_name)
    try:
        v0 = evaluator.evaluate(equation, 0.0)
        v1 = evaluator.evaluate(equation, 1.0)
        v2 = evaluator.evaluate(equation, 2.0)
    except Exception:
        return 1.0, 0.0

    factor = v1 - v0
    if abs((v2 - v1) - factor) > 1e-9:
        return 1.0, 0.0
    return factor, v0


def element_metadata(node: ET.Element) -> dict[str, Any]:
    embedded = node.find("EMBEDDEDDATA")
    if embedded is None:
        return {}

    bits = parse_int(embedded.get("mmedelementsizebits"), 8)
    type_flags = parse_int(embedded.get("mmedtypeflags"), 0)
    signed = bool(type_flags & 0x01)
    floating = bool(type_flags & 0x100)

    return {
        "address": embedded.get("mmedaddress"),
        "element_bits": bits,
        "datatype": infer_dtype(bits, signed=signed, floating=floating),
        "cols": parse_int(embedded.get("mmedcolcount"), 1),
        "rows": parse_int(embedded.get("mmedrowcount"), 1),
        "major_stride_bits": parse_int(embedded.get("mmedmajorstridebits"), 0),
        "minor_stride_bits": parse_int(embedded.get("mmedminorstridebits"), 0),
    }


def axis_metadata(axis: ET.Element) -> dict[str, Any]:
    embedded_meta = element_metadata(axis)
    math_node = axis.find("MATH")
    equation = None
    variable = None
    if math_node is not None:
        var_node = math_node.find("VAR")
        if var_node is not None:
            variable = var_node.get("id")
        equation = math_node.get("equation")
    factor, offset = derive_linear_math(equation, variable)

    labels = [label.get("value", "") for label in axis.findall("LABEL")]
    index_count = parse_int(axis.findtext("indexcount"), embedded_meta.get("cols", 1))

    return {
        **embedded_meta,
        "id": axis.get("id", ""),
        "units": axis.findtext("units"),
        "index_count": index_count,
        "decimal_places": parse_int(axis.findtext("decimalpl"), 0),
        "factor": factor,
        "offset": offset,
        "equation": equation or "X",
        "labels": labels,
    }


def parse_categories(root: ET.Element) -> dict[str, str]:
    categories: dict[str, str] = {}
    header = root.find("XDFHEADER")
    if header is None:
        return categories
    for category in header.findall("CATEGORY"):
        index = category.get("index")
        name = category.get("name")
        if index and name:
            categories[str(parse_int(index))] = name
    return categories


def parse_table(node: ET.Element, categories: dict[str, str]) -> dict[str, Any]:
    axes = {axis.get("id", ""): axis_metadata(axis) for axis in node.findall("XDFAXIS")}
    z_axis = axes.get("z", {})

    category_names = []
    for category_mem in node.findall("CATEGORYMEM"):
        category_id = category_mem.get("category")
        if category_id is not None:
            category_names.append(categories.get(str(parse_int(category_id)), category_id))

    return {
        "name": clean_name(node.findtext("title", "unnamed_table")),
        "title": node.findtext("title", "unnamed_table"),
        "type": "table",
        "unique_id": node.get("uniqueid"),
        "flags": node.get("flags"),
        "address": z_axis.get("address"),
        "shape": [z_axis.get("rows", 1), z_axis.get("cols", 1)],
        "datatype": z_axis.get("datatype", "u1"),
        "major_stride_bits": z_axis.get("major_stride_bits", 0),
        "minor_stride_bits": z_axis.get("minor_stride_bits", 0),
        "factor": z_axis.get("factor", 1.0),
        "offset": z_axis.get("offset", 0.0),
        "units": z_axis.get("units"),
        "categories": category_names,
        "x_axis": axes.get("x"),
        "y_axis": axes.get("y"),
        "z_axis": z_axis,
    }


def parse_flag(node: ET.Element, categories: dict[str, str]) -> dict[str, Any]:
    meta = element_metadata(node)

    category_names = []
    for category_mem in node.findall("CATEGORYMEM"):
        category_id = category_mem.get("category")
        if category_id is not None:
            category_names.append(categories.get(str(parse_int(category_id)), category_id))

    return {
        "name": clean_name(node.findtext("title", "unnamed_flag")),
        "title": node.findtext("title", "unnamed_flag"),
        "type": "flag",
        "unique_id": node.get("uniqueid"),
        "address": meta.get("address"),
        "datatype": meta.get("datatype", "u1"),
        "mask": node.findtext("mask"),
        "categories": category_names,
        "description": node.findtext("description"),
    }


def toml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def write_string_list(lines: list[str], key: str, values: list[str]) -> None:
    rendered = ", ".join(toml_string(value) for value in values)
    lines.append(f"{key} = [{rendered}]")


def write_stride_bits(lines: list[str], meta: dict[str, Any], prefix: str = "") -> None:
    key = f"{prefix}." if prefix else ""
    major = meta.get("major_stride_bits", 0) or 0
    minor = meta.get("minor_stride_bits", 0) or 0
    if major > 0:
        lines.append(f"{key}major_stride_bits = {major}")
    if minor > 0:
        lines.append(f"{key}minor_stride_bits = {minor}")


def write_axis(lines: list[str], prefix: str, axis: dict[str, Any] | None) -> None:
    if not axis:
        return

    for field in ("address", "units", "datatype", "equation"):
        value = axis.get(field)
        if value:
            lines.append(f"{prefix}.{field} = {toml_string(str(value))}")

    lines.append(f"{prefix}.index_count = {axis.get('index_count', 1)}")
    lines.append(f"{prefix}.decimal_places = {axis.get('decimal_places', 0)}")
    lines.append(f"{prefix}.factor = {axis.get('factor', 1.0)!r}")
    lines.append(f"{prefix}.offset = {axis.get('offset', 0.0)!r}")
    write_stride_bits(lines, axis, prefix)

    labels = axis.get("labels") or []
    if labels:
        write_string_list(lines, f"{prefix}.labels", labels)


def render_toml(definition_title: str, tables: list[dict[str, Any]], flags: list[dict[str, Any]], source_name: str) -> str:
    lines = [
        "# ==========================================",
        "# TinyRom ECU Definition File",
        f"# Auto-generated from: {source_name}",
        "# ==========================================",
        "",
        "[metadata]",
        f"title = {toml_string(definition_title)}",
        f"table_count = {len(tables)}",
        f"flag_count = {len(flags)}",
        "",
    ]

    for table in tables:
        lines.append(f"[tables.{table['name']}]")
        lines.append(f'title = {toml_string(table["title"])}')
        if table.get("unique_id"):
            lines.append(f'unique_id = {toml_string(table["unique_id"])}')
        if table.get("flags"):
            lines.append(f'flags = {toml_string(table["flags"])}')
        if table.get("address"):
            lines.append(f'address = {toml_string(table["address"])}')
        lines.append(f"shape = [{table['shape'][0]}, {table['shape'][1]}]")
        lines.append(f'datatype = {toml_string(table["datatype"])}')
        write_stride_bits(lines, table)
        lines.append(f"factor = {table['factor']!r}")
        lines.append(f"offset = {table['offset']!r}")
        if table.get("units"):
            lines.append(f'units = {toml_string(table["units"])}')
        if table.get("categories"):
            write_string_list(lines, "categories", table["categories"])

        write_axis(lines, "x_axis", table.get("x_axis"))
        write_axis(lines, "y_axis", table.get("y_axis"))
        write_axis(lines, "z_axis", table.get("z_axis"))
        lines.append("")

    for flag in flags:
        lines.append(f"[flags.{flag['name']}]")
        lines.append(f'title = {toml_string(flag["title"])}')
        if flag.get("unique_id"):
            lines.append(f'unique_id = {toml_string(flag["unique_id"])}')
        if flag.get("address"):
            lines.append(f'address = {toml_string(flag["address"])}')
        lines.append(f'datatype = {toml_string(flag["datatype"])}')
        if flag.get("mask"):
            lines.append(f'mask = {toml_string(flag["mask"])}')
        if flag.get("description"):
            lines.append(f'description = {toml_string(flag["description"])}')
        if flag.get("categories"):
            write_string_list(lines, "categories", flag["categories"])
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def convert_xdf(xdf_path: str, toml_path: str) -> None:
    source_path = Path(xdf_path)
    output_path = Path(toml_path)
    root = read_xml_root(source_path)

    categories = parse_categories(root)
    tables = [parse_table(node, categories) for node in root.findall("XDFTABLE")]
    flags = [parse_flag(node, categories) for node in root.findall("XDFFLAG")]

    definition_title = root.findtext("XDFHEADER/deftitle", source_path.stem)
    output = render_toml(definition_title, tables, flags, source_path.name)
    output_path.write_text(output, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert unlocked TunerPro XDF XML into TinyRom TOML.")
    parser.add_argument("input", help="Path to the unlocked .xdf or .xml file")
    parser.add_argument("output", nargs="?", help="Destination .toml path")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_suffix(".toml")
    convert_xdf(str(input_path), str(output_path))
    print(f"[+] Wrote {output_path}")


if __name__ == "__main__":
    main()
