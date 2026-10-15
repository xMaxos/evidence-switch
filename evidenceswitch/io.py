"""Strict JSONL input and non-overwriting output helpers."""

import json
from pathlib import Path


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


def parse_json(value):
    return json.loads(value, object_pairs_hook=_unique_object,
                      parse_constant=_invalid_constant)


def read_jsonl(path, *, allow_empty=False):
    rows = []
    with Path(path).open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = parse_json(line)
            except ValueError as error:
                raise ValueError(f"Invalid JSON at line {number} in {path}") from error
            if not isinstance(row, dict):
                raise ValueError(f"Expected an object at line {number} in {path}")
            rows.append(row)
    if not rows and not allow_empty:
        raise ValueError(f"Empty JSONL file: {path}")
    return rows


def write_jsonl(path, rows):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write("\n")
