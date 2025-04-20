#!/usr/bin/env python3
"""
merge_json.py
-------------
Concatenate two JSON payloads “one after the other”.

Typical use‑case: both files contain a top‑level *list* (e.g., a list of
dialogues you just converted).  The script will do:

    merged = data_from_file1 + data_from_file2

If either file contains a dictionary instead of a list, the script falls back
to a shallow merge of the dictionary keys (keys in file2 override file1).

Usage:
    python merge_json.py FIRST.json SECOND.json OUTPUT.json
    # Use "-" for STDIN / STDOUT if you prefer pipes, e.g.:
    cat a.json b.json | python merge_json.py - - merged.json
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

def read_json(path: str) -> Any:
    if path == "-":
        return json.load(sys.stdin)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def write_json(obj: Any, path: str) -> None:
    if path == "-":
        json.dump(obj, sys.stdout, ensure_ascii=False, indent=2)
        return
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def merge(a: Any, b: Any) -> Any:
    """
    If both *a* and *b* are lists   -> return a + b
    If both are dicts               -> dict(**a, **b)  (b overrides)
    Otherwise                       -> raise TypeError
    """
    if isinstance(a, list) and isinstance(b, list):
        return a + b
    if isinstance(a, dict) and isinstance(b, dict):
        merged = a.copy()
        merged.update(b)
        return merged
    raise TypeError("Cannot merge: types do not match ({} vs {})".format(type(a), type(b)))

def main() -> None:
    parser = argparse.ArgumentParser(description="Concatenate two JSON files.")
    parser.add_argument("first",  help="First input JSON file or '-' for STDIN")
    parser.add_argument("second", help="Second input JSON file or '-' for STDIN")
    parser.add_argument("output", help="Output JSON file or '-' for STDOUT")
    args = parser.parse_args()

    data1 = read_json(args.first)
    data2 = read_json(args.second)
    merged = merge(data1, data2)
    write_json(merged, args.output)

if __name__ == "__main__":
    main()
