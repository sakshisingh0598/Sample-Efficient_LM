#!/usr/bin/env python3
"""
convert_dialogues.py
--------------------
Read a file whose top level is a JSON list of records like

    { "persona": ..., "scenario": ..., "dialogue": [...], "image_text": "..." }

Remove all `"image_text"` keys, then transform each dialogue into a list of
turns (`role = user|assistant`) and write the result back to JSON.

Usage
-----
    python convert_dialogues.py INPUT_FILE OUTPUT_FILE
    # Use "-" for STDIN / STDOUT if you like:
    python convert_dialogues.py - -
"""

import argparse
import json
import sys
from typing import List, Dict, Any


def drop_image_text(records: List[Dict[str, Any]]) -> None:
    """In‑place removal of any 'image_text' fields."""
    for rec in records:
        rec.pop("image_text", None)   # quietly ignore if the key isn’t present


def convert_dialogues(records: List[Dict[str, Any]]) -> List[List[Dict[str, str]]]:
    """Return the ChatGPT‑style nested list of conversations."""
    conversations: List[List[Dict[str, str]]] = []

    for rec in records:
        turns: List[Dict[str, str]] = [
            {
                "role": "user" if idx % 2 == 0 else "assistant",
                "content": line["text"],
            }
            for idx, line in enumerate(rec.get("dialogue", []))
        ]
        if turns:
            conversations.append(turns)

    return conversations


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove image_text and reformat dialogue JSON.")
    parser.add_argument("input",  help="Input JSON file or '-' for STDIN")
    parser.add_argument("output", help="Output JSON file or '-' for STDOUT")
    args = parser.parse_args()

    # ---------- read ----------
    if args.input == "-":
        data = json.load(sys.stdin)
    else:
        with open(args.input, "r", encoding="utf-8") as f:
            data = json.load(f)

    # ---------- process ----------
    drop_image_text(data)
    transformed = convert_dialogues(data)

    # ---------- write ----------
    if args.output == "-":
        json.dump(transformed, sys.stdout, ensure_ascii=False, indent=2)
    else:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(transformed, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
