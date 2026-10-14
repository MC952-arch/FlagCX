#!/usr/bin/env python3
"""Show the DTK definitions needed to diagnose Hygon CI build failures."""

import re
import sys
from pathlib import Path
from typing import Optional


def print_numbered(text: str, first_line: int = 1) -> None:
    for line_number, line in enumerate(text.splitlines(), first_line):
        print(f"{line_number:5}: {line}")


def print_env_file(path: Path) -> None:
    print(f"=== {path} (complete file) ===", flush=True)
    if not path.is_file():
        print("File not found", flush=True)
        return
    print_numbered(path.read_text(errors="replace"))
    print("=== end DTK env.sh ===", flush=True)


def enum_end(text: str, opening_brace: int) -> Optional[int]:
    depth = 0
    for index in range(opening_brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                semicolon = text.find(";", index + 1)
                return semicolon + 1 if semicolon >= 0 else None
    return None


def print_handle_type_definitions(root: Path) -> None:
    print(f"=== hipMemRangeHandleType enum definitions under {root} ===", flush=True)
    if not root.is_dir():
        print("Include directory not found", flush=True)
        return

    # DTK may use a named enum or an anonymous enum with a typedef after `}`.
    pattern = re.compile(r"\benum(?:\s+[A-Za-z_]\w*)?\s*\{")
    found = False
    references = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in {".h", ".hpp"}:
            continue
        text = path.read_text(errors="replace")
        if "hipMemRangeHandleType" not in text:
            continue
        for line_number, line in enumerate(text.splitlines(), 1):
            if "hipMemRangeHandleType" in line:
                references.append(f"{path}:{line_number}: {line}")
        for match in pattern.finditer(text):
            end = enum_end(text, text.index("{", match.start(), match.end()))
            if end is None:
                continue
            start = text.rfind("\n", 0, match.start()) + 1
            if "hipMemRangeHandleType" not in text[start:end]:
                continue
            first_line = text.count("\n", 0, start) + 1
            print(f"--- {path}:{first_line} ---")
            print_numbered(text[start:end], first_line)
            found = True

    if not found:
        print("No complete hipMemRangeHandleType enum definition found")
    print("=== all hipMemRangeHandleType references ===")
    print("\n".join(references) if references else "No references found", flush=True)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: show_hygon_dtk_info.py <env.sh> <hip-include-dir>")
    print_env_file(Path(sys.argv[1]))
    print_handle_type_definitions(Path(sys.argv[2]))
