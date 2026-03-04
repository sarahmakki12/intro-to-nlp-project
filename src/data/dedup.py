#!/usr/bin/env python3
"""
Deduplicate the English training corpus by capping line occurrences.

Produces two output files:
  - en_capped.txt:  every line kept up to MAX_COUNT times (preserves original order)
  - en_unique.txt:  strictly one copy of each line (first-occurrence order)
"""

from collections import Counter
from pathlib import Path

MAX_COUNT = 10
INPUT = Path("data/training/en.txt")
CAPPED_OUTPUT = Path("data/training/en_capped.txt")
UNIQUE_OUTPUT = Path("data/training/en_unique.txt")


def main():
    if not INPUT.exists():
        print(f"Error: {INPUT} not found")
        return

    lines = INPUT.read_text(encoding="utf-8").splitlines()
    total = len(lines)

    counts: Counter[str] = Counter()
    seen: set[str] = set()
    capped: list[str] = []
    unique: list[str] = []

    for line in lines:
        if line not in seen:
            seen.add(line)
            unique.append(line)
        counts[line] += 1
        if counts[line] <= MAX_COUNT:
            capped.append(line)

    CAPPED_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    CAPPED_OUTPUT.write_text("\n".join(capped) + "\n", encoding="utf-8")
    UNIQUE_OUTPUT.write_text("\n".join(unique) + "\n", encoding="utf-8")

    removed = total - len(capped)
    print(f"Input:       {total:>8,} lines")
    print(f"Capped (≤{MAX_COUNT}): {len(capped):>8,} lines")
    print(f"Unique:      {len(unique):>8,} lines")
    print(f"Removed:     {removed:>8,} lines ({removed / total * 100:.1f}%)")


if __name__ == "__main__":
    main()
