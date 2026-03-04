#!/usr/bin/env python3
"""
Combine all .txt files under data/cleaned/ into a single file at
data/combined/en.txt (one line per line from each source file).
"""

from pathlib import Path


def main():
    cleaned_dir = Path("data/cleaned")
    output_path = Path("data/training/en.txt")

    if not cleaned_dir.exists():
        print(f"Error: {cleaned_dir} not found")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for txt_file in sorted(cleaned_dir.rglob("*.txt")):
            with open(txt_file, encoding="utf-8") as f:
                for line in f:
                    line = line.rstrip("\n")
                    if line:
                        out.write(line + "\n")
                        total += 1

    print(f"Combined {total} lines into {output_path}")


if __name__ == "__main__":
    main()
