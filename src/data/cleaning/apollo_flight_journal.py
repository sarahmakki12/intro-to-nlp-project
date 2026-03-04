#!/usr/bin/env python3
"""
Extract and clean astronaut dialogue from Apollo Flight Journal HTML transcripts.

For each HTML file in data/raw/Apollo Flight Journal/, produces a cleaned .txt
file in data/cleaned/Apollo Flight Journal/ containing only astronaut lines with
timestamps, speaker labels, and annotations removed.
"""

import html
import os
import re
from pathlib import Path

CREW = {
    7: ["Schirra", "Eisele", "Cunningham"],
    8: ["Borman", "Lovell", "Anders"],
    9: ["McDivitt", "Scott", "Schweickart"],
    10: ["Stafford", "Young", "Cernan"],
    11: ["Armstrong", "Collins", "Aldrin"],
    12: ["Conrad", "Gordon", "Bean"],
    13: ["Lovell", "Swigert", "Haise"],
    14: ["Shepard", "Roosa", "Mitchell"],
    15: ["Scott", "Worden", "Irwin"],
    16: ["Young", "Mattingly", "Duke"],
    17: ["Cernan", "Evans", "Schmitt"],
}

ROLE_LABELS = {"cdr", "cmp", "lmp", "sc", "spacecraft"}

CC_DIV_RE = re.compile(
    r"<div(?:\s+class=\"(?:cc|onboard)\")?[^>]*>"
    r"((?:<a\s+name=|<b>).*?)"
    r"</div>",
    re.DOTALL,
)

TIMESTAMP_SPEAKER_RE = re.compile(
    r"^(?:\d{3}:\d{2}:\d{2}\s+)?"  # optional GET timestamp
    r"([A-Za-z][A-Za-z0-9 /'-]*?)"  # speaker name
    r"(?:\s*\([^)]*\))?"  # optional qualifier like (onboard)
    r"\s*:\s*"  # colon separator
)

BRACKET_RE = re.compile(r"\s*\[[^\]]*\]")
GARBLE_BRACKET_RE = re.compile(r"\[.*?garble.*?\]", re.IGNORECASE)


def strip_html(text: str) -> str:
    text = re.sub(r"<a\s+[^>]*>", "", text)
    text = re.sub(r"</a>", "", text)
    text = re.sub(r"</?(?:sub|sup|b|i|em|strong|font|span|div|p|br\s*/?)>", " ", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_cc_line(raw_html: str, crew_lower: set[str]) -> str | None:
    text = strip_html(raw_html)
    if not text:
        return None

    if GARBLE_BRACKET_RE.search(text):
        return None

    m = TIMESTAMP_SPEAKER_RE.match(text)
    if not m:
        return None

    speaker = m.group(1).strip().lower()
    if speaker not in crew_lower and speaker not in ROLE_LABELS:
        return None

    dialogue = text[m.end():].strip()
    if not dialogue:
        return None

    dialogue = BRACKET_RE.sub("", dialogue)
    dialogue = re.sub(r"\s+", " ", dialogue).strip()

    if not dialogue or len(dialogue) < 2:
        return None

    return dialogue


def process_file(input_path: Path, output_path: Path, mission: int) -> int:
    crew_lower = {n.lower() for n in CREW.get(mission, [])}

    with open(input_path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    lines = []
    for raw in CC_DIV_RE.findall(content):
        cleaned = clean_cc_line(raw, crew_lower)
        if cleaned:
            lines.append(cleaned)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    return len(lines)


def main():
    raw_dir = Path("data/raw/Apollo Flight Journal")
    clean_dir = Path("data/cleaned/Apollo Flight Journal")

    if not raw_dir.exists():
        print(f"Error: {raw_dir} not found")
        return

    total = 0
    for mission_dir in sorted(raw_dir.iterdir()):
        if not mission_dir.is_dir():
            continue

        m = re.search(r"Apollo\s+(\d+)", mission_dir.name)
        if not m:
            continue
        mission = int(m.group(1))

        mission_lines = 0
        for html_file in sorted(mission_dir.glob("*.html")):
            out = clean_dir / mission_dir.name / html_file.with_suffix(".txt").name
            n = process_file(html_file, out, mission)
            mission_lines += n

        print(f"{mission_dir.name}: {mission_lines} lines")
        total += mission_lines

    print(f"\nTotal: {total} astronaut lines extracted")


if __name__ == "__main__":
    main()
