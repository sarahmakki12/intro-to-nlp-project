#!/usr/bin/env python3
"""
Extract and clean astronaut dialogue from Apollo Lunar Surface Journal HTML.

For each HTML file in data/raw/Apollo Lunar Surface Journal/, produces a cleaned
.txt file in data/cleaned/Apollo Lunar Surface Journal/ containing only crew
dialogue with timestamps, speaker labels, and annotations removed.

ALSJ format: <b>HHH:MM:SS</b> Speaker: dialogue text<p>
Timestamps may also use "xx" for unknown seconds: <b>HHH:MM:xx</b>
Anchors may wrap the timestamp or precede the <b> tag.
Commentary lives in <blockquote> blocks and is excluded.
"""

import html
import re
from pathlib import Path

# CDR, CMP, LMP for each mission
CREW = {
    11: ["Armstrong", "Collins", "Aldrin"],
    12: ["Conrad", "Gordon", "Bean"],
    14: ["Shepard", "Roosa", "Mitchell"],
    15: ["Scott", "Worden", "Irwin"],
    16: ["Young", "Mattingly", "Duke"],
    17: ["Cernan", "Evans", "Schmitt"],
}

# Matches <b>[optional anchor]TIMESTAMP[/anchor]</b>
# Timestamps: 102:15:02 or 185:58:xx
TIMESTAMP_RE = (
    r"<b>\s*(?:<a\s[^>]*>\s*)?"
    r"\d{3}:\d{2}:(?:\d{2}|xx)"
    r"\s*(?:</a>)?\s*</b>"
)

# Full dialogue entry: timestamp block, speaker, colon, dialogue text.
# Dialogue text runs until <p>, <blockquote, next timestamp, or EOF.
DIALOGUE_RE = re.compile(
    TIMESTAMP_RE
    + r"\s+"
    + r"([A-Za-z][A-Za-z /'-]*?)"       # group 1: speaker name
    + r"(?:\s*\([^)]*\))?"              # optional qualifier like (onboard)
    + r"\s*:\s*"                         # colon separator
    + r"(.*?)"                           # group 2: dialogue text
    + r"(?:<p>|<p\s*/>|(?=\s*<blockquote)|(?=\s*" + TIMESTAMP_RE + r")|$)",
    re.DOTALL,
)

GARBLE_RE = re.compile(r"garble", re.IGNORECASE)
BRACKET_RE = re.compile(r"\s*\[[^\]]*\]")
PAREN_RE = re.compile(r"\s*\([^)]*\)")

FILL_IN_WORDS = {
    "i", "you", "he", "she", "we", "they", "it", "me", "him", "her",
    "us", "them", "my", "your", "his", "our", "their", "its", "yours",
    "the", "a", "an", "to", "on", "in", "at", "of", "for", "by",
    "with", "from", "up", "out", "off", "is", "am", "are", "was",
    "were", "be", "been", "do", "did", "does", "have", "has", "had",
    "will", "would", "could", "should", "can", "may", "might",
    "and", "or", "but", "not", "if", "so", "then", "that", "this",
    "just", "also", "i'm", "he's", "she's", "it's", "we're",
    "they're", "you're", "i'll", "won't", "don't", "didn't",
    "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't",
    "couldn't", "shouldn't", "wouldn't", "can't",
}


def has_fill_in_paren(text: str) -> bool:
    for m in re.finditer(r"\(([^)]*)\)", text):
        content = m.group(1).strip().lower()
        words = content.split()
        if 1 <= len(words) <= 3 and all(w in FILL_IN_WORDS for w in words):
            return True
    return False


def strip_html(text: str) -> str:
    text = re.sub(r"<a\s+[^>]*>", "", text)
    text = re.sub(r"</a>", "", text)
    text = re.sub(r"</?(?:sub|sup|b|i|em|strong|font|span|div|p|br\s*/?)>", " ", text)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def clean_dialogue(raw_html: str) -> str | None:
    text = strip_html(raw_html)
    if not text:
        return None

    if GARBLE_RE.search(text):
        return None

    if has_fill_in_paren(text):
        return None

    text = PAREN_RE.sub("", text)
    text = BRACKET_RE.sub("", text)
    text = re.sub(r"\s*\([^)]*$", "", text)
    text = re.sub(r"\s*\[[^\]]*$", "", text)
    text = re.sub(r"[(){}\[\]]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    if not text or len(text) < 2:
        return None

    return text


def process_file(input_path: Path, output_path: Path, mission: int) -> int:
    crew_lower = {n.lower() for n in CREW.get(mission, [])}

    with open(input_path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    lines = []
    for m in DIALOGUE_RE.finditer(content):
        speaker = m.group(1).strip().lower()
        if speaker not in crew_lower and speaker != "lm crew":
            continue

        dialogue = clean_dialogue(m.group(2))
        if dialogue:
            lines.append(dialogue)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    return len(lines)


def main():
    raw_dir = Path("data/raw/Apollo Lunar Surface Journal")
    clean_dir = Path("data/cleaned/Apollo Lunar Surface Journal")

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
