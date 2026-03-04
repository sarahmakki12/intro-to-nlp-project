#!/usr/bin/env python3
"""
Extract and clean astronaut dialogue from Spacelog transcript files.

For each transcript in data/raw/Spacelog/, produces a cleaned .txt file in
data/cleaned/Spacelog/ containing only crew dialogue with timestamps, speaker
labels, and annotations removed.

Spacelog format:
    [HH:MM:SS:FF]
    SPEAKER: dialogue text
    (optional continuation lines)

Metadata lines (_page, _tape) and [glossary:...] tags are stripped.
"""

import re
from pathlib import Path

# Crew speaker codes per mission (only actual spacecraft crew)
CREW_SPEAKERS = {
    "Mercury-Redstone 3": {"p"},
    "Mercury-Redstone 4": {"bell 7"},
    "Mercury-Atlas 6": {"p"},
    "Mercury-Atlas 7": {"p"},
    "Mercury-Atlas 8": {"p"},
    "Gemini 3": {"c", "p", "molly_brown"},
    "Gemini 4": {"c", "p"},
    "Gemini 6": {"c6", "p6"},
    "Gemini 8": {"c", "p"},
}

TIMESTAMP_RE = re.compile(r"^\[[-\d:]+\]\s*$")
META_RE = re.compile(r"^_(?:page|tape)\s*:")
GLOSSARY_RE = re.compile(r"\[glossary:([^\]]*)\]")
GARBLE_RE = re.compile(r"garble", re.IGNORECASE)
PAREN_RE = re.compile(r"\s*\([^)]*\)")
BRACKET_RE = re.compile(r"\s*\[[^\]]*\]")

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


def unwrap_fill_in_parens(text: str) -> str:
    """Replace (word) with word when content is 1-3 common fill-in words."""
    def _replace(m: re.Match) -> str:
        content = m.group(1).strip()
        words = content.lower().split()
        if 1 <= len(words) <= 3 and all(w in FILL_IN_WORDS for w in words):
            return content
        return m.group(0)
    return re.sub(r"\(([^)]*)\)", _replace, text)


def parse_entries(text: str) -> list[tuple[str, str]]:
    """Parse transcript into (speaker, dialogue) tuples."""
    entries = []
    current_speaker = None
    current_lines: list[str] = []

    for line in text.splitlines():
        line = line.strip()

        if not line or TIMESTAMP_RE.match(line) or META_RE.match(line):
            continue

        speaker_match = re.match(r"^([A-Za-z0-9_ -]+?):\s*(.*)", line)
        if speaker_match:
            if current_speaker and current_lines:
                entries.append((current_speaker, " ".join(current_lines)))
            current_speaker = speaker_match.group(1)
            dialogue = speaker_match.group(2).strip()
            current_lines = [dialogue] if dialogue else []
        elif current_speaker:
            current_lines.append(line)

    if current_speaker and current_lines:
        entries.append((current_speaker, " ".join(current_lines)))

    return entries


def clean_dialogue(text: str) -> str | None:
    text = GLOSSARY_RE.sub(r"\1", text)
    text = re.sub(r"<[^>]+>", "", text)

    if GARBLE_RE.search(text):
        return None

    text = unwrap_fill_in_parens(text)
    text = PAREN_RE.sub("", text)
    text = BRACKET_RE.sub("", text)
    text = re.sub(r"\s*\([^)]*$", "", text)
    text = re.sub(r"\s*\[[^\]]*$", "", text)
    text = re.sub(r"[(){}\[\]]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    if text == "...":
        return None

    if not text or len(text) < 2:
        return None

    return text


def process_file(input_path: Path, output_path: Path, mission: str) -> int:
    crew = CREW_SPEAKERS.get(mission, set())

    with open(input_path, encoding="utf-8", errors="replace") as f:
        content = f.read()

    entries = parse_entries(content)

    lines = []
    for speaker, dialogue in entries:
        if speaker.lower() not in crew:
            continue
        cleaned = clean_dialogue(dialogue)
        if cleaned:
            lines.append(cleaned)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line + "\n")

    return len(lines)


def main():
    raw_dir = Path("data/raw/Spacelog")
    clean_dir = Path("data/cleaned/Spacelog")

    if not raw_dir.exists():
        print(f"Error: {raw_dir} not found")
        return

    total = 0
    for mission_dir in sorted(raw_dir.iterdir()):
        if not mission_dir.is_dir():
            continue

        transcript = mission_dir / "transcript.txt"
        if not transcript.exists():
            continue

        mission = mission_dir.name
        out = clean_dir / mission / "transcript.txt"
        n = process_file(transcript, out, mission)
        print(f"{mission}: {n} lines")
        total += n

    print(f"\nTotal: {total} astronaut lines extracted")


if __name__ == "__main__":
    main()
