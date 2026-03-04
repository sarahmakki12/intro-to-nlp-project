#!/usr/bin/env python3
"""
Translate the deduplicated English training corpus into target languages
using Google Translate via deep-translator.

Features:
  - Batches lines into single API requests (up to --batch-chars characters)
  - Rate limiting with configurable delay and exponential backoff on errors
  - Checkpoint/resume: tracks progress per language so interrupted runs continue
  - CLI flags for language selection, limits, and tuning
"""

import argparse
import sys
import time
from pathlib import Path

from deep_translator import GoogleTranslator

ALL_LANGUAGES = ["ru", "zh-CN", "ja", "hi", "ar", "ko", "fr", "de", "it"]
LANG_FILE_NAMES = {
    "ru": "ru", "zh-CN": "zh", "ja": "ja", "hi": "hi",
    "ar": "ar", "ko": "ko", "fr": "fr", "de": "de", "it": "it",
}
LANG_DISPLAY = {
    "ru": "Russian", "zh-CN": "Chinese (Simplified)", "ja": "Japanese",
    "hi": "Hindi", "ar": "Arabic", "ko": "Korean",
    "fr": "French", "de": "German", "it": "Italian",
}


def parse_args():
    p = argparse.ArgumentParser(description="Translate English corpus to target languages")
    p.add_argument("--source", type=Path, default=Path("data/training/en_unique.txt"))
    p.add_argument("--output-dir", type=Path, default=Path("data/training"))
    p.add_argument("--languages", nargs="+", default=None,
                   help=f"Target languages (default: all). Choices: {list(LANG_FILE_NAMES.values())}")
    p.add_argument("--batch-chars", type=int, default=4500,
                   help="Max characters per API request (default: 4500)")
    p.add_argument("--delay", type=float, default=0.5,
                   help="Seconds between API requests (default: 0.5)")
    p.add_argument("--limit", type=int, default=None,
                   help="Only translate the first N lines (for testing)")
    p.add_argument("--max-retries", type=int, default=5)
    return p.parse_args()


def resolve_languages(requested):
    """Map user-friendly codes (e.g. 'zh') to API codes (e.g. 'zh-CN')."""
    if requested is None:
        return list(ALL_LANGUAGES)

    file_to_api = {v: k for k, v in LANG_FILE_NAMES.items()}
    resolved = []
    for lang in requested:
        if lang in file_to_api:
            resolved.append(file_to_api[lang])
        elif lang in LANG_FILE_NAMES:
            resolved.append(lang)
        else:
            print(f"Unknown language: {lang}")
            sys.exit(1)
    return resolved


def progress_path(output_dir: Path, lang_code: str) -> Path:
    return output_dir / f".progress_{LANG_FILE_NAMES[lang_code]}"


def load_progress(output_dir: Path, lang_code: str) -> int:
    p = progress_path(output_dir, lang_code)
    if p.exists():
        return int(p.read_text().strip())
    return 0


def save_progress(output_dir: Path, lang_code: str, line_idx: int):
    progress_path(output_dir, lang_code).write_text(str(line_idx))


def make_batches(lines: list[str], max_chars: int) -> list[list[str]]:
    batches: list[list[str]] = []
    current: list[str] = []
    current_chars = 0

    for line in lines:
        line_len = len(line) + 1  # +1 for the \n separator
        if current and current_chars + line_len > max_chars:
            batches.append(current)
            current = []
            current_chars = 0
        current.append(line)
        current_chars += line_len

    if current:
        batches.append(current)
    return batches


def translate_single(translator: GoogleTranslator, line: str,
                     delay: float, max_retries: int) -> str:
    """Translate a single line with retries. Returns empty string on total failure."""
    for attempt in range(max_retries):
        try:
            result = translator.translate(line)
            return result if result else ""
        except Exception as e:
            wait = delay * (2 ** attempt)
            if attempt < max_retries - 1:
                time.sleep(wait)
    return ""


def translate_batch(translator: GoogleTranslator, batch: list[str],
                    delay: float, max_retries: int) -> list[str]:
    """Translate a batch of lines, joining with newlines for a single API call."""
    joined = "\n".join(batch)

    for attempt in range(max_retries):
        try:
            result = translator.translate(joined)
            translated_lines = result.split("\n")

            if len(translated_lines) == len(batch):
                return translated_lines

            # Google merged/split lines; fall back to line-by-line
            break

        except Exception as e:
            wait = delay * (2 ** attempt)
            print(f"    Error: {e} — retrying in {wait:.1f}s (attempt {attempt + 1}/{max_retries})")
            time.sleep(wait)
    else:
        print(f"    Batch failed after {max_retries} attempts, falling back to line-by-line")

    translated_lines = []
    for line in batch:
        single = translate_single(translator, line, delay, max_retries)
        translated_lines.append(single)
        time.sleep(delay)

    assert len(translated_lines) == len(batch), \
        f"Line count mismatch: {len(translated_lines)} != {len(batch)}"
    return translated_lines


def translate_language(lines: list[str], lang_code: str, args):
    fname = LANG_FILE_NAMES[lang_code]
    output_file = args.output_dir / f"{fname}.txt"
    display = LANG_DISPLAY[lang_code]

    done = load_progress(args.output_dir, lang_code)
    remaining = lines[done:]

    if not remaining:
        print(f"[{fname}] Already complete ({done}/{len(lines)} lines)")
        return

    print(f"[{fname}] Translating to {display}: {len(remaining)} lines remaining "
          f"(starting from line {done})")

    batches = make_batches(remaining, args.batch_chars)
    translator = GoogleTranslator(source="en", target=lang_code)

    mode = "a" if done > 0 else "w"
    with open(output_file, mode, encoding="utf-8") as out:
        lines_done = done
        for i, batch in enumerate(batches):
            translated = translate_batch(translator, batch, args.delay, args.max_retries)

            for tline in translated:
                out.write(tline + "\n")

            lines_done += len(batch)
            save_progress(args.output_dir, lang_code, lines_done)

            if (i + 1) % 100 == 0 or (i + 1) == len(batches):
                pct = lines_done / len(lines) * 100
                print(f"    [{fname}] {lines_done:>7,}/{len(lines):,} lines ({pct:.1f}%) "
                      f"— batch {i + 1}/{len(batches)}")

            time.sleep(args.delay)

    print(f"[{fname}] Done: {lines_done:,} lines written to {output_file}")


def main():
    args = parse_args()

    if not args.source.exists():
        print(f"Error: source file {args.source} not found")
        sys.exit(1)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    lines = args.source.read_text(encoding="utf-8").splitlines()
    if args.limit:
        lines = lines[:args.limit]

    languages = resolve_languages(args.languages)

    print(f"Source: {args.source} ({len(lines):,} lines)")
    print(f"Languages: {[LANG_FILE_NAMES[l] for l in languages]}")
    print(f"Batch size: {args.batch_chars} chars, delay: {args.delay}s")
    print()

    for lang_code in languages:
        translate_language(lines, lang_code, args)
        print()


if __name__ == "__main__":
    main()
