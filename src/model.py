#!/usr/bin/env python3
"""
Character-level n-gram language model with backoff for next-character prediction.
"""

import pickle
from collections import Counter, defaultdict
from pathlib import Path

CORPUS_DIR = Path("data/training")
CORPUS_FILES = [
    "en_unique.txt", "ru.txt", "zh.txt", "ja.txt", "hi.txt",
    "ar.txt", "ko.txt", "fr.txt", "de.txt", "it.txt",
]
CHECKPOINT_EVERY = 50_000


class CharNgramModel:

    def __init__(self, max_order: int = 6):
        self.max_order = max_order
        self.ngram_counts: dict[int, dict[str, Counter]] = {}
        self.global_counts: Counter = Counter()

    def _init_accumulators(self):
        return {
            order: defaultdict(Counter) for order in range(self.max_order + 1)
        }, Counter()

    def _record(self, ngram_counts, global_counts, context: str, target: str):
        """Record one (context, target) observation across all orders."""
        global_counts[target] += 1
        ngram_counts[0][""][target] += 1
        for order in range(1, self.max_order + 1):
            if len(context) >= order:
                ngram_counts[order][context[-order:]][target] += 1

    def _snapshot(self, ngram_counts, global_counts):
        """Copy accumulators into self so save() works mid-training."""
        self.ngram_counts = {
            order: dict(prefixes) for order, prefixes in ngram_counts.items()
        }
        self.global_counts = global_counts

    def _print_stats(self, label: str):
        vocab_size = len(self.global_counts)
        total_chars = sum(self.global_counts.values())
        print(f"{label}: {total_chars:,} character observations, "
              f"vocab size {vocab_size:,} unique chars")
        for order in range(self.max_order + 1):
            n_prefixes = len(self.ngram_counts.get(order, {}))
            print(f"  order {order}: {n_prefixes:,} prefixes")

    def train(self, contexts: list[str], targets: list[str]):
        """Train from explicit (context, target) pairs (e.g. CSV data)."""
        ngram_counts, global_counts = self._init_accumulators()
        for context, target in zip(contexts, targets):
            self._record(ngram_counts, global_counts, context.lower(), target.lower())
        self._snapshot(ngram_counts, global_counts)
        self._print_stats("CSV training")

    def train_from_text(self, text_files: list[Path], work_dir: str | None = None,
                        model_name: str = "model", limit: int | None = None):
        """Train by sliding an n-gram window across each line in raw text files.
        Each line is treated as an independent sentence.
        Saves intermediate checkpoints after each file and every CHECKPOINT_EVERY lines."""
        ngram_counts, global_counts = self._init_accumulators()
        total_lines = 0

        for fi, fpath in enumerate(text_files):
            file_lines = 0
            since_checkpoint = 0
            print(f"  [{fi+1}/{len(text_files)}] Processing {fpath.name}...", flush=True)
            with open(fpath, encoding="utf-8") as f:
                for raw in f:
                    line = raw.rstrip("\n").lower()
                    if not line:
                        continue
                    file_lines += 1
                    since_checkpoint += 1
                    for i, ch in enumerate(line):
                        context = line[:i]
                        self._record(ngram_counts, global_counts, context, ch)

                    if limit and file_lines >= limit:
                        break

                    if work_dir and since_checkpoint >= CHECKPOINT_EVERY:
                        self._snapshot(ngram_counts, global_counts)
                        self.save(work_dir, name=model_name)
                        print(f"    checkpoint at {file_lines:,} lines", flush=True)
                        since_checkpoint = 0

            total_lines += file_lines
            print(f"    {file_lines:,} lines", flush=True)

            if work_dir:
                self._snapshot(ngram_counts, global_counts)
                self.save(work_dir, name=model_name)
                print(f"    checkpoint saved after {fpath.name}", flush=True)

        print(f"  Total: {total_lines:,} lines from {len(text_files)} files")
        self._snapshot(ngram_counts, global_counts)
        self._print_stats("Text training")

    def predict(self, context: str, n_guesses: int = 3) -> str:
        context = context.lower()

        for order in range(self.max_order, -1, -1):
            if order == 0:
                prefix = ""
            else:
                if len(context) < order:
                    continue
                prefix = context[-order:]

            counts = self.ngram_counts.get(order, {}).get(prefix)
            if not counts:
                continue

            total = sum(counts.values())
            if total < 3 and order > 0:
                continue

            top = counts.most_common(n_guesses)
            chars = [ch for ch, _ in top]
            if len(chars) < n_guesses:
                seen = set(chars)
                for ch, _ in self.global_counts.most_common():
                    if ch not in seen:
                        chars.append(ch)
                        seen.add(ch)
                        if len(chars) >= n_guesses:
                            break
            return "".join(chars[:n_guesses])

        # Absolute fallback: most common characters globally
        if self.global_counts:
            top = self.global_counts.most_common(n_guesses)
            return "".join(ch for ch, _ in top)

        return " et"

    def save(self, work_dir: str, name: str = "model"):
        path = Path(work_dir) / f"{name}.pkl"
        tmp_path = path.with_suffix(".tmp.pkl")
        data = {
            "max_order": self.max_order,
            "ngram_counts": self.ngram_counts,
            "global_counts": self.global_counts,
        }
        with open(tmp_path, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp_path.rename(path)
        size_mb = path.stat().st_size / (1024 * 1024)
        print(f"Model saved to {path} ({size_mb:.1f} MB)")

    @classmethod
    def load(cls, work_dir: str, name: str = "model") -> "CharNgramModel":
        path = Path(work_dir) / f"{name}.pkl"
        with open(path, "rb") as f:
            data = pickle.load(f)
        model = cls(max_order=data["max_order"])
        model.ngram_counts = data["ngram_counts"]
        model.global_counts = data["global_counts"]
        return model
