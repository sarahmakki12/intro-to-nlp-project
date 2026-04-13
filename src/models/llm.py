#!/usr/bin/env python3
"""
distilgpt2-based character prediction model.

At inference time this model:
1. Tokenizes the context with distilgpt2's byte-level BPE tokenizer
   (every Unicode character is representable without <unk>)
2. Runs a single forward pass and reads last-position logits
3. Projects the full vocab probability distribution onto characters via a
   precomputed (vocab_size × num_chars) matrix multiply — all on GPU — then
   returns the top-3 characters

The merged model weights (distilgpt2 base + LoRA adapter, produced by
llm_train.py) live at work/distilgpt2_finetuned/.
"""

from __future__ import annotations

from pathlib import Path

import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

MODEL_SUBDIR = "distilgpt2_finetuned"


class LLMCharModel:

    def __init__(self, model_dir: Path):
        print(f"Loading tokenizer from {model_dir}", flush=True)
        self.tokenizer = GPT2TokenizerFast.from_pretrained(
            str(model_dir), local_files_only=True
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading model from {model_dir} (device: {self.device})", flush=True)
        self.model = GPT2LMHeadModel.from_pretrained(
            str(model_dir), local_files_only=True
        )
        self.model.to(self.device)
        self.model.eval()
        if self.device.type == "cpu":
            torch.set_num_threads(8)

        print("Building character projection matrix...", flush=True)
        self._proj, self._chars = self._build_proj_matrix()
        print(f"  {len(self._chars)} unique first-characters in vocabulary",
              flush=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_proj_matrix(self) -> tuple[torch.Tensor, list[str]]:
        """Build a (vocab_size, num_chars) projection matrix on the model device.

        proj[token_id, char_idx] = 1 if the token's decoded string starts with
        chars[char_idx] (lowercased), else 0.  A single matmul then converts a
        (batch, vocab_size) probability matrix into (batch, num_chars) character
        scores with no Python-level loops or GPU↔CPU round-trips.
        """
        char_to_ids: dict[str, list[int]] = {}
        for token_id in range(self.tokenizer.vocab_size):
            token = self.tokenizer.convert_ids_to_tokens(token_id)
            if token is None:
                continue
            try:
                token_str = self.tokenizer.convert_tokens_to_string([token])
            except Exception:
                continue
            if not token_str:
                continue
            first_char = token_str[0].lower()
            char_to_ids.setdefault(first_char, []).append(token_id)

        chars = sorted(char_to_ids.keys())
        char_to_idx = {c: i for i, c in enumerate(chars)}

        proj = torch.zeros(self.tokenizer.vocab_size, len(chars))
        for c, ids in char_to_ids.items():
            proj[ids, char_to_idx[c]] = 1.0

        return proj.to(self.device), chars

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, context: str, n_guesses: int = 3) -> str:
        return self.predict_batch([context], n_guesses=n_guesses)[0]

    def predict_batch(self, contexts: list[str], n_guesses: int = 3,
                      max_length: int = 256) -> list[str]:
        """Run a batched forward pass and return one prediction string per context."""
        inputs = self.tokenizer(
            contexts,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True,
            padding_side="left",  # left-pad so the last real token is always at the right
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.inference_mode():
            logits = self.model(**inputs).logits  # (batch, seq_len, vocab)

        # Gather the last real token's logits for each sequence
        attention_mask = inputs["attention_mask"]
        last_positions = attention_mask.sum(dim=1) - 1  # (batch,)
        last_logits = logits[
            torch.arange(len(contexts), device=self.device), last_positions
        ]  # (batch, vocab)

        # Project vocab probs → character scores in one matmul, entirely on device
        probs = torch.softmax(last_logits, dim=-1)          # (batch, vocab)
        char_scores = probs @ self._proj                     # (batch, num_chars)
        top_indices = char_scores.topk(n_guesses, dim=-1).indices  # (batch, n)
        top_indices_cpu = top_indices.tolist()

        results: list[str] = []
        for row in top_indices_cpu:
            chars = [self._chars[idx] for idx in row]
            while len(chars) < n_guesses:
                chars.append(" ")
            results.append("".join(chars))
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @classmethod
    def load(cls, work_dir: str) -> "LLMCharModel":
        model_dir = Path(work_dir) / MODEL_SUBDIR
        if not model_dir.exists():
            raise FileNotFoundError(
                f"LLM model not found at {model_dir}.\n"
                "Train the model first by running the Colab notebook "
                "(colab_train.ipynb) or:\n"
                "  python src/models/llm_train.py --work_dir work"
            )
        return cls(model_dir)
