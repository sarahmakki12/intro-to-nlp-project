#!/usr/bin/env python3
"""
distilgpt2-based character prediction model.

At inference time this model:
1. Tokenizes the context with distilgpt2's byte-level BPE tokenizer
   (every Unicode character is representable without <unk>)
2. Runs a single forward pass and reads last-position logits
3. Aggregates softmax probabilities by the first decoded character of each
   vocabulary token, then returns the top-3 characters

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

        print("Building character→token-id map...", flush=True)
        self._char_to_token_ids: dict[str, torch.Tensor] = self._build_char_map()
        print(f"  {len(self._char_to_token_ids)} unique first-characters in vocabulary",
              flush=True)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_char_map(self) -> dict[str, torch.Tensor]:
        """Map each lowercase first-character to the tensor of token IDs whose
        decoded string starts with that character.  Built once at load time."""
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
        return {c: torch.tensor(ids, dtype=torch.long)
                for c, ids in char_to_ids.items()}

    def _logits_to_top_chars(self, logits: torch.Tensor, n: int) -> str:
        """Convert last-position logits (vocab_size,) into a top-n char string."""
        probs = torch.softmax(logits, dim=-1)
        char_scores = {c: probs[ids].sum().item()
                       for c, ids in self._char_to_token_ids.items()}
        top = sorted(char_scores, key=char_scores.get, reverse=True)[:n]
        # Pad with space if the vocabulary somehow doesn't cover n chars
        while len(top) < n:
            top.append(" ")
        return "".join(top)

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

        attention_mask = inputs["attention_mask"]
        results: list[str] = []
        for i in range(len(contexts)):
            # Find the index of the last non-padding token
            last_pos = int(attention_mask[i].sum().item()) - 1
            results.append(self._logits_to_top_chars(logits[i, last_pos], n_guesses))
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
