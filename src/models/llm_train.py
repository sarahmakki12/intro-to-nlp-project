#!/usr/bin/env python3
"""
LoRA CLM fine-tuning of distilgpt2 on the multilingual astronaut corpus.

Run this on a GPU (Google Colab, Kaggle Kernels, or any CUDA machine).
CPU training is supported but will be very slow.

Usage:
    python src/models/llm_train.py [options]

Key options:
    --work_dir DIR      Where to save checkpoints and the final merged model
                        (default: work)
    --data_dir DIR      Directory containing *.txt training files
                        (default: data/training)
    --epochs N          Training epochs (default: 3)
    --batch_size N      Per-device batch size (default: 32)
    --max_length N      Max token length per example (default: 128)
    --lora_r N          LoRA rank (default: 16)
    --limit N           Use only the first N lines per file (for quick tests)

After training the script merges the LoRA adapter into the base model weights
and saves the result to {work_dir}/distilgpt2_finetuned/.  Copy that directory
into your submission's work/ folder before packaging.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="LoRA CLM fine-tune distilgpt2 on the astronaut corpus",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--work_dir", default="work")
    p.add_argument("--data_dir", default="data/training")
    p.add_argument("--open_dev_dir", default="data/open-dev",
                   help="Path to open-dev directory (input.txt + answer.txt); "
                        "set to empty string to skip")
    p.add_argument("--open_dev_limit", type=int, default=None,
                   help="Max lines to use from open-dev (omit for all)")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_length", type=int, default=128)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--limit", type=int, default=None,
                   help="Max lines per file (omit for full corpus)")
    p.add_argument("--learning_rate", type=float, default=3e-4)
    return p.parse_args()


CORPUS_FILES = [
    "en_unique.txt", "ru.txt", "zh.txt", "ja.txt", "hi.txt",
    "ar.txt", "ko.txt", "fr.txt", "de.txt", "it.txt",
]


def load_texts(data_dir: Path, limit: int | None) -> list[str]:
    texts: list[str] = []
    for fname in CORPUS_FILES:
        fpath = data_dir / fname
        if not fpath.exists():
            print(f"  WARNING: {fpath} not found, skipping")
            continue
        with open(fpath, encoding="utf-8") as f:
            lines = f.readlines()
        if limit:
            lines = lines[:limit]
        file_texts = [line.rstrip("\n") for line in lines if line.strip()]
        texts.extend(file_texts)
        print(f"  {fname}: {len(file_texts):,} lines")
    return texts


def load_open_dev(open_dev_dir: Path, limit: int | None) -> list[str]:
    """Reconstruct full texts from open-dev by appending each answer character
    to its context.  These are kept as independent training lines."""
    input_file = open_dev_dir / "input.txt"
    answer_file = open_dev_dir / "answer.txt"
    if not input_file.exists() or not answer_file.exists():
        print(f"  WARNING: open-dev files not found in {open_dev_dir}, skipping")
        return []
    with open(input_file, encoding="utf-8") as f:
        inputs = [line.rstrip("\n") for line in f]
    with open(answer_file, encoding="utf-8") as f:
        answers = [line.rstrip("\n") for line in f]
    texts = [ctx + ans for ctx, ans in zip(inputs, answers) if ctx or ans]
    if limit:
        texts = texts[:limit]
    print(f"  open-dev: {len(texts):,} lines")
    return texts


def main() -> None:
    args = parse_args()

    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import (
        DataCollatorForLanguageModeling,
        GPT2LMHeadModel,
        GPT2TokenizerFast,
        Trainer,
        TrainingArguments,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # ------------------------------------------------------------------ data
    data_dir = Path(args.data_dir)
    print(f"Loading training data from {data_dir} ...")
    texts = load_texts(data_dir, args.limit)

    if args.open_dev_dir:
        print(f"Loading open-dev data from {args.open_dev_dir} ...")
        texts.extend(load_open_dev(Path(args.open_dev_dir), args.open_dev_limit))

    print(f"Total: {len(texts):,} lines")

    # -------------------------------------------------------------- tokenizer
    print("Loading tokenizer (distilgpt2) ...")
    tokenizer = GPT2TokenizerFast.from_pretrained("distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(batch: dict) -> dict:
        out = tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )
        # For CLM the labels are identical to input_ids; the Trainer handles
        # shifting internally when using GPT2LMHeadModel.
        out["labels"] = out["input_ids"].copy()
        return out

    print("Tokenizing ...")
    dataset = Dataset.from_dict({"text": texts})
    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        num_proc=4,
        desc="Tokenizing",
    )
    tokenized.set_format(type="torch")

    # --------------------------------------------------------------- LoRA model
    print("Loading base model (distilgpt2) ...")
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_r * 2,
        # Target the combined QKV projection and the output projection in each
        # attention block — these give the best accuracy/parameter trade-off
        # for GPT-2-style models.
        target_modules=["c_attn", "c_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # ---------------------------------------------------------------- training
    lora_ckpt_dir = Path(args.work_dir) / "lora_checkpoints"
    training_args = TrainingArguments(
        output_dir=str(lora_ckpt_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=torch.cuda.is_available(),
        save_strategy="epoch",
        logging_steps=200,
        dataloader_num_workers=2,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    print("Training ...")
    trainer.train()

    # ------------------------------------------------- merge and save
    print("Merging LoRA adapter into base weights ...")
    merged = model.merge_and_unload()

    save_dir = Path(args.work_dir) / "distilgpt2_finetuned"
    save_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(save_dir))
    tokenizer.save_pretrained(str(save_dir))

    size_mb = (
        sum(f.stat().st_size for f in save_dir.rglob("*") if f.is_file())
        / (1024 ** 2)
    )
    print(f"Saved merged model to {save_dir}  ({size_mb:.0f} MB)")
    print(
        "\nNext step: copy work/distilgpt2_finetuned/ into your submission's "
        "work/ directory, then run:\n"
        "  python src/myprogram.py test --work_dir work "
        "--test_data data/open-dev/input.txt --test_output pred.txt"
    )


if __name__ == "__main__":
    main()
