#!/usr/bin/env python
import csv
import os
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from pathlib import Path


def is_csv(path: str) -> bool:
    return path.endswith(".csv")


def load_train_csv(fname: str) -> tuple[list[str], list[str]]:
    contexts, targets = [], []
    with open(fname, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            contexts.append(row["context"])
            targets.append(row["prediction"])
    return contexts, targets


def load_test_data(fname: str) -> tuple[list[str], list[str]]:
    """Returns (ids, contexts). For CSV files ids come from the file; for
    plain text files ids are sequential line numbers."""
    ids, contexts = [], []
    if is_csv(fname):
        with open(fname, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                ids.append(row["id"])
                contexts.append(row["context"])
    else:
        with open(fname, encoding="utf-8") as f:
            for i, line in enumerate(f):
                ids.append(str(i))
                contexts.append(line.rstrip("\n"))
    return ids, contexts


def write_predictions(ids: list[str], preds: list[str], fname: str):
    if is_csv(fname):
        with open(fname, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "prediction"])
            for row_id, pred in zip(ids, preds):
                writer.writerow([row_id, pred])
    else:
        with open(fname, "w", encoding="utf-8") as f:
            for pred in preds:
                f.write(f"{pred}\n")


if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument("mode", choices=("train", "test"), help="what to run")
    parser.add_argument("--work_dir", help="where to save/load model", default="work")
    parser.add_argument("--model", choices=("llm", "ngram"), default="llm",
                        help="model type: llm (distilgpt2, default) or ngram")

    # LLM options (training + inference)
    parser.add_argument("--batch_size", type=int, default=32,
                        help="batch size for LLM inference / training")
    parser.add_argument("--epochs", type=int, default=3,
                        help="LLM training epochs")
    parser.add_argument("--max_length", type=int, default=128,
                        help="max token length per training example")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--learning_rate", type=float, default=3e-4,
                        help="LLM learning rate")
    parser.add_argument("--open_dev_dir", default="data/open-dev",
                        help="open-dev directory to include in LLM training "
                             "(set to empty string to skip)")
    parser.add_argument("--open_dev_limit", type=int, default=None,
                        help="max lines from open-dev for LLM training")

    # N-gram training options
    parser.add_argument("--train_data", help="path to training CSV (uses context/prediction pairs)")
    parser.add_argument("--train_files", nargs="+",
                        help="specific .txt files for raw text training (overrides default corpus)")
    parser.add_argument("--test_data", help="path to test data (CSV or txt)")
    parser.add_argument("--test_output", help="path to write predictions", default="pred.txt")
    parser.add_argument("--max_order", type=int, default=6, help="max n-gram order")
    parser.add_argument("--limit", type=int, default=None,
                        help="only use the first N lines from each text file")
    parser.add_argument("--model_name", type=str, default="ngram_o4_50k",
                        help="n-gram model filename (without .pkl)")
    args = parser.parse_args()

    if args.mode == "train":
        os.makedirs(args.work_dir, exist_ok=True)

        if args.model == "llm":
            import subprocess
            import sys

            cmd = [
                sys.executable, "src/models/llm_train.py",
                "--work_dir", args.work_dir,
                "--epochs", str(args.epochs),
                "--batch_size", str(args.batch_size),
                "--max_length", str(args.max_length),
                "--lora_r", str(args.lora_r),
                "--learning_rate", str(args.learning_rate),
                "--open_dev_dir", args.open_dev_dir,
            ]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            if args.open_dev_limit:
                cmd += ["--open_dev_limit", str(args.open_dev_limit)]
            subprocess.run(cmd, check=True)

        else:
            # N-gram training (unchanged from original)
            from models.ngram import CharNgramModel, CORPUS_DIR, CORPUS_FILES

            if args.train_data:
                if args.model_name is None:
                    args.model_name = f"csv_o{args.max_order}"
            else:
                if args.model_name is None:
                    limit_tag = f"_{args.limit // 1000}k" if args.limit else "_full"
                    args.model_name = f"ngram_o{args.max_order}{limit_tag}"
            print(f"Model name: {args.model_name}")

            model = CharNgramModel(max_order=args.max_order)

            if args.train_data:
                print(f"Loading training data from {args.train_data}")
                contexts, targets = load_train_csv(args.train_data)
                print(f"Training model (max_order={args.max_order})")
                model.train(contexts, targets)
                model.save(args.work_dir, name=args.model_name)
            else:
                if args.train_files:
                    txt_files = [Path(f) for f in args.train_files]
                else:
                    txt_files = [CORPUS_DIR / f for f in CORPUS_FILES]
                missing = [f for f in txt_files if not f.exists()]
                if missing:
                    parser.error(f"Missing files: {[str(f) for f in missing]}")
                limit_str = f", limit={args.limit} lines/file" if args.limit else ""
                print(f"Training on {len(txt_files)} text files "
                      f"(max_order={args.max_order}{limit_str})")
                model.train_from_text(txt_files, work_dir=args.work_dir,
                                      model_name=args.model_name, limit=args.limit)

    elif args.mode == "test":
        if not args.test_data:
            parser.error("--test_data is required for test mode")

        if args.test_output == "pred.txt" and is_csv(args.test_data):
            args.test_output = "pred.csv"

        print(f"Loading test data from {args.test_data}")
        ids, contexts = load_test_data(args.test_data)
        print(f"  {len(contexts):,} examples")

        if args.model == "llm":
            from models.llm import LLMCharModel
            model = LLMCharModel.load(args.work_dir)

            print(f"Predicting {len(contexts):,} examples "
                  f"(batch_size={args.batch_size})")
            preds: list[str] = []
            bs = args.batch_size
            n_batches = (len(contexts) + bs - 1) // bs
            for i in range(0, len(contexts), bs):
                batch = contexts[i:i + bs]
                preds.extend(model.predict_batch(batch))
                batch_num = i // bs + 1
                if batch_num % 100 == 0 or batch_num == n_batches:
                    print(f"  {min(i + bs, len(contexts)):,} / {len(contexts):,}",
                          flush=True)

        else:
            from models.ngram import CharNgramModel
            print(f"Loading n-gram model {args.model_name} from {args.work_dir}")
            model = CharNgramModel.load(args.work_dir, name=args.model_name)
            print(f"Predicting {len(contexts):,} examples")
            preds = [model.predict(ctx) for ctx in contexts]

        print(f"Writing predictions to {args.test_output}")
        write_predictions(ids, preds, args.test_output)
        print("Done")
