"""Evaluate a trained model and optionally fail below a quality bar (CI eval gate).

    python -m skillex.evaluate --model YOUR_USER/jobbert-skill-extractor --min-f1 0.55
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile

from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
)

from skillex.crf import load_crfs
from skillex.data import load_skillspan, tokenize_dataset
from skillex.metrics import make_compute_metrics
from skillex.trainer import TwoHeadTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Local path or Hub id")
    parser.add_argument("--dataset", default="jjzha/skillspan")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None, help="Evaluate on first N examples")
    parser.add_argument("--min-f1", type=float, default=None)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForTokenClassification.from_pretrained(args.model)
    crfs = load_crfs(args.model)
    ds = load_skillspan(args.dataset)
    split = ds[args.split]
    if args.limit:
        split = split.select(range(min(args.limit, len(split))))
    tokenized = tokenize_dataset({args.split: split}, tokenizer)[args.split]

    with tempfile.TemporaryDirectory() as tmp:
        trainer = TwoHeadTrainer(
            model=model,
            args=TrainingArguments(output_dir=tmp, per_device_eval_batch_size=32, report_to="none"),
            processing_class=tokenizer,
            data_collator=DataCollatorForTokenClassification(tokenizer),
            compute_metrics=make_compute_metrics((lambda: crfs) if crfs else None),
        )
        metrics = trainer.evaluate(tokenized, metric_key_prefix=args.split)

    print(json.dumps(metrics, indent=2))
    f1 = metrics[f"{args.split}_f1"]
    if args.min_f1 is not None and f1 < args.min_f1:
        print(f"EVAL GATE FAILED: f1={f1:.4f} < {args.min_f1}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
