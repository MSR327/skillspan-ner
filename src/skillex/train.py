"""Fine-tune a token-classification model on SkillSpan.

    python -m skillex.train --config configs/jobbert.yaml
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
import yaml
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    set_seed,
)

from skillex.crf import BIOCRF, save_crfs
from skillex.data import load_skillspan, tokenize_dataset
from skillex.labels import ID2LABEL, LABEL2ID, NUM_OUTPUTS
from skillex.metrics import make_compute_metrics
from skillex.trainer import TwoHeadTrainer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--set", nargs="*", default=[], metavar="KEY=VALUE",
        help="Override config values, e.g. --set seed=43 learning_rate=3e-5",
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    for item in args.set:
        key, _, value = item.partition("=")
        if key not in cfg and key != "max_samples":
            raise SystemExit(f"--set: unknown config key {key!r}")
        parsed = yaml.safe_load(value)  # "true" -> bool, "8" -> int, ...
        if isinstance(parsed, str):
            try:  # PyYAML (YAML 1.1) reads "3e-5" as a string; we want a float
                parsed = float(parsed)
            except ValueError:
                pass
        cfg[key] = parsed
    print("config:", json.dumps(cfg))

    set_seed(cfg["seed"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["base_model"])
    model = AutoModelForTokenClassification.from_pretrained(
        cfg["base_model"],
        num_labels=NUM_OUTPUTS,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        # Some checkpoints (e.g. deberta-v3-base) are stored in fp16 and transformers 5 keeps
        # that dtype; fp16 mixed precision needs fp32 master weights or GradScaler crashes.
        dtype=torch.float32,
    )
    # Some tokenizers (incl. JobBERT's) ship without model_max_length; without it the
    # inference pipeline won't chunk long job posts and BERT crashes past 512 tokens.
    tokenizer.model_max_length = min(
        tokenizer.model_max_length, model.config.max_position_embeddings
    )

    use_crf = bool(cfg.get("crf", False))
    if use_crf:
        # One CRF per head; as a submodule its params get optimized and checkpointed with the model.
        model.skillex_crf = torch.nn.ModuleList([BIOCRF(), BIOCRF()])
        compute_metrics = make_compute_metrics(lambda: [c.to_dict() for c in model.skillex_crf])
    else:
        compute_metrics = make_compute_metrics()

    raw = load_skillspan(cfg["dataset"])
    if cfg.get("max_samples"):  # quick debug runs
        raw = raw.filter(lambda _, i: i < cfg["max_samples"], with_indices=True)
    ds = tokenize_dataset(raw, tokenizer, cfg["max_length"])

    out_dir = Path(cfg["output_dir"])
    training_args = TrainingArguments(
        output_dir=str(out_dir),
        run_name=cfg["run_name"],
        learning_rate=cfg["learning_rate"],
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"] * 2,
        weight_decay=cfg["weight_decay"],
        warmup_steps=cfg["warmup_ratio"],  # transformers 5.x: float in [0,1) = ratio
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="crf_f1" if use_crf else "f1",
        fp16=cfg.get("fp16", False),
        logging_steps=50,
        report_to=cfg.get("report_to", "none"),
        seed=cfg["seed"],
    )

    trainer = TwoHeadTrainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()

    # Validation picked the checkpoint; test is touched exactly once, here.
    test_metrics = trainer.evaluate(ds["test"], metric_key_prefix="test")
    print(json.dumps(test_metrics, indent=2))
    metrics_dir = Path("metrics")
    metrics_dir.mkdir(exist_ok=True)
    # Validation curve per epoch: lets us justify decoding choices on validation, not test.
    val_history = [
        {k: v for k, v in e.items() if k.startswith("eval_") or k == "epoch"}
        for e in trainer.state.log_history
        if "eval_f1" in e
    ]
    record = {**test_metrics, "config": cfg, "val_history": val_history}
    (metrics_dir / f"{cfg['run_name']}.json").write_text(json.dumps(record, indent=2))

    trainer.save_model(str(out_dir / "best"))
    if use_crf:  # stock HF classes ignore extra modules on load, so store transitions as JSON
        for d in (out_dir, out_dir / "best"):
            save_crfs(model.skillex_crf, d)
    # Intermediate checkpoints hold optimizer state (~3x model size); drop them to save disk.
    for ckpt in out_dir.glob("checkpoint-*"):
        shutil.rmtree(ckpt, ignore_errors=True)
    if cfg.get("hub_model_id"):
        push_best(out_dir / "best", cfg["hub_model_id"], test_metrics, cfg["run_name"])


def push_best(best_dir: Path, repo_id: str, test_metrics: dict, run_name: str) -> None:
    """Upload only the selected checkpoint (+ CRF json), not every epoch's checkpoint.

    Auth comes from the HF_TOKEN environment variable (a Kaggle secret), never from code.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(repo_id, exist_ok=True)
    f1 = test_metrics.get("test_crf_f1", test_metrics.get("test_vit_f1", test_metrics["test_f1"]))
    api.upload_folder(
        repo_id=repo_id, folder_path=str(best_dir), commit_message=f"{run_name}: test F1 {f1:.4f}"
    )
    print(f"pushed {best_dir} -> https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
