from __future__ import annotations

from datasets import DatasetDict, load_dataset

from skillex.labels import align_labels, encode_pair


def load_skillspan(name: str = "jjzha/skillspan") -> DatasetDict:
    """Load SkillSpan; pack both tag columns into one int per word (`ner_tags`), losing nothing."""
    ds = load_dataset(name)

    def _pack(ex):
        return {"ner_tags": encode_pair(ex["tags_skill"], ex["tags_knowledge"])}

    return ds.map(_pack, remove_columns=["tags_skill", "tags_knowledge"])


def tokenize_dataset(ds: DatasetDict, tokenizer, max_length: int = 256) -> DatasetDict:
    def _tok(batch):
        enc = tokenizer(
            batch["tokens"], is_split_into_words=True, truncation=True, max_length=max_length
        )
        enc["labels"] = [
            align_labels(enc.word_ids(i), tags) for i, tags in enumerate(batch["ner_tags"])
        ]
        return enc

    ds = DatasetDict(ds)  # also accepts a plain {split: Dataset} dict
    return DatasetDict(
        {name: split.map(_tok, batched=True, remove_columns=split.column_names) for name, split in ds.items()}
    )
