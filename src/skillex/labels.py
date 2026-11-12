"""Label schema: two independent BIO taggers (SKILL, KNOWLEDGE) sharing one encoder.

SkillSpan annotates skills and knowledge in separate columns because they can
overlap (e.g. "QA" is KNOWLEDGE inside the SKILL "promoting all QA relevant
topics"). A single BIO sequence can't represent that, so each type gets its own.

Encoding tricks (so we can reuse a stock `AutoModelForTokenClassification`):
- Model has 6 outputs: [0:3] = SKILL O/B/I, [3:6] = KNOWLEDGE O/B/I.
- Training label per token packs both heads into one int: skill * 3 + knowledge.

Kept free of torch/transformers imports so it is fast to unit-test in CI.
"""

from __future__ import annotations

TYPES = ("SKILL", "KNOWLEDGE")
BIO = ("O", "B", "I")
BIO2ID = {t: i for i, t in enumerate(BIO)}
NUM_BIO = len(BIO)
NUM_OUTPUTS = NUM_BIO * len(TYPES)
IGNORE_INDEX = -100  # ignored by the loss

# Names for the model's 6 output columns (stored in config.json for readability).
ID2LABEL = {h * NUM_BIO + i: f"{t}-{b}" for h, t in enumerate(TYPES) for i, b in enumerate(BIO)}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}


def encode_pair(skill_tags: list[str], knowledge_tags: list[str]) -> list[int]:
    """Pack per-word SKILL and KNOWLEDGE BIO tags into one int per word (0..8)."""
    return [
        BIO2ID[s] * NUM_BIO + BIO2ID[k] for s, k in zip(skill_tags, knowledge_tags, strict=True)
    ]


def decode_pair(label: int) -> tuple[int, int]:
    """Inverse of encode_pair for one value: -> (skill_bio_id, knowledge_bio_id)."""
    return divmod(label, NUM_BIO)


def align_labels(word_ids: list[int | None], word_labels: list[int]) -> list[int]:
    """Map word-level labels onto subword tokens.

    Only the first subword of each word carries the label; special tokens and
    continuation subwords get IGNORE_INDEX so they don't affect loss or metrics.
    """
    labels, prev = [], None
    for wid in word_ids:
        if wid is None or wid == prev:
            labels.append(IGNORE_INDEX)
        else:
            labels.append(word_labels[wid])
        prev = wid
    return labels


def bio_ids_to_tags(ids: list[int], etype: str) -> list[str]:
    return [BIO[i] if i == 0 else f"{BIO[i]}-{etype}" for i in ids]


def spans_from_tags(tokens: list[str], tags: list[str]) -> list[dict]:
    """Collapse BIO tags into spans: [{"type", "text", "start", "end"}] (word indices, end exclusive).

    A stray I- tag (no preceding B-/I- of the same type) starts a new span,
    matching seqeval/conlleval's default behaviour.
    """
    spans: list[dict] = []
    cur: dict | None = None
    for i, tag in enumerate(tags):
        prefix, _, etype = tag.partition("-")
        if prefix == "I" and cur is not None and cur["type"] == etype:
            cur["end"] = i + 1
            continue
        if cur is not None:
            spans.append(cur)
            cur = None
        if prefix in ("B", "I"):
            cur = {"type": etype, "start": i, "end": i + 1}
    if cur is not None:
        spans.append(cur)
    for s in spans:
        s["text"] = " ".join(tokens[s["start"] : s["end"]])
    return spans
