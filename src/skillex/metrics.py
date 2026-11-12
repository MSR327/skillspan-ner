"""Span-level exact-match P/R/F1, scored per head and micro-averaged over both.

Same rule as seqeval's default (a span counts only if type, start and end all
match), but it can score SKILL and KNOWLEDGE independently, so overlapping
spans are counted correctly. Reports argmax (plain keys), fixed-rule Viterbi
(`vit_*`) and, for CRF models, learned-transition Viterbi (`crf_*`).
"""

from __future__ import annotations

import numpy as np

from skillex.decode import decode_head
from skillex.labels import IGNORE_INDEX, NUM_BIO, TYPES, bio_ids_to_tags, spans_from_tags


def _prf(tp: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def _span_set(ids: list[int], etype: str) -> set[tuple[int, int]]:
    tags = bio_ids_to_tags(ids, etype)
    return {(s["start"], s["end"]) for s in spans_from_tags([""] * len(tags), tags)}


def span_scores(
    logits: np.ndarray, label_ids: np.ndarray, decode: str = "argmax", crfs: list | None = None
) -> dict:
    """logits: [N, L, 6]; label_ids: [N, L] packed (skill*3+knowledge) or IGNORE_INDEX."""
    counts = {t: [0, 0, 0] for t in TYPES}  # tp, n_pred, n_gold
    for h, etype in enumerate(TYPES):
        head_logits = logits[..., h * NUM_BIO : (h + 1) * NUM_BIO]
        for row_logits, label_row in zip(head_logits, label_ids):
            keep = label_row != IGNORE_INDEX
            gold_ids = (label_row[keep] // NUM_BIO) if h == 0 else (label_row[keep] % NUM_BIO)
            gold = _span_set(gold_ids.tolist(), etype)
            crf = crfs[h] if crfs else None
            pred = _span_set(decode_head(row_logits[keep], decode, crf), etype)
            c = counts[etype]
            c[0] += len(gold & pred)
            c[1] += len(pred)
            c[2] += len(gold)

    out = {}
    for etype, (tp, n_pred, n_gold) in counts.items():
        p, r, f = _prf(tp, n_pred, n_gold)
        key = etype.lower()
        out |= {f"precision_{key}": p, f"recall_{key}": r, f"f1_{key}": f}
    tp, n_pred, n_gold = (sum(c[i] for c in counts.values()) for i in range(3))
    out["precision"], out["recall"], out["f1"] = _prf(tp, n_pred, n_gold)
    return out


def make_compute_metrics(get_crfs=None):
    """get_crfs: callable returning current per-head CRF params (live during training)."""

    def compute_metrics(eval_pred) -> dict:
        logits, label_ids = (np.asarray(x) for x in eval_pred)
        out = span_scores(logits, label_ids, "argmax")
        vit = span_scores(logits, label_ids, "viterbi")
        out |= {f"vit_{k}": v for k, v in vit.items()}
        if get_crfs is not None:
            crf = span_scores(logits, label_ids, "crf", get_crfs())
            out |= {f"crf_{k}": v for k, v in crf.items()}
        return out

    return compute_metrics


compute_metrics = make_compute_metrics()
