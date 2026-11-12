"""Span-level exact-match P/R/F1, scored per head and micro-averaged over both.

Same rule as seqeval's default (a span counts only if type, start and end all
match), but it can score SKILL and KNOWLEDGE independently, so overlapping
spans are counted correctly.
"""

from __future__ import annotations

import numpy as np

from skillex.labels import IGNORE_INDEX, NUM_BIO, TYPES, bio_ids_to_tags, spans_from_tags


def _prf(tp: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def span_scores(logits: np.ndarray, label_ids: np.ndarray) -> dict:
    """logits: [N, L, 6]; label_ids: [N, L] packed (skill*3+knowledge) or IGNORE_INDEX."""
    counts = {t: [0, 0, 0] for t in TYPES}  # tp, n_pred, n_gold
    for h, etype in enumerate(TYPES):
        preds = logits[..., h * NUM_BIO : (h + 1) * NUM_BIO].argmax(-1)
        for pred_row, label_row in zip(preds, label_ids):
            keep = label_row != IGNORE_INDEX
            gold_ids = (label_row[keep] // NUM_BIO) if h == 0 else (label_row[keep] % NUM_BIO)
            dummy = [""] * int(keep.sum())
            gold = {(s["start"], s["end"]) for s in spans_from_tags(dummy, bio_ids_to_tags(gold_ids.tolist(), etype))}
            pred = {(s["start"], s["end"]) for s in spans_from_tags(dummy, bio_ids_to_tags(pred_row[keep].tolist(), etype))}
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


def compute_metrics(eval_pred) -> dict:
    logits, label_ids = eval_pred
    return span_scores(np.asarray(logits), np.asarray(label_ids))
