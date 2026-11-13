"""Error analysis for span extraction: why is a span wrong, not just how many.

Each gold span ends up in exactly one category, and so does each predicted span that is not
an exact match:

    correct      exact boundaries, same type
    too_short    prediction sits inside the gold span        (gold "manage a team", pred "manage")
    too_long     prediction contains the gold span           (gold "SQL", pred "SQL databases")
    shifted      partial overlap, neither contains the other
    wrong_type   overlaps a gold span of the other type only
    missed       gold span with no overlapping prediction    (false negative)
    spurious     prediction overlapping no gold span at all  (false positive)

Spans are (start, end) word indices, end exclusive, per type.
"""

from __future__ import annotations

from collections import Counter

Span = tuple[int, int]
BOUNDARY = ("too_short", "too_long", "shifted")


def _overlap(a: Span, b: Span) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _boundary_kind(gold: Span, pred: Span) -> str:
    if gold[0] <= pred[0] and pred[1] <= gold[1]:
        return "too_short"
    if pred[0] <= gold[0] and gold[1] <= pred[1]:
        return "too_long"
    return "shifted"


def categorize(gold: set[Span], pred: set[Span], other_gold: set[Span]) -> tuple[Counter, Counter]:
    """Categorize one sentence, one span type.

    Returns (gold_counts, pred_counts). gold_counts covers every gold span (correct,
    boundary kinds, missed). pred_counts covers every predicted span that is not exact
    (boundary kinds, wrong_type, spurious).
    """
    g_counts, p_counts = Counter(), Counter()
    for g in gold:
        if g in pred:
            g_counts["correct"] += 1
            continue
        hits = [p for p in pred if _overlap(g, p)]
        # The most informative overlap: the prediction sharing the most words with g.
        g_counts[
            _boundary_kind(g, max(hits, key=lambda p: min(g[1], p[1]) - max(g[0], p[0])))
            if hits
            else "missed"
        ] += 1
    for p in pred - gold:
        hits = [g for g in gold if _overlap(g, p)]
        if hits:
            p_counts[_boundary_kind(max(hits, key=lambda g: min(g[1], p[1]) - max(g[0], p[0])), p)] += 1
        elif any(_overlap(p, o) for o in other_gold):
            p_counts["wrong_type"] += 1
        else:
            p_counts["spurious"] += 1
    return g_counts, p_counts


def partial_prf(gold_sets: list[set[Span]], pred_sets: list[set[Span]]) -> dict:
    """Lenient P/R/F1: a span counts as found if any same-type span overlaps it."""
    tp_p = sum(sum(any(_overlap(p, g) for g in G) for p in P) for G, P in zip(gold_sets, pred_sets))
    tp_r = sum(sum(any(_overlap(g, p) for p in P) for g in G) for G, P in zip(gold_sets, pred_sets))
    n_p = sum(map(len, pred_sets))
    n_g = sum(map(len, gold_sets))
    prec = tp_p / n_p if n_p else 0.0
    rec = tp_r / n_g if n_g else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"precision": prec, "recall": rec, "f1": f1}
