"""Constrained BIO decoding.

Per-token argmax can emit invalid sequences such as O -> I, which span scoring
then turns into an extra (often spurious) span. Viterbi finds the highest-
scoring sequence that obeys BIO rules: I may only follow B or I.
"""

from __future__ import annotations

import numpy as np

from skillex.labels import BIO2ID

_O, _B, _I = BIO2ID["O"], BIO2ID["B"], BIO2ID["I"]
_NEG = -1e9

# ALLOWED[prev, cur]: everything except O -> I.
_TRANSITION = np.zeros((3, 3))
_TRANSITION[_O, _I] = _NEG
_START = np.array([0.0, 0.0, _NEG])  # can't start with I


def _log_softmax(x: np.ndarray) -> np.ndarray:
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def viterbi_bio(scores: np.ndarray) -> list[int]:
    """scores: [T, 3] logits for O/B/I at each word. Returns the best valid tag ids."""
    if len(scores) == 0:
        return []
    emit = _log_softmax(np.asarray(scores, dtype=np.float64))
    best = _START + emit[0]
    back = np.zeros((len(emit), 3), dtype=np.int64)
    for t in range(1, len(emit)):
        cand = best[:, None] + _TRANSITION  # [prev, cur]
        back[t] = cand.argmax(0)
        best = cand.max(0) + emit[t]
    path = [int(best.argmax())]
    for t in range(len(emit) - 1, 0, -1):
        path.append(int(back[t, path[-1]]))
    return path[::-1]


def decode_head(scores: np.ndarray, method: str = "argmax") -> list[int]:
    if method == "viterbi":
        return viterbi_bio(scores)
    if method == "argmax":
        return np.asarray(scores).argmax(-1).tolist()
    raise ValueError(f"unknown decode method {method!r}")
