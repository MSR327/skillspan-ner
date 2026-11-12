import itertools

import numpy as np

from skillex.decode import decode_head, viterbi_bio

O, B, I = 0, 1, 2


def _scores(tags, conf=5.0):
    s = np.zeros((len(tags), 3))
    for t, tag in enumerate(tags):
        s[t, tag] = conf
    return s


def test_viterbi_matches_argmax_on_valid_sequence():
    s = _scores([O, B, I, I, O, B])
    assert viterbi_bio(s) == decode_head(s, "argmax") == [O, B, I, I, O, B]


def test_viterbi_never_emits_o_then_i():
    rng = np.random.default_rng(0)
    for _ in range(200):
        path = viterbi_bio(rng.normal(size=(12, 3)) * 3)
        assert path[0] != I
        assert all(not (a == O and b == I) for a, b in itertools.pairwise(path))


def test_viterbi_repairs_stray_inside():
    # argmax says O, I, I (invalid); I is only weakly preferred over B at position 1.
    s = np.array([[5.0, 0.0, 0.0], [0.0, 2.0, 2.5], [0.0, 0.0, 5.0]])
    assert decode_head(s, "argmax") == [O, I, I]
    assert viterbi_bio(s) == [O, B, I]


def test_empty():
    assert viterbi_bio(np.zeros((0, 3))) == []
