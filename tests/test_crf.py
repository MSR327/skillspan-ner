import itertools

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from skillex.crf import BIOCRF, compact
from skillex.decode import viterbi_bio
from skillex.labels import IGNORE_INDEX


def _random_crf(seed=0):
    torch.manual_seed(seed)
    crf = BIOCRF()
    with torch.no_grad():
        crf.raw_trans.normal_(0, 1e-3)  # x SCALE(1000) -> transitions ~ N(0, 1)
        crf.raw_start.normal_(0, 1e-3)
    return crf


def _score(crf, em, path):
    trans, start = crf.transitions.detach(), crf.start.detach()
    s = start[path[0]] + em[0, path[0]]
    for t in range(1, len(path)):
        s = s + trans[path[t - 1], path[t]] + em[t, path[t]]
    return float(s)


def test_nll_matches_brute_force():
    crf = _random_crf()
    torch.manual_seed(1)
    em = torch.randn(4, 3)
    gold = [1, 2, 0, 1]
    scores = [_score(crf, em, list(p)) for p in itertools.product(range(3), repeat=4)]
    log_z = float(torch.logsumexp(torch.tensor(scores), 0))
    expected = log_z - _score(crf, em, gold)
    got = crf.nll(em[None], torch.tensor([gold]), torch.ones(1, 4, dtype=torch.bool))
    assert got.item() == pytest.approx(expected, rel=1e-5, abs=1e-4)


def test_nll_respects_mask():
    crf = _random_crf()
    em = torch.randn(1, 5, 3)
    tags = torch.tensor([[1, 2, 0, 0, 0]])
    short = crf.nll(em[:, :3], tags[:, :3], torch.ones(1, 3, dtype=torch.bool))
    masked = crf.nll(em, tags, torch.tensor([[True, True, True, False, False]]))
    assert masked.item() == pytest.approx(short.item(), rel=1e-5)


def test_learned_viterbi_matches_brute_force_argmax():
    crf = _random_crf(3)
    torch.manual_seed(4)
    em = torch.randn(4, 3)
    best = max(itertools.product(range(3), repeat=4), key=lambda p: _score(crf, em, list(p)))
    d = crf.to_dict()
    assert viterbi_bio(em.numpy(), d["transitions"], d["start"]) == list(best)


def test_o_to_i_stays_forbidden():
    crf = _random_crf()
    d = crf.to_dict()
    em = np.array([[5.0, 0, 0], [0, 0, 5.0]])  # emissions scream O then I
    path = viterbi_bio(em, d["transitions"], d["start"])
    assert path != [0, 2]


def test_compact_gathers_first_subwords_and_drops_empty_rows():
    logits = torch.arange(2 * 5 * 3, dtype=torch.float).reshape(2, 5, 3)
    logits = torch.cat([logits, torch.zeros(1, 5, 3)])
    labels = torch.tensor(
        [
            [IGNORE_INDEX, 1, IGNORE_INDEX, 2, IGNORE_INDEX],
            [IGNORE_INDEX, 0, 1, 2, IGNORE_INDEX],
            [IGNORE_INDEX] * 5,  # no labelled words -> dropped
        ]
    )
    em, tags, mask = compact(logits, labels)
    assert em.shape == (2, 3, 3)
    assert tags.tolist() == [[1, 2, 0], [0, 1, 2]]
    assert mask.tolist() == [[True, True, False], [True, True, True]]
    assert torch.equal(em[0, 0], logits[0, 1]) and torch.equal(em[0, 1], logits[0, 3])


def test_gradients_reach_transitions():
    crf = BIOCRF()
    em = torch.randn(2, 6, 3, requires_grad=True)
    tags = torch.tensor([[1, 2, 2, 0, 1, 2], [0, 0, 1, 2, 0, 0]])
    crf.nll(em, tags, torch.ones(2, 6, dtype=torch.bool)).backward()
    assert crf.raw_trans.grad is not None and crf.raw_trans.grad.abs().sum() > 0
    assert torch.isfinite(em.grad).all()
