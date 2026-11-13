from skillex.errors import categorize, partial_prf


def test_each_category():
    gold = {(0, 3), (5, 6), (8, 10), (12, 14), (20, 22)}
    pred = {
        (0, 3),  # correct
        (5, 7),  # too_long  (contains gold 5-6)
        (8, 9),  # too_short (inside gold 8-10)
        (13, 16),  # shifted (partial overlap with 12-14)
        (30, 31),  # spurious
        (40, 42),  # wrong_type (overlaps other-type gold)
    }
    g, p = categorize(gold, pred, other_gold={(40, 41)})
    assert g == {"correct": 1, "too_long": 1, "too_short": 1, "shifted": 1, "missed": 1}
    assert p == {"too_long": 1, "too_short": 1, "shifted": 1, "spurious": 1, "wrong_type": 1}


def test_every_gold_span_counted_once():
    gold = {(0, 4)}
    pred = {(0, 1), (2, 4)}  # gold split in two
    g, p = categorize(gold, pred, set())
    assert sum(g.values()) == 1 and g["too_short"] == 1
    assert p["too_short"] == 2


def test_partial_prf():
    m = partial_prf([{(0, 3)}, {(5, 6)}], [{(1, 2)}, set()])
    assert m["precision"] == 1.0 and m["recall"] == 0.5
