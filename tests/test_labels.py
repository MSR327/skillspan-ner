import numpy as np

from skillex.labels import (
    IGNORE_INDEX,
    NUM_BIO,
    align_labels,
    bio_ids_to_tags,
    decode_pair,
    encode_pair,
    spans_from_tags,
)
from skillex.metrics import span_scores


def test_encode_pair_keeps_overlap():
    # "promoting all QA topics": whole phrase is SKILL, "QA" is also KNOWLEDGE.
    skill = ["B", "I", "I", "I"]
    know = ["O", "O", "B", "O"]
    packed = encode_pair(skill, know)
    assert [decode_pair(x) for x in packed] == [(1, 0), (2, 0), (2, 1), (2, 0)]


def test_align_labels_first_subword_only():
    # [CLS] python  pro ##gramming  [SEP]
    word_ids = [None, 0, 1, 1, None]
    assert align_labels(word_ids, [5, 3]) == [IGNORE_INDEX, 5, 3, IGNORE_INDEX, IGNORE_INDEX]


def test_spans_from_tags():
    tokens = ["Strong", "Python", "and", "team", "leadership", "skills"]
    tags = ["O", "B-KNOWLEDGE", "O", "B-SKILL", "I-SKILL", "O"]
    spans = spans_from_tags(tokens, tags)
    assert [(s["type"], s["text"]) for s in spans] == [
        ("KNOWLEDGE", "Python"),
        ("SKILL", "team leadership"),
    ]


def test_stray_inside_tag_starts_new_span():
    spans = spans_from_tags(["a", "b", "c"], ["B-SKILL", "I-KNOWLEDGE", "I-KNOWLEDGE"])
    assert [(s["type"], s["text"]) for s in spans] == [("SKILL", "a"), ("KNOWLEDGE", "b c")]


def test_bio_ids_to_tags():
    assert bio_ids_to_tags([0, 1, 2], "SKILL") == ["O", "B-SKILL", "I-SKILL"]


def _logits_for(skill_ids, know_ids):
    out = np.full((1, len(skill_ids), 2 * NUM_BIO), -10.0)
    for t, (s, k) in enumerate(zip(skill_ids, know_ids)):
        out[0, t, s] = 10.0
        out[0, t, NUM_BIO + k] = 10.0
    return out


def test_span_scores_perfect_with_overlap():
    skill, know = [1, 2, 2, 2], [0, 0, 1, 0]
    labels = np.array([[IGNORE_INDEX] + [s * NUM_BIO + k for s, k in zip(skill, know)]])
    logits = np.concatenate([np.zeros((1, 1, 2 * NUM_BIO)), _logits_for(skill, know)], axis=1)
    m = span_scores(logits, labels)
    assert m["f1"] == m["f1_skill"] == m["f1_knowledge"] == 1.0


def test_span_scores_partial_boundary_is_wrong():
    # Gold SKILL spans words 0-2; prediction stops at word 1 -> no credit (exact match).
    labels = np.array([[1 * NUM_BIO, 2 * NUM_BIO, 2 * NUM_BIO]])
    logits = _logits_for([1, 2, 0], [0, 0, 0])
    m = span_scores(logits, labels)
    assert m["f1_skill"] == 0.0 and m["recall_skill"] == 0.0


def test_split_words_matches_skillspan_style():
    from skillex.predict import split_words

    text = "Python, AWS and Node.js; mentor juniors."
    assert [text[a:b] for a, b in split_words(text)] == [
        "Python", ",", "AWS", "and", "Node.js", ";", "mentor", "juniors", ".",
    ]


def test_split_sentences_breaks_on_punctuation_and_newlines():
    from skillex.predict import split_sentences, split_words

    text = "Know SQL. Mentor juniors!\nTeam player"
    words = split_words(text)
    sents = split_sentences(text, words)
    assert [[text[slice(*words[i])] for i in s] for s in sents] == [
        ["Know", "SQL", "."], ["Mentor", "juniors", "!"], ["Team", "player"],
    ]
