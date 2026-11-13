"""Split raw text into words and sentences the way SkillSpan's training data is split.

Kept free of torch/transformers so it is cheap to import and test.
"""

from __future__ import annotations

import re

# SkillSpan's training words split punctuation off ("AWS" "," not "AWS,"), so split raw
# text the same way before tokenizing; otherwise spans swallow commas and full stops.
_WORD = re.compile(r"\w+(?:['’.\-]\w+)*|\S")


def split_words(text: str) -> list[tuple[int, int]]:
    """Char (start, end) of each word: runs of word chars (keeping "Node.js", "e-mail") or single symbols."""
    return [m.span() for m in _WORD.finditer(text)]


def split_sentences(text: str, words: list[tuple[int, int]]) -> list[list[int]]:
    """Group word indices into sentences: break after . ! ? or at a newline.

    The model was trained on single sentences; feeding a whole multi-sentence posting
    at once makes it miss most spans, so each sentence is tagged on its own.
    """
    sents, cur = [], []
    for i, (a, b) in enumerate(words):
        if cur and "\n" in text[words[cur[-1]][1] : a]:
            sents.append(cur)
            cur = []
        cur.append(i)
        if text[a:b] in {".", "!", "?"}:
            sents.append(cur)
            cur = []
    if cur:
        sents.append(cur)
    return sents
