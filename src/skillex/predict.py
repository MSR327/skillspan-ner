from __future__ import annotations

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from skillex.crf import load_crfs
from skillex.decode import decode_head
from skillex.labels import NUM_BIO, TYPES, bio_ids_to_tags, spans_from_tags
from skillex.text import split_sentences, split_words


class SkillExtractor:
    """Extract SKILL and KNOWLEDGE spans (possibly overlapping) from raw text.

    Text is split into sentences (the training unit) and tagged as one batch.
    A sentence longer than the model's limit is split into overlapping windows; each
    word takes its prediction from the window where it sits farthest from an edge.
    Each word is labeled by its first subword, matching training.
    """

    def __init__(
        self,
        model_id: str,
        device: str = "cpu",
        stride: int = 32,
        decode: str = "viterbi",
        batch_size: int = 16,
    ):
        self.decode = decode  # "viterbi" (BIO-constrained, +1.9 F1 in sweep) or "argmax"
        self.crfs = load_crfs(model_id)  # learned transitions if trained with crf: true
        if self.crfs and decode == "viterbi":
            self.decode = "crf"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForTokenClassification.from_pretrained(model_id).to(device).eval()
        self.max_length = min(
            self.tokenizer.model_max_length, self.model.config.max_position_embeddings
        )
        self.device, self.stride, self.batch_size = device, stride, batch_size

    @torch.inference_mode()
    def extract(self, text: str) -> dict:
        words = split_words(text)
        if not words:
            return {"skills": [], "knowledge": [], "spans": []}
        sents = split_sentences(text, words)
        enc = self.tokenizer(
            [[text[a:b] for a, b in (words[i] for i in sent)] for sent in sents],
            is_split_into_words=True,
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            padding=True,
            return_tensors="pt",
        )
        window_sent = enc.pop("overflow_to_sample_mapping").tolist()
        # Run sentences in small batches: a long posting (~150 sentences) in one forward
        # pass runs out of GPU memory. Each batch is trimmed to its own longest sentence.
        n_win, width = enc["input_ids"].shape
        probs = torch.zeros(n_win, width, self.model.config.num_labels)
        for b in range(0, n_win, self.batch_size):
            rows = slice(b, b + self.batch_size)
            n = int(enc["attention_mask"][rows].sum(1).max())
            batch = {k: v[rows, :n].to(self.device) for k, v in enc.items()}
            probs[rows, :n] = self.model(**batch).logits.float().cpu()

        best: dict[int, tuple[int, list]] = {}  # global word -> (edge distance, logits)
        for w in range(probs.shape[0]):
            sent = sents[window_sent[w]]
            word_ids = enc.word_ids(w)
            real = [t for t, wid in enumerate(word_ids) if wid is not None]
            for rank, t in enumerate(real):
                wid = word_ids[t]
                if word_ids[t - 1] == wid:  # continuation subword: no prediction
                    continue
                gid = sent[wid]
                edge = min(rank, len(real) - 1 - rank)
                if gid in best and best[gid][0] >= edge:
                    continue
                best[gid] = (edge, probs[w, t])

        spans = []
        for sent in sents:
            spans += self._decode_sentence(text, words, [i for i in sent if i in best], best)
        spans.sort(key=lambda s: (s["start"], s["type"]))
        return {
            "skills": sorted({s["text"] for s in spans if s["type"] == "SKILL"}),
            "knowledge": sorted({s["text"] for s in spans if s["type"] == "KNOWLEDGE"}),
            "spans": spans,
        }

    def _decode_sentence(self, text, words, order, best) -> list[dict]:
        if not order:
            return []
        spans = []
        for h, etype in enumerate(TYPES):
            head = torch.stack([best[i][1][h * NUM_BIO : (h + 1) * NUM_BIO] for i in order])
            word_probs = head.softmax(-1)
            ids = decode_head(head.numpy(), self.decode, self.crfs[h] if self.crfs else None)
            tags = bio_ids_to_tags(ids, etype)
            for s in spans_from_tags([""] * len(order), tags):
                c0, c1 = words[order[s["start"]]][0], words[order[s["end"] - 1]][1]
                score = sum(float(word_probs[j, ids[j]]) for j in range(s["start"], s["end"]))
                spans.append(
                    {
                        "type": etype,
                        "text": text[c0:c1],
                        "start": c0,
                        "end": c1,
                        "score": round(score / (s["end"] - s["start"]), 4),
                    }
                )
        return spans
