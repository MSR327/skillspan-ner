from __future__ import annotations

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from skillex.decode import decode_head
from skillex.labels import NUM_BIO, TYPES, bio_ids_to_tags, spans_from_tags


class SkillExtractor:
    """Extract SKILL and KNOWLEDGE spans (possibly overlapping) from raw text.

    Long texts are split into overlapping windows; each word takes its
    prediction from the window where it sits farthest from an edge (most context).
    Each word is labeled by its first subword, matching training.
    """

    def __init__(
        self, model_id: str, device: str = "cpu", stride: int = 32, decode: str = "argmax"
    ):
        self.decode = decode  # "argmax" or "viterbi" (BIO-constrained)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForTokenClassification.from_pretrained(model_id).to(device).eval()
        self.max_length = min(
            self.tokenizer.model_max_length, self.model.config.max_position_embeddings
        )
        self.device, self.stride = device, stride

    @torch.inference_mode()
    def extract(self, text: str) -> dict:
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            stride=self.stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding=True,
            return_tensors="pt",
        )
        offsets = enc.pop("offset_mapping")
        enc.pop("overflow_to_sample_mapping", None)
        probs = self.model(**{k: v.to(self.device) for k, v in enc.items()}).logits.cpu()

        char_span: dict[int, list[int]] = {}  # word -> [char_start, char_end]
        best: dict[int, tuple[int, list]] = {}  # word -> (edge distance, [(bio_id, prob) per head])
        for w in range(probs.shape[0]):
            word_ids = enc.word_ids(w)
            real = [t for t, wid in enumerate(word_ids) if wid is not None]
            for rank, t in enumerate(real):
                wid = word_ids[t]
                s, e = offsets[w, t].tolist()
                cs = char_span.setdefault(wid, [s, e])
                cs[0], cs[1] = min(cs[0], s), max(cs[1], e)
                if word_ids[t - 1] == wid:  # continuation subword: no prediction
                    continue
                edge = min(rank, len(real) - 1 - rank)
                if wid in best and best[wid][0] >= edge:
                    continue
                best[wid] = (edge, probs[w, t])

        order = sorted(best)
        spans = []
        for h, etype in enumerate(TYPES):
            head = torch.stack([best[i][1][h * NUM_BIO : (h + 1) * NUM_BIO] for i in order])
            word_probs = head.softmax(-1)
            ids = decode_head(head.numpy(), self.decode)
            tags = bio_ids_to_tags(ids, etype)
            for s in spans_from_tags([""] * len(order), tags):
                c0, c1 = char_span[order[s["start"]]][0], char_span[order[s["end"] - 1]][1]
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
        spans.sort(key=lambda s: (s["start"], s["type"]))
        return {
            "skills": sorted({s["text"] for s in spans if s["type"] == "SKILL"}),
            "knowledge": sorted({s["text"] for s in spans if s["type"] == "KNOWLEDGE"}),
            "spans": spans,
        }
