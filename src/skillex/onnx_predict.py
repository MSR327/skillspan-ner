"""Torch-free skill extraction: ONNX Runtime on CPU + numpy CRF decoding.

Same algorithm as `skillex.predict.SkillExtractor` (sentence split, overlapping windows,
first-subword labels, per-head CRF Viterbi), but needs only onnxruntime, tokenizers and numpy,
so it fits small CPU hosts (e.g. a 1 GB Streamlit app). Uses `onnx/model_quantized.onnx`
from the model repo (271 MB; test F1 0.6309 vs 0.6318 for fp32).

    from skillex.onnx_predict import OnnxSkillExtractor
    ex = OnnxSkillExtractor("srish327/deberta-v3-skillspan-crf")
    ex.extract("Strong SQL and stakeholder management skills.")
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from skillex.decode import decode_head
from skillex.labels import NUM_BIO, TYPES, bio_ids_to_tags, spans_from_tags
from skillex.text import split_sentences, split_words

CRF_FILE = "skillex_crf.json"


def _file(model_id: str, name: str) -> Path:
    local = Path(model_id, name)
    if local.exists():
        return local
    from huggingface_hub import hf_hub_download

    return Path(hf_hub_download(model_id, name))


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


class OnnxSkillExtractor:
    def __init__(
        self,
        model_id: str,
        onnx_file: str = "onnx/model_quantized.onnx",
        stride: int = 32,
        batch_size: int = 8,
        max_length: int = 512,
        threads: int | None = None,
    ):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        self.session = ort.InferenceSession(
            str(_file(model_id, onnx_file)), opts, providers=["CPUExecutionProvider"]
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        try:
            self.crfs = json.loads(_file(model_id, CRF_FILE).read_text())
        except Exception:  # noqa: BLE001 - model trained without a CRF
            self.crfs = None
        self.decode = "crf" if self.crfs else "viterbi"
        self.max_length = min(self.tokenizer.model_max_length, max_length)
        self.stride, self.batch_size = stride, batch_size

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
            return_tensors="np",
        )
        window_sent = enc["overflow_to_sample_mapping"].tolist()
        ids, mask = enc["input_ids"].astype(np.int64), enc["attention_mask"].astype(np.int64)
        n_win, width = ids.shape
        logits = np.zeros((n_win, width, NUM_BIO * len(TYPES)), dtype=np.float32)
        for b in range(0, n_win, self.batch_size):
            rows = slice(b, b + self.batch_size)
            n = int(mask[rows].sum(1).max())  # trim each batch to its own longest window
            (out,) = self.session.run(None, {"input_ids": ids[rows, :n], "attention_mask": mask[rows, :n]})
            logits[rows, :n] = out

        best: dict[int, tuple[int, np.ndarray]] = {}  # word -> (distance from window edge, logits)
        for w in range(n_win):
            sent = sents[window_sent[w]]
            word_ids = enc.word_ids(w)
            real = [t for t, wid in enumerate(word_ids) if wid is not None]
            for rank, t in enumerate(real):
                wid = word_ids[t]
                if word_ids[t - 1] == wid:  # continuation subword
                    continue
                gid = sent[wid]
                edge = min(rank, len(real) - 1 - rank)
                if gid in best and best[gid][0] >= edge:
                    continue
                best[gid] = (edge, logits[w, t])

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
            head = np.stack([best[i][1][h * NUM_BIO : (h + 1) * NUM_BIO] for i in order])
            probs = _softmax(head.astype(np.float64))
            ids = decode_head(head, self.decode, self.crfs[h] if self.crfs else None)
            for s in spans_from_tags([""] * len(order), bio_ids_to_tags(ids, etype)):
                c0, c1 = words[order[s["start"]]][0], words[order[s["end"] - 1]][1]
                score = sum(float(probs[j, ids[j]]) for j in range(s["start"], s["end"]))
                spans.append({"type": etype, "text": text[c0:c1], "start": c0, "end": c1,
                              "score": round(score / (s["end"] - s["start"]), 4)})
        return spans
