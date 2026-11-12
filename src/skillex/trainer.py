"""Trainer whose loss treats the model's 6 outputs as two independent 3-way heads.

Optionally each head gets a CRF (model.skillex_crf, a ModuleList of BIOCRF), in
which case the loss is the CRF negative log-likelihood instead of cross-entropy.
"""

from __future__ import annotations

import torch.nn.functional as F
from transformers import Trainer

from skillex.crf import compact
from skillex.labels import IGNORE_INDEX, NUM_BIO


def _head_labels(labels, h):
    ignore = labels == IGNORE_INDEX
    safe = labels.clamp(min=0)
    y = (safe // NUM_BIO) if h == 0 else (safe % NUM_BIO)
    return y.masked_fill(ignore, IGNORE_INDEX)


def two_head_loss(logits, labels, crfs=None):
    """Sum over heads: cross-entropy, or CRF NLL if crfs is given."""
    loss = 0.0
    for h in range(2):
        head_logits = logits[..., h * NUM_BIO : (h + 1) * NUM_BIO]
        y = _head_labels(labels, h)
        if crfs is None:
            loss = loss + F.cross_entropy(
                head_logits.reshape(-1, NUM_BIO), y.reshape(-1), ignore_index=IGNORE_INDEX
            )
        else:
            em, tags, mask = compact(head_logits, y)
            if len(em):
                loss = loss + crfs[h].nll(em, tags, mask)
    return loss


class TwoHeadTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        # Don't pass labels to the model: its built-in 6-way loss would misread the packed ids.
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        crfs = getattr(self.accelerator.unwrap_model(model), "skillex_crf", None)
        loss = two_head_loss(outputs.logits, labels, crfs)
        return (loss, outputs) if return_outputs else loss
