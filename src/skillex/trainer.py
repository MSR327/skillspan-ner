"""Trainer whose loss treats the model's 6 outputs as two independent 3-way heads."""

from __future__ import annotations

import torch.nn.functional as F
from transformers import Trainer

from skillex.labels import IGNORE_INDEX, NUM_BIO


def two_head_loss(logits, labels):
    """Sum of cross-entropies: SKILL head on logits[..., :3], KNOWLEDGE head on logits[..., 3:]."""
    ignore = labels == IGNORE_INDEX
    safe = labels.clamp(min=0)
    skill_y = (safe // NUM_BIO).masked_fill(ignore, IGNORE_INDEX)
    know_y = (safe % NUM_BIO).masked_fill(ignore, IGNORE_INDEX)
    loss_skill = F.cross_entropy(
        logits[..., :NUM_BIO].reshape(-1, NUM_BIO), skill_y.reshape(-1), ignore_index=IGNORE_INDEX
    )
    loss_know = F.cross_entropy(
        logits[..., NUM_BIO:].reshape(-1, NUM_BIO), know_y.reshape(-1), ignore_index=IGNORE_INDEX
    )
    return loss_skill + loss_know


class TwoHeadTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs["labels"]
        # Don't pass labels to the model: its built-in 6-way loss would misread the packed ids.
        outputs = model(**{k: v for k, v in inputs.items() if k != "labels"})
        loss = two_head_loss(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss
