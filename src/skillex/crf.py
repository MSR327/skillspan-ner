"""Linear-chain CRF over BIO tags, one per head, trained on first-subword positions.

Unlike decode.viterbi_bio's fixed rules, a CRF *learns* transition scores
(e.g. how likely I -> I is inside long skill spans). The O -> I ban stays hard.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch import nn

from skillex.labels import BIO2ID, IGNORE_INDEX, NUM_BIO

_O, _I = BIO2ID["O"], BIO2ID["I"]
_NEG = -1e4
CRF_FILE = "skillex_crf.json"


class BIOCRF(nn.Module):
    # Transitions start at 0 while the encoder uses lr ~3e-5; scaling the raw
    # parameter gives them an effectively larger step under Adam (scale-invariant
    # gradients), so they actually learn within a few epochs.
    SCALE = 1000.0

    def __init__(self):
        super().__init__()
        self.raw_trans = nn.Parameter(torch.zeros(NUM_BIO, NUM_BIO))
        self.raw_start = nn.Parameter(torch.zeros(NUM_BIO))
        mask = torch.zeros(NUM_BIO, NUM_BIO)
        mask[_O, _I] = _NEG
        start_mask = torch.zeros(NUM_BIO)
        start_mask[_I] = _NEG
        self.register_buffer("trans_mask", mask, persistent=False)
        self.register_buffer("start_mask", start_mask, persistent=False)

    @property
    def transitions(self) -> torch.Tensor:  # [prev, cur]
        return self.raw_trans * self.SCALE + self.trans_mask

    @property
    def start(self) -> torch.Tensor:
        return self.raw_start * self.SCALE + self.start_mask

    def nll(self, emissions: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean negative log-likelihood. emissions [B,T,3], tags [B,T], mask [B,T] (prefix-True)."""
        emissions = emissions.float()
        trans, start = self.transitions, self.start
        B, T, _ = emissions.shape
        ar = torch.arange(B, device=emissions.device)

        # log partition function (forward algorithm)
        alpha = start + emissions[:, 0]
        for t in range(1, T):
            nxt = torch.logsumexp(alpha[:, :, None] + trans[None] + emissions[:, t, None, :], dim=1)
            alpha = torch.where(mask[:, t, None], nxt, alpha)
        log_z = torch.logsumexp(alpha, dim=1)

        # score of the gold path
        gold = start[tags[:, 0]] + emissions[ar, 0, tags[:, 0]]
        for t in range(1, T):
            step = trans[tags[:, t - 1], tags[:, t]] + emissions[ar, t, tags[:, t]]
            gold = gold + step * mask[:, t]
        return (log_z - gold).mean()

    def to_dict(self) -> dict:
        return {"transitions": self.transitions.tolist(), "start": self.start.tolist()}


def compact(logits: torch.Tensor, labels: torch.Tensor):
    """Gather the labelled (first-subword) positions of each row to the front.

    Returns emissions [B',T,C], labels [B',T], mask [B',T]; rows with no labelled
    positions are dropped.
    """
    keep = labels != IGNORE_INDEX
    lengths = keep.sum(1)
    rows = lengths > 0
    logits, labels, keep, lengths = logits[rows], labels[rows], keep[rows], lengths[rows]
    T = int(lengths.max()) if len(lengths) else 0
    order = torch.sort((~keep).int(), dim=1, stable=True).indices[:, :T]
    em = logits.gather(1, order[..., None].expand(-1, -1, logits.shape[-1]))
    lab = labels.gather(1, order)
    mask = torch.arange(T, device=labels.device)[None] < lengths[:, None]
    return em, lab.masked_fill(~mask, 0), mask


def save_crfs(crfs, out_dir: str | Path) -> None:
    Path(out_dir, CRF_FILE).write_text(json.dumps([c.to_dict() for c in crfs]))


def load_crfs(model_dir: str | Path) -> list[dict] | None:
    """Return per-head {"transitions", "start"} if the model was trained with a CRF.

    Works for local dirs; for Hub ids, tries to download the file.
    """
    p = Path(model_dir, CRF_FILE)
    if not p.exists():
        try:
            from huggingface_hub import hf_hub_download

            p = Path(hf_hub_download(str(model_dir), CRF_FILE))
        except Exception:  # noqa: BLE001 - no CRF file is the normal case
            return None
    return json.loads(p.read_text())
