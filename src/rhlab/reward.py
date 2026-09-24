"""A learned reward model: the imperfect stand-in for the exact oracle verifier.

Given a full sequence ``prompt | chain`` it outputs ``P(chain is correct)``. It is
trained only on ``(chain, oracle_label)`` rows, so it must *guess* the verifier
from data. Trained on few rows it is a leaky proxy — it assigns high scores to
certain wrong chains — and that leak is exactly what lets the policy over-optimize
the proxy while true accuracy stalls (Goodhart / reward hacking).
"""

from __future__ import annotations

import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import NVOCAB, PAD
from .model import Block


class RewardModel(nn.Module):
    def __init__(self, d_model: int = 64, n_head: int = 4, n_layer: int = 2,
                 max_len: int = 16):
        super().__init__()
        self.tok = nn.Embedding(NVOCAB, d_model, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 1)
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
            if m.padding_idx is not None:
                with torch.no_grad():
                    m.weight[m.padding_idx].fill_(0)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B, T) -> reward logit (B,). Bidirectional pooling for scoring."""
        b, t = ids.shape
        pos = torch.arange(t, device=ids.device).unsqueeze(0).expand(b, t)
        x = self.tok(ids) + self.pos(pos)
        for blk in self.blocks:
            x = blk(x, None)  # no causal mask: this is a classifier, not a LM
        x = self.ln_f(x).mean(dim=1)
        return self.head(x).squeeze(-1)

    @torch.no_grad()
    def score(self, ids: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward(ids))


def _row_tensors(rows: list[tuple[list[int], int]]):
    ids = torch.tensor([r[0] for r in rows], dtype=torch.long)
    labels = torch.tensor([r[1] for r in rows], dtype=torch.float)
    return ids, labels


def fit_rm(rm: RewardModel, rows: list[tuple[list[int], int]], *, steps: int,
           batch: int, lr: float, seed: int) -> None:
    if not rows:
        return
    torch.manual_seed(seed)
    ids, labels = _row_tensors(rows)
    opt = torch.optim.AdamW(rm.parameters(), lr=lr)
    n = ids.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch, n),))
        logits = rm(ids[idx])
        loss = F.binary_cross_entropy_with_logits(logits, labels[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(rm.parameters(), 1.0)
        opt.step()


@torch.no_grad()
def rm_accuracy(rm: RewardModel, rows: list[tuple[list[int], int]], *,
                threshold: float = 0.5) -> float:
    """How well the reward model reproduces the oracle label on (chain, label) rows."""
    if not rows:
        return 0.0
    ids, labels = _row_tensors(rows)
    pred = (rm.score(ids) >= threshold).float()
    return float((pred == labels).float().mean())


def new_rm(seed: int, **kw) -> RewardModel:
    torch.manual_seed(seed)
    return RewardModel(**kw)


def make_rm_rows(examples, rng: random.Random) -> list[tuple[list[int], int]]:
    from .data import make_labelled
    return make_labelled(examples, rng)
