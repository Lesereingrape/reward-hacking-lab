"""The tiny decoder-only policy transformer, sized so the whole loop runs on CPU.

Same nanoGPT-shaped architecture the other labs use (learned token + position
embeddings, a couple of causal attention/MLP blocks, a linear head) at a few tens
of thousands of parameters, so a multi-round self-improvement loop with sampling,
reward-model scoring and fine-tuning completes in CPU seconds.

For the reward-hacking study the two primitives that matter are ``generate_cot``
— the policy draws its own candidate chains, which a reward model then ranks — and
``seq_logprob``/``sft_loss`` — the policy is fine-tuned on the chains it is told
are best. Whether "best" comes from the exact oracle verifier or from an imperfect
learned reward model is the whole experiment.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import NVOCAB, PAD


class Block(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        a, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x


class TinyTransformer(nn.Module):
    def __init__(self, d_model: int = 64, n_head: int = 4, n_layer: int = 2,
                 max_len: int = 16):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.tok = nn.Embedding(NVOCAB, d_model, padding_idx=PAD)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList(Block(d_model, n_head) for _ in range(n_layer))
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, NVOCAB)
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

    def _mask(self, t: int, device) -> torch.Tensor:
        return torch.triu(torch.full((t, t), float("-inf"), device=device), diagonal=1)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (B, T) -> logits (B, T, NVOCAB)."""
        b, t = ids.shape
        pos = torch.arange(t, device=ids.device).unsqueeze(0).expand(b, t)
        x = self.tok(ids) + self.pos(pos)
        mask = self._mask(t, ids.device)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln_f(x))

    def _cot_logprobs(self, prompt: torch.Tensor, cot: torch.Tensor) -> torch.Tensor:
        """Per-position logprob (B, C) of each CoT token under teacher forcing."""
        x = torch.cat([prompt, cot[:, :-1]], dim=1)
        logits = self.forward(x)
        p = prompt.shape[1]
        preds = logits[:, p - 1 : p - 1 + cot.shape[1], :]
        logp = F.log_softmax(preds, dim=-1)
        return logp.gather(-1, cot.unsqueeze(-1)).squeeze(-1)

    def sft_loss(self, prompt: torch.Tensor, cot: torch.Tensor) -> torch.Tensor:
        lp = self._cot_logprobs(prompt, cot)
        return -lp.mean()

    def seq_logprob(self, prompt: torch.Tensor, cot: torch.Tensor) -> torch.Tensor:
        """Sum logprob of the CoT given the prompt, shape (B,)."""
        return self._cot_logprobs(prompt, cot).sum(dim=1)

    @torch.no_grad()
    def generate_cot(self, prompt: torch.Tensor, n_tokens: int = 5,
                     greedy: bool = True, temperature: float = 1.0) -> torch.Tensor:
        self.eval()
        x = prompt.clone()
        for _ in range(n_tokens):
            logits = self.forward(x)
            nxt = logits[:, -1, :]
            if greedy:
                tok = nxt.argmax(dim=-1, keepdim=True)
            else:
                tok = torch.multinomial(
                    F.softmax(nxt / max(temperature, 1e-5), dim=-1), 1)
            x = torch.cat([x, tok], dim=1)
        return x[:, prompt.shape[1] :]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
