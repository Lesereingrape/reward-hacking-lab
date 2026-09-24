"""Rejection-sampling fine-tuning with a pluggable reward: oracle vs learned RM.

One round of self-improvement: sample N candidate chains per prompt, keep the one
the *selector* scores highest, then fine-tune the policy on those kept chains. If
the selector is the exact verifier the kept chains are truly correct and the policy
improves (the STaR/RFT regime). If the selector is an imperfect learned reward
model, the policy starts keeping chains that merely *score* well — some of them
wrong — and true accuracy stalls or falls while the reward model's own score keeps
climbing. That divergence is the reward-hacking curve this lab measures.
"""

from __future__ import annotations

import torch

from .data import Example, oracle_correct
from .model import TinyTransformer
from .reward import RewardModel
from .train import fit_pairs


@torch.no_grad()
def sample_candidates(model: TinyTransformer, prompts: torch.Tensor, n: int,
                      temperature: float) -> torch.Tensor:
    """(P, n, COT) chains sampled independently per prompt."""
    reps = prompts.repeat_interleave(n, dim=0)
    cots = model.generate_cot(reps, n_tokens=5, greedy=False, temperature=temperature)
    return cots.reshape(prompts.shape[0], n, -1)


def oracle_scores(examples: list[Example], cand: torch.Tensor) -> torch.Tensor:
    """(P, n) 1.0 where the sampled chain reconstructs a+b, else 0.0."""
    out = torch.zeros_like(cand[:, :, 0], dtype=torch.float)
    cl = cand.tolist()
    for p, ex in enumerate(examples):
        for k in range(cand.shape[1]):
            out[p, k] = 1.0 if oracle_correct(ex, cl[p][k]) else 0.0
    return out


def rm_scores(rm: RewardModel, examples: list[Example], cand: torch.Tensor) -> torch.Tensor:
    """(P, n) reward-model probability for each sampled chain."""
    n = cand.shape[1]
    flat_ids: list[list[int]] = []
    for p, ex in enumerate(examples):
        for k in range(n):
            flat_ids.append(ex.prompt() + cand[p, k].tolist())
    scores = rm.score(torch.tensor(flat_ids, dtype=torch.long)).view(len(examples), n)
    return scores


def select(examples: list[Example], cand: torch.Tensor, scores: torch.Tensor
           ) -> list[tuple[Example, list[int]]]:
    """Keep the argmax-scored chain per prompt (ties -> first, which may be wrong)."""
    best = scores.argmax(dim=1).tolist()
    cl = cand.tolist()
    return [(examples[p], cl[p][best[p]]) for p in range(len(examples))]


@torch.no_grad()
def proxy_of_policy(rm: RewardModel, model: TinyTransformer, examples: list[Example],
                    batch: int = 256) -> float:
    """Mean reward-model score on the policy's own greedy chains (the proxy metric)."""
    from .train import tensors
    prompts, _ = tensors(examples)
    out = 0.0
    cnt = 0
    for start in range(0, prompts.shape[0], batch):
        chunk = prompts[start : start + batch]
        cots = model.generate_cot(chunk, n_tokens=5, greedy=True)
        ids = torch.cat([chunk, cots], dim=1)
        out += float(rm.score(ids).sum())
        cnt += ids.shape[0]
    return out / max(cnt, 1)


def selection_precision(examples: list[Example], pairs: list[tuple[Example, list[int]]]
                        ) -> float:
    return sum(int(oracle_correct(ex, cot)) for ex, cot in pairs) / max(len(pairs), 1)


@torch.no_grad()
def collect_labeled(policy: TinyTransformer, examples: list[Example], *, k: int,
                    temperature: float) -> list[tuple[list[int], int]]:
    """(full_ids, oracle_label) rows drawn from the policy's OWN samples.

    This is the realistic reward-model training distribution: the RM learns to
    predict the verifier's judgement on the kinds of chains the selector will
    actually face, not on hand-crafted wrong answers. A small sample of these
    rows leaves a leaky proxy (reward hacking); a large sample tracks the oracle.
    """
    prompts = torch.tensor([e.prompt() for e in examples], dtype=torch.long)
    policy.eval()
    cand = sample_candidates(policy, prompts, k, temperature)
    cl = cand.tolist()
    rows: list[tuple[list[int], int]] = []
    for p, ex in enumerate(examples):
        for j in range(k):
            rows.append((ex.prompt() + cl[p][j], int(oracle_correct(ex, cl[p][j]))))
    return rows


def rft_round(policy: TinyTransformer, train_ex: list[Example], rm: RewardModel | None,
              *, n: int, temperature: float, steps: int, batch: int, lr: float,
              seed: int) -> dict:
    """One sample->select->fine-tune step; returns diagnostics (does not touch eval)."""
    prompts = torch.tensor([e.prompt() for e in train_ex], dtype=torch.long)
    policy.eval()
    cand = sample_candidates(policy, prompts, n, temperature)
    scores = rm_scores(rm, train_ex, cand) if rm is not None else oracle_scores(train_ex, cand)
    pairs = select(train_ex, cand, scores)
    fit_pairs(policy, pairs, steps=steps, batch=batch, lr=lr, seed=seed)
    return {"sel_precision": selection_precision(train_ex, pairs),
            "n_kept": len(pairs)}
