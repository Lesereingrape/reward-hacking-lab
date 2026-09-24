"""Policy pre-training and oracle-verifier evaluation for the tiny transformer.

Plain PyTorch, CPU only. ``fit`` trains next-token cross-entropy on gold chains;
``evaluate_answer`` samples the model's *own* greedy chain and checks the
reconstructed sum against ``a + b`` with the exact oracle verifier, so reported
accuracy is real generated arithmetic, never teacher forcing.
"""

from __future__ import annotations

import random

import torch

from .data import Example, oracle_correct, sample_examples
from .model import TinyTransformer


def new_model(seed: int, **kw) -> TinyTransformer:
    torch.manual_seed(seed)
    return TinyTransformer(**kw)


def tensors(examples: list[Example]) -> tuple[torch.Tensor, torch.Tensor]:
    prompts = torch.tensor([e.prompt() for e in examples], dtype=torch.long)
    cots = torch.tensor([e.cot() for e in examples], dtype=torch.long)
    return prompts, cots


def tensors_pairs(pairs: list[tuple[Example, list[int]]]):
    prompts = torch.tensor([ex.prompt() for ex, _ in pairs], dtype=torch.long)
    cots = torch.tensor([cot for _, cot in pairs], dtype=torch.long)
    return prompts, cots


def fit_pairs(model: TinyTransformer, pairs: list[tuple[Example, list[int]]], *,
              steps: int, batch: int, lr: float, seed: int) -> None:
    """Fine-tune on arbitrary (Example, chain) supervision — gold or self-generated."""
    if not pairs:
        return
    torch.manual_seed(seed)
    prompts, cots = tensors_pairs(pairs)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    n = prompts.shape[0]
    for _ in range(steps):
        idx = torch.randint(0, n, (min(batch, n),))
        loss = model.sft_loss(prompts[idx], cots[idx])
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def fit(model: TinyTransformer, examples: list[Example], *, steps: int,
        batch: int, lr: float, seed: int) -> None:
    fit_pairs(model, [(ex, ex.cot()) for ex in examples],
              steps=steps, batch=batch, lr=lr, seed=seed)


@torch.no_grad()
def evaluate_answer(model: TinyTransformer, examples: list[Example], *,
                    batch: int = 256, greedy: bool = True) -> float:
    model.eval()
    prompts, _ = tensors(examples)
    n = prompts.shape[0]
    ok = 0
    for start in range(0, n, batch):
        chunk = prompts[start : start + batch]
        cots = model.generate_cot(chunk, n_tokens=COT, greedy=greedy).tolist()
        for ex, cot in zip(examples[start : start + batch], cots, strict=True):
            ok += int(oracle_correct(ex, cot))
    return ok / n


COT = 5


def make_dataset(n_seed: int, n_eval: int, seed: int, lo: int = 10, hi: int = 99):
    rng = random.Random(seed)
    seedset = sample_examples(n_seed, rng, lo, hi)
    evalset = sample_examples(n_eval, random.Random(seed + 10_000), lo, hi)
    return seedset, evalset
