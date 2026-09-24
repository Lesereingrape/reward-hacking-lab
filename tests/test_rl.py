from __future__ import annotations

import random

import torch

from rhlab.data import Example, oracle_correct, sample_examples
from rhlab.model import TinyTransformer
from rhlab.rl import (
    collect_labeled,
    oracle_scores,
    rft_round,
    sample_candidates,
    select,
    selection_precision,
)


def test_oracle_selector_keeps_a_correct_chain_when_present():
    exs = [Example(47, 58), Example(12, 34)]  # 105, 46
    # hand-build candidates: for prompt 0 include the gold chain, for prompt 1 not
    gold0 = exs[0].cot()
    wrong1 = [t + 0 for t in exs[1].cot()]
    wrong1[0] = (wrong1[0] + 1)  # break the units digit
    cand = torch.tensor([[gold0, gold0], [wrong1, wrong1]], dtype=torch.long)
    scores = oracle_scores(exs, cand)
    assert scores[0].max() == 1.0 and scores[1].max() == 0.0
    pairs = select(exs, cand, scores)
    assert oracle_correct(exs[0], pairs[0][1])


def test_collect_labeled_shapes_and_labels():
    torch.manual_seed(0)
    model = TinyTransformer()
    rng = random.Random(0)
    rows = collect_labeled(model, sample_examples(10, rng), k=3, temperature=1.0)
    assert len(rows) == 30
    assert all(len(ids) == 11 and lab in (0, 1) for ids, lab in rows)


def test_rft_round_runs_and_bounds_precision():
    torch.manual_seed(0)
    model = TinyTransformer()
    rng = random.Random(1)
    train = sample_examples(24, rng)
    diag = rft_round(model, train, None, n=4, temperature=1.0,
                     steps=5, batch=16, lr=1e-3, seed=0)
    assert 0.0 <= diag["sel_precision"] <= 1.0
    assert diag["n_kept"] == len(train)


def test_sample_candidates_shape():
    torch.manual_seed(0)
    model = TinyTransformer()
    prompts = torch.tensor([e.prompt() for e in sample_examples(5, random.Random(0))])
    cand = sample_candidates(model, prompts, n=6, temperature=1.0)
    assert cand.shape == (5, 6, 5)


def test_selection_precision_exact():
    exs = [Example(47, 58), Example(12, 34)]
    pairs = [(exs[0], exs[0].cot()), (exs[1], exs[1].cot())]
    assert selection_precision(exs, pairs) == 1.0
