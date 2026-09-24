from __future__ import annotations

import random

import torch

from rhlab.data import make_labelled, sample_examples
from rhlab.reward import fit_rm, new_rm, rm_accuracy


def test_rm_output_shapes_and_range():
    rm = new_rm(0, d_model=32, n_layer=1)
    rng = random.Random(0)
    rows = make_labelled(sample_examples(8, rng), rng)
    ids = torch.tensor([r[0] for r in rows], dtype=torch.long)
    logit = rm(ids)
    assert logit.shape == (ids.shape[0],)
    prob = rm.score(ids)
    assert torch.all((prob >= 0) & (prob <= 1))


def test_rm_learns_separable_signal():
    rng = random.Random(1)
    train_rows = make_labelled(sample_examples(120, rng), rng)
    test_rows = make_labelled(sample_examples(60, rng), rng)
    rm0 = new_rm(2, d_model=48, n_layer=2)
    base = rm_accuracy(rm0, test_rows)
    fit_rm(rm0, train_rows, steps=300, batch=64, lr=3e-3, seed=2)
    after = rm_accuracy(rm0, test_rows)
    assert after > base
    assert 0.0 <= after <= 1.0
