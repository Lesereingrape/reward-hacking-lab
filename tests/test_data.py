from __future__ import annotations

import random

from rhlab.data import (
    COT_LEN,
    Example,
    make_labelled,
    make_wrong,
    oracle_correct,
    parse_answer,
    sample_examples,
)


def test_cot_reconstructs_sum():
    ex = Example(a=47, b=58)
    assert ex.target == 105
    assert parse_answer(ex.cot()) == 105
    assert oracle_correct(ex, ex.cot())


def test_wrong_chain_is_not_correct():
    rng = random.Random(0)
    for ex in sample_examples(100, rng):
        assert not oracle_correct(ex, make_wrong(ex, rng))


def test_labelled_rows_balanced_and_shaped():
    rng = random.Random(1)
    rows = make_labelled(sample_examples(20, rng), rng)
    labels = {r[1] for r in rows}
    assert labels == {0, 1}
    for ids, lab in rows:
        assert len(ids) == 6 + COT_LEN
        assert lab in (0, 1)


def test_sample_range():
    rng = random.Random(2)
    for ex in sample_examples(50, rng, 10, 99):
        assert 10 <= ex.a <= 99 and 10 <= ex.b <= 99
