"""Column addition as a fully-verifiable task, plus labelled chains for a reward model.

The model sees two 2-digit operands and emits the digit-by-digit column chain,
right-to-left, interleaving each result digit with the carry it makes::

    prompt:  a1 a0 + b1 b0 =
    chain :  o0 c1 o1 c2 o2

with ``a0+b0 = o0 + 10*c1``, ``a1+b1+c1 = o1 + 10*c2`` and ``o2 = c2``. Reading
``o2 o1 o0`` reconstructs the answer, checked exactly against ``a + b``.

That exact check is the *oracle* reward. The whole study is about what happens
when you optimise against an *imperfect learned* reward model that tries to
predict the oracle from data instead of computing it — Goodhart / reward-model
over-optimization. So ``make_labelled`` produces (chain, oracle_correctness)
training pairs for the reward model from plausible good and bad chains.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

PAD, BOS, EOS, PLUS, EQ = 0, 1, 2, 3, 4
DIGIT0 = 5
VOCAB = ["<pad>", "<bos>", "<eos>", "+", "="] + [str(d) for d in range(10)]
NVOCAB = len(VOCAB)
COT_LEN = 5


def digit_token(d: int) -> int:
    return DIGIT0 + d


def token_to_digit(tok: int) -> int:
    return tok - DIGIT0


@dataclass(frozen=True)
class Example:
    a: int
    b: int

    @property
    def target(self) -> int:
        return self.a + self.b

    def prompt(self) -> list[int]:
        a1, a0 = divmod(self.a, 10)
        b1, b0 = divmod(self.b, 10)
        return [digit_token(a1), digit_token(a0), PLUS,
                digit_token(b1), digit_token(b0), EQ]

    def cot(self) -> list[int]:
        a1, a0 = divmod(self.a, 10)
        b1, b0 = divmod(self.b, 10)
        s0 = a0 + b0
        o0, c1 = s0 % 10, s0 // 10
        s1 = a1 + b1 + c1
        o1, c2 = s1 % 10, s1 // 10
        return [digit_token(x) for x in (o0, c1, o1, c2, c2)]


def sample_examples(n: int, rng: random.Random, lo: int = 10, hi: int = 99) -> list[Example]:
    hi = min(hi, 99)
    lo = max(lo, 10)
    return [Example(a=rng.randint(lo, hi), b=rng.randint(lo, hi)) for _ in range(n)]


def parse_answer(cot_tokens: list[int]) -> int | None:
    if len(cot_tokens) < COT_LEN:
        return None
    if any(not (DIGIT0 <= t <= DIGIT0 + 9) for t in cot_tokens[:COT_LEN]):
        return None
    o0, _c1, o1, _c2, o2 = (token_to_digit(t) for t in cot_tokens[:COT_LEN])
    return 100 * o2 + 10 * o1 + o0


def oracle_correct(ex: Example, cot_tokens: list[int]) -> bool:
    """The ground-truth reward: does the reconstructed answer equal a+b?"""
    ans = parse_answer(cot_tokens)
    return ans is not None and ans == ex.target


def _bump(tok: int, delta: int) -> int:
    return digit_token((token_to_digit(tok) + delta) % 10)


def make_wrong(ex: Example, rng: random.Random) -> list[int]:
    """A plausible incorrect chain of valid shape (a dropped carry or a slipped digit)."""
    gold = ex.cot()
    a1, a0 = divmod(ex.a, 10)
    b1, b0 = divmod(ex.b, 10)

    def no_carry() -> list[int]:
        o0 = (a0 + b0) % 10
        c1 = (a0 + b0) // 10
        o1 = (a1 + b1) % 10
        return [digit_token(x) for x in (o0, c1, o1, 0, 0)]

    def drop_final() -> list[int]:
        return [gold[0], gold[1], gold[2], gold[3], digit_token(0)]

    def slip() -> list[int]:
        bad = list(gold)
        bad[0] = _bump(bad[0], rng.choice((-1, 1)))
        return bad

    for cand in (rng.choice([no_carry, drop_final, slip])(), slip()):
        if not oracle_correct(ex, cand):
            return cand
    return slip()


def make_labelled(examples: list[Example], rng: random.Random) -> list[tuple[list[int], int]]:
    """(full_ids, label) pairs teaching the reward model the oracle on gold vs wrong chains."""
    rows: list[tuple[list[int], int]] = []
    for ex in examples:
        rows.append((ex.prompt() + ex.cot(), 1))
        rows.append((ex.prompt() + make_wrong(ex, rng), 0))
    rng.shuffle(rows)
    return rows
