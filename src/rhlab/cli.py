"""Tiny CLI: ``rhlab demo`` runs one seed of the published study live.

    rhlab demo                 # the six rounds the README table has
    rhlab demo --rounds 3      # a shorter look, clearly labelled as shorter

The round count defaults to ``rhlab.study.ROUNDS`` — the value
``results/hacking.json`` records — because a demo that stops halfway prints a curve
whose collapse has not happened yet, in front of a table that describes the full
trajectory.
"""

from __future__ import annotations

import argparse

from rhlab.study import ROUNDS, VARIANTS, run_seed


def demo(rounds: int, seed: int) -> None:
    rec = run_seed(seed, rounds=rounds)
    label = "" if rounds == ROUNDS else f"  (truncated: {rounds} of {ROUNDS} rounds)"
    print(f"seed {seed}{label}")
    print(f"{'arm':9s} {'rm_hold':>7s}   true_acc[/proxy] per round")
    for name in VARIANTS:
        data = rec[name]
        hold = data["rm_holdout_acc"]
        hold_s = f"{hold:.2f}" if hold is not None else "--"
        line = "  ".join(
            f"{c['true_acc']:.2f}"
            + (f"/{c['proxy']:.2f}" if c.get("proxy") is not None else "")
            for c in data["curve"]
        )
        print(f"{name:9s} {hold_s:>7s}   {line}")


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="rhlab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="single-seed run of the published reward-hacking curve")
    d.add_argument("--rounds", type=int, default=ROUNDS)
    d.add_argument("--seed", type=int, default=0)
    return ap


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.cmd == "demo":
        demo(args.rounds, args.seed)


if __name__ == "__main__":
    main()
