"""Tiny CLI: ``rhlab demo`` runs one short seed live and prints the hacking curve."""

from __future__ import annotations

import argparse

from rhlab.study import run_seed


def demo(rounds: int, seed: int) -> None:
    rec = run_seed(seed, rounds=rounds)
    print(f"{'arm':9s} {'rm_hold':>7s}   true_acc[/proxy] per round")
    for name, data in rec.items():
        hold = data["rm_holdout_acc"]
        hold_s = f"{hold:.2f}" if hold is not None else "--"
        line = "  ".join(
            f"{c['true_acc']:.2f}"
            + (f"/{c['proxy']:.2f}" if c.get("proxy") is not None else "")
            for c in data["curve"]
        )
        print(f"{name:9s} {hold_s:>7s}   {line}")


def main() -> None:
    ap = argparse.ArgumentParser(prog="rhlab")
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("demo", help="quick single-seed reward-hacking curve")
    d.add_argument("--rounds", type=int, default=3)
    d.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.cmd == "demo":
        demo(args.rounds, args.seed)


if __name__ == "__main__":
    main()
