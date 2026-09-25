"""Reusable study logic for the reward-hacking comparison.

Lives in the package (not just the experiment script) so ``rhlab demo`` and
``experiments/run_study.py`` share exactly one implementation of the loop. See the
module docstring of ``experiments/run_study.py`` for the experimental design.

The reward model is trained on the *policy's own sampled chains labelled by the
exact verifier* (an RLAIF-style proxy). Budget is the number of such labelled rows:
a small budget leaves a leaky proxy that the policy exploits, a large budget tracks
the oracle far better — which is what makes the two RM arms diverge in reward-hacking
severity while sharing the same selection algorithm.
"""

from __future__ import annotations

import platform
import random
import sys

import torch

from rhlab.data import sample_examples
from rhlab.model import TinyTransformer, count_parameters
from rhlab.reward import fit_rm, new_rm, rm_accuracy
from rhlab.rl import collect_labeled, proxy_of_policy, rft_round
from rhlab.train import evaluate_answer, fit, new_model

SEEDS = (0, 1, 2)
ROUNDS = 6
N_TRAIN = 600
N_EVAL = 500
N_CAND = 8
TEMP = 1.0
PRETRAIN_STEPS = 200
RFT_STEPS = 250
BATCH = 128
LR = 1e-3
RM_LABEL_K = 5          # sampled chains per prompt when building RM rows
RM_POOL_N = 400         # prompts used to source RM rows
RM_HOLDOUT_N = 300
# per-arm reward-model spec: labelled rows + capacity + optimisation steps
RM_SPECS = {
    "rm_small": {"rows": 150, "d_model": 64, "n_layer": 2, "steps": 500},
    "rm_large": {"rows": 2500, "d_model": 64, "n_layer": 2, "steps": 900},
}
VARIANTS = ("oracle", "rm_small", "rm_large")

POLICY_PARAMS = count_parameters(TinyTransformer())


def _base_policy(seed: int):
    rng = random.Random(seed)
    train_ex = sample_examples(N_TRAIN, rng)
    eval_ex = sample_examples(N_EVAL, random.Random(seed + 1000))
    model = new_model(seed)
    fit(model, train_ex, steps=PRETRAIN_STEPS, batch=BATCH, lr=2e-3, seed=seed)
    return model, train_ex, eval_ex


def _build_rm(base, seed: int, spec: dict):
    pool = sample_examples(RM_POOL_N, random.Random(seed + 77))
    rows = collect_labeled(base, pool, k=RM_LABEL_K, temperature=TEMP)
    # balance the sample distribution with gold positives (an oracle-labelled set)
    rows += [(ex.prompt() + ex.cot(), 1) for ex in pool]
    hold = sample_examples(RM_HOLDOUT_N, random.Random(seed + 88))
    hold_rows = collect_labeled(base, hold, k=RM_LABEL_K, temperature=TEMP)
    sampled = random.Random(seed).sample(rows, min(spec["rows"], len(rows)))
    rm = new_rm(seed, d_model=spec["d_model"], n_layer=spec["n_layer"])
    fit_rm(rm, sampled, steps=spec["steps"], batch=128, lr=2e-3, seed=seed)
    return rm, rm_accuracy(rm, hold_rows)


def _run_variant(kind: str, seed, base, train_ex, eval_ex, rm, rounds: int):
    policy = new_model(seed)
    policy.load_state_dict(base.state_dict())
    curve = []
    for _round in range(rounds):
        diag = rft_round(policy, train_ex, rm if kind == "rm" else None,
                         n=N_CAND, temperature=TEMP, steps=RFT_STEPS,
                         batch=BATCH, lr=LR, seed=seed + 1)
        curve.append({
            "true_acc": round(evaluate_answer(policy, eval_ex), 4),
            "proxy": round(proxy_of_policy(rm, policy, eval_ex), 4) if rm is not None else None,
            "sel_precision": round(diag["sel_precision"], 4),
        })
    return curve


def environment() -> dict:
    """The machine a run must be reproduced on to be bit-exact.

    Float reduction order over a batch follows the thread count and the torch build,
    so the artifact names its environment instead of claiming a reproducibility that
    only holds inside it.
    """
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "threads": torch.get_num_threads(),
        "device": "cpu",
    }


def run_seed(seed: int, rounds: int = ROUNDS):
    base, train_ex, eval_ex = _base_policy(seed)
    out = {
        "seed": seed,
        "oracle": {"base_acc": round(evaluate_answer(base, eval_ex), 4),
                   "rm_holdout_acc": None,
                   "curve": _run_variant("oracle", seed, base, train_ex, eval_ex, None, rounds)}
    }
    for name, spec in RM_SPECS.items():
        rm, hold = _build_rm(base, seed, spec)
        curve = _run_variant("rm", seed, base, train_ex, eval_ex, rm, rounds)
        out[name] = {"base_acc": out["oracle"]["base_acc"], "rm_holdout_acc": round(hold, 4),
                     "curve": curve}
    return out


def aggregate(per_seed, rounds: int = ROUNDS):
    agg = {}
    for v in VARIANTS:
        recs = [rec[v] for rec in per_seed]
        curves = [r["curve"] for r in recs]
        rmh = [r["rm_holdout_acc"] for r in recs if r["rm_holdout_acc"] is not None]
        curve_rows = []
        for r in range(rounds):
            snap = [c[r] for c in curves]
            accs = [s["true_acc"] for s in snap]
            m = sum(accs) / len(accs)
            std = (sum((a - m) ** 2 for a in accs) / len(accs)) ** 0.5
            proxies = [s["proxy"] for s in snap if s["proxy"] is not None]
            curve_rows.append({
                "round": r + 1,
                "true_acc": round(m, 4),
                "true_acc_std": round(std, 4),
                "proxy": round(sum(proxies) / len(proxies), 4) if proxies else None,
                "sel_precision": round(sum(s["sel_precision"] for s in snap) / len(snap), 4),
            })
        agg[v] = {
            "base_acc": round(sum(r["base_acc"] for r in recs) / len(recs), 4),
            "rm_holdout_acc": round(sum(rmh) / len(rmh), 4) if rmh else None,
            "curve": curve_rows,
            "final_true_acc": curve_rows[-1]["true_acc"],
            "peak_true_acc": round(max(c["true_acc"] for c in curve_rows), 4),
            "final_proxy": curve_rows[-1]["proxy"],
            "final_sel_precision": curve_rows[-1]["sel_precision"],
            # The raw per-seed trajectories behind the mean curve: a test recomputes
            # the published means from these, and the README says in words whether
            # every seed hacks or only some of them do.
            "true_acc_per_seed": {str(r["seed"]): [c["true_acc"] for c in r[v]["curve"]]
                                  for r in per_seed},
        }
    return agg


def build_results(per_seed, rounds: int = ROUNDS, runtime: float = 0.0) -> dict:
    return {
        "config": {
            "seeds": list(SEEDS),
            "rounds": rounds,
            "n_train": N_TRAIN,
            "n_eval": N_EVAL,
            "n_candidates": N_CAND,
            "temperature": TEMP,
            "pretrain_steps": PRETRAIN_STEPS,
            "rft_steps": RFT_STEPS,
            "rm_label_k": RM_LABEL_K,
            "rm_specs": RM_SPECS,
            "params_policy": POLICY_PARAMS,
        },
        "variants": aggregate(per_seed, rounds),
        "environment": environment(),
        "runtime_sec": round(runtime, 1),
    }
