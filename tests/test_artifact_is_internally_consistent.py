"""The committed artifact must reconcile with itself, not only with the README.

``variants[arm]["curve"]`` publishes a mean over seeds; ``true_acc_per_seed`` is the
material that mean is made of. Checking one against the other is what catches a
results file where the headline curve and the seed rows came from different runs —
and it is also the only mechanical way to answer "does every seed reward-hack, or did
one reward model get drawn badly?".
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("oracle", "rm_small", "rm_large")


def _data() -> dict:
    return json.loads((ROOT / "results" / "hacking.json").read_text(encoding="utf-8"))


def test_published_curves_are_the_mean_of_their_seeds():
    data = _data()
    assert len(data["config"]["seeds"]) == len(data["variants"]["oracle"]["true_acc_per_seed"])
    for arm in ARMS:
        v = data["variants"][arm]
        per = v["true_acc_per_seed"]
        for i, row in enumerate(v["curve"]):
            across = statistics.fmean(vals[i] for vals in per.values())
            assert abs(across - row["true_acc"]) < 2e-4, f"{arm} round {row['round']}"
            std = statistics.pstdev(vals[i] for vals in per.values())
            assert abs(std - row["true_acc_std"]) < 2e-4, f"{arm} round {row['round']} std"
        finals = statistics.fmean(vals[-1] for vals in per.values())
        assert abs(finals - v["final_true_acc"]) < 2e-4, arm
        # peak is a max over the same curve, so it can never sit below the endpoint.
        assert v["peak_true_acc"] >= v["final_true_acc"] - 2e-4, arm


def test_only_the_learned_reward_arms_have_a_holdout_number():
    data = _data()
    assert data["variants"]["oracle"]["rm_holdout_acc"] is None
    for arm in ("rm_small", "rm_large"):
        hold = data["variants"][arm]["rm_holdout_acc"]
        assert hold is not None and 0.0 <= hold <= 1.0, arm


def test_the_run_records_the_environment_it_is_bit_exact_under():
    env = _data()["environment"]
    assert env["device"] == "cpu"
    assert env["threads"] >= 1
    assert env["torch"].startswith("2.")
    assert env["python"] and env["platform"]
