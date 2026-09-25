"""Render the README results block directly from results/hacking.json.

The README numbers are mechanically tied to the committed artifact: run
``python experiments/run_study.py`` then ``python experiments/make_report.py --write``
to splice the rendered block back between the RESULTS markers. A test asserts the
README already equals this, so nothing is hand-copied.
"""

from __future__ import annotations

import json
from pathlib import Path

ARM_ORDER = ("oracle", "rm_small", "rm_large")
BASE_LABEL = {"oracle": "Oracle verifier"}


def _label(cfg: dict, arm: str) -> str:
    """Label learned-RM arms with their *committed* row budget, never a hardcode."""
    if arm in BASE_LABEL:
        return BASE_LABEL[arm]
    return f"Learned RM ({cfg['rm_specs'][arm]['rows']:,} rows)"


def _curve(data: dict, arm: str, fields: list[str]) -> str:
    header = "| round | " + " | ".join(fields) + " |"
    sep = "|------:|" + "|".join(["-------:"] * len(fields)) + "|"
    lines = [header, sep]
    for row in data["variants"][arm]["curve"]:
        cells = []
        for f in fields:
            v = row[f]
            cells.append("—" if v is None else f"{v:.3f}")
        lines.append(f"| {row['round']} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _trend(a: dict, field: str, plural: bool = False) -> str:
    """'climbs' / 'falls' / 'holds' for one arm's field, round 1 to its last round."""
    return _verb(a["curve"][0][field], a["curve"][-1][field], plural)


def _verb(start: float, end: float, plural: bool = False) -> str:
    word = "climb" if end > start else "fall" if end < start else "hold"
    return word if plural else word + "s"


def _arc(start: float, end: float) -> str:
    return f"{_verb(start, end)} from {start:.3f} to {end:.3f}"


def build(data: dict) -> str:
    cfg = data["config"]
    v = data["variants"]
    orc, sm, lg = v["oracle"], v["rm_small"], v["rm_large"]
    out: list[str] = []

    out.append(
        "*Every figure below is produced by `experiments/run_study.py` on CPU and "
        "committed as [`results/hacking.json`](results/hacking.json); the tables are "
        "rendered by `experiments/make_report.py`. Same policy, same best-of-N "
        f"candidate budget (N={cfg['n_candidates']}), same fine-tuning schedule; only "
        f"the *reward source* differs. Mean over {len(cfg['seeds'])} seeds.*"
    )
    out.append("")
    out.append(f"- policy: **{cfg['params_policy']:,}** parameters "
               "(decoder-only transformer, CPU-only)")
    out.append(f"- shared SFT base, graded by the exact verifier: "
               f"**{orc['base_acc']:.3f}**")
    env = data["environment"]
    out.append(f"- measured under: Python {env['python']} on {env['platform']}, "
               f"torch {env['torch']}, {env['threads']} CPU threads, {env['device']} "
               "(a rerun is bit-exact only inside this environment)")
    out.append("")

    out.append("### Final held-out accuracy vs the reward you optimise\n")
    out.append("| reward source | RM agreement w/ oracle | final true acc | final proxy score | selection precision |")
    out.append("|---|---:|---:|---:|---:|")
    for arm in ARM_ORDER:
        a = v[arm]
        hold = "—" if a["rm_holdout_acc"] is None else f"{a['rm_holdout_acc']:.3f}"
        proxy = "—" if a["final_proxy"] is None else f"{a['final_proxy']:.3f}"
        out.append(f"| {_label(cfg, arm)} | {hold} | {a['final_true_acc']:.3f} | "
                   f"{proxy} | {a['final_sel_precision']:.3f} |")
    out.append("")
    if _trend(sm, "proxy") == _trend(lg, "proxy"):
        proxy_clause = (
            f"even though both proxy scores {_trend(sm, 'proxy', True)} between the "
            f"first and last round (rm_small {sm['curve'][0]['proxy']:.3f} -> "
            f"{sm['final_proxy']:.3f}, rm_large {lg['curve'][0]['proxy']:.3f} -> "
            f"{lg['final_proxy']:.3f})"
        )
    else:
        proxy_clause = (f"even though the proxy for rm_small {_trend(sm, 'proxy')} and "
                        f"for rm_large {_trend(lg, 'proxy')} across those rounds")
    out.append(
        f"The oracle-verifier arm **{_arc(orc['base_acc'], orc['final_true_acc'])}** in "
        "true accuracy over the same six rounds — this is the verified-reward "
        "self-improvement result. The two learned-reward arms go the other way: "
        f"rm_small ends at {sm['final_true_acc']:.3f} and rm_large at "
        f"{lg['final_true_acc']:.3f}, {proxy_clause}, and the fraction of kept "
        f"chains that are actually correct drops to "
        f"{sm['final_sel_precision']:.3f}/{lg['final_sel_precision']:.3f}. "
        "A proxy the selector is getting better at satisfying, next to a true metric it "
        "is not, *is* reward hacking."
    )
    out.append("")

    out.append("### The Goodhart divergence (learned RM, small budget)\n")
    out.append(_curve(data, "rm_small", ["true_acc", "proxy", "sel_precision"]))
    out.append("")
    out.append(
        f"Across the rounds the reward model's own score for the policy "
        f"{_trend(sm, 'proxy')} (proxy {sm['curve'][0]['proxy']:.3f} -> "
        f"{sm['final_proxy']:.3f}) while the verifier's verdict on the same policy "
        f"{_trend(sm, 'true_acc')} (true accuracy {sm['curve'][0]['true_acc']:.3f} -> "
        f"{sm['final_true_acc']:.3f}). The policy learns to produce chains the RM "
        "scores highly that are not correct — optimising the map, not the territory."
    )
    out.append("")

    out.append("### Does the collapse need an unlucky seed?\n")
    out.append("| arm | final true acc, seed by seed | seeds below their own start |")
    out.append("|-----|-----------------------------:|----------------------------:|")
    for arm in ARM_ORDER:
        per = v[arm]["true_acc_per_seed"]
        cells = ", ".join(f"seed {s}: {vals[-1]:.3f}"
                          for s, vals in sorted(per.items(), key=lambda kv: int(kv[0])))
        n_down = sum(1 for vals in per.values() if vals[-1] < vals[0])
        out.append(f"| {_label(cfg, arm)} | {cells} | {n_down}/{len(per)} |")
    out.append("")
    per_sm = v["rm_small"]["true_acc_per_seed"]
    n_down = sum(1 for vals in per_sm.values() if vals[-1] < vals[0])
    if n_down == len(per_sm):
        out.append(
            "Every seed's small-budget arm ends below where that seed started, so "
            "over-optimisation is what the setup does, not what one badly-drawn reward "
            "model happens to do. The oracle arm's seeds are listed in the same table "
            "for the same reason: a control is only a control if it holds seed by seed."
        )
    elif n_down == 0:
        out.append(
            "No seed's small-budget arm ends below its own start, so this study does "
            "*not* reproduce a collapse seed after seed — the mean curve above "
            "describes an effect that individual runs escape."
        )
    else:
        out.append(
            f"{n_down} of {len(per_sm)} seeds end below their own start, so the "
            "collapse is partly which reward model got drawn. The mean curve "
            "overstates how inevitable it is, and the honest reading is 'a learned "
            "reward can be exploited, here often enough to matter'."
        )
    out.append("")

    out.append("### Verified-reward control (oracle)\n")
    out.append(_curve(data, "oracle", ["true_acc", "sel_precision"]))
    out.append("")

    out.append("### Does a better reward model save you?\n")
    out.append("| RM labelled rows | RM agreement w/ oracle | final true acc |")
    out.append("|---:|---:|---:|")
    for arm in ("rm_small", "rm_large"):
        out.append(f"| {cfg['rm_specs'][arm]['rows']} | {v[arm]['rm_holdout_acc']:.3f} | "
                   f"{v[arm]['final_true_acc']:.3f} |")
    out.append("")
    ratio = cfg["rm_specs"]["rm_large"]["rows"] / cfg["rm_specs"]["rm_small"]["rows"]
    agreement = _arc(sm["rm_holdout_acc"], lg["rm_holdout_acc"])
    if lg["final_true_acc"] >= orc["base_acc"]:
        verdict = ("and the larger reward model holds true accuracy at or above the "
                   "base, so at this budget more proxy data does largely fix it")
    elif lg["final_true_acc"] > sm["final_true_acc"]:
        verdict = "so a better proxy softened the failure without removing it"
    else:
        verdict = "so a better proxy bought nothing at all"
    out.append(
        f"The reward model's agreement with the oracle {agreement} as labelled rows go "
        f"up {ratio:.0f}x, yet held-out "
        f"accuracy ended at {sm['final_true_acc']:.3f} (small) and "
        f"{lg['final_true_acc']:.3f} (large) against the {orc['base_acc']:.3f} base, "
        f"{verdict}. At this scale the argument is that best-of-N selection is an "
        "adversarial argmax over the proxy, so it hunts out whatever mistakes the "
        "reward model still has; the question of how many remain is answered by the "
        "numbers, not by the story."
    )
    out.append("")

    out.append("### Honest limitations\n")
    out.append("- Deliberately toy. A real RLHF reward model is far stronger than a "
               "mean-pooled ~100k-param classifier, so the absolute size of the "
               "collapse is exaggerated; the *direction* — proxy up while the true "
               "metric goes the other way under an imperfect reward, and the verified "
               f"reward {_trend(orc, 'true_acc')} in the same schedule — is the "
               "reproducible point.")
    out.append("- The oracle arm largely re-demonstrates verified self-improvement "
               "(see the companion starlab); its role here is the control that makes "
               "the proxy-vs-true divergence legible.")
    out.append("- Best-of-N (an offline selection method), not a full PPO loop: this "
               "isolates over-optimization of a proxy reward from the noise of an "
               "on-policy gradient.")
    return "\n".join(out)


def _write(path: Path, block: str) -> None:
    text = path.read_text(encoding="utf-8")
    start, end = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
    head, _, rest = text.partition(start)
    _, _, tail = rest.partition(end)
    nl = "\n"
    path.write_text(f"{head}{start}{nl}{block}{nl}{end}{tail}", encoding="utf-8")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(prog="make_report")
    ap.add_argument("--write", action="store_true",
                    help="splice the block into README.md instead of printing it")
    ap.add_argument("--results", default="results/hacking.json")
    args = ap.parse_args()
    rendered = build(json.loads(Path(args.results).read_text(encoding="utf-8")))
    if args.write:
        _write(Path("README.md"), rendered)
        print("README results block rewritten")
    else:
        print(rendered)
