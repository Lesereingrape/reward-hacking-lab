"""Render the README results block directly from results/hacking.json.

The README numbers are mechanically tied to the committed artifact: run
``python experiments/run_study.py`` then ``python experiments/make_report.py`` and
paste the output between the RESULTS markers. A test asserts the README already
equals this, so nothing is hand-copied.
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
    out.append(
        f"The oracle-verifier arm **climbs** from {orc['base_acc']:.3f} to "
        f"{orc['final_true_acc']:.3f} true accuracy — this is the verified-reward "
        "self-improvement result. The two learned-reward arms do the opposite: they "
        f"**fall** to {sm['final_true_acc']:.3f} and {lg['final_true_acc']:.3f} while "
        "their proxy score *rises*, and the fraction of kept chains that are actually "
        f"correct drops to {sm['final_sel_precision']:.3f}/{lg['final_sel_precision']:.3f}. "
        "That gap between a climbing proxy and a collapsing true metric *is* reward hacking."
    )
    out.append("")

    out.append("### The Goodhart divergence (learned RM, small budget)\n")
    out.append(_curve(data, "rm_small", ["true_acc", "proxy", "sel_precision"]))
    out.append("")
    out.append(
        "Round after round the reward model is more pleased with the policy "
        f"(proxy {sm['curve'][0]['proxy']:.3f} -> {sm['final_proxy']:.3f}) while the "
        f"verifier disagrees (true accuracy {sm['curve'][0]['true_acc']:.3f} -> "
        f"{sm['final_true_acc']:.3f}). The policy learns to produce chains the RM "
        "scores highly that are not correct — optimising the map, not the territory."
    )
    out.append("")

    out.append("### Verified-reward control (oracle)\n")
    out.append(_curve(data, "oracle", ["true_acc", "sel_precision"]))
    out.append("")

    out.append("### More reward-model data sharpens the proxy but does not save you\n")
    out.append("| RM labelled rows | RM agreement w/ oracle | final true acc |")
    out.append("|---:|---:|---:|")
    for arm in ("rm_small", "rm_large"):
        out.append(f"| {cfg['rm_specs'][arm]['rows']} | {v[arm]['rm_holdout_acc']:.3f} | "
                   f"{v[arm]['final_true_acc']:.3f} |")
    out.append("")
    ratio = cfg["rm_specs"]["rm_large"]["rows"] / cfg["rm_specs"]["rm_small"]["rows"]
    out.append(
        f"{ratio:.0f}x more reward-model data lifted agreement with the oracle from "
        f"{sm['rm_holdout_acc']:.3f} to {lg['rm_holdout_acc']:.3f}, yet held-out "
        f"accuracy still collapsed ({sm['final_true_acc']:.3f} -> "
        f"{lg['final_true_acc']:.3f}). At this scale the failure is *structural*: "
        "best-of-N selection is an adversarial argmax over the proxy, so it hunts out "
        "the reward model's mistakes no matter how many rows it saw. The lever that "
        "works here is a *verifiable* reward, not a bigger learned one."
    )
    out.append("")

    out.append("### Honest limitations\n")
    out.append("- Deliberately toy. A real RLHF reward model is far stronger than a "
               "mean-pooled ~100k-param classifier, so the absolute collapse is "
               "exaggerated; the *direction* (proxy up, true down under imperfect "
               "rewards; verified rewards keep climbing) is the reproducible point.")
    out.append("- The oracle arm largely re-demonstrates verified self-improvement "
               "(see the companion starlab); its role here is the control that makes "
               "the proxy-vs-true divergence legible.")
    out.append("- Best-of-N (an offline selection method), not a full PPO loop: this "
               "isolates over-optimization of a proxy reward from the noise of an "
               "on-policy gradient.")
    return "\n".join(out)


if __name__ == "__main__":
    print(build(json.loads(Path("results/hacking.json").read_text(encoding="utf-8"))))
