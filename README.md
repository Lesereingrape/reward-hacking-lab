# reward-hacking-lab — what happens when you optimise a learned reward instead of a verifiable one

**reward-hacking-lab** is a ~780-line, CPU-only study of *reward-model
over-optimisation* (Goodharting) inside an RLHF-style self-improvement loop. A small
transformer improves itself by rejection-sampling fine-tuning: each round it samples
best-of-N candidate chains, keeps the ones its reward ranks highest, and fine-tunes on
them. The only thing that varies between arms is the **reward source** — an exact
verifier vs a learned reward model — and the result is the textbook failure mode of
aligning to a proxy, measured end to end with real held-out accuracy.

![ci](https://github.com/Lesereingrape/reward-hacking-lab/actions/workflows/ci.yml/badge.svg)

## The setup

- **Policy:** a from-scratch decoder-only transformer (~103k params) emits the
  digit-by-digit column-addition carry chain for two 2-digit operands; the answer is
  reconstructed and checked by an **exact verifier**, so "accuracy" is real generated
  arithmetic, never teacher forcing.
- **Self-improvement:** best-of-N rejection sampling. Sample N=8 chains per prompt,
  select the top one by the reward, SFT on the selected chains, repeat for 6 rounds.
- **Reward sources** (the arm that changes):
  - **Oracle verifier** — the true correctness signal (a control).
  - **Learned RM** — a mean-pooled classifier trained on policy-sampled chains
    labelled by the oracle. Imperfect, exactly like a real reward model.
- **Metrics, always together:** held-out **true accuracy** (the territory) and the
  **proxy score** the reward assigns (the map), plus selection precision.

## The finding

When you optimise the **verifiable** reward, true accuracy climbs. When you optimise a
**learned** reward, the proxy climbs *while true accuracy collapses* — the policy
discovers chains the reward model likes that are simply wrong. Scaling the reward
model's training data sharpens its agreement with the oracle yet still does not stop
the collapse, because best-of-N selection is an adversarial argmax over the proxy. On
this toy the lever that works is a *verifiable* reward, not a bigger learned one.

## Quickstart

```bash
pip install -e .                    # torch is the only runtime dependency
python -m rhlab demo                # one seed of the published six-round curve
python experiments/run_study.py     # full 3-seed study -> results/hacking.json
python experiments/make_report.py --write   # splice the block into README.md
```

## Results

<!-- RESULTS:START -->
*Every figure below is produced by `experiments/run_study.py` on CPU and committed as [`results/hacking.json`](results/hacking.json); the tables are rendered by `experiments/make_report.py`. Same policy, same best-of-N candidate budget (N=8), same fine-tuning schedule; only the *reward source* differs. Mean over 3 seeds.*

- policy: **103,055** parameters (decoder-only transformer, CPU-only)
- shared SFT base, graded by the exact verifier: **0.609**
- measured under: Python 3.13.7 on Windows-11-10.0.26200-SP0, torch 2.14.0+cpu, 8 CPU threads, cpu (a rerun is bit-exact only inside this environment)

### Final held-out accuracy vs the reward you optimise

| reward source | RM agreement w/ oracle | final true acc | final proxy score | selection precision |
|---|---:|---:|---:|---:|
| Oracle verifier | — | 0.825 | — | 0.927 |
| Learned RM (150 rows) | 0.560 | 0.139 | 0.751 | 0.155 |
| Learned RM (2,500 rows) | 0.647 | 0.154 | 0.621 | 0.171 |

The oracle-verifier arm **climbs from 0.609 to 0.825** in true accuracy over the same six rounds — this is the verified-reward self-improvement result. The two learned-reward arms go the other way: rm_small ends at 0.139 and rm_large at 0.154, even though both proxy scores climb between the first and last round (rm_small 0.650 -> 0.751, rm_large 0.550 -> 0.621), and the fraction of kept chains that are actually correct drops to 0.155/0.171. A proxy the selector is getting better at satisfying, next to a true metric it is not, *is* reward hacking.

### The Goodhart divergence (learned RM, small budget)

| round | true_acc | proxy | sel_precision |
|------:|-------:|-------:|-------:|
| 1 | 0.303 | 0.650 | 0.296 |
| 2 | 0.191 | 0.705 | 0.203 |
| 3 | 0.171 | 0.719 | 0.173 |
| 4 | 0.147 | 0.728 | 0.163 |
| 5 | 0.137 | 0.751 | 0.158 |
| 6 | 0.139 | 0.751 | 0.155 |

Across the rounds the reward model's own score for the policy climbs (proxy 0.650 -> 0.751) while the verifier's verdict on the same policy falls (true accuracy 0.303 -> 0.139). The policy learns to produce chains the RM scores highly that are not correct — optimising the map, not the territory.

### Does the collapse need an unlucky seed?

| arm | final true acc, seed by seed | seeds below their own start |
|-----|-----------------------------:|----------------------------:|
| Oracle verifier | seed 0: 0.786, seed 1: 0.720, seed 2: 0.970 | 1/3 |
| Learned RM (150 rows) | seed 0: 0.082, seed 1: 0.086, seed 2: 0.248 | 3/3 |
| Learned RM (2,500 rows) | seed 0: 0.102, seed 1: 0.118, seed 2: 0.242 | 3/3 |

Every seed's small-budget arm ends below where that seed started, so over-optimisation is what the setup does, not what one badly-drawn reward model happens to do. The oracle arm's seeds are listed in the same table for the same reason: a control is only a control if it holds seed by seed.

### Verified-reward control (oracle)

| round | true_acc | sel_precision |
|------:|-------:|-------:|
| 1 | 0.727 | 0.892 |
| 2 | 0.774 | 0.926 |
| 3 | 0.775 | 0.928 |
| 4 | 0.793 | 0.928 |
| 5 | 0.803 | 0.926 |
| 6 | 0.825 | 0.927 |

### Does a better reward model save you?

| RM labelled rows | RM agreement w/ oracle | final true acc |
|---:|---:|---:|
| 150 | 0.560 | 0.139 |
| 2500 | 0.647 | 0.154 |

The reward model's agreement with the oracle climbs from 0.560 to 0.647 as labelled rows go up 17x, yet held-out accuracy ended at 0.139 (small) and 0.154 (large) against the 0.609 base, so a better proxy softened the failure without removing it. At this scale the argument is that best-of-N selection is an adversarial argmax over the proxy, so it hunts out whatever mistakes the reward model still has; the question of how many remain is answered by the numbers, not by the story.

### Honest limitations

- Deliberately toy. A real RLHF reward model is far stronger than a mean-pooled ~100k-param classifier, so the absolute size of the collapse is exaggerated; the *direction* — proxy up while the true metric goes the other way under an imperfect reward, and the verified reward climbs in the same schedule — is the reproducible point.
- The oracle arm largely re-demonstrates verified self-improvement (see the companion starlab); its role here is the control that makes the proxy-vs-true divergence legible.
- Best-of-N (an offline selection method), not a full PPO loop: this isolates over-optimization of a proxy reward from the noise of an on-policy gradient.
<!-- RESULTS:END -->

## Reproducing and honesty

The `Results` block is generated, not typed: `tests/test_readme_matches_results.py`
asserts the README equals `make_report.build(results/hacking.json)`, so no number can
drift from the committed artifact, and `tests/test_readme_size_claims.py` checks the
hand-written "~780-line" claim above against `src/`.
`tests/test_demo_matches_published_study.py` pins `python -m rhlab demo` to the same
six-round budget the table reports, so the command a reader tries first cannot be a
smaller, differently-tuned experiment than the numbers printed above it.
`tests/test_artifact_is_internally_consistent.py` recomputes every published mean and
final value from the per-seed curves stored in the JSON, and the block itself states
seed by seed whether the collapse is universal or one unlucky reward model — because a
3/3 result and a 2/3 result deserve different sentences.

A rerun is bit-exact only inside the environment the artifact records (Python, torch
build, CPU thread count); float reduction order over a batch follows the thread count,
so elsewhere expect the same shape rather than the same digits. To check the claim
inside that environment, write a second run somewhere disposable and diff it field for
field:

```bash
python experiments/run_study.py --out /tmp/again.json
```

We ran that diff: the scratch rerun came back with exactly one differing field in the
whole artifact, `runtime_sec` (697.3s against the published 874.6s), while the three
arms' curves, their per-seed accuracy traces and the environment block were identical.

Std-devs and per-seed spreads are across the 3 seeded runs, which is 3 runs; where a
gap between two arms sits inside that spread the text says so instead of selling it.

## License

MIT
