# reward-hacking-lab — what happens when you optimise a learned reward instead of a verifiable one

**reward-hacking-lab** is a ~400-line, CPU-only study of *reward-model
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
python -m rhlab demo                # one seeded run, seconds
python experiments/run_study.py     # full 3-seed study -> results/hacking.json
python experiments/make_report.py   # re-render the README block from the JSON
```

## Results

<!-- RESULTS:START -->
*Every figure below is produced by `experiments/run_study.py` on CPU and committed as [`results/hacking.json`](results/hacking.json); the tables are rendered by `experiments/make_report.py`. Same policy, same best-of-N candidate budget (N=8), same fine-tuning schedule; only the *reward source* differs. Mean over 3 seeds.*

- policy: **103,055** parameters (decoder-only transformer, CPU-only)
- shared SFT base, graded by the exact verifier: **0.613**

### Final held-out accuracy vs the reward you optimise

| reward source | RM agreement w/ oracle | final true acc | final proxy score | selection precision |
|---|---:|---:|---:|---:|
| Oracle verifier | — | 0.841 | — | 0.927 |
| Learned RM (150 rows) | 0.569 | 0.147 | 0.722 | 0.159 |
| Learned RM (2,500 rows) | 0.671 | 0.193 | 0.616 | 0.227 |

The oracle-verifier arm **climbs** from 0.613 to 0.841 true accuracy — this is the verified-reward self-improvement result. The two learned-reward arms do the opposite: they **fall** to 0.147 and 0.193 while their proxy score *rises*, and the fraction of kept chains that are actually correct drops to 0.159/0.227. That gap between a climbing proxy and a collapsing true metric *is* reward hacking.

### The Goodhart divergence (learned RM, small budget)

| round | true_acc | proxy | sel_precision |
|------:|-------:|-------:|-------:|
| 1 | 0.350 | 0.616 | 0.301 |
| 2 | 0.220 | 0.684 | 0.204 |
| 3 | 0.176 | 0.716 | 0.171 |
| 4 | 0.163 | 0.710 | 0.167 |
| 5 | 0.172 | 0.724 | 0.163 |
| 6 | 0.147 | 0.722 | 0.159 |

Round after round the reward model is more pleased with the policy (proxy 0.616 -> 0.722) while the verifier disagrees (true accuracy 0.350 -> 0.147). The policy learns to produce chains the RM scores highly that are not correct — optimising the map, not the territory.

### Verified-reward control (oracle)

| round | true_acc | sel_precision |
|------:|-------:|-------:|
| 1 | 0.730 | 0.880 |
| 2 | 0.775 | 0.917 |
| 3 | 0.793 | 0.921 |
| 4 | 0.821 | 0.922 |
| 5 | 0.824 | 0.924 |
| 6 | 0.841 | 0.927 |

### More reward-model data sharpens the proxy but does not save you

| RM labelled rows | RM agreement w/ oracle | final true acc |
|---:|---:|---:|
| 150 | 0.569 | 0.147 |
| 2500 | 0.671 | 0.193 |

17x more reward-model data lifted agreement with the oracle from 0.569 to 0.671, yet held-out accuracy still collapsed (0.147 -> 0.193). At this scale the failure is *structural*: best-of-N selection is an adversarial argmax over the proxy, so it hunts out the reward model's mistakes no matter how many rows it saw. The lever that works here is a *verifiable* reward, not a bigger learned one.

### Honest limitations

- Deliberately toy. A real RLHF reward model is far stronger than a mean-pooled ~100k-param classifier, so the absolute collapse is exaggerated; the *direction* (proxy up, true down under imperfect rewards; verified rewards keep climbing) is the reproducible point.
- The oracle arm largely re-demonstrates verified self-improvement (see the companion starlab); its role here is the control that makes the proxy-vs-true divergence legible.
- Best-of-N (an offline selection method), not a full PPO loop: this isolates over-optimization of a proxy reward from the noise of an on-policy gradient.
<!-- RESULTS:END -->
