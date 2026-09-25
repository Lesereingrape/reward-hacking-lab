"""Run the full multi-seed reward-hacking study and write results/hacking.json.

Three arms of rejection-sampling fine-tuning that differ ONLY in how the best of N
sampled chains is *selected* per prompt:

* ``oracle``   — the exact verifier keeps truly-correct chains: the clean STaR/RFT
  control whose held-out accuracy climbs round after round.
* ``rm_small`` / ``rm_large`` — a learned reward model trained on 150 / 2,500 labelled
  chains keeps the chains *it thinks* are best. The leaky (small) model lets the
  policy over-optimize the proxy: reward-model score keeps rising while true
  accuracy flattens or falls and selection precision decays — reward hacking. The
  better (large) reward model tracks the oracle more closely, showing the gap is a
  function of proxy quality, not of RL itself.

The loop logic lives in ``rhlab.study`` (shared with ``rhlab demo``), and the
artifact records the per-seed trajectories plus the environment they are bit-exact in.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from rhlab.study import SEEDS, build_results, run_seed


def main(out: str = "results/hacking.json") -> None:
    t0 = time.time()
    per_seed = [run_seed(s) for s in SEEDS]
    results = build_results(per_seed, runtime=time.time() - t0)
    dest = Path(out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results["variants"], indent=2))
    print(f"runtime={results['runtime_sec']}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(prog="run_study")
    ap.add_argument("--out", default="results/hacking.json",
                    help="where to write the artifact (default: results/hacking.json)")
    main(ap.parse_args().out)
