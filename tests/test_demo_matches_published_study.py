"""``rhlab demo`` must run the study the README table reports, not a shorter one.

The demo used to stop at three rounds while ``results/hacking.json`` averaged six, so
it printed a curve that cut off before the proxy collapse the table describes.
"""

from __future__ import annotations

import json
from pathlib import Path

from rhlab.cli import _parser
from rhlab.study import ROUNDS, SEEDS

ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    data = json.loads((ROOT / "results" / "hacking.json").read_text(encoding="utf-8"))
    return data["config"]


def test_demo_defaults_match_the_committed_study():
    config = _config()
    args = _parser().parse_args(["demo"])
    assert config["rounds"] == ROUNDS
    assert tuple(config["seeds"]) == SEEDS
    assert args.rounds == ROUNDS
    assert args.seed in config["seeds"]
