"""Génère un rapport à partir des artefacts"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from simplex_warmstart.params import load_params
from simplex_warmstart.reporting import build_report

GATE_PATH = Path("metrics/gate.json")
DRIFT_PATH = Path("metrics/drift.json")
EVAL_PATH = Path("metrics/eval.json")
OUTPUT_PATH = Path("report.md")


def metrics_diff(base_ref: str) -> str:
    result = subprocess.run(
        ["dvc", "metrics", "diff", base_ref, "--md"],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return f"_Unavailable comparison : {result.stderr.strip().splitlines()[-1:]}_"
    return result.stdout.strip()


def load_or_default(path: Path, default: object) -> object:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default="master")
    args = parser.parse_args()

    gate = load_or_default(
        GATE_PATH,
        {
            "passed": False,
            "checks": [{"name": "pipeline", "value": 0.0, "threshold": 0.0, "passed": False}],
        },
    )
    report = build_report(
        gate=gate,
        drift=load_or_default(DRIFT_PATH, {"status": "not_applicable"}),
        eval_results=load_or_default(EVAL_PATH, []),
        batches=load_params()["data"]["batches"],
        metrics_diff=metrics_diff(args.base),
        base_ref=args.base,
    )

    OUTPUT_PATH.write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
