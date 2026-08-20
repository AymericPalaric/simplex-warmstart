"""E5 : vérifier si le nouveau batch drift"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from ..drift import compute_drift
from ..features import DESCRIPTOR_COLS
from ..params import load_params

INPUT = Path("data/processed/experiments.parquet")
METRICS_PATH = Path("metrics/drift.json")
HTML_PATH = Path("reports/drift.html")

# MONITORED_COLS = [*DESCRIPTOR_COLS, TARGET_COL]
MONITORED_COLS = [
    *DESCRIPTOR_COLS,
]


def build_evidently_report(
    ref: pd.DataFrame, cur: pd.DataFrame, columns: list[str], output: Path
) -> bool:
    try:
        from evidently import DataDefinition, Dataset, Report
        from evidently.presets import DataDriftPreset
    except ImportError:
        return False

    data_def = DataDefinition(numerical_columns=list(columns))
    snapshot = Report([DataDriftPreset()]).run(
        reference_data=Dataset.from_pandas(ref[columns], data_definition=data_def),
        current_data=Dataset.from_pandas(cur[columns], data_definition=data_def),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    snapshot.save_html(str(output))
    return True


def main():
    cfg = load_params()["drift"]
    df = pd.read_parquet(INPUT)
    latest = int(df["batch_id"].max())

    METRICS_PATH.parent.mkdir(parents=True, exist_ok=True)
    if latest == 0:
        payload = {
            "status": "not_applicable",
            "reason": "one batch : no past reference",
            "dataset_drift": False,
            "drift_share": 0.0,
            "new_categories": {},
            "columns": [],
        }
        METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print("One batch, no comparison available")
        return

    ref = df[df["batch_id"] < latest]
    cur = df[df["batch_id"] == latest]

    payload = compute_drift(
        ref,
        cur,
        MONITORED_COLS,
        psi_thresh=cfg["psi_threshold"],
        share_thresh=cfg["share_threshold"],
    )
    payload["status"] = "ok"
    payload["latest_batch"] = latest
    payload["html_report"] = build_evidently_report(ref, cur, MONITORED_COLS, HTML_PATH)

    METRICS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    flag = "DRIFT" if payload["dataset_drift"] else "STABLE"
    print(
        f"Batch {latest} vs {latest} previous batch(es) : {flag} "
        f"({payload['drift_share']:.0%} of columns)"
    )


if __name__ == "__main__":
    main()
