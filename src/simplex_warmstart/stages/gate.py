"""E6 : vérif qualité du modèle"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ..params import load_params

METRICS = Path("metrics/eval.json")
REPORT = Path("metrics/gate.json")


def main():
    thresholds = load_params()["gates"]
    results = json.loads(METRICS.read_text(encoding="utf-8"))

    overall = results["test"]
    family_rmse = {
        key.removeprefix("test_"): scores["rmse"]
        for key, scores in results.items()
        if key.startswith("test_")
    }
    worst_family, worst_rmse = max(family_rmse.items(), key=lambda kv: kv[1])
    gap = worst_rmse - overall["rmse"]

    checks = [
        {
            "name": "test_rmse",
            "value": overall["rmse"],
            "threshold": thresholds["max_test_rmse"],
            "passed": overall["rmse"] <= thresholds["max_test_rmse"],
        },
        {
            "name": "test_r2",
            "value": overall["r2"],
            "threshold": thresholds["min_test_r2"],
            "passed": overall["r2"] >= thresholds["min_test_r2"],
        },
        {
            "name": f"family_gap ({worst_family})",
            "value": gap,
            "threshold": thresholds["max_family_rmse_gap"],
            "passed": gap <= thresholds["max_family_rmse_gap"],
        },
    ]

    passed = all(check["passed"] for check in checks)

    REPORT.write_text(json.dumps({"passed": passed, "checks": checks}, indent=2), encoding="utf-8")

    for check in checks:
        status = "OK " if check["passed"] else "ECHEC"
        print(f"{status} {check['name']}: {check['value']:.4f} (seuil {check['threshold']})")

    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
