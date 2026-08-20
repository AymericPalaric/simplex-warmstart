from simplex_warmstart.reporting import MARKER, build_report

GATE_OK = {
    "passed": True,
    "checks": [{"name": "test_rmse", "value": 0.42, "threshold": 0.55, "passed": True}],
}
GATE_KO = {
    "passed": False,
    "checks": [{"name": "test_rmse", "value": 0.91, "threshold": 0.55, "passed": False}],
}
DRIFT = {
    "status": "ok",
    "dataset_drift": True,
    "drift_share": 0.8,
}
EVAL = {"test": {"rmse": 0.42, "mae": 0.33, "r2": 0.94}}
BATCHES = [{"id": 0, "n_studies": 120, "seed": 0, "families": ["esters"]}]


def build(gate):
    return build_report(
        gate=gate,
        drift=DRIFT,
        eval_results=EVAL,
        batches=BATCHES,
        metrics_diff="",
        base_ref="master",
    )


def test_report_carries_the_sticky_marker():
    assert build(GATE_OK).startswith(MARKER)


def test_failed_gate_is_visible_in_the_headline():
    assert "❌" in build(GATE_KO).splitlines()[1]


def test_empty_diff_is_replaced_by_a_message():
    assert "No diff" in build(GATE_OK)


def test_drift_detector_reported():
    assert "⚠️" in build(GATE_OK)
