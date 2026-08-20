"""Rendu md du rapport de retrain"""

from __future__ import annotations

MARKER = "<!-- simplex-warmstart-report -->"


def render_gate_table(gate: dict) -> str:
    rows = ["| Control | Value | Threshold | Verdict |", "| --- | --- | --- | --- |"]
    for check in gate["checks"]:
        verdict = "🟩" if check["passed"] else "🟥"
        rows.append(f"| {check['name']} | {check['value']:.4f} | {check['threshold']} | {verdict}")
    return "\n".join(rows)


def render_family_table(results: dict) -> str:
    rows = ["| Family | RMSE | MAE | R² |", "| --- | --- | --- | --- |"]
    for key, scores in sorted(results.items()):
        if not key.startswith("test_"):
            continue
        family = key.removeprefix("test_")
        rows.append(
            f"| {family} | {scores['rmse']:.4f} | {scores['mae']:.4f} | {scores['r2']:.4f} |"
        )
    return "\n".join(rows)


def render_manifest_table(batches: list[dict]) -> str:
    rows = ["| Batch | Studies | Families | Seed |", "| --- | --- | --- | --- |"]
    for batch in batches:
        rows.append(
            f"| {batch['id']} | {batch['n_studies']} "
            f"| {' '.join(batch['families'])} | {batch['seed']} |"
        )
    return "\n".join(rows)


def render_drift_section(drift: dict) -> str:
    if drift.get("status") == "not_applicable":
        return "_One single batch : no comparison available._"

    lines = []
    if drift.get("dataset_drift"):
        lines.append(
            f"> ⚠️ **Distribution drift detected (OOD)** — "
            f"{drift['drift_share']:.0%} of monitored columns changed."
        )
    for column, values in drift.get("new_categories", {}).items():
        lines.append(f"> 🆕 New value(s) in `{column}` : {', '.join(values)}.")
    if not lines:
        lines.append("> ✅ Distributions are stable wrt previous batches.")

    lines += ["", "| Column | PSI | KS | Verdict |", "| --- | --- | --- | --- |"]
    for column in sorted(drift.get("columns", []), key=lambda c: -c["psi"])[:8]:
        verdict = "⚠️" if column["drifted"] else "—"
        lines.append(
            f"| {column['column']} | {column['psi']:.3f} | {column['ks']:.3f} | {verdict} |"
        )
    return "\n".join(lines)


def build_report(
    *,
    gate: dict,
    drift: dict,
    eval_results: dict,
    batches: list[dict],
    metrics_diff: str,
    base_ref: str,
) -> str:
    headline = "✅ Admissible model" if gate["passed"] else "❌ Quality gate failed"

    sections = [
        MARKER,
        f"## Retrain Report — {headline}",
        "",
        "### Quality gate",
        render_gate_table(gate),
        "",
        "### Data drift",
        render_drift_section(drift),
        "",
        f"### Metrics diff (vs `{base_ref}`)",
        metrics_diff or "_No diff detected._",
        "",
        "### Performance per family",
        render_family_table(eval_results),
        "",
        "<details><summary>Data manifest</summary>",
        "",
        render_manifest_table(batches),
        "",
        "</details>",
    ]
    return "\n".join(sections) + "\n"
