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


def build_report(
    *,
    gate: dict,
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
