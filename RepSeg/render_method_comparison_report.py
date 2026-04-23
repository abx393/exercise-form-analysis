#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", "/tmp")

import matplotlib.pyplot as plt
import numpy as np


METHODS = [
    ("FINALREP", "finalrep_reps", "#0f766e"),
    ("plot_multi_accel_updated", "plot_multi_reps", "#b45309"),
]

FIXED_PERSON_REP_COUNTS = {
    "Abhinav": 15,
    "Salvador": 15,
}

BG = "#f6f1e8"
PANEL = "#fffdf7"
GRID = "#d8ccb8"
TEXT = "#1f2937"
MUTED = "#6b7280"
ACTUAL = "#111827"


def _is_actual_truth_source(source: str) -> bool:
    return source in {"inferred_ground_truth", "manual_person_rule"}


def _actual_truth_count(rows: Sequence[Dict[str, Any]]) -> int:
    return sum(1 for r in rows if _is_actual_truth_source(str(r.get("reference_source", ""))))


def _baseline_count(rows: Sequence[Dict[str, Any]]) -> int:
    return sum(1 for r in rows if str(r.get("reference_source", "")) == "plot_multi_baseline")


def _safe_int(value: Any) -> int | None:
    if value in ("", None):
        return None
    try:
        return int(float(value))
    except Exception:
        return None


def _safe_float(value: Any) -> float | None:
    if value in ("", None):
        return None
    try:
        out = float(value)
    except Exception:
        return None
    if not math.isfinite(out):
        return None
    return out


def _read_rows(path: Path) -> List[Dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as f:
        raw_rows = list(csv.DictReader(f))

    rows: List[Dict[str, Any]] = []
    for row in raw_rows:
        gt = _safe_int(row.get("ground_truth_reps"))
        ref = _safe_int(row.get("reference_reps"))
        person = str(row.get("person", "")).strip()
        finalrep = _safe_int(row.get("finalrep_reps"))
        plot_multi = _safe_int(row.get("plot_multi_reps"))
        finalrep_err = _safe_int(row.get("finalrep_abs_error"))
        plot_multi_err = _safe_int(row.get("plot_multi_abs_error"))
        finalrep_ref_err = _safe_int(row.get("finalrep_reference_abs_error"))
        plot_multi_ref_err = _safe_int(row.get("plot_multi_reference_abs_error"))
        if person in FIXED_PERSON_REP_COUNTS:
            gt = FIXED_PERSON_REP_COUNTS[person]
            ref = FIXED_PERSON_REP_COUNTS[person]
        if ref is None:
            ref = gt if gt is not None else plot_multi
        reference_source = row.get("reference_source", "")
        if not reference_source:
            reference_source = "inferred_ground_truth" if gt is not None else ("plot_multi_baseline" if plot_multi is not None else "")
        if person in FIXED_PERSON_REP_COUNTS:
            reference_source = "manual_person_rule"
        if finalrep_ref_err is None and ref is not None and finalrep is not None:
            finalrep_ref_err = abs(finalrep - ref)
        if plot_multi_ref_err is None and ref is not None and plot_multi is not None:
            plot_multi_ref_err = abs(plot_multi - ref)
        reference_winner = row.get("reference_winner", "")
        if not reference_winner and finalrep_ref_err is not None and plot_multi_ref_err is not None:
            if finalrep_ref_err < plot_multi_ref_err:
                reference_winner = "FINALREP"
            elif plot_multi_ref_err < finalrep_ref_err:
                reference_winner = "plot_multi_accel_updated"
            else:
                reference_winner = "tie"
        if ref is None:
            continue
        rows.append(
            {
                **row,
                "ground_truth_reps": gt,
                "reference_reps": ref,
                "reference_source": reference_source,
                "reference_winner": reference_winner,
                "finalrep_reps": finalrep,
                "plot_multi_reps": plot_multi,
                "finalrep_abs_error": finalrep_err,
                "plot_multi_abs_error": plot_multi_err,
                "finalrep_reference_abs_error": finalrep_ref_err,
                "plot_multi_reference_abs_error": plot_multi_ref_err,
            }
        )
    return rows


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    vals = sorted(values)
    mid = len(vals) // 2
    if len(vals) % 2 == 0:
        return 0.5 * (vals[mid - 1] + vals[mid])
    return vals[mid]


def _metrics(rows: Sequence[Dict[str, Any]], pred_key: str, truth_key: str = "reference_reps") -> Dict[str, Any]:
    scored = [r for r in rows if r.get(pred_key) is not None]
    preds = [int(r[pred_key]) for r in scored]
    actual = [int(r[truth_key]) for r in scored]
    errors = [p - a for p, a in zip(preds, actual)]
    abs_errors = [abs(e) for e in errors]
    rel_accuracy = [max(0.0, 1.0 - (abs(e) / a)) for e, a in zip(errors, actual) if a > 0]
    pred_ratio = [p / a for p, a in zip(preds, actual) if a > 0]

    return {
        "sessions_scored": len(scored),
        "exact_match_rate": _mean([1.0 if e == 0 else 0.0 for e in errors]),
        "within_1_rate": _mean([1.0 if abs(e) <= 1 else 0.0 for e in errors]),
        "mae": _mean(abs_errors),
        "rmse": math.sqrt(_mean([float(e * e) for e in errors]) or 0.0) if errors else None,
        "median_abs_error": _median(abs_errors),
        "mean_signed_error": _mean(errors),
        "mean_relative_accuracy": _mean(rel_accuracy),
        "mean_prediction_ratio": _mean(pred_ratio),
    }


def _exercise_summary(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_exercise: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        by_exercise.setdefault(str(row["exercise"]), []).append(row)

    out: List[Dict[str, Any]] = []
    for exercise in sorted(by_exercise):
        subset = by_exercise[exercise]
        item: Dict[str, Any] = {
            "exercise": exercise,
            "sessions_scored": len(subset),
            "actual_truth_sessions": _actual_truth_count(subset),
            "baseline_sessions": _baseline_count(subset),
        }
        for method_name, pred_key, _ in METHODS:
            m = _metrics(subset, pred_key)
            prefix = "finalrep" if method_name == "FINALREP" else "plot_multi"
            item[f"{prefix}_mae"] = m["mae"]
            item[f"{prefix}_exact_match_rate"] = m["exact_match_rate"]
            item[f"{prefix}_within_1_rate"] = m["within_1_rate"]
        out.append(item)
    return out


def _method_table(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for method_name, pred_key, _ in METHODS:
        m = _metrics(rows, pred_key)
        out.append(
            {
                "method": method_name,
                "actual_truth_sessions": _actual_truth_count(rows),
                "baseline_sessions": _baseline_count(rows),
                **m,
            }
        )
    return out


def _fmt_num(value: Any, digits: int = 2) -> str:
    if value is None:
        return ""
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _fmt_pct(value: float | None) -> str:
    return "" if value is None else f"{100.0 * value:.1f}%"


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _set_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": BG,
            "axes.facecolor": PANEL,
            "axes.edgecolor": GRID,
            "axes.labelcolor": TEXT,
            "axes.titleweight": "bold",
            "xtick.color": TEXT,
            "ytick.color": TEXT,
            "text.color": TEXT,
            "font.size": 11,
            "grid.color": GRID,
            "grid.alpha": 0.7,
            "axes.grid": True,
            "grid.linestyle": "-",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _plot_dashboard(rows: Sequence[Dict[str, Any]], method_rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    _set_style()
    fig = plt.figure(figsize=(15, 10), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.15])

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    labels = [r["method"] for r in method_rows]
    x = np.arange(len(labels))
    exact = [100.0 * float(r["exact_match_rate"] or 0.0) for r in method_rows]
    within1 = [100.0 * float(r["within_1_rate"] or 0.0) for r in method_rows]
    colors = [c for _, _, c in METHODS]

    ax1.bar(x - 0.18, exact, width=0.34, color=colors, label="Exact match")
    ax1.bar(x + 0.18, within1, width=0.34, color="#cbd5e1", edgecolor="#94a3b8", label="Within 1 rep")
    ax1.set_title("Accuracy Rate")
    ax1.set_ylabel("Sessions (%)")
    ax1.set_xticks(x, labels)
    ax1.set_ylim(0, 110)
    ax1.legend(frameon=False, loc="upper left")
    for i, value in enumerate(exact):
        ax1.text(i - 0.18, value + 3, f"{value:.0f}%", ha="center", va="bottom", fontsize=10)
    for i, value in enumerate(within1):
        ax1.text(i + 0.18, value + 3, f"{value:.0f}%", ha="center", va="bottom", fontsize=10, color=MUTED)

    mae = [float(r["mae"] or 0.0) for r in method_rows]
    rmse = [float(r["rmse"] or 0.0) for r in method_rows]
    ax2.bar(x - 0.18, mae, width=0.34, color=colors, label="MAE")
    ax2.bar(x + 0.18, rmse, width=0.34, color="#ddd6fe", edgecolor="#a78bfa", label="RMSE")
    ax2.set_title("Error Magnitude")
    ax2.set_ylabel("Reps")
    ax2.set_xticks(x, labels)
    ax2.legend(frameon=False, loc="upper left")
    for i, value in enumerate(mae):
        ax2.text(i - 0.18, value + 0.08, f"{value:.2f}", ha="center", va="bottom", fontsize=10)
    for i, value in enumerate(rmse):
        ax2.text(i + 0.18, value + 0.08, f"{value:.2f}", ha="center", va="bottom", fontsize=10, color=MUTED)

    actual = np.asarray([int(r["reference_reps"]) for r in rows], dtype=float)
    finalrep = np.asarray([int(r["finalrep_reps"]) for r in rows], dtype=float)
    plot_multi = np.asarray([int(r["plot_multi_reps"]) for r in rows], dtype=float)
    max_reps = max(np.max(actual), np.max(finalrep), np.max(plot_multi)) + 1.0

    ax3.scatter(actual, finalrep, s=90, color=METHODS[0][2], alpha=0.9, label="FINALREP")
    ax3.scatter(actual, plot_multi, s=90, color=METHODS[1][2], alpha=0.75, marker="s", label="plot_multi_accel_updated")
    ax3.plot([0, max_reps], [0, max_reps], color=ACTUAL, linestyle="--", linewidth=1.4, label="Perfect prediction")
    ax3.set_title("Reference vs Predicted Reps")
    ax3.set_xlabel("Reference reps")
    ax3.set_ylabel("Predicted reps")
    ax3.set_xlim(0, max_reps)
    ax3.set_ylim(0, max_reps)
    ax3.legend(frameon=False, loc="upper left")

    session_labels = [f"{r['exercise'][:2].upper()}-{idx + 1}" for idx, r in enumerate(rows)]
    order = np.argsort(
        np.asarray([abs(int(r["plot_multi_reference_abs_error"] or 0)) for r in rows])
        - np.asarray([abs(int(r["finalrep_reference_abs_error"] or 0)) for r in rows])
    )
    ordered_labels = [session_labels[i] for i in order]
    ordered_finalrep = [int(rows[i]["finalrep_reference_abs_error"] or 0) for i in order]
    ordered_plot = [int(rows[i]["plot_multi_reference_abs_error"] or 0) for i in order]
    y = np.arange(len(rows))

    ax4.barh(y - 0.18, ordered_finalrep, height=0.34, color=METHODS[0][2], label="FINALREP")
    ax4.barh(y + 0.18, ordered_plot, height=0.34, color=METHODS[1][2], label="plot_multi_accel_updated")
    ax4.set_title("Absolute Error by Session")
    ax4.set_xlabel("Absolute error to reference (reps)")
    ax4.set_yticks(y, ordered_labels)
    ax4.invert_yaxis()
    ax4.legend(frameon=False, loc="lower right")

    fig.suptitle("Rep Counting Method Comparison", fontsize=18, fontweight="bold")
    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_session_lines(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    _set_style()
    fig, ax = plt.subplots(figsize=(16, 6.5), constrained_layout=True)

    x = np.arange(len(rows))
    actual = np.asarray([int(r["reference_reps"]) for r in rows], dtype=float)
    finalrep = np.asarray([int(r["finalrep_reps"]) for r in rows], dtype=float)
    plot_multi = np.asarray([int(r["plot_multi_reps"]) for r in rows], dtype=float)
    labels = [Path(str(r["relative_path"])).name for r in rows]

    ax.plot(x, actual, color=ACTUAL, linewidth=2.6, marker="o", label="Reference reps")
    ax.plot(x, finalrep, color=METHODS[0][2], linewidth=2.4, marker="o", label="FINALREP")
    ax.plot(x, plot_multi, color=METHODS[1][2], linewidth=2.4, marker="s", label="plot_multi_accel_updated")
    ax.set_title("Session-by-Session Prediction Trace")
    ax.set_ylabel("Rep count")
    ax.set_xticks(x, [f"S{i + 1}" for i in x])
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.set_xlim(-0.4, len(rows) - 0.6)

    for i, label in enumerate(labels):
        ax.text(i, -0.12, label, rotation=35, ha="right", va="top", fontsize=8, color=MUTED, transform=ax.get_xaxis_transform())

    fig.savefig(out_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _html_table(rows: Sequence[Dict[str, Any]], columns: Sequence[str], title: str, pct_cols: Iterable[str] = ()) -> str:
    pct_cols = set(pct_cols)
    head = "".join(f"<th>{html.escape(col.replace('_', ' ').title())}</th>" for col in columns)
    body_rows = []
    for row in rows:
        cells = []
        for col in columns:
            value = row.get(col)
            if col in pct_cols:
                text = _fmt_pct(_safe_float(value))
            elif isinstance(value, float):
                text = _fmt_num(value)
            else:
                text = _fmt_num(value)
            cells.append(f"<td>{html.escape(text)}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    body = "".join(body_rows)
    return (
        f"<section class='card'>"
        f"<h2>{html.escape(title)}</h2>"
        f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"
        f"</section>"
    )


def _write_html_report(
    out_path: Path,
    root_dir: Path,
    method_rows: Sequence[Dict[str, Any]],
    exercise_rows: Sequence[Dict[str, Any]],
    session_rows: Sequence[Dict[str, Any]],
) -> None:
    method_table = _html_table(
        method_rows,
        [
            "method",
            "sessions_scored",
            "actual_truth_sessions",
            "baseline_sessions",
            "exact_match_rate",
            "within_1_rate",
            "mae",
            "rmse",
            "median_abs_error",
            "mean_signed_error",
            "mean_relative_accuracy",
            "mean_prediction_ratio",
        ],
        "Method Summary",
        pct_cols={"exact_match_rate", "within_1_rate", "mean_relative_accuracy"},
    )
    exercise_table = _html_table(
        exercise_rows,
        [
            "exercise",
            "sessions_scored",
            "actual_truth_sessions",
            "baseline_sessions",
            "finalrep_mae",
            "plot_multi_mae",
            "finalrep_exact_match_rate",
            "plot_multi_exact_match_rate",
            "finalrep_within_1_rate",
            "plot_multi_within_1_rate",
        ],
        "Exercise Summary",
        pct_cols={
            "finalrep_exact_match_rate",
            "plot_multi_exact_match_rate",
            "finalrep_within_1_rate",
            "plot_multi_within_1_rate",
        },
    )
    session_table = _html_table(
        session_rows,
        [
            "relative_path",
            "reference_source",
            "reference_reps",
            "ground_truth_reps",
            "finalrep_reps",
            "plot_multi_reps",
            "finalrep_reference_abs_error",
            "plot_multi_reference_abs_error",
            "reference_winner",
        ],
        "Scored Sessions",
    )

    best = min(method_rows, key=lambda row: float(row["mae"] or 1e9))
    generated_at = datetime.now(timezone.utc).isoformat()
    html_text = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Rep Method Comparison Report</title>
  <style>
    :root {{
      --bg: {BG};
      --panel: {PANEL};
      --ink: {TEXT};
      --muted: {MUTED};
      --line: {GRID};
      --accent-a: {METHODS[0][2]};
      --accent-b: {METHODS[1][2]};
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Avenir Next", "Segoe UI", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(15,118,110,0.10), transparent 28rem),
        radial-gradient(circle at top right, rgba(180,83,9,0.12), transparent 28rem),
        var(--bg);
      color: var(--ink);
      line-height: 1.45;
    }}
    .wrap {{
      max-width: 1280px;
      margin: 0 auto;
      padding: 40px 24px 64px;
    }}
    .hero {{
      display: grid;
      grid-template-columns: 2fr 1fr;
      gap: 18px;
      margin-bottom: 24px;
    }}
    .card {{
      background: rgba(255,253,247,0.92);
      border: 1px solid var(--line);
      border-radius: 20px;
      padding: 20px 22px;
      box-shadow: 0 18px 40px rgba(31,41,55,0.06);
      backdrop-filter: blur(8px);
      margin-bottom: 20px;
    }}
    h1, h2, h3 {{ margin: 0 0 12px; }}
    h1 {{
      font-size: clamp(2rem, 5vw, 3.4rem);
      line-height: 1.02;
      letter-spacing: -0.04em;
    }}
    h2 {{ font-size: 1.2rem; }}
    p {{ margin: 0 0 10px; color: var(--muted); }}
    .metrics {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .metric {{
      padding: 14px 16px;
      border-radius: 14px;
      background: white;
      border: 1px solid var(--line);
    }}
    .metric .label {{
      display: block;
      font-size: 0.85rem;
      color: var(--muted);
      margin-bottom: 4px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .metric .value {{
      font-size: 1.65rem;
      font-weight: 700;
      color: var(--ink);
    }}
    .charts {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 20px;
    }}
    img {{
      width: 100%;
      display: block;
      border-radius: 16px;
      border: 1px solid var(--line);
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 0.95rem;
      background: white;
      border-radius: 14px;
      overflow: hidden;
    }}
    th, td {{
      padding: 11px 12px;
      border-bottom: 1px solid rgba(216,204,184,0.55);
      text-align: left;
      white-space: nowrap;
    }}
    thead th {{
      background: #f4ece0;
      font-size: 0.82rem;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      color: var(--muted);
    }}
    tbody tr:nth-child(even) {{
      background: #fffaf2;
    }}
    @media (max-width: 900px) {{
      .hero {{ grid-template-columns: 1fr; }}
      .metrics {{ grid-template-columns: 1fr 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="card">
        <h1>Rep Prediction vs Reference Reps</h1>
        <p>Dataset: {html.escape(str(root_dir))}</p>
        <p>The charts below compare each model's predicted rep count against the reference repetition count for every scored session.</p>
        <p>Reference rule: use actual reps when they can be inferred from the session name or when the person-level rule supplies them, otherwise use <strong>plot_multi_accel_updated</strong> as the fallback baseline.</p>
        <p>Best overall by MAE: <strong>{html.escape(str(best['method']))}</strong></p>
      </div>
      <div class="card">
        <h2>At A Glance</h2>
        <div class="metrics">
          <div class="metric"><span class="label">Scored Sessions</span><span class="value">{len(session_rows)}</span></div>
          <div class="metric"><span class="label">Actual Truth Sessions</span><span class="value">{_actual_truth_count(session_rows)}</span></div>
          <div class="metric"><span class="label">Baseline Sessions</span><span class="value">{_baseline_count(session_rows)}</span></div>
          <div class="metric"><span class="label">Best MAE</span><span class="value">{_fmt_num(best['mae'])}</span></div>
          <div class="metric"><span class="label">Generated</span><span class="value" style="font-size:1rem">{html.escape(generated_at[:19])}</span></div>
        </div>
      </div>
    </section>
    <section class="card charts">
      <h2>Dashboard</h2>
      <img src="method_comparison_dashboard.png" alt="Method comparison dashboard" />
    </section>
    <section class="card charts">
      <h2>Prediction Trace</h2>
      <img src="method_comparison_session_trace.png" alt="Prediction trace by session" />
    </section>
    {method_table}
    {exercise_table}
    {session_table}
  </div>
</body>
</html>
"""
    out_path.write_text(html_text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render tables and charts for method_comparison.csv.")
    parser.add_argument("root", type=str, help="Dataset root that contains rep_analysis_reports/method_comparison.csv")
    parser.add_argument("--out-dir", type=str, default=None, help="Directory for rendered report assets")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (root / "rep_analysis_reports")
    comparison_csv = out_dir / "method_comparison.csv"
    if not comparison_csv.exists():
        raise SystemExit(f"Comparison CSV not found: {comparison_csv}")

    rows = _read_rows(comparison_csv)
    if not rows:
        raise SystemExit("No scored rows with ground-truth reps were found in method_comparison.csv")

    method_rows = _method_table(rows)
    exercise_rows = _exercise_summary(rows)
    session_rows = [
        {
            "relative_path": r["relative_path"],
                "reference_source": r["reference_source"],
                "reference_reps": r["reference_reps"],
                "ground_truth_reps": r["ground_truth_reps"],
                "finalrep_reps": r["finalrep_reps"],
                "plot_multi_reps": r["plot_multi_reps"],
                "finalrep_reference_abs_error": r["finalrep_reference_abs_error"],
                "plot_multi_reference_abs_error": r["plot_multi_reference_abs_error"],
                "reference_winner": r["reference_winner"],
            }
        for r in rows
    ]

    method_summary_csv = out_dir / "method_summary.csv"
    exercise_summary_csv = out_dir / "exercise_summary.csv"
    scored_sessions_csv = out_dir / "scored_sessions_comparison.csv"
    metrics_json = out_dir / "method_summary.json"
    dashboard_png = out_dir / "method_comparison_dashboard.png"
    session_trace_png = out_dir / "method_comparison_session_trace.png"
    html_report = out_dir / "method_comparison_report.html"

    _write_csv(method_summary_csv, method_rows)
    _write_csv(exercise_summary_csv, exercise_rows)
    _write_csv(scored_sessions_csv, session_rows)
    metrics_json.write_text(
        json.dumps(
            {
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "root": str(root),
                "method_summary": method_rows,
                "exercise_summary": exercise_rows,
                "scored_sessions": session_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    _plot_dashboard(rows, method_rows, dashboard_png)
    _plot_session_lines(rows, session_trace_png)
    _write_html_report(html_report, root, method_rows, exercise_rows, session_rows)

    print(f"Wrote {method_summary_csv}")
    print(f"Wrote {exercise_summary_csv}")
    print(f"Wrote {scored_sessions_csv}")
    print(f"Wrote {metrics_json}")
    print(f"Wrote {dashboard_png}")
    print(f"Wrote {session_trace_png}")
    print(f"Wrote {html_report}")


if __name__ == "__main__":
    main()
