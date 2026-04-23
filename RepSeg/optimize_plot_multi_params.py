#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

import batch_rep_boundary_report as batch


TUNABLE_FIELDS = (
    "lowpass_hz",
    "prominence_factor",
    "min_dur_factor",
    "min_period_s",
    "max_period_s",
    "candidate_signals",
)

SIGNAL_CHOICES = ("mag", "x", "y", "z", "xy", "xz", "yz")


@dataclass(frozen=True)
class SessionSpec:
    session_dir: str
    relative_path: str
    exercise: str
    person: str
    session: str
    expected_reps: int


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Random-search optimizer for plot_multi_accel_updated using weighted "
            "per-exercise MAE. Trials run sequentially; sessions within each "
            "trial are evaluated in parallel with threads."
        )
    )
    parser.add_argument("root", type=str, help="Dataset root directory")
    parser.add_argument("--out-dir", type=str, default=None, help="Output directory for optimizer artifacts")
    parser.add_argument("--template-reps", type=int, default=5, help="Template rep count passed to analyze_with_plot_multi")
    parser.add_argument("--trials", type=int, default=250, help="Number of random trials to evaluate, excluding baseline")
    parser.add_argument("--workers", type=int, default=max(1, min(16, (os.cpu_count() or 8))), help="Thread count used within each trial")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--exercise",
        action="append",
        default=[],
        help="Restrict optimization to one or more exercises, e.g. --exercise lunges --exercise situps",
    )
    parser.add_argument(
        "--exercise-weight",
        action="append",
        default=[],
        help="Exercise weight override in the form exercise=weight, e.g. lunges=2.0",
    )
    parser.add_argument(
        "--failure-penalty-factor",
        type=float,
        default=1.5,
        help="Multiplier applied to expected reps when a session evaluation fails",
    )
    parser.add_argument(
        "--failure-penalty-floor",
        type=float,
        default=10.0,
        help="Minimum absolute error assigned when a session evaluation fails",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="How many top trials to retain in the leaderboard outputs",
    )
    parser.add_argument(
        "--shuffle-signals",
        action="store_true",
        help="Allow candidate signal order to be mutated during search",
    )
    return parser


def _parse_weight_overrides(items: Sequence[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Invalid --exercise-weight value: {item!r}. Expected exercise=weight.")
        exercise, raw_weight = item.split("=", 1)
        exercise_key = batch._canonical_exercise_name(exercise)
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise SystemExit(f"Invalid weight for {exercise!r}: {raw_weight!r}") from exc
        if not math.isfinite(weight) or weight <= 0:
            raise SystemExit(f"Weight for {exercise!r} must be a positive finite number.")
        out[exercise_key] = weight
    return out


def _collect_labeled_sessions(root: Path, exercises: Optional[set[str]]) -> List[SessionSpec]:
    sessions: List[SessionSpec] = []
    for session_dir in batch._iter_session_dirs(root):
        if session_dir.name == "rep_analysis_reports":
            continue
        meta = batch._session_metadata(root, session_dir)
        exercise_key = batch._canonical_exercise_name(meta["exercise"])
        if exercises and exercise_key not in exercises:
            continue
        expected = batch.infer_expected_reps(meta)
        if expected is None:
            continue
        sessions.append(
            SessionSpec(
                session_dir=str(session_dir),
                relative_path=meta["relative_path"],
                exercise=exercise_key,
                person=meta["person"],
                session=meta["session"],
                expected_reps=int(expected),
            )
        )
    return sessions


def _default_exercise_weights(sessions: Sequence[SessionSpec], overrides: Mapping[str, float]) -> Dict[str, float]:
    exercises = sorted({session.exercise for session in sessions})
    weights = {exercise: 1.0 for exercise in exercises}
    weights.update(overrides)
    total = sum(weights.values())
    if total <= 0:
        raise SystemExit("Exercise weights must sum to a positive value.")
    return {exercise: weights[exercise] / total for exercise in exercises}


def _base_search_config(exercises: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    return {
        exercise: {field: deepcopy(batch._activity_config(exercise)[field]) for field in TUNABLE_FIELDS}
        for exercise in exercises
    }


def _bounded_float(rng: random.Random, low: float, high: float, digits: int = 3) -> float:
    return round(rng.uniform(low, high), digits)


def _mutated_signal_order(rng: random.Random, current: Sequence[str]) -> List[str]:
    seen = [name for name in current if name in SIGNAL_CHOICES]
    remaining = [name for name in SIGNAL_CHOICES if name not in seen]
    order = list(seen) + remaining
    rng.shuffle(order)
    keep = max(4, len(seen))
    return order[:keep]


def _sample_trial_config(
    rng: random.Random,
    base_config: Mapping[str, Mapping[str, Any]],
    *,
    shuffle_signals: bool,
) -> Dict[str, Dict[str, Any]]:
    sampled: Dict[str, Dict[str, Any]] = {}
    for exercise, base in base_config.items():
        min_period = _bounded_float(rng, max(0.25, 0.6 * float(base["min_period_s"])), min(3.5, 1.8 * float(base["min_period_s"])))
        max_period_low = max(min_period + 0.5, 0.7 * float(base["max_period_s"]))
        max_period_high = max(max_period_low + 0.1, min(12.0, 1.6 * float(base["max_period_s"])))
        sampled[exercise] = {
            "lowpass_hz": _bounded_float(rng, 1.5, 5.0),
            "prominence_factor": _bounded_float(rng, 0.15, 0.65),
            "min_dur_factor": _bounded_float(rng, 0.40, 0.85),
            "min_period_s": min_period,
            "max_period_s": _bounded_float(rng, max_period_low, max_period_high),
            "candidate_signals": (
                _mutated_signal_order(rng, base["candidate_signals"])
                if shuffle_signals
                else list(base["candidate_signals"])
            ),
        }
    return sampled


def _apply_config_overrides(overrides: Mapping[str, Mapping[str, Any]]) -> Dict[str, Dict[str, Any]]:
    original = deepcopy(batch.ACTIVITY_CONFIGS)
    updated = deepcopy(original)
    for exercise, config in overrides.items():
        updated[exercise] = {**updated.get(exercise, {}), **deepcopy(dict(config))}
    batch.ACTIVITY_CONFIGS = updated
    return original


def _restore_configs(original: Mapping[str, Mapping[str, Any]]) -> None:
    batch.ACTIVITY_CONFIGS = deepcopy(dict(original))


def _evaluate_session(session: SessionSpec, template_reps: int) -> Dict[str, Any]:
    try:
        result = batch.analyze_with_plot_multi(
            Path(session.session_dir),
            template_reps=template_reps,
            expected_reps=session.expected_reps,
            exercise_name=session.exercise,
        )
        estimated = int(result["estimated_total_reps"])
        abs_error = abs(estimated - session.expected_reps)
        return {
            "relative_path": session.relative_path,
            "exercise": session.exercise,
            "person": session.person,
            "session": session.session,
            "expected_reps": session.expected_reps,
            "estimated_reps": estimated,
            "abs_error": abs_error,
            "failed": False,
            "error_text": "",
        }
    except Exception as exc:
        return {
            "relative_path": session.relative_path,
            "exercise": session.exercise,
            "person": session.person,
            "session": session.session,
            "expected_reps": session.expected_reps,
            "estimated_reps": None,
            "abs_error": None,
            "failed": True,
            "error_text": str(exc),
        }


def _evaluate_trial_sessions(
    sessions: Sequence[SessionSpec],
    *,
    template_reps: int,
    workers: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(_evaluate_session, session, template_reps): session.relative_path
            for session in sessions
        }
        for future in as_completed(future_map):
            rows.append(future.result())
    rows.sort(key=lambda row: row["relative_path"])
    return rows


def _score_trial(
    rows: Sequence[Dict[str, Any]],
    *,
    exercise_weights: Mapping[str, float],
    failure_penalty_factor: float,
    failure_penalty_floor: float,
) -> Dict[str, Any]:
    by_exercise: Dict[str, List[float]] = {exercise: [] for exercise in exercise_weights}
    session_rows: List[Dict[str, Any]] = []

    for row in rows:
        expected = int(row["expected_reps"])
        error_value: float
        if row["failed"]:
            error_value = max(failure_penalty_floor, failure_penalty_factor * expected)
        else:
            error_value = float(row["abs_error"])
        by_exercise[row["exercise"]].append(error_value)
        session_rows.append({**row, "scored_abs_error": error_value})

    exercise_metrics: Dict[str, Dict[str, Any]] = {}
    weighted_mae = 0.0
    for exercise, weight in exercise_weights.items():
        errors = by_exercise.get(exercise, [])
        if not errors:
            exercise_mae = None
            weighted_component = None
        else:
            exercise_mae = float(sum(errors) / len(errors))
            weighted_component = float(weight * exercise_mae)
            weighted_mae += weighted_component
        exercise_metrics[exercise] = {
            "weight": float(weight),
            "sessions": len(errors),
            "mae": exercise_mae,
            "median_abs_error": float(statistics.median(errors)) if errors else None,
            "weighted_component": weighted_component,
            "max_abs_error": float(max(errors)) if errors else None,
            "failed_sessions": sum(1 for row in session_rows if row["exercise"] == exercise and row["failed"]),
        }

    overall_errors = [row["scored_abs_error"] for row in session_rows]
    return {
        "weighted_mae": float(weighted_mae),
        "overall_mae": float(sum(overall_errors) / len(overall_errors)) if overall_errors else None,
        "failed_sessions": sum(1 for row in session_rows if row["failed"]),
        "exercise_metrics": exercise_metrics,
        "session_rows": session_rows,
    }


def _trial_row(trial_index: int, trial_kind: str, overrides: Mapping[str, Mapping[str, Any]], metrics: Mapping[str, Any]) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "trial_index": trial_index,
        "trial_kind": trial_kind,
        "weighted_mae": metrics["weighted_mae"],
        "overall_mae": metrics["overall_mae"],
        "failed_sessions": metrics["failed_sessions"],
    }
    for exercise, exercise_metrics in sorted(metrics["exercise_metrics"].items()):
        row[f"{exercise}_mae"] = exercise_metrics["mae"]
        row[f"{exercise}_weight"] = exercise_metrics["weight"]
        row[f"{exercise}_failed_sessions"] = exercise_metrics["failed_sessions"]
        row[f"{exercise}_candidate_signals"] = ",".join(overrides[exercise]["candidate_signals"])
        for key in ("lowpass_hz", "prominence_factor", "min_dur_factor", "min_period_s", "max_period_s"):
            row[f"{exercise}_{key}"] = overrides[exercise][key]
    return row


def _write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Root directory not found: {root}")

    selected_exercises = {batch._canonical_exercise_name(name) for name in args.exercise} if args.exercise else None
    sessions = _collect_labeled_sessions(root, selected_exercises)
    if not sessions:
        raise SystemExit("No labeled sessions found for optimization.")

    exercises = sorted({session.exercise for session in sessions})
    weight_overrides = _parse_weight_overrides(args.exercise_weight)
    exercise_weights = _default_exercise_weights(sessions, weight_overrides)
    base_config = _base_search_config(exercises)
    rng = random.Random(args.seed)

    out_dir = Path(args.out_dir).expanduser().resolve() if args.out_dir else (root / "rep_analysis_reports" / "plot_multi_optimizer")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Optimizing plot_multi_accel_updated on {len(sessions)} labeled sessions across {len(exercises)} exercises")
    print(f"Exercises: {', '.join(exercises)}")
    print("Exercise weights: " + ", ".join(f"{exercise}={exercise_weights[exercise]:.3f}" for exercise in exercises))
    print(f"Trials: baseline + {args.trials} random")
    print(f"Workers per trial: {args.workers}")

    leaderboard_rows: List[Dict[str, Any]] = []
    best_trial: Optional[Dict[str, Any]] = None
    best_metrics: Optional[Dict[str, Any]] = None
    best_session_rows: List[Dict[str, Any]] = []

    trial_configs: List[tuple[str, Dict[str, Dict[str, Any]]]] = [("baseline", deepcopy(base_config))]
    for _ in range(args.trials):
        trial_configs.append(
            (
                "random",
                _sample_trial_config(
                    rng,
                    base_config,
                    shuffle_signals=bool(args.shuffle_signals),
                ),
            )
        )

    for trial_index, (trial_kind, overrides) in enumerate(trial_configs):
        original_configs = _apply_config_overrides(overrides)
        try:
            rows = _evaluate_trial_sessions(
                sessions,
                template_reps=args.template_reps,
                workers=args.workers,
            )
        finally:
            _restore_configs(original_configs)

        metrics = _score_trial(
            rows,
            exercise_weights=exercise_weights,
            failure_penalty_factor=float(args.failure_penalty_factor),
            failure_penalty_floor=float(args.failure_penalty_floor),
        )
        row = _trial_row(trial_index, trial_kind, overrides, metrics)
        leaderboard_rows.append(row)

        if best_metrics is None or metrics["weighted_mae"] < best_metrics["weighted_mae"]:
            best_trial = {
                "trial_index": trial_index,
                "trial_kind": trial_kind,
                "config": deepcopy(overrides),
            }
            best_metrics = deepcopy(metrics)
            best_session_rows = deepcopy(metrics["session_rows"])

        print(
            f"trial {trial_index:04d} [{trial_kind}] "
            f"weighted_mae={metrics['weighted_mae']:.3f} "
            f"overall_mae={metrics['overall_mae']:.3f} "
            f"failed={metrics['failed_sessions']}"
        )

    leaderboard_rows.sort(key=lambda row: (float(row["weighted_mae"]), float(row["overall_mae"]), int(row["failed_sessions"])))
    top_rows = leaderboard_rows[: max(1, int(args.top_k))]

    assert best_trial is not None
    assert best_metrics is not None

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "root": str(root),
        "sessions_scored": len(sessions),
        "exercise_weights": exercise_weights,
        "seed": int(args.seed),
        "trials_requested": int(args.trials),
        "workers": int(args.workers),
        "template_reps": int(args.template_reps),
        "failure_penalty_factor": float(args.failure_penalty_factor),
        "failure_penalty_floor": float(args.failure_penalty_floor),
        "shuffle_signals": bool(args.shuffle_signals),
        "best_trial": best_trial,
        "best_metrics": best_metrics,
    }

    summary_path = out_dir / "plot_multi_optimizer_summary.json"
    leaderboard_path = out_dir / "plot_multi_optimizer_leaderboard.csv"
    best_sessions_path = out_dir / "plot_multi_optimizer_best_sessions.csv"

    summary_path.write_text(json.dumps(summary, indent=2, default=batch._json_default))
    _write_csv(leaderboard_path, top_rows)
    _write_csv(best_sessions_path, best_session_rows)

    print("")
    print(f"Wrote summary JSON: {summary_path}")
    print(f"Wrote leaderboard CSV: {leaderboard_path}")
    print(f"Wrote best-session CSV: {best_sessions_path}")
    print(f"Best weighted MAE: {best_metrics['weighted_mae']:.3f}")
    print(f"Best overall MAE: {best_metrics['overall_mae']:.3f}")
    for exercise, metrics in sorted(best_metrics["exercise_metrics"].items()):
        print(
            f"  {exercise}: mae={metrics['mae']:.3f} "
            f"weight={metrics['weight']:.3f} "
            f"failed={metrics['failed_sessions']}"
        )


if __name__ == "__main__":
    main()
