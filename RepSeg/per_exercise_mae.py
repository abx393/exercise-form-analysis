#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path


FIXED_PERSON_REP_COUNTS = {
    "Abhinav": 15,
    "Salvador": 15,
}


def infer_expected(person: str, session: str) -> int | None:
    if person in FIXED_PERSON_REP_COUNTS:

        print('FIXED GT ', FIXED_PERSON_REP_COUNTS[person])
        return FIXED_PERSON_REP_COUNTS[person]

    match = re.search(r"(?i)(?<!\d)(\d+)x(\d+)(?!\d)", session)
    print('session', session)
    if match:
        sets, reps = map(int, match.groups())
        if 1 <= sets <= 10 and 1 <= reps <= 40:
            #print('GT sets x reps', sets * reps)
            return sets * reps

    match = re.search(r"(?i)(?<!\d)(\d+)x(?:_|-|$)", session)
    if match:
        reps = int(match.group(1))
        if 1 <= reps <= 40:
            #print('GT reps', reps)
            return reps
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute per-exercise MAE from a batch rep report session_summary.csv.")
    parser.add_argument(
        "summary_csv",
        nargs="?",
        default="/Users/nhburke/Desktop/BioSensing/GitHubClone/Outputs/session_summary.csv",
        help="Path to session_summary.csv",
    )
    args = parser.parse_args()

    path = Path(args.summary_csv).expanduser().resolve()
    errs: dict[str, list[int]] = defaultdict(list)
    all_errors: list[int] = []

    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            expected = infer_expected(str(row["person"]), str(row["session"]))
            if expected is None:
                continue
            estimated = int(row["estimated_total_reps"])
            print('estimated', estimated)
            abs_error = abs(estimated - expected)
            print('abs_error', abs_error)
            errs[str(row["exercise"])].append(abs_error)
            all_errors.append(abs_error)

    for exercise in sorted(errs):
        vals = errs[exercise]
        print(f"{exercise}: MAE={sum(vals)/len(vals):.3f} (n={len(vals)})")

    if all_errors:
        print(f"overall: MAE={sum(all_errors)/len(all_errors):.3f} (n={len(all_errors)})")


if __name__ == "__main__":
    main()
