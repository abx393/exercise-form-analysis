"""
convert_apple_device_csv.py
----------------------------
Converts Apple device accelerometer CSV files exported from apps like
Physics Toolbox or SensorLog into the standard format expected by the
exercise pipeline (classify_exercise.py / plot_multi_accel.py).

Transformations applied:
  1. Rename acceleration columns to accel_x, accel_y, accel_z:
       x, y, z             -> accel_x, accel_y, accel_z
       accelerationX/Y/Z   -> accel_x, accel_y, accel_z
  2. Build absolute_timestamp = time + seconds_elapsed
       The `time` column holds the Unix epoch of the recording start
       (same value for every row). Adding `seconds_elapsed` gives a
       unique, monotonically increasing timestamp per sample.
  3. Drop all columns except absolute_timestamp, accel_x, accel_y, accel_z.

Usage:
    # Single file
    python convert_apple_device_csv.py Accelerometer.csv

    # Multiple files
    python convert_apple_device_csv.py Accelerometer.csv Headphone.csv

    # Custom output path (only valid when converting a single file)
    python convert_apple_device_csv.py Accelerometer.csv -o output.csv

    # Convert all CSVs in a directory tree in-place
    python convert_apple_device_csv.py data/ --recursive
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Column name mappings
# ---------------------------------------------------------------------------

# Maps any recognised source column name -> canonical name.
# Keys are lower-cased for case-insensitive matching.
ACCEL_COL_MAP = {
    'x':             'accel_x',
    'y':             'accel_y',
    'z':             'accel_z',
    'accelerationx': 'accel_x',
    'accelerationy': 'accel_y',
    'accelerationz': 'accel_z',
}

REQUIRED_OUTPUT_COLS = ['absolute_timestamp', 'accel_x', 'accel_y', 'accel_z']


# ---------------------------------------------------------------------------
# Conversion logic
# ---------------------------------------------------------------------------

def convert(input_path: Path, output_path: Path) -> bool:
    """
    Convert a single Apple device CSV file.

    Returns True on success, False if the file could not be converted
    (e.g. missing required columns).
    """
    try:
        df = pd.read_csv(input_path)
    except Exception as e:
        print(f"  ERROR reading {input_path}: {e}")
        return False

    # Lower-case column index for case-insensitive lookup
    col_lower = {c.lower(): c for c in df.columns}

    # ------------------------------------------------------------------
    # 1. Build absolute_timestamp from time + seconds_elapsed
    # ------------------------------------------------------------------
    time_col    = col_lower.get('time')
    elapsed_col = col_lower.get('seconds_elapsed')

    if time_col is None or elapsed_col is None:
        print(f"  ERROR {input_path.name}: missing 'time' or "
              f"'seconds_elapsed' column")
        return False

    # Detect whether `time` is a fixed recording-start epoch (same value for
    # every row, as in Headphone.csv) or already a unique per-row timestamp
    # (as in AccelerometerUncalibrated.csv).
    #
    # If fixed: absolute_timestamp = time + seconds_elapsed converted to ns.
    #   Adding in float64 loses precision because `time` (~1e18) already
    #   exhausts float64's ~15-16 significant digits, so `seconds_elapsed`
    #   vanishes. Fix: cast to int64 and do integer arithmetic.
    # If per-row: use `time` directly — it is already a correct timestamp.
    time_series = df[time_col].astype('int64')
    if time_series.nunique() == 1:
        # Fixed epoch: add seconds_elapsed (converted to ns) as int64
        elapsed_ns = (df[elapsed_col] * 1_000_000_000).round().astype('int64')
        df['absolute_timestamp'] = time_series + elapsed_ns
        print(f"    timestamp mode: fixed epoch + seconds_elapsed")
    else:
        # Per-row timestamps: use time column directly
        df['absolute_timestamp'] = time_series
        print(f"    timestamp mode: per-row time column")

    # ------------------------------------------------------------------
    # 2. Rename acceleration columns
    # ------------------------------------------------------------------
    rename_map = {}
    for src_lower, dst in ACCEL_COL_MAP.items():
        if src_lower in col_lower:
            rename_map[col_lower[src_lower]] = dst

    missing = [d for d in ['accel_x', 'accel_y', 'accel_z']
               if d not in rename_map.values()]
    if missing:
        print(f"  ERROR {input_path.name}: could not find source columns "
              f"for: {missing}")
        print(f"    Available columns: {list(df.columns)}")
        return False

    df = df.rename(columns=rename_map)

    # ------------------------------------------------------------------
    # 3. Keep only the four required output columns
    # ------------------------------------------------------------------
    df = df[REQUIRED_OUTPUT_COLS]

    # ------------------------------------------------------------------
    # 4. Write output
    # ------------------------------------------------------------------
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  {input_path.name} -> {output_path}  ({len(df):,} rows)")
    return True


def output_path_for(input_path: Path, output_arg: Path | None,
                    suffix: str = '_converted') -> Path:
    """
    Derive an output path.
    - If --output was given explicitly, use it (single-file mode only).
    - Otherwise, place the output alongside the input with a suffix added
      before the extension: e.g. Accelerometer.csv -> Accelerometer_converted.csv
    """
    if output_arg is not None:
        return output_arg
    return input_path.with_stem(input_path.stem + suffix)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Convert Apple device accelerometer CSVs to the standard '
                    'pipeline format.')
    parser.add_argument('inputs', nargs='+', metavar='PATH',
                        help='One or more CSV files or directories to convert.')
    parser.add_argument('-o', '--output', metavar='PATH', default=None,
                        help='Output file path. Only valid when converting a '
                             'single file. Defaults to <input>_converted.csv.')
    parser.add_argument('--recursive', action='store_true',
                        help='When a directory is given, recurse into '
                             'subdirectories and convert all CSV files found.')
    parser.add_argument('--suffix', default='_converted',
                        help='Suffix added to the output filename when --output '
                             'is not given (default: _converted).')
    args = parser.parse_args()

    # Collect all input files
    input_files: list[Path] = []
    for raw in args.inputs:
        p = Path(raw)
        if p.is_dir():
            pattern = '**/*.csv' if args.recursive else '*.csv'
            found   = sorted(p.glob(pattern))
            if not found:
                print(f"No CSV files found in {p}")
            input_files.extend(found)
        elif p.is_file():
            input_files.append(p)
        else:
            print(f"WARNING: {p} does not exist, skipping.")

    if not input_files:
        print("No input files to process.")
        sys.exit(1)

    output_arg = Path(args.output) if args.output else None
    if output_arg is not None and len(input_files) > 1:
        parser.error('--output can only be used with a single input file.')

    # Convert
    n_ok, n_fail = 0, 0
    for fpath in input_files:
        out = output_path_for(fpath, output_arg, suffix=args.suffix)
        ok  = convert(fpath, out)
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    print(f"\nDone: {n_ok} converted, {n_fail} failed.")
    if n_fail:
        sys.exit(1)


if __name__ == '__main__':
    main()
