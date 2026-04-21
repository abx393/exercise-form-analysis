"""
fit_to_csv.py
-------------
Converts a Garmin FIT-exported CSV file into a clean time-series CSV with
columns: absolute_timestamp, accel_x, accel_y, accel_z

All timestamps are in seconds (float). Acceleration values are in milli-g.

Usage:
    python fit_to_csv.py <input.csv>
    python fit_to_csv.py <input.csv> --output <output.csv>
"""

import csv
import sys
import argparse
from pathlib import Path


def parse_fit_csv(filepath):
    """
    Parse a Garmin FIT-exported CSV and extract accelerometer samples.

    Handles the export quirk where each row is wrapped in outer quotes
    and inner values are double-quoted (e.g. ""value"").

    Returns a list of (absolute_timestamp, accel_x, accel_y, accel_z) tuples
    sorted by timestamp.
    """
    with open(filepath, newline='', encoding='utf-8-sig') as f:
        raw = f.read()

    # Strip outer quotes and unescape inner "" -> "
    lines = []
    for line in raw.splitlines():
        line = line.strip()
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
        line = line.replace('""', '"')
        lines.append(line)

    samples = []
    reader = csv.reader(lines)

    for row in reader:
        if len(row) < 4:
            continue
        if row[0].strip().lower() != 'data':
            continue
        if row[2].strip().lower() != 'accelerometer_data':
            continue

        # Build field->value dict. Layout: Type, LocalNum, Message,
        # Field1, Value1, Units1, Field2, Value2, Units2, ...
        fields = {}
        col = 3
        while col + 1 < len(row):
            field = row[col].strip()
            value = row[col + 1].strip()
            if field:
                fields[field] = value
            col += 3

        timestamp_s  = fields.get('timestamp')
        timestamp_ms = fields.get('timestamp_ms')
        accel_x_raw  = fields.get('calibrated_accel_x')
        accel_y_raw  = fields.get('calibrated_accel_y')
        accel_z_raw  = fields.get('calibrated_accel_z')
        offsets_raw  = fields.get('sample_time_offset')

        if not all([timestamp_s, accel_x_raw, accel_y_raw,
                    accel_z_raw, offsets_raw]):
            continue

        try:
            ts_s  = int(timestamp_s)
            ts_ms = int(timestamp_ms) if timestamp_ms else 0
            xs    = [float(v) / 1000 for v in accel_x_raw.split('|')]
            ys    = [float(v) / 1000 for v in accel_y_raw.split('|')]
            zs    = [float(v) / 1000 for v in accel_z_raw.split('|')]
            offs  = [int(v)   for v in offsets_raw.split('|')]
        except ValueError:
            continue

        n = min(len(xs), len(ys), len(zs), len(offs))
        for i in range(n):
            t = ts_s + (ts_ms + offs[i]) / 1000.0
            samples.append((t, xs[i], ys[i], zs[i]))

    if not samples:
        raise ValueError(
            "No accelerometer_data rows found. "
            "Check that the file is a Garmin FIT CSV export."
        )

    samples.sort(key=lambda s: s[0])
    return samples


def write_csv(samples, output_path):
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['absolute_timestamp', 'accel_x', 'accel_y', 'accel_z'])
        for t, x, y, z in samples:
            writer.writerow([f'{t:.3f}', f'{x:.6f}', f'{y:.6f}',
                f'{z:.6f}'])

def main():
    parser = argparse.ArgumentParser(
        description='Convert a Garmin FIT CSV export to a clean accelerometer CSV.')
    parser.add_argument('input', help='Garmin FIT-exported CSV file')
    parser.add_argument('--exercise', '-o',
                        help='The exercise the user performed')
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = f'garmin_watch_IMU_{args.exercise}_{input_path.stem}.csv'

    print(f"Input:  {input_path}")
    samples = parse_fit_csv(input_path)

    duration = samples[-1][0] - samples[0][0]
    fs_est   = len(samples) / duration if duration > 0 else 0
    print(f"Parsed: {len(samples):,} samples  |  {duration:.3f} s  |  ~{fs_est:.0f} Hz")

    write_csv(samples, output_path)
    print(f"Output: {output_path}")


if __name__ == '__main__':
    main()
