"""
plot_accel.py
-------------
Plots X, Y, Z acceleration from a clean accelerometer CSV file
(as produced by fit_to_csv.py).

Expected CSV columns: absolute_timestamp, accel_x, accel_y, accel_z
  - absolute_timestamp: seconds (float)
  - accel_x/y/z:        milli-g

Usage:
    python plot_accel.py <accel.csv>
    python plot_accel.py <accel.csv> --remove-gravity
    python plot_accel.py <accel.csv> --save-png output.png
"""

import csv
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_csv(filepath):
    times, xs, ys, zs = [], [], [], []
    with open(filepath, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            times.append(float(row['absolute_timestamp']))
            xs.append(float(row['accel_x']))
            ys.append(float(row['accel_y']))
            zs.append(float(row['accel_z']))
    if not times:
        raise ValueError("CSV file is empty or has no data rows.")
    times = np.array(times)
    return times, np.array(xs), np.array(ys), np.array(zs)


def remove_gravity(times, xs, ys, zs, cutoff_hz=0.5):
    dt = np.median(np.diff(times))
    fs = 1.0 / dt if dt > 0 else 100.0
    try:
        from scipy.signal import butter, filtfilt
        nyq  = fs / 2.0
        b, a = butter(2, cutoff_hz / nyq, btype='high')
        xs   = filtfilt(b, a, xs)
        ys   = filtfilt(b, a, ys)
        zs   = filtfilt(b, a, zs)
        print(f"  Gravity removal: Butterworth high-pass (>{cutoff_hz:.1f} Hz, fs={fs:.0f} Hz)")
    except ImportError:
        window = max(3, int(fs * 2))
        def rolling_subtract(arr, w):
            return arr - np.convolve(arr, np.ones(w) / w, mode='same')
        xs = rolling_subtract(xs, window)
        ys = rolling_subtract(ys, window)
        zs = rolling_subtract(zs, window)
        print("  Gravity removal: rolling-mean subtraction (scipy not available)")
    return xs, ys, zs


def plot(times, xs, ys, zs, title, save_path=None):
    sample = times[len(times) // 2]
    if sample >= 1e15:
        times = [t / 1e9 for t in times]
    elif sample >= 1e11:
        times = [t / 1e3 for t in times]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(times, xs, color='#e74c3c', linewidth=0.7, label='accel_x', alpha=0.9)
    ax.plot(times, ys, color='#2ecc71', linewidth=0.7, label='accel_y', alpha=0.9)
    ax.plot(times, zs, color='#3498db', linewidth=0.7, label='accel_z', alpha=0.9)
    ax.set_xlabel("Time (seconds)", fontsize=12)
    ax.set_ylabel("Acceleration (milli-g)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.legend(loc='upper right', fontsize=11)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, which='major', linestyle='--', alpha=0.5)
    ax.grid(True, which='minor', linestyle=':', alpha=0.25)

    duration = times[-1] - times[0] if len(times) > 1 else 0
    fs_est   = len(times) / duration if duration > 0 else 0

    ax.annotate(
        f"{len(times):,} samples  |  {duration:.2f} s  |  ~{fs_est:.0f} Hz",
        xy=(0.01, 0.02), xycoords='axes fraction', fontsize=9, color='gray'
    )
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Plot accelerometer X/Y/Z data from a clean CSV file.')
    parser.add_argument('csv_file', help='Clean accelerometer CSV (from fit_to_csv.py)')
    parser.add_argument('--remove-gravity', action='store_true',
                        help='High-pass filter to remove static gravity')
    parser.add_argument('--save-png', metavar='PATH',
                        help='Save plot as PNG instead of displaying it')
    args = parser.parse_args()

    print(f"Loading: {args.csv_file}")
    times, xs, ys, zs = load_csv(args.csv_file)

    title = f"{Path(args.csv_file).name}"
    if args.remove_gravity:
        xs, ys, zs = remove_gravity(times, xs, ys, zs)
        title += " (gravity removed)"

    plot(times, xs, ys, zs, title=title, save_path=args.save_png)

if __name__ == '__main__':
    main()
