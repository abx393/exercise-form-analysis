"""
plot_multi_accel.py
--------------------
Plots X, Y, Z accelerometer data from 3 devices on a single figure,
synchronized to their overlapping time window. Performs peak and valley
detection on the vector magnitude of each device and annotates them on
every axis subplot:
  ▼  (downward triangle) = peak   (high-acceleration moment)
  ▲  (upward triangle)   = valley (low-acceleration moment)

Rep segmentation is performed on a single primary device (default: device 0,
typically the wrist) using valley-to-valley boundaries. The minimum separation
between valleys is estimated automatically from the autocorrelation of the
low-pass filtered magnitude signal. A bandpass filter is then applied — with
cutoffs derived adaptively from the ACF-estimated rep period — to further
remove DC drift and high-frequency noise before valley detection. This keeps
segmentation clean without discarding within-rep jitter that is useful for
form analysis. The raw low-pass signal is retained for DTW scoring so that
high-frequency deviations in form still contribute to anomaly scores. A manual
override (--min-separation) is available for edge cases. The first N reps
(default: 5) are used as a good-form template. Every rep is scored against
that template using DTW on the raw vector magnitude of each device
independently. Per-device scores are normalised to a common scale and then
combined via a weighted sum (equal weights by default, overridable with
--weights). Reps whose combined score exceeds the threshold are flagged as
anomalous and highlighted in red.

Each device's CSV must have at minimum these columns:
    absolute_timestamp, accel_x, accel_y, accel_z

Timestamp units and epochs are detected automatically:
  - Seconds      (~1e9):  treated as Unix epoch seconds
  - Milliseconds (~1e12): divided by 1e3
  - Nanoseconds  (~1e18): divided by 1e9
  - Garmin epoch (~1e9 but resolves to pre-2010): offset by +631065600 s

Usage:
    python plot_multi_accel.py <file1.csv> <file2.csv> <file3.csv> \\
        --labels "Bose" "Garmin" "Samsung" \\
        --primary 1 \\
        --template-reps 5 \\
        --anomaly-threshold 2.0 \\
        --weights 0.5 0.3 0.2 \\
        --save-png output.png
"""

import argparse
import datetime
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from signal_utils import (
    detect_and_convert_timestamps,
    load_signal_csv,
    load_recording,
    compute_sync_window,
    trim_to_window,
    estimate_rep_period_acf,
    bandpass_filter,
    detect_peaks_valleys,
    segment_reps,
)


# ---------------------------------------------------------------------------
# CSV loading  (simple path: files already have absolute_timestamp + accel_x/y/z)
# ---------------------------------------------------------------------------

def load_device_csv(filepath):
    """
    Load a pre-converted accelerometer CSV with absolute_timestamp, accel_x/y/z.
    Returns (timestamps_unix_s, xs, ys, zs) sorted by timestamp.
    Falls back to load_signal_csv from signal_utils for other formats.
    """
    result = load_signal_csv(filepath, sensor='accel')
    if result is None:
        raise ValueError(f"Could not load accelerometer data from {filepath}")
    return result


# ---------------------------------------------------------------------------
# DTW
# ---------------------------------------------------------------------------

def dtw_distance(a, b):
    """
    Compute the DTW distance between two 1-D signals using raw amplitude values.

    The total accumulated cost is divided by the warping path length so that
    longer reps are not penalised purely for having more samples (length
    normalisation). No amplitude normalisation is applied: differences in
    peak height are a meaningful bad-form signal and should contribute to
    the score.

    Parameters
    ----------
    a, b : 1-D numpy arrays

    Returns
    -------
    float : length-normalised DTW distance
    """
    n, m = len(a), len(b)
    # Initialise cost matrix with infinity
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost    = abs(float(a[i - 1]) - float(b[j - 1]))
            D[i, j] = cost + min(D[i - 1, j],      # insertion
                                  D[i, j - 1],      # deletion
                                  D[i - 1, j - 1])  # match

    # Trace back path length for normalisation
    i, j = n, m
    path_len = 0
    while i > 0 or j > 0:
        path_len += 1
        if i == 0:
            j -= 1
        elif j == 0:
            i -= 1
        else:
            step = np.argmin([D[i - 1, j], D[i, j - 1], D[i - 1, j - 1]])
            if step == 0:
                i -= 1
            elif step == 1:
                j -= 1
            else:
                i -= 1
                j -= 1

    return D[n, m] / max(path_len, 1)


# ---------------------------------------------------------------------------
# Template building
# ---------------------------------------------------------------------------

def build_template(segments, mag_f, n_template=5):
    """
    Build a good-form template from the first `n_template` rep segments.

    Uses the medoid strategy: picks the single rep (among the template reps)
    that has the lowest total DTW distance to all others. This avoids
    averaging artifacts while still being representative of the group.

    Parameters
    ----------
    segments   : list of (start_idx, end_idx) tuples from segment_reps()
    mag_f      : filtered magnitude array
    n_template : number of early reps to treat as good-form examples

    Returns
    -------
    template : 1-D numpy array — the medoid rep's magnitude signal
    """
    n = min(n_template, len(segments))
    if n < 1:
        raise ValueError("Need at least 1 rep to build a template.")

    segs = [mag_f[s:e] for s, e in segments[:n]]

    if n == 1:
        return segs[0].copy()

    # Compute pairwise DTW distances among the template reps
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = dtw_distance(segs[i], segs[j])
            dist_matrix[i, j] = d
            dist_matrix[j, i] = d

    # Medoid = rep with lowest sum of distances to all others
    medoid_idx = int(np.argmin(dist_matrix.sum(axis=1)))
    print(f"    Template medoid: rep {medoid_idx + 1} of {n} "
          f"(total DTW sum = {dist_matrix[medoid_idx].sum():.4f})")
    return segs[medoid_idx].copy()


# ---------------------------------------------------------------------------
# Rep scoring
# ---------------------------------------------------------------------------

def score_reps(segments, devices, templates, weights,
               n_template=5, threshold_sigma=2.0):
    """
    Score every rep against the good-form template for each device, then
    combine into a single weighted anomaly score.

    Per-device scores are computed on raw amplitude (preserving depth-of-rep
    information), then normalised to zero-mean / unit-variance across reps
    before weighting. Normalisation is done per device so that a device whose
    magnitude signal naturally lives on a larger scale (e.g. wrist vs.
    headphones) does not dominate the combined score simply due to scale.

    The anomaly threshold is person-adaptive: it is set at
    `threshold_sigma` standard deviations above the mean combined score of
    the template reps themselves.

    Parameters
    ----------
    segments        : list of (start_idx, end_idx) tuples
    devices         : list of device dicts (must contain 'mag_f' and 'label')
    templates       : list of 1-D arrays, one template per device,
                      from build_template()
    weights         : 1-D array of per-device weights (normalised to sum-to-1
                      internally)
    n_template      : number of template reps (used to compute threshold)
    threshold_sigma : how many sigma above template-rep mean marks an anomaly

    Returns
    -------
    per_device_scores : 2-D array, shape (n_devices, n_reps) - raw DTW
                        distances per device before normalisation
    combined_scores   : 1-D array, shape (n_reps,) - weighted combined score
    threshold         : scalar anomaly threshold on the combined score
    anomalous         : boolean array, True where combined score > threshold
    """
    n_reps    = len(segments)
    n_devices = len(devices)

    # --- Raw DTW scores: shape (n_devices, n_reps) ---
    raw = np.zeros((n_devices, n_reps))
    for d_i, (dev, tmpl) in enumerate(zip(devices, templates)):
        mag_f = dev['mag_f']
        for r_i, (s, e) in enumerate(segments):
            raw[d_i, r_i] = dtw_distance(mag_f[s:e], tmpl)

    # --- Per-device z-score normalisation across reps ---
    # Equalises scale across devices without discarding amplitude information
    # within a device: each device's scores are centred and scaled by their
    # own cross-rep distribution, not by their raw amplitude units.
    normed = np.zeros_like(raw)
    for d_i in range(n_devices):
        #mu    = float(np.mean(raw[d_i]))
        sigma = float(np.std(raw[d_i]))
        sigma = max(sigma, 1e-6)
        #normed[d_i] = (raw[d_i] - mu) / sigma
        normed[d_i] = raw[d_i] / sigma

    # --- Weighted sum across devices ---
    w = np.array(weights, dtype=float)
    w = w / w.sum()                                       # normalise to sum-to-1
    combined = (normed * w[:, np.newaxis]).sum(axis=0)    # shape (n_reps,)

    # --- Person-adaptive threshold from template reps ---
    template_combined = combined[:min(n_template, n_reps)]
    t_mu    = float(np.mean(template_combined))
    t_sigma = float(np.std(template_combined))
    t_sigma = max(t_sigma, 1e-6)
    threshold = t_mu + threshold_sigma * t_sigma

    anomalous = combined > threshold
    return raw, combined, threshold, anomalous


def plot_bandpass(devices, segments, anomalous, primary_label,
                  save_path=None):
    """
    Diagnostic figure showing three filter layers for each device:
      - Raw vector magnitude (light grey)
      - Low-pass filtered magnitude (blue)
      - Bandpass filtered magnitude (orange, used for segmentation)

    Valley markers are drawn on the bandpass signal to show exactly what
    the segmentation algorithm sees. Rep shading is also applied so you
    can verify that valley boundaries align correctly with the movement.

    One subplot per device, all sharing the same x-axis.

    Parameters
    ----------
    devices      : list of ordered device dicts (must contain mag_f, mag_bp,
                   times, xs, ys, zs, valley_times, bp_low_hz, bp_high_hz)
    segments     : list of (start_idx, end_idx) rep boundary tuples
    anomalous    : boolean array (n_reps,)
    primary_label: label of the primary device (for title annotation)
    save_path    : if given, save PNG here instead of displaying
    """
    n         = len(devices)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    fig.suptitle(
        "Bandpass filter diagnostic  —  "
        "grey=raw mag · blue=low-pass · orange=bandpass (used for segmentation)",
        fontsize=12, fontweight='bold')

    primary_times = np.array(devices[0]['times'])

    for ax, dev in zip(axes, devices):
        times = np.array(dev['times'])
        xs    = np.array(dev['xs'])
        ys    = np.array(dev['ys'])
        zs    = np.array(dev['zs'])
        mag_raw = np.sqrt(xs**2 + ys**2 + zs**2)
        mag_f   = np.array(dev['mag_f'])
        mag_bp  = np.array(dev['mag_bp'])

        # Rep shading
        for rep_i, (s, e) in enumerate(segments):
            t_s = primary_times[s]
            t_e = primary_times[min(e, len(primary_times) - 1)]
            color = '#e74c3c' if anomalous[rep_i] else '#2ecc71'
            ax.axvspan(t_s, t_e, color=color, alpha=0.10, zorder=0)
            ax.text((t_s + t_e) / 2, 1.0, str(rep_i + 1),
                    transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=7, color='#555555')

        # Three signal layers
        ax.plot(times, mag_raw, color='#bdc3c7', linewidth=0.6,
                alpha=0.8, label='Raw magnitude', zorder=1)
        ax.plot(times, mag_f,   color='#2980b9', linewidth=0.9,
                alpha=0.85, label='Low-pass', zorder=2)
        ax.plot(times, mag_bp,  color='#e67e22', linewidth=1.1,
                alpha=0.95, label='Bandpass (segmentation)', zorder=3)

        # Valley markers on bandpass signal
        valley_times = np.array(dev['valley_times'])
        if len(valley_times) > 0:
            valley_bp_vals = np.interp(valley_times, times, mag_bp)
            y_span         = mag_raw.max() - mag_raw.min()
            offset         = y_span * 0.06
            ax.scatter(valley_times,
                       valley_bp_vals - offset,
                       marker='^', color='#c0392b', s=55, zorder=5,
                       label=f'Valley ({len(valley_times)})')

        primary_tag = '  \u2605 primary' if dev['label'] == primary_label else ''
        ax.set_title(
            f"{dev['label']}{primary_tag}   "
            f"bandpass [{dev['bp_low_hz']:.2f} – {dev['bp_high_hz']:.2f}] Hz",
            fontsize=11, loc='left')
        ax.set_ylabel("Magnitude\n(native units)", fontsize=9)
        ax.legend(loc='upper right', fontsize=9, ncol=4)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(True, which='major', linestyle='--', alpha=0.5)
        ax.grid(True, which='minor', linestyle=':', alpha=0.25)

    axes[-1].set_xlabel("Time (seconds from sync point)", fontsize=11)
    axes[-1].xaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved bandpass plot: {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main signal + anomaly plot
# ---------------------------------------------------------------------------

def plot_all(devices, segments, per_device_scores, combined_scores,
             threshold, anomalous, weights, primary_label, save_path=None):
    """
    One subplot per device showing X/Y/Z lines, peak/valley markers, and
    rep shading. A final subplot shows a stacked bar chart of the per-device
    weighted DTW contribution to the combined anomaly score.

    Rep shading:
      green = good-form rep (combined score within threshold)
      red   = anomalous rep (combined score exceeds threshold)

    Peaks   -> inverted triangle (v) above the signal
    Valleys -> upright triangle  (^) below the signal
    All device subplots share the same x-axis.

    Parameters
    ----------
    devices           : list of device dicts (label, times, xs, ys, zs, ...)
    segments          : list of (start_idx, end_idx) rep boundary tuples
    per_device_scores : 2-D array (n_devices, n_reps) of raw DTW distances
    combined_scores   : 1-D array (n_reps,) of weighted combined scores
    threshold         : scalar anomaly threshold on combined_scores
    anomalous         : boolean array (n_reps,)
    weights           : 1-D array of normalised per-device weights
    primary_label     : label string of the primary (segmentation) device
    save_path         : if given, save PNG here instead of displaying
    """
    axis_colors  = {'X': '#e74c3c', 'Y': '#2ecc71', 'Z': '#3498db'}
    peak_color   = '#c0392b'
    valley_color = '#27ae60'

    # Distinct colours for device bars in the stacked chart
    device_bar_colors = [
        '#3498db', '#e67e22', '#9b59b6',
        '#1abc9c', '#e74c3c', '#f1c40f',
    ]

    n_subplots = len(devices) + 1   # device signal axes + stacked score chart
    fig, axes  = plt.subplots(n_subplots, 1,
                               figsize=(14, 4 * n_subplots),
                               sharex=False)
    device_axes = axes[:len(devices)]
    score_ax    = axes[-1]

    # Share x-axis across device signal subplots only
    for i in range(len(devices) - 1):
        device_axes[i].get_shared_x_axes().joined(
            device_axes[i], device_axes[i + 1])

    fig.suptitle("Multi-Device Accelerometer Comparison (synchronised)",
                 fontsize=14, fontweight='bold')

    # Primary device times used to map segment indices to time values
    primary_times = np.array(devices[0]['times'])   # device 0 is always primary

    def shade_reps(ax):
        """Draw per-rep background shading and rep-number labels."""
        for rep_i, (s, e) in enumerate(segments):
            t_s = primary_times[s]
            t_e = primary_times[min(e, len(primary_times) - 1)]
            color = '#e74c3c' if anomalous[rep_i] else '#2ecc71'
            ax.axvspan(t_s, t_e, color=color, alpha=0.13, zorder=0)
            ax.text((t_s + t_e) / 2, 1.0, str(rep_i + 1),
                    transform=ax.get_xaxis_transform(),
                    ha='center', va='bottom', fontsize=7, color='#555555')

    # ------------------------------------------------------------------
    # Device signal subplots
    # ------------------------------------------------------------------
    for ax, dev in zip(device_axes, devices):
        times        = np.array(dev['times'])
        xs           = np.array(dev['xs'])
        ys           = np.array(dev['ys'])
        zs           = np.array(dev['zs'])
        peak_times   = dev['peak_times']
        valley_times = dev['valley_times']

        shade_reps(ax)

        ax.plot(times, xs, color=axis_colors['X'],
                linewidth=0.7, alpha=0.9, label='X')
        ax.plot(times, ys, color=axis_colors['Y'],
                linewidth=0.7, alpha=0.9, label='Y')
        ax.plot(times, zs, color=axis_colors['Z'],
                linewidth=0.7, alpha=0.9, label='Z')

        y_min = min(xs.min(), ys.min(), zs.min())
        y_max = max(xs.max(), ys.max(), zs.max())
        y_span = y_max - y_min
        marker_offset = y_span * 0.06

        if len(peak_times) > 0:
            ax.scatter(peak_times,
                       np.full_like(peak_times, y_max + marker_offset),
                       marker='v', color=peak_color, s=60, zorder=5,
                       label=f'Peak ({len(peak_times)})')

        if len(valley_times) > 0:
            ax.scatter(valley_times,
                       np.full_like(valley_times, y_min - marker_offset),
                       marker='^', color=valley_color, s=60, zorder=5,
                       label=f'Valley ({len(valley_times)})')

        ax.set_ylim(y_min - y_span * 0.15, y_max + y_span * 0.15)

        dur = times[-1] - times[0] if len(times) > 1 else 1
        fs  = len(times) / dur
        primary_tag = '  \u2605 primary' if dev['label'] == primary_label else ''
        ax.set_title(
            f"{dev['label']}{primary_tag}   "
            f"({len(times):,} samples, ~{fs:.0f} Hz)",
            fontsize=11, loc='left')
        ax.set_ylabel("Acceleration\n(native units)", fontsize=9)
        ax.legend(loc='upper right', fontsize=9, ncol=5)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(True, which='major', linestyle='--', alpha=0.5)
        ax.grid(True, which='minor', linestyle=':', alpha=0.25)
        ax.set_xlabel("Time (seconds from sync point)", fontsize=9)
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    # ------------------------------------------------------------------
    # Stacked bar chart: per-device weighted contribution to combined score
    # ------------------------------------------------------------------
    # The stacked bars show each device's weighted normalised score.
    # Because normalised scores can be negative (below the cross-rep mean),
    # we shift everything up by the minimum so bars always start at zero and
    # the stacking remains visually interpretable. The threshold line is
    # shifted by the same offset.
    n_devices = len(devices)
    n_reps    = len(combined_scores)
    rep_nums  = np.arange(1, n_reps + 1)

    w = np.array(weights, dtype=float)
    w = w / w.sum()

    # Recompute normalised scores for plotting (same logic as score_reps)
    normed = np.zeros_like(per_device_scores)
    for d_i in range(n_devices):
        mu    = float(np.mean(per_device_scores[d_i]))
        sigma = float(np.std(per_device_scores[d_i]))
        sigma = max(sigma, 1e-6)
        #normed[d_i] = (per_device_scores[d_i] - mu) / sigma
        normed[d_i] = per_device_scores[d_i] / sigma

    weighted = normed * w[:, np.newaxis]   # (n_devices, n_reps)

    # Shift so the minimum combined value sits at y=0
    #shift     = combined_scores.min()
    shift = 0
    threshold_shifted = threshold - shift

    bottoms = np.zeros(n_reps)
    for d_i, dev in enumerate(devices):
        bar_vals = weighted[d_i] - shift / n_devices   # distribute shift evenly
        color    = device_bar_colors[d_i % len(device_bar_colors)]
        score_ax.bar(rep_nums, bar_vals, bottom=bottoms,
                     color=color, edgecolor='white', linewidth=0.4,
                     alpha=0.85, zorder=3,
                     label=f"{dev['label']} (w={w[d_i]:.2f})")
        bottoms += bar_vals

    # Threshold line and anomaly highlights
    score_ax.axhline(threshold_shifted, color='#c0392b', linewidth=1.5,
                     linestyle='--', zorder=5,
                     label=f'Anomaly threshold ({threshold:.3f})')

    for rep_i in range(n_reps):
        if anomalous[rep_i]:
            score_ax.axvspan(rep_i + 0.5, rep_i + 1.5,
                             color='#e74c3c', alpha=0.08, zorder=0)

    score_ax.set_xlabel("Rep number", fontsize=11)
    score_ax.set_ylabel("Weighted normalised\nDTW score", fontsize=9)
    score_ax.set_title(
        f"Per-rep DTW anomaly score (stacked by device)  —  "
        f"{int(anomalous.sum())} of {n_reps} reps flagged",
        fontsize=11, loc='left')
    score_ax.set_xticks(rep_nums)
    score_ax.legend(loc='upper left', fontsize=9, ncol=n_devices + 1)
    score_ax.grid(True, which='major', linestyle='--', alpha=0.5, axis='y')
    score_ax.set_xlim(0.5, n_reps + 0.5)

    # Colour x-tick labels red for anomalous reps
    for tick, a in zip(score_ax.get_xticklabels(), anomalous):
        tick.set_color('#c0392b' if a else '#333333')

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Saved: {save_path}")
    else:
        plt.show()

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Plot synchronised accelerometer data from multiple '
                    'devices with rep segmentation and DTW-based anomaly '
                    'detection.')
    parser.add_argument('files', nargs='+', metavar='CSV',
                        help='Accelerometer CSV files (at least 1)')
    parser.add_argument('--labels', nargs='+', metavar='LABEL',
                        default=None,
                        help='Display labels for each device '
                             '(default: filenames)')
    parser.add_argument('--lowpass-hz', type=float, default=5.0,
                        help='Low-pass filter cutoff for peak detection '
                             '(default: 5 Hz)')
    parser.add_argument('--min-separation', type=float, default=None,
                        help='Minimum seconds between peaks/valleys. When '
                             'omitted (default), the value is derived '
                             'automatically from the autocorrelation of the '
                             'filtered magnitude signal, making the script '
                             'exercise-agnostic. Supply this flag to override '
                             'the ACF estimate for edge cases.')
    parser.add_argument('--prominence', type=float, default=0.5,
                        help='Peak prominence threshold as a multiple of '
                             'signal IQR (default: 0.5)')
    parser.add_argument('--primary', type=int, default=0,
                        help='Index of the device to use for rep segmentation '
                             '(default: 0, i.e. first file)')
    parser.add_argument('--template-reps', type=int, default=5,
                        help='Number of early reps to use as good-form '
                             'template (default: 5)')
    parser.add_argument('--anomaly-threshold', type=float, default=2.0,
                        help='Anomaly threshold in standard deviations above '
                             'template-rep mean DTW score (default: 2.0)')
    parser.add_argument('--weights', nargs='+', type=float, default=None,
                        metavar='W',
                        help='Per-device weights for the combined DTW score '
                             '(one value per file, e.g. --weights 0.5 0.3 0.2). '
                             'Values are normalised to sum to 1 internally. '
                             'Default: equal weights across all devices.')
    parser.add_argument('--trim-margin', type=float, default=2.0,
                        help='Seconds to trim from each end of the sync '
                             'window to remove startup/shutdown noise '
                             '(default: 2.0)')
    parser.add_argument('--save-png', metavar='PATH',
                        help='Save main plot as PNG instead of displaying it')
    parser.add_argument('--save-bp-png', metavar='PATH',
                        help='Save bandpass diagnostic plot as PNG. '
                             'If omitted the plot is shown interactively '
                             '(or skipped if --save-png is also omitted and '
                             'you prefer not to see it).')
    args = parser.parse_args()

    labels = args.labels or [Path(f).stem for f in args.files]
    if len(labels) != len(args.files):
        parser.error('--labels must have the same number of entries as files')

    if args.primary >= len(args.files):
        parser.error(f'--primary {args.primary} out of range '
                     f'(only {len(args.files)} files provided)')

    # Resolve weights: default to equal; validate length if provided
    if args.weights is None:
        weights = [1.0] * len(args.files)
    else:
        if len(args.weights) != len(args.files):
            parser.error(f'--weights must have the same number of entries as '
                         f'files ({len(args.files)}), got {len(args.weights)}')
        if any(w < 0 for w in args.weights):
            parser.error('--weights values must be non-negative')
        weights = args.weights
    weights = np.array(weights, dtype=float)
    weights = weights / weights.sum()   # normalise once here for display

    # ------------------------------------------------------------------
    # Load all devices
    # ------------------------------------------------------------------
    all_data = []
    for filepath, label in zip(args.files, labels):
        print(f"\nLoading [{label}]: {filepath}")
        ts, xs, ys, zs = load_device_csv(filepath)
        print(f"    {len(ts):,} samples  |  "
              f"{ts[0]:.3f} → {ts[-1]:.3f}  ({ts[-1]-ts[0]:.1f} s)")
        all_data.append({'label': label,
                         'ts': ts, 'xs': xs, 'ys': ys, 'zs': zs})

    # ------------------------------------------------------------------
    # Find overlapping window with trim margin
    # ------------------------------------------------------------------
    all_signals = {d['label']: {'ts': d['ts']} for d in all_data}
    t_start, t_end = compute_sync_window(all_signals,
                                          trim_margin_s=args.trim_margin)

    dt_start = datetime.datetime.fromtimestamp(t_start, tz=datetime.timezone.utc)
    print(f"\nSync window: {t_end - t_start:.2f} s  "
          f"(from {dt_start.strftime('%Y-%m-%d %H:%M:%S UTC')})")

    # ------------------------------------------------------------------
    # Trim, normalise time, detect peaks/valleys for every device
    # ------------------------------------------------------------------
    devices = []
    for d in all_data:
        ts, xs, ys, zs = trim_to_window(
            d['ts'], d['xs'], d['ys'], d['zs'], t_start, t_end)
        t0    = ts[0]
        times = ts - t0

        peak_times, valley_times, valley_idx, mag_f, mag_bp, \
            bp_low_hz, bp_high_hz, min_sep_used = \
            detect_peaks_valleys(
                times, xs, ys, zs,
                lowpass_hz=args.lowpass_hz,
                min_separation_s=args.min_separation,
                prominence_factor=args.prominence,
            )

        dur = times[-1] if len(times) > 1 else 0
        fs  = len(times) / dur if dur > 0 else 0
        print(f"  [{d['label']}] {len(times):,} samples (~{fs:.0f} Hz)  |  "
              f"min_sep={min_sep_used:.2f}s  |  "
              f"bp=[{bp_low_hz:.2f},{bp_high_hz:.2f}] Hz  |  "
              f"{len(peak_times)} peaks, {len(valley_times)} valleys")

        devices.append({'label':        d['label'],
                        'times':        times,
                        'xs':           xs,
                        'ys':           ys,
                        'zs':           zs,
                        'peak_times':   peak_times,
                        'valley_times': valley_times,
                        'valley_idx':   valley_idx,
                        'mag_f':        mag_f,
                        'mag_bp':       mag_bp,
                        'bp_low_hz':    bp_low_hz,
                        'bp_high_hz':   bp_high_hz})

    # ------------------------------------------------------------------
    # Rep segmentation on primary device
    # ------------------------------------------------------------------
    primary   = devices[args.primary]
    p_label   = primary['label']
    p_valley  = primary['valley_idx']
    p_mag_f   = primary['mag_f']
    p_times   = primary['times']

    print(f"\nSegmenting reps on primary device: [{p_label}]")
    segments = segment_reps(p_valley,
                             n_samples=len(p_mag_f),
                             n_template=args.template_reps)

    if len(segments) < args.template_reps + 1:
        raise ValueError(
            f"Only {len(segments)} rep(s) detected on [{p_label}]. "
            f"Need at least {args.template_reps + 1} "
            f"({args.template_reps} template + 1 to score).")

    print(f"    {len(segments)} reps detected")
    for i, (s, e) in enumerate(segments):
        dur = p_times[min(e, len(p_times)-1)] - p_times[s]
        print(f"      Rep {i+1:2d}: t={p_times[s]:.2f}s – "
              f"{p_times[min(e, len(p_times)-1)]:.2f}s  ({dur:.2f}s)")

    # ------------------------------------------------------------------
    # Build one template per device, then score all reps across devices
    # ------------------------------------------------------------------
    # Re-order devices so primary is first (plot_all expects this)
    ordered_devices = (
        [devices[args.primary]] +
        [d for i, d in enumerate(devices) if i != args.primary]
    )
    # Weights must follow the same reordering
    ordered_weights = np.concatenate([
        weights[args.primary:args.primary + 1],
        np.delete(weights, args.primary)
    ])

    print(f"\nBuilding good-form templates from first "
          f"{args.template_reps} reps (one per device) ...")
    templates = []
    for dev in ordered_devices:
        print(f"  [{dev['label']}]")
        tmpl = build_template(segments, dev['mag_f'],
                               n_template=args.template_reps)
        templates.append(tmpl)

    print("\nScoring all reps across all devices ...")
    print(f"  Weights: " +
          ", ".join(f"{d['label']}={w:.3f}"
                    for d, w in zip(ordered_devices, ordered_weights)))

    per_device_scores, combined_scores, threshold, anomalous = score_reps(
        segments=segments,
        devices=ordered_devices,
        templates=templates,
        weights=ordered_weights,
        n_template=args.template_reps,
        threshold_sigma=args.anomaly_threshold,
    )

    tp, fp, tn, fn = 0, 0, 0, 0

    print(f"  Anomaly threshold (combined): {threshold:.4f}")
    header = "  Rep  |  Combined  |  " + "  ".join(
        f"{d['label']:>10}" for d in ordered_devices)
    print(header)
    print("  " + "-" * (len(header) - 2))
    for i in range(len(combined_scores)):
        device_cols = "  ".join(
            f"{per_device_scores[d_i, i]:>10.4f}"
            for d_i in range(len(ordered_devices)))
        flag = "  <- ANOMALOUS" if anomalous[i] else ""
        if i >= 10:
            if anomalous[i]:
                tp += 1
            else:
                fn += 1
        else:
            if anomalous[i]:
                fp += 1
            else:
                tn += 1

        print(f"  {i+1:3d}  |  {combined_scores[i]:8.4f}  |  "
              f"{device_cols}{flag}")
    print('TP', tp, 'FP', fp, 'TN', tn, 'FN', fn)

    # ------------------------------------------------------------------
    # Bandpass diagnostic plot
    # ------------------------------------------------------------------
    plot_bandpass(ordered_devices, segments, anomalous,
                  primary_label=p_label,
                  save_path=args.save_bp_png)

    # ------------------------------------------------------------------
    # Main signal + anomaly plot
    # ------------------------------------------------------------------
    plot_all(ordered_devices, segments,
             per_device_scores, combined_scores,
             threshold, anomalous,
             weights=ordered_weights,
             primary_label=p_label,
             save_path=args.save_png)


if __name__ == '__main__':
    main()
