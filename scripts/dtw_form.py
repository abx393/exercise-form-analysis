"""
dtw_form.py
-----------
Analyses exercise form deviation using DTW (Dynamic Time Warping) on
multi-device accelerometer data.

Two operating modes:

  File mode  (default): provide CSV files directly on the command line.
      python dtw_form.py file1.csv file2.csv file3.csv \\
          --labels "Bose" "Garmin" "Samsung" --primary 1

  Directory mode: provide a data directory instead with --data-dir.
      python dtw_form.py --data-dir <data_dir> \\
          --primary-device watch

      Directory structure expected:
          <data_dir>/
              <exercise>/
                  <subject>/
                      <recording_id>/
                          <sensor_files>

      Each recording is analysed independently. Results are printed per
      recording and plots are saved to --save-dir.

Rep segmentation uses valley-to-valley boundaries on the primary device.
The minimum separation is derived automatically from the ACF of the
low-pass filtered magnitude. A bandpass filter then further cleans the
signal before valley detection.

The first N reps (--template-reps, default 5) are treated as good form.
Every rep is scored against the DTW template on each device independently.
Per-device scores are ratio-normalised (divided by the mean template-rep
score for that device), then combined via a weighted sum. A score of 1.0
means the rep matches the good-form template exactly; higher means worse.
Scores are always >= 0 and never cancel across devices.

Timestamp formats, column naming conventions, and device inference all
follow the same rules as classify_exercise.py and autoencoder_form.py.

Usage (file mode):
    python dtw_form.py <file1.csv> ... \\
        --labels "Bose" "Garmin" "Samsung" \\
        --primary 1 \\
        --template-reps 5 \\
        --anomaly-threshold 2.0 \\
        --weights 0.5 0.3 0.2 \\
        --save-png output.png

Usage (directory mode):
    python dtw_form.py --data-dir <data_dir> \\
        --primary-device watch \\
        --save-dir ./results
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

    Normalisation strategy: ratio normalisation per device.
    Each device's raw DTW scores are divided by the mean DTW score of
    its own template reps. This maps the good-form baseline to 1.0 on
    every device, so a score of 1.0 means "matches the template exactly"
    and higher means worse. Scores are always >= 0, so a below-average rep
    on one device can never cancel an above-average rep on another.
    Amplitude differences across reps within the same device are fully
    preserved — a shallow rep that scores 1.8 on the watch still scores
    1.8 after normalisation.

    Parameters
    ----------
    segments        : list of (start_idx, end_idx) tuples
    devices         : list of device dicts (must contain 'mag_f' and 'label')
    templates       : list of 1-D arrays, one template per device,
                      from build_template()
    weights         : 1-D array of per-device weights (normalised to sum-to-1
                      internally)
    n_template      : number of template reps (used to compute normalisation
                      reference and anomaly threshold)
    threshold_sigma : how many sigma above template-rep mean marks an anomaly

    Returns
    -------
    per_device_scores : 2-D array, shape (n_devices, n_reps) — raw DTW
                        distances per device before normalisation
    combined_scores   : 1-D array, shape (n_reps,) — weighted combined score
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
            seg = mag_f[s:e]
            if len(seg) < 2 or len(tmpl) < 2:
                raw[d_i, r_i] = np.nan
            else:
                raw[d_i, r_i] = dtw_distance(seg, tmpl)

    # --- Ratio normalisation: divide by mean of template-rep scores ---
    # Each device's scores are divided by its own good-form baseline so
    # that 1.0 = good form on every device, regardless of the device's
    # native magnitude scale.  Scores remain non-negative.
    # NaN entries (empty rep windows) are kept as NaN and treated as 0
    # in the weighted sum so they don't inflate other devices' contributions.
    normed = np.zeros_like(raw)
    for d_i in range(n_devices):
        template_mean = float(
            np.nanmean(raw[d_i, :min(n_template, n_reps)]))
        template_mean = max(template_mean, 1e-8)
        normed[d_i]   = raw[d_i] / template_mean
    # Replace NaN with 0 so they contribute nothing to the weighted sum
    normed = np.where(np.isfinite(normed), normed, 0.0)

    # --- Weighted sum across devices ---
    w = np.array(weights, dtype=float)
    w = w / w.sum()                                       # normalise to sum-to-1
    combined = (normed * w[:, np.newaxis]).sum(axis=0)    # shape (n_reps,)

    # --- Person-adaptive threshold from template reps ---
    # Template reps should all be close to 1.0 after ratio normalisation;
    # the threshold is set relative to their actual spread.
    template_combined = combined[:min(n_template, n_reps)]
    t_mu    = float(np.nanmean(template_combined))
    t_sigma = float(np.nanstd(template_combined))
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
             threshold, anomalous, weights, primary_label,
             title_prefix='', save_path=None):
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
    for i in range(1, len(devices)):
        device_axes[i].sharex(device_axes[0])

    fig.suptitle(f"{title_prefix}Multi-Device Accelerometer — DTW Form Analysis",
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
    # With ratio normalisation, scores are always >= 0 and anchor at 1.0
    # for a perfect good-form rep. No shift is needed; bars stack cleanly
    # from zero. The threshold line sits at its true combined value.
    n_devices = len(devices)
    n_reps    = len(combined_scores)
    rep_nums  = np.arange(1, n_reps + 1)

    w = np.array(weights, dtype=float)
    w = w / w.sum()

    # Recompute ratio-normalised scores for plotting (mirrors score_reps)
    normed = np.zeros_like(per_device_scores)
    for d_i in range(n_devices):
        n_tmpl        = min(5, n_reps)   # use up to 5 template reps
        template_mean = float(np.mean(per_device_scores[d_i, :n_tmpl]))
        template_mean = max(template_mean, 1e-8)
        normed[d_i]   = per_device_scores[d_i] / template_mean

    weighted = normed * w[:, np.newaxis]   # (n_devices, n_reps)

    bottoms = np.zeros(n_reps)
    for d_i, dev in enumerate(devices):
        color = device_bar_colors[d_i % len(device_bar_colors)]
        score_ax.bar(rep_nums, weighted[d_i], bottom=bottoms,
                     color=color, edgecolor='white', linewidth=0.4,
                     alpha=0.85, zorder=3,
                     label=f"{dev['label']} (w={w[d_i]:.2f})")
        print('dev', dev)
        print('weighted[d_i]', weighted[d_i])
        bottoms += weighted[d_i]

    # Threshold line and anomaly highlights
    score_ax.axhline(threshold, color='#c0392b', linewidth=1.5,
                     linestyle='--', zorder=5,
                     label=f'Anomaly threshold ({threshold:.3f})')

    for rep_i in range(n_reps):
        if anomalous[rep_i]:
            score_ax.axvspan(rep_i + 0.5, rep_i + 1.5,
                             color='#e74c3c', alpha=0.08, zorder=0)

    # Reference line at 1.0 = perfect good-form match
    score_ax.axhline(1.0, color='#27ae60', linewidth=0.8,
                     linestyle=':', zorder=4, label='Good-form baseline (1.0)')

    score_ax.set_xlabel("Rep number", fontsize=11)
    score_ax.set_ylabel("Weighted ratio-normalised\nDTW score  (1.0 = template)",
                        fontsize=9)
    score_ax.set_title(
        f"Per-rep DTW anomaly score (stacked by device)  —  "
        f"{int(anomalous.sum())} of {n_reps} reps flagged",
        fontsize=11, loc='left')
    score_ax.set_xticks(rep_nums)
    score_ax.legend(loc='upper left', fontsize=9, ncol=n_devices + 2)
    score_ax.grid(True, which='major', linestyle='--', alpha=0.5, axis='y')
    score_ax.set_xlim(0.5, n_reps + 0.5)
    score_ax.set_ylim(bottom=0)

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
# Core pipeline (shared by both modes)
# ---------------------------------------------------------------------------

def run_recording(all_data, labels, primary_idx, weights,
                  template_reps, anomaly_threshold,
                  lowpass_hz, min_separation, prominence,
                  trim_margin, save_png=None, save_bp_png=None,
                  title_prefix=''):
    """
    Run the full DTW form-scoring pipeline on a set of already-loaded
    device data dicts and produce plots.

    Parameters
    ----------
    all_data        : list of dicts with keys label, ts, xs, ys, zs
    labels          : list of device label strings (same order as all_data)
    primary_idx     : index into all_data of the primary segmentation device
    weights         : 1-D numpy array of per-device weights (will be sliced
                      to match all_data length and renormalised if needed)
    template_reps   : number of good-form template reps
    anomaly_threshold : sigma multiplier for anomaly threshold
    lowpass_hz, min_separation, prominence, trim_margin : signal params
    save_png        : path to save main plot, or None for interactive
    save_bp_png     : path to save bandpass plot, or None for interactive
    title_prefix    : prepended to plot suptitle (e.g. recording ID)

    Returns
    -------
    dict with keys: segments, combined_scores, threshold, anomalous,
                    per_device_scores, ordered_devices
    or None on failure.
    """
    # Sync window
    all_signals = {d['label']: {'ts': d['ts']} for d in all_data}
    try:
        t_start, t_end = compute_sync_window(all_signals,
                                              trim_margin_s=trim_margin)
    except ValueError as e:
        print(f"  [skip] {e}")
        return None

    dt_start = datetime.datetime.fromtimestamp(
        t_start, tz=datetime.timezone.utc)
    print(f"  Sync window: {t_end - t_start:.2f}s  "
          f"(from {dt_start.strftime('%Y-%m-%d %H:%M:%S UTC')})")

    # Trim, filter, detect peaks/valleys per device
    devices = []
    for d in all_data:
        ts, xs, ys, zs = trim_to_window(
            d['ts'], d['xs'], d['ys'], d['zs'], t_start, t_end)
        times = ts - ts[0]

        peak_times, valley_times, valley_idx, mag_f, mag_bp, \
            bp_low_hz, bp_high_hz, min_sep_used = \
            detect_peaks_valleys(
                times, xs, ys, zs,
                lowpass_hz=lowpass_hz,
                min_separation_s=min_separation,
                prominence_factor=prominence,
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

    # Segmentation on primary device
    primary = devices[primary_idx]
    p_label = primary['label']
    p_valley = primary['valley_idx']
    p_mag_f  = primary['mag_f']
    p_times  = primary['times']

    print(f"\n  Segmenting reps on primary device: [{p_label}]")
    segments = segment_reps(p_valley,
                             n_samples=len(p_mag_f),
                             n_template=template_reps)

    if len(segments) < template_reps + 1:
        print(f"  [skip] only {len(segments)} rep(s) detected "
              f"(need {template_reps + 1})")
        return None

    print(f"  {len(segments)} reps detected")
    for i, (s, e) in enumerate(segments):
        dur = p_times[min(e, len(p_times) - 1)] - p_times[s]
        print(f"    Rep {i+1:2d}: t={p_times[s]:.2f}s – "
              f"{p_times[min(e, len(p_times)-1)]:.2f}s  ({dur:.2f}s)")

    # Re-order so primary is first
    ordered_devices = (
        [devices[primary_idx]] +
        [d for i, d in enumerate(devices) if i != primary_idx]
    )
    # Slice + renormalise weights to match device count
    w = weights[:len(all_data)].copy()
    ordered_weights = np.concatenate([
        w[primary_idx:primary_idx + 1],
        np.delete(w, primary_idx)
    ])
    ordered_weights = ordered_weights / ordered_weights.sum()

    # Build templates
    print(f"\n  Building good-form templates from first {template_reps} reps ...")
    templates = []
    for dev in ordered_devices:
        print(f"    [{dev['label']}]")
        tmpl = build_template(segments, dev['mag_f'],
                               n_template=template_reps)
        templates.append(tmpl)

    # Score
    print("  Scoring all reps ...")
    print(f"  Weights: " + ", ".join(
        f"{d['label']}={w:.3f}"
        for d, w in zip(ordered_devices, ordered_weights)))

    per_device_scores, combined_scores, threshold, anomalous = score_reps(
        segments=segments,
        devices=ordered_devices,
        templates=templates,
        weights=ordered_weights,
        n_template=template_reps,
        threshold_sigma=anomaly_threshold,
    )

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
        print(f"  {i+1:3d}  |  {combined_scores[i]:8.4f}  |  "
              f"{device_cols}{flag}")

    # Plots
    plot_bandpass(ordered_devices, segments, anomalous,
                  primary_label=p_label,
                  save_path=save_bp_png)

    plot_all(ordered_devices, segments,
             per_device_scores, combined_scores,
             threshold, anomalous,
             weights=ordered_weights,
             primary_label=p_label,
             title_prefix=title_prefix,
             save_path=save_png)

    return {
        'segments':          segments,
        'combined_scores':   combined_scores,
        'threshold':         threshold,
        'anomalous':         anomalous,
        'per_device_scores': per_device_scores,
        'ordered_devices':   ordered_devices,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='DTW-based exercise form deviation analysis. '
                    'Operates in file mode (CSV files as positional args) '
                    'or directory mode (--data-dir).')

    # File mode args
    parser.add_argument('files', nargs='*', metavar='CSV',
                        help='Accelerometer CSV files (file mode). '
                             'Omit when using --data-dir.')
    parser.add_argument('--labels', nargs='+', metavar='LABEL',
                        default=None,
                        help='Display labels for each file (file mode only; '
                             'default: filenames)')
    parser.add_argument('--primary', type=int, default=0,
                        help='Index of the primary segmentation device '
                             '(file mode; default: 0)')

    # Directory mode args
    parser.add_argument('--data-dir', metavar='PATH', default=None,
                        help='Root data directory for batch processing. '
                             'Structure: <exercise>/<subject>/<recording_id>/')
    parser.add_argument('--primary-device', default='headphones',
                        choices=['watch', 'headphones', 'phone'],
                        help='Primary segmentation device for directory mode '
                             '(default: watch)')
    parser.add_argument('--save-dir', metavar='PATH', default='./dtw_results',
                        help='Output directory for plots in directory mode '
                             '(default: ./dtw_results)')

    # Shared args
    parser.add_argument('--lowpass-hz', type=float, default=5.0,
                        help='Low-pass filter cutoff (default: 5 Hz)')
    parser.add_argument('--min-separation', type=float, default=None,
                        help='Min seconds between valleys (default: auto ACF)')
    parser.add_argument('--prominence', type=float, default=0.5,
                        help='Valley prominence factor x IQR (default: 0.5)')
    parser.add_argument('--template-reps', type=int, default=5,
                        help='Good-form template reps (default: 5)')
    parser.add_argument('--anomaly-threshold', type=float, default=2.0,
                        help='Sigma multiplier for anomaly threshold '
                             '(default: 2.0)')
    parser.add_argument('--weights', nargs='+', type=float, default=None,
                        metavar='W',
                        help='Per-device weights (file mode). '
                             'Normalised to sum-to-1 internally. '
                             'Default: equal weights.')
    parser.add_argument('--trim-margin', type=float, default=2.0,
                        help='Seconds to trim from each end of sync window '
                             '(default: 2.0)')
    parser.add_argument('--save-png', metavar='PATH',
                        help='Save main plot (file mode only)')
    parser.add_argument('--save-bp-png', metavar='PATH',
                        help='Save bandpass diagnostic plot (file mode only)')
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validate mode
    # ------------------------------------------------------------------
    if args.data_dir and args.files:
        parser.error('Provide either positional CSV files or --data-dir, '
                     'not both.')
    if not args.data_dir and not args.files:
        parser.error('Provide either positional CSV files or --data-dir.')

    # ------------------------------------------------------------------
    # FILE MODE
    # ------------------------------------------------------------------
    if args.files:
        labels = args.labels or [Path(f).stem for f in args.files]
        if len(labels) != len(args.files):
            parser.error('--labels must have the same number of entries '
                         'as files')
        if args.primary >= len(args.files):
            parser.error(f'--primary {args.primary} out of range '
                         f'(only {len(args.files)} files)')

        if args.weights is None:
            weights = np.ones(len(args.files))
        else:
            if len(args.weights) != len(args.files):
                parser.error('--weights must have the same number of entries '
                             f'as files ({len(args.files)}), '
                             f'got {len(args.weights)}')
            if any(w < 0 for w in args.weights):
                parser.error('--weights values must be non-negative')
            weights = np.array(args.weights, dtype=float)
        weights = weights / weights.sum()

        all_data = []
        for filepath, label in zip(args.files, labels):
            print(f"\nLoading [{label}]: {filepath}")
            ts, xs, ys, zs = load_device_csv(filepath)
            print(f"    {len(ts):,} samples  |  "
                  f"{ts[0]:.3f} -> {ts[-1]:.3f}  ({ts[-1]-ts[0]:.1f}s)")
            all_data.append({'label': label,
                             'ts': ts, 'xs': xs, 'ys': ys, 'zs': zs})

        run_recording(
            all_data        = all_data,
            labels          = labels,
            primary_idx     = args.primary,
            weights         = weights,
            template_reps   = args.template_reps,
            anomaly_threshold = args.anomaly_threshold,
            lowpass_hz      = args.lowpass_hz,
            min_separation  = args.min_separation,
            prominence      = args.prominence,
            trim_margin     = args.trim_margin,
            save_png        = args.save_png,
            save_bp_png     = args.save_bp_png,
        )

    # ------------------------------------------------------------------
    # DIRECTORY MODE
    # ------------------------------------------------------------------
    else:
        data_dir = Path(args.data_dir)
        save_dir = Path(args.save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        for ex_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
            exercise = ex_dir.name
            for subj_dir in sorted(p for p in ex_dir.iterdir()
                                   if p.is_dir()):
                subject = subj_dir.name
                for rec_dir in sorted(p for p in subj_dir.iterdir()
                                      if p.is_dir()):
                    rec_id = f"{exercise}/{subject}/{rec_dir.name}"
                    print(f"\n{'='*60}")
                    print(f"Recording: {rec_id}")
                    print(f"{'='*60}")

                    signals = load_recording(rec_dir)
                    if not signals:
                        print("  [skip] no recognised sensor files")
                        continue

                    # Build all_data list from signals dict;
                    # resolve primary device index
                    all_data     = []
                    primary_idx  = 0
                    primary_key  = (args.primary_device, 'accel')
                    for key, d in signals.items():
                        dev_label = f"{key[0]}_{key[1]}"
                        all_data.append({
                            'label': dev_label,
                            'ts': d['ts'], 'xs': d['xs'],
                            'ys': d['ys'], 'zs': d['zs'],
                        })
                    # Find primary device index; fall back to 0
                    for i, key in enumerate(signals.keys()):
                        if key == primary_key:
                            primary_idx = i
                            break

                    weights = np.ones(len(all_data)) / len(all_data)

                    # Build output paths
                    rec_slug  = rec_id.replace('/', '_')
                    save_png  = save_dir / f"{rec_slug}_scores.png"
                    save_bp   = save_dir / f"{rec_slug}_bandpass.png"

                    run_recording(
                        all_data          = all_data,
                        labels            = [d['label'] for d in all_data],
                        primary_idx       = primary_idx,
                        weights           = weights,
                        template_reps     = args.template_reps,
                        anomaly_threshold = args.anomaly_threshold,
                        lowpass_hz        = args.lowpass_hz,
                        min_separation    = args.min_separation,
                        prominence        = args.prominence,
                        trim_margin       = args.trim_margin,
                        save_png          = str(save_png),
                        save_bp_png       = str(save_bp),
                        title_prefix      = f"{rec_id}  ",
                    )


if __name__ == '__main__':
    main()

