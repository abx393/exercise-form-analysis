"""
classify_exercise.py
---------------------
Classifies repetition-based exercises from multi-device accelerometer
(and optionally gyroscope) data.

Directory structure expected:
    <data_dir>/
        <exercise_name>/          e.g. situp, pushup, squat
            <subject>/            e.g. alice, bob
                <recording_id>/   e.g. 1, 2, 3
                    <sensor_files>

Supported file formats
----------------------
Garmin / Samsung / Bose (already-converted):
    Columns: absolute_timestamp, accel_x, accel_y, accel_z
    Gyro:    absolute_timestamp, gyro_x, gyro_y, gyro_z

Apple iPhone (Accelerometer.csv / Gyroscope.csv):
    Columns: time, seconds_elapsed, x, y, z
    Timestamp: per-row nanosecond epoch in `time`, or fixed epoch + seconds_elapsed

Apple AirPods (Headphone.csv):
    Columns: time, seconds_elapsed, accelerationX/Y/Z, rotationRateX/Y/Z, ...
    Yields two signal streams: headphones/accel and headphones/gyro

Device and sensor type are inferred from filename keywords:
    Device:
      "bose" | "headphone" | "Headphone"   → headphones
      "garmin" | "watch"                   → watch
      "samsung" | "phone" | "Accelerometer"
        | "Gyroscope"                      → phone
    Sensor:
      "gyro" | "Gyroscope" | "rotationRate"
        | "gyroscope" in name              → gyro
      otherwise                            → accel

Rep segmentation uses the primary device's accelerometer signal:
    1. Low-pass filter the vector magnitude
    2. Estimate rep period from the autocorrelation
    3. Detect valleys with that period as the minimum separation
    4. Apply the same time-boundary segments to all signals

Features are keyed as {device}_{sensor}_{feature}, e.g.:
    watch_accel_mag_mean, headphones_gyro_corr_xy

Model: Random Forest (recording-level train/test split to avoid data leakage).

Usage:
    python classify_exercise.py <data_dir> [--primary-device watch]
                                            [--lowpass-hz 5.0]
                                            [--prominence 0.5]
                                            [--test-size 0.2]
                                            [--n-estimators 200]
                                            [--save-model model.joblib]
                                            [--save-features features.csv]
                                            [--save-plot importance.png]
"""
import argparse
import warnings
from pathlib import Path
from collections import defaultdict, Counter

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.manifold import TSNE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from signal_utils import (
    load_recording,
    compute_sync_window,
    trim_to_window,
    compute_fs,
    lowpass_filter,
    detect_valleys,
    segment_reps,
)

warnings.filterwarnings('ignore', category=RuntimeWarning)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def dominant_frequency(signal, fs):
    """Frequency (Hz) of the highest-amplitude FFT bin above 0 Hz."""
    n      = len(signal)
    if n < 4:
        return 0.0
    fft    = np.abs(np.fft.rfft(signal - signal.mean()))
    freqs  = np.fft.rfftfreq(n, d=1.0 / fs)
    fft[0] = 0.0   # zero DC
    return float(freqs[np.argmax(fft)])


def top3_power_ratio(signal):
    """Fraction of total FFT power in the top 3 magnitude bins."""
    fft   = np.abs(np.fft.rfft(signal - signal.mean())) ** 2
    total = fft.sum()
    if total < 1e-12:
        return 0.0
    top3  = np.sort(fft)[-3:].sum()
    return float(top3 / total)


def zero_crossing_rate(signal):
    """Number of zero crossings per sample."""
    centred = signal - signal.mean()
    zc      = np.sum(np.diff(np.sign(centred)) != 0)
    return float(zc) / max(len(signal) - 1, 1)


def safe_pearson(a, b):
    """Pearson r, returns 0 if either signal has zero variance."""
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def extract_device_features(mag_f_seg, xs_seg, ys_seg, zs_seg,
                              fs, rep_duration_s):
    """
    Compute all features for one rep on one device.

    Returns a flat dict of {feature_name: value}.
    """
    feats = {}

    # --- Magnitude features (orientation-invariant) ---
    mag = mag_f_seg   # already filtered
    feats['mag_mean']     = float(np.mean(mag))
    feats['mag_std']      = float(np.std(mag))
    feats['mag_min']      = float(np.min(mag))
    feats['mag_max']      = float(np.max(mag))
    feats['mag_range']    = feats['mag_max'] - feats['mag_min']
    feats['mag_rms']      = float(np.sqrt(np.mean(mag**2)))
    feats['mag_skew']     = float(_safe_skewness(mag))
    feats['mag_kurt']     = float(_safe_kurtosis(mag))
    feats['mag_dom_freq'] = dominant_frequency(mag, fs)
    feats['mag_top3_pwr'] = top3_power_ratio(mag)

    # Fraction of rep spent above the magnitude mean (phase asymmetry)
    above_mean = float(np.sum(mag > feats['mag_mean'])) / max(len(mag), 1)
    feats['mag_frac_above_mean'] = above_mean

    # --- Per-axis features ---
    for axis_name, axis_sig in [('x', xs_seg), ('y', ys_seg), ('z', zs_seg)]:
        feats[f'{axis_name}_mean'] = float(np.mean(axis_sig))
        feats[f'{axis_name}_std']  = float(np.std(axis_sig))
        feats[f'{axis_name}_zcr']  = zero_crossing_rate(axis_sig)

    # --- Cross-axis correlations ---
    feats['corr_xy'] = safe_pearson(xs_seg, ys_seg)
    feats['corr_xz'] = safe_pearson(xs_seg, zs_seg)
    feats['corr_yz'] = safe_pearson(ys_seg, zs_seg)

    # --- Rep timing ---
    feats['rep_duration_s'] = float(rep_duration_s)

    return feats


def _safe_skewness(x):
    mu, sd = np.mean(x), np.std(x)
    if sd < 1e-12:
        return 0.0
    return float(np.mean(((x - mu) / sd) ** 3))


def _safe_kurtosis(x):
    mu, sd = np.mean(x), np.std(x)
    if sd < 1e-12:
        return 0.0
    return float(np.mean(((x - mu) / sd) ** 4) - 3.0)


def features_for_recording(signals, primary_device='watch',
                            lowpass_hz=5.0, prominence_factor=0.5,
                            trim_margin_s=2.0):
    """
    Segment reps from the primary device's accelerometer signal and extract
    features from all available (device, sensor) streams.

    Parameters
    ----------
    signals       : dict keyed by (device, sensor) tuples, as returned by
                    load_recording()
    primary_device: device label to use for segmentation (default: 'watch').
                    Always uses the 'accel' sensor of this device.
                    Falls back to the first available accel signal if the
                    chosen device is not present.

    Returns
    -------
    list of dicts, one per rep. Feature names are prefixed as
    {device}_{sensor}_{feature}, e.g. 'watch_accel_mag_mean'.
    'rep_duration_s' is added once, unprefixed, as a shared feature.
    Returns empty list if segmentation fails.
    """
    if not signals:
        return []

    try:
        t_start, t_end = compute_sync_window(signals, trim_margin_s)
    except ValueError as e:
        print(f"    [skip] {e}")
        return []

    # Trim all signals to the common window and normalise time to 0
    trimmed = {}
    for key, d in signals.items():
        ts, xs, ys, zs = trim_to_window(
            d['ts'], d['xs'], d['ys'], d['zs'], t_start, t_end)
        if len(ts) < 10:
            print(f"    [skip] {key} has too few samples after trim")
            return []
        times = ts - ts[0]
        trimmed[key] = {'times': times, 'xs': xs, 'ys': ys, 'zs': zs}

    # ------------------------------------------------------------------
    # Select the primary accel signal for segmentation
    # ------------------------------------------------------------------
    primary_key = (primary_device, 'accel')
    if primary_key not in trimmed:
        # Fall back to any available accel signal
        accel_keys  = [k for k in trimmed if k[1] == 'accel']
        if not accel_keys:
            print("    [skip] no accelerometer signal available for segmentation")
            return []
        primary_key = accel_keys[0]
        print(f"    [warn] primary {(primary_device, 'accel')} not found, "
              f"using {primary_key}")

    pd_sig = trimmed[primary_key]

    valley_idx, _, min_sep = detect_valleys(
        pd_sig['times'], pd_sig['xs'], pd_sig['ys'], pd_sig['zs'],
        lowpass_hz=lowpass_hz,
        prominence_factor=prominence_factor,
    )
    segments = segment_reps(valley_idx)

    if len(segments) < 2:
        print(f"    [skip] only {len(segments)} rep(s) detected")
        return []

    print(f"    {len(segments)} reps detected "
          f"(min_sep={min_sep:.2f}s, primary={primary_key})")

    # ------------------------------------------------------------------
    # Extract features per rep per (device, sensor)
    # ------------------------------------------------------------------

    # Build the NaN stub keys once — used when a signal has too few samples
    # in a rep window. Keys follow the {device}_{sensor}_{feat} convention.
    _feat_suffixes = [
        'mag_mean', 'mag_std', 'mag_min', 'mag_max', 'mag_range',
        'mag_rms', 'mag_skew', 'mag_kurt', 'mag_dom_freq', 'mag_top3_pwr',
        'mag_frac_above_mean',
        'x_mean', 'x_std', 'x_zcr',
        'y_mean', 'y_std', 'y_zcr',
        'z_mean', 'z_std', 'z_zcr',
        'corr_xy', 'corr_xz', 'corr_yz',
    ]

    all_rep_features = []

    for s_idx, e_idx in segments:
        t_rep_start = pd_sig['times'][s_idx]
        t_rep_end   = pd_sig['times'][min(e_idx, len(pd_sig['times']) - 1)]
        rep_dur     = float(t_rep_end - t_rep_start)
        if rep_dur <= 0:
            continue

        rep_feats = {'rep_duration_s': rep_dur}

        for (device, sensor), ddata in trimmed.items():
            prefix = f'{device}_{sensor}'
            mask   = ((ddata['times'] >= t_rep_start) &
                      (ddata['times'] <= t_rep_end))
            xs_s = ddata['xs'][mask]
            ys_s = ddata['ys'][mask]
            zs_s = ddata['zs'][mask]

            if len(xs_s) < 4:
                stub = {f'{prefix}_{k}': np.nan for k in _feat_suffixes}
                rep_feats.update(stub)
                continue

            fs_d    = compute_fs(ddata['times'][mask])
            mag_s   = np.sqrt(xs_s**2 + ys_s**2 + zs_s**2)
            mag_s_f = lowpass_filter(mag_s, fs_d, lowpass_hz) \
                      if len(mag_s) > 9 else mag_s

            dev_feats = extract_device_features(
                mag_s_f, xs_s, ys_s, zs_s, fs_d, rep_dur)

            for k, v in dev_feats.items():
                if k != 'rep_duration_s':
                    rep_feats[f'{prefix}_{k}'] = v

        all_rep_features.append(rep_feats)

    return all_rep_features


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------

def build_dataset(data_dir, primary_device='watch',
                  lowpass_hz=5.0, prominence_factor=0.5,
                  trim_margin_s=2.0):
    """
    Walk data_dir/<exercise>/<subject>/<recording_id>/ and build a flat
    feature table.

    Returns
    -------
    rows        : list of feature dicts
    labels      : list of exercise name strings (one per rep)
    group_ids   : list of recording ID strings (one per rep, for grouped CV).
                  Format: "{exercise}/{subject}/{recording_id}" so that the
                  stratified split keeps whole recordings together.
    """
    data_dir = Path(data_dir)
    rows, labels, group_ids = [], [], []

    exercise_dirs = sorted(p for p in data_dir.iterdir() if p.is_dir())

    for ex_dir in exercise_dirs:
        exercise  = ex_dir.name
        subj_dirs = sorted(p for p in ex_dir.iterdir() if p.is_dir())

        if not subj_dirs:
            print(f"  [{exercise}] no subject subdirectories found, skipping")
            continue

        print(f"\n[{exercise}]")
        for subj_dir in subj_dirs:
            subject  = subj_dir.name
            rec_dirs = sorted(p for p in subj_dir.iterdir() if p.is_dir())

            if not rec_dirs:
                print(f"  [{exercise}/{subject}] no recording subdirectories, "
                      f"skipping")
                continue

            print(f"  Subject: {subject}")
            for rec_dir in rec_dirs:
                rec_id = f"{exercise}/{subject}/{rec_dir.name}"
                print(f"    Recording: {rec_dir.name}")

                signals = load_recording(rec_dir)
                if len(signals) == 0:
                    print("      [skip] no recognised sensor files")
                    continue

                rep_feats = features_for_recording(
                    signals,
                    primary_device=primary_device,
                    lowpass_hz=lowpass_hz,
                    prominence_factor=prominence_factor,
                    trim_margin_s=trim_margin_s,
                )

                for rf in rep_feats:
                    rows.append(rf)
                    labels.append(exercise)
                    group_ids.append(rec_id)

    return rows, labels, group_ids


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def _build_matrices(rows, labels, group_ids):
    """
    Convert rows/labels/group_ids into imputed numpy arrays.

    Returns (X, y, g, feature_names, imputer).
    """
    import pandas as pd

    df = pd.DataFrame(rows)
    df = df.ffill().fillna(0.0)
    feature_names = list(df.columns)

    X = df.values.astype(float)
    y = np.array(labels)
    g = np.array(group_ids)

    imputer = SimpleImputer(strategy='median')
    X       = imputer.fit_transform(X)

    return X, y, g, feature_names, imputer


def _subjects_from_group_ids(group_ids):
    """
    Extract unique subject names from group IDs formatted as
    'exercise/subject/recording_id'.
    """
    subjects = {}
    for gid in group_ids:
        parts   = gid.split('/')
        subject = parts[1] if len(parts) >= 3 else parts[0]
        subjects[subject] = subjects.get(subject, 0) + 1
    return sorted(subjects.keys())


# ---------------------------------------------------------------------------
# LOSO cross-validation
# ---------------------------------------------------------------------------

def run_loso_cv(rows, labels, group_ids, n_estimators=200, random_state=42):
    """
    Leave-one-subject-out cross-validation.

    Subject names are extracted from group_ids, which are expected to follow
    the format 'exercise/subject/recording_id'. For each held-out subject:
      - Train a Random Forest on all reps from all other subjects.
      - Predict on all reps from the held-out subject.

    This is the honest generalisation estimate: the model never sees any
    data from the held-out subject during training.

    Returns
    -------
    all_y_true  : (N,) array of true exercise labels
    all_y_pred  : (N,) array of predicted labels
    all_subjects: (N,) array of subject IDs (one per rep)
    fold_clfs   : list of (subject, fitted_clf) tuples — one per fold,
                  useful for inspecting per-fold feature importances
    feature_names: list of feature name strings
    """
    X, y, g, feature_names, _ = _build_matrices(rows, labels, group_ids)
    

    # Subject for each rep (second path component)
    subj_per_rep = np.array([
        gid.split('/')[1] if len(gid.split('/')) >= 3 else gid.split('/')[0]
        for gid in group_ids
    ])
    subjects = sorted(set(subj_per_rep))

    if len(subjects) < 2:
        print("  Need at least 2 subjects for LOSO CV.")
        return None, None, None, [], feature_names

    all_y_true   = []
    all_y_pred   = []
    all_subjects = []
    fold_clfs    = []

    print(f"\nLOSO CV — {len(subjects)} subjects")

    for held_out in subjects:
        train_mask = subj_per_rep != held_out
        test_mask  = subj_per_rep == held_out

        X_train, y_train = X[train_mask], y[train_mask]
        X_test,  y_test  = X[test_mask],  y[test_mask]

        if len(set(y_train)) < 2:
            print(f"  [{held_out}] skip — training set has < 2 classes")
            continue

        clf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        acc = (y_pred == y_test).mean()
        print(f"  Held-out: {held_out:20s}  "
              f"train={len(X_train):4d} reps  "
              f"test={len(X_test):4d} reps  "
              f"acc={acc:.3f}")

        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(y_pred.tolist())
        all_subjects.extend([held_out] * len(y_test))
        fold_clfs.append((held_out, clf))

    return (np.array(all_y_true),
            np.array(all_y_pred),
            np.array(all_subjects),
            fold_clfs,
            feature_names)


# ---------------------------------------------------------------------------
# Final model (trained on all data)
# ---------------------------------------------------------------------------

def train_final_model(rows, labels, group_ids,
                      n_estimators=200, random_state=42):
    """
    Train a single Random Forest on all available data.

    This is used to produce the saved model artifact after LOSO evaluation
    has already given an honest accuracy estimate. The feature importance
    plot is produced from this model since it sees all the data.

    Returns (clf, feature_names, imputer).
    """
    X, y, g, feature_names, imputer = _build_matrices(
        rows, labels, group_ids)
    tsne = TSNE(n_components=2, perplexity=30, random_state=random_state)
    X_embedded = tsne.fit_transform(X)
    cmap = {'pushups': 'blue', 'pullups': 'green', 'situps': 'black',
            'lunges': 'red', 'squats': 'purple', 'bench': 'brown'}
    colors = [cmap[label] for label in labels]
    scatter = plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=colors)
    #plt.colorbar()
    plt.title("2D t-SNE Dimensionality Reduction of Exercise Features")
    plt.show()

    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    clf.fit(X, y)

    # Print per-class training counts
    counts = {c: int((y == c).sum()) for c in sorted(set(y))}
    print(f"  Final model trained on {len(X)} reps: "
          + ", ".join(f"{c}={n}" for c, n in counts.items()))

    return clf, feature_names, imputer


# ---------------------------------------------------------------------------
# Reporting and plotting
# ---------------------------------------------------------------------------

def print_loso_summary(y_true, y_pred, subjects):
    """Print combined LOSO classification report and per-subject accuracy."""
    print("\n" + "=" * 60)
    print("LOSO CROSS-VALIDATION RESULTS")
    print("=" * 60)

    # Per-subject accuracy
    for subj in sorted(set(subjects)):
        mask = subjects == subj
        acc  = (y_pred[mask] == y_true[mask]).mean()
        n    = mask.sum()
        print(f"  {subj:20s}  acc={acc:.3f}  ({n} reps)")

    # Aggregate classification report across all folds
    print("\nAggregate (all folds combined):")
    print(classification_report(y_true, y_pred, zero_division=0))

    print("CONFUSION MATRIX")
    classes = sorted(set(y_true) | set(y_pred))
    cm      = confusion_matrix(y_true, y_pred, labels=classes)
    col_w   = max(len(c) for c in classes) + 2
    header  = " " * col_w + "".join(f"{c:>{col_w}}" for c in classes)
    print(header)
    for true_c, row in zip(classes, cm):
        print(f"{true_c:<{col_w}}" +
              "".join(f"{v:>{col_w}}" for v in row))
    print()


def plot_loso_f1(y_true, y_pred, subjects, save_path=None):
    """
    Bar chart showing per-subject F1 score for each exercise class,
    plus overall accuracy per subject.
    """
    from sklearn.metrics import f1_score

    subj_list  = sorted(set(subjects))
    classes    = sorted(set(y_true))
    x          = np.arange(len(subj_list))
    bar_width  = 0.8 / (len(classes) + 1)

    colors = ['#3498db', '#e67e22', '#2ecc71', '#9b59b6', '#e74c3c']
    fig, ax = plt.subplots(figsize=(max(8, len(subj_list) * 1.5), 5))

    for i, cls in enumerate(classes):
        f1s = []
        for subj in subj_list:
            mask   = subjects == subj
            yt, yp = y_true[mask], y_pred[mask]
            # Binary F1 for this class vs all others
            yt_bin = (yt == cls).astype(int)
            yp_bin = (yp == cls).astype(int)
            f1s.append(f1_score(yt_bin, yp_bin, zero_division=0))
        offset = (i - len(classes) / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, f1s, bar_width,
                      label=f'F1 ({cls})',
                      color=colors[i % len(colors)],
                      edgecolor='white', linewidth=0.4, alpha=0.85)

    # Overall accuracy per subject as a line
    accs = [(y_pred[subjects == s] == y_true[subjects == s]).mean()
            for s in subj_list]
    ax.plot(x, accs, 'k--o', linewidth=1.2, markersize=5,
            label='Overall accuracy', zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(subj_list, rotation=30, ha='right', fontsize=9)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_ylim(0, 1.08)
    ax.set_title("LOSO CV — per-subject F1 by exercise class", fontsize=11)
    ax.legend(fontsize=9, loc='lower right')
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  LOSO F1 plot saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


def plot_feature_importance(clf, feature_names, top_n=30, save_path=None):
    importances = clf.feature_importances_
    idx         = np.argsort(importances)[-top_n:][::-1]
    top_names   = [feature_names[i] for i in idx]
    top_vals    = importances[idx]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.28)))
    ax.barh(range(top_n), top_vals[::-1],
            color='#3498db', edgecolor='white', linewidth=0.4)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names[::-1], fontsize=9)
    ax.set_xlabel("Mean decrease in impurity (feature importance)", fontsize=10)
    ax.set_title(f"Top {top_n} features — Random Forest (final model)",
                 fontsize=12)
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.grid(True, axis='x', linestyle='--', alpha=0.5)
    ax.grid(True, axis='x', which='minor', linestyle=':', alpha=0.25)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"  Feature importance plot saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train an exercise classifier from multi-device '
                    'accelerometer data with LOSO cross-validation.')
    parser.add_argument('data_dir',
                        help='Root data directory. Structure: '
                             '<data_dir>/<exercise>/<subject>/<recording_id>/<csvs>')
    parser.add_argument('--primary-device', default='watch',
                        choices=['watch', 'headphones', 'phone'],
                        help='Device used for rep segmentation (default: watch)')
    parser.add_argument('--lowpass-hz', type=float, default=5.0,
                        help='Low-pass filter cutoff in Hz (default: 5.0)')
    parser.add_argument('--prominence', type=float, default=0.5,
                        help='Valley prominence factor x IQR (default: 0.5)')
    parser.add_argument('--n-estimators', type=int, default=200,
                        help='Number of trees in the Random Forest '
                             '(default: 200)')
    parser.add_argument('--trim-margin', type=float, default=2.0,
                        help='Seconds to trim from each end of the sync '
                             'window to remove startup/shutdown noise '
                             '(default: 2.0)')
    parser.add_argument('--save-model', metavar='PATH', default=None,
                        help='Save the final model (trained on all data) '
                             'to this path (joblib format)')
    parser.add_argument('--save-features', metavar='PATH', default=None,
                        help='Save the full feature matrix to this CSV path')
    parser.add_argument('--save-plot', metavar='PATH', default=None,
                        help='Save the feature importance plot to this path')
    parser.add_argument('--save-loso-plot', metavar='PATH', default=None,
                        help='Save the LOSO per-subject F1 plot to this path')
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Build dataset
    # ------------------------------------------------------------------
    print("Building dataset ...")
    rows, labels, group_ids = build_dataset(
        args.data_dir,
        primary_device=args.primary_device,
        lowpass_hz=args.lowpass_hz,
        prominence_factor=args.prominence,
        trim_margin_s=args.trim_margin,
    )

    if len(rows) == 0:
        print("\nNo reps extracted. Check data directory structure and "
              "device file naming.")
        return

    print(f"\nTotal reps extracted: {len(rows)}")
    for ex, cnt in sorted(Counter(labels).items()):
        print(f"  {ex}: {cnt} reps")

    # ------------------------------------------------------------------
    # Optionally save feature matrix
    # ------------------------------------------------------------------
    if args.save_features:
        import pandas as pd
        df = pd.DataFrame(rows)
        df.insert(0, 'exercise', labels)
        df.insert(1, 'recording', group_ids)
        df.to_csv(args.save_features, index=False)
        print(f"\nFeature matrix saved: {args.save_features}")

    if len(set(labels)) < 2:
        print("\nNeed at least 2 exercise classes to train a classifier.")
        return

    # ------------------------------------------------------------------
    # LOSO cross-validation (honest generalisation estimate)
    # ------------------------------------------------------------------
    y_true, y_pred, subjects, fold_clfs, feature_names = run_loso_cv(
        rows, labels, group_ids,
        n_estimators=args.n_estimators,
    )

    if y_true is not None and len(y_true) > 0:
        print_loso_summary(y_true, y_pred, subjects)
        plot_loso_f1(y_true, y_pred, subjects,
                     save_path=args.save_loso_plot)

    # ------------------------------------------------------------------
    # Final model trained on all data
    # ------------------------------------------------------------------
    print("\nTraining final model on all data ...")
    clf, feature_names, imputer = train_final_model(
        rows, labels, group_ids,
        n_estimators=args.n_estimators,
    )

    if args.save_model:
        joblib.dump({'model': clf, 'imputer': imputer,
                     'feature_names': feature_names}, args.save_model)
        print(f"Model saved: {args.save_model}")

    # ------------------------------------------------------------------
    # Feature importance plot (from final model)
    # ------------------------------------------------------------------
    plot_feature_importance(
        clf, feature_names,
        top_n=min(30, len(feature_names)),
        save_path=args.save_plot,
    )


if __name__ == '__main__':
    main()

