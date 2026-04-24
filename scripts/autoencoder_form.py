"""
autoencoder_form.py
--------------------
Detects exercise form deviation using a 1D CNN autoencoder trained on
good-form reps, evaluated with leave-one-subject-out cross-validation.

Directory structure expected:
    <data_dir>/
        <exercise>/
            <subject>/
                <recording_id>/
                    <sensor_files>   (same formats as classify_exercise.py)

Form labelling is automatic: the first N reps in each set are labelled
good form, any remaining reps are labelled bad form.

Pipeline
--------
1. Walk the directory tree, segment every recording into reps using the
   same ACF → bandpass → valley detection pipeline as the other scripts.
2. Normalise each recording's magnitude signal by the min/max of its own
   good-form reps (so amplitude is preserved relative to that recording's
   baseline).
3. Resample every rep to a fixed length (--rep-length, default 128).
4. Leave-one-subject-out CV: for each held-out subject,
     a. Train a 1D CNN autoencoder on good-form reps from all other subjects.
     b. Encode all reps from the held-out subject.
     c. Compute the good-form centroid from the first N encodings.
     d. Anomaly score = Euclidean distance in latent space to that centroid.
     e. Threshold at mean + k*std of the good-form distances.
5. Aggregate per-rep predictions across all folds, report binary metrics
   (precision, recall, F1, AUC-ROC).
6. Produce plots:
     - Per-set anomaly score bar chart (one figure per recording).
     - 2D PCA projection of latent vectors, coloured good/bad.
     - LOSO F1 per subject fold summary.

Usage:
    python autoencoder_form.py <data_dir>
        [--primary-device watch]
        [--n-good 10]
        [--rep-length 128]
        [--latent-dim 16]
        [--epochs 100]
        [--lr 1e-3]
        [--batch-size 32]
        [--threshold-sigma 2.0]
        [--lowpass-hz 5.0]
        [--prominence 0.5]
        [--trim-margin 2.0]
        [--save-dir ./results]
"""

import argparse
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.metrics import (
    precision_recall_fscore_support, roc_auc_score, classification_report,
)
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from signal_utils import (
    load_recording,
    compute_sync_window,
    trim_to_window,
    compute_fs,
    lowpass_filter,
    detect_valleys,
    segment_reps,
    load_rep_boundaries,
    match_recording_to_boundaries,
)

warnings.filterwarnings('ignore', category=RuntimeWarning)


# ---------------------------------------------------------------------------
# 1D CNN Autoencoder
# ---------------------------------------------------------------------------

class RepEncoder(nn.Module):
    """
    Three-stage strided 1D CNN encoder.

    Input : (batch, 1, rep_length)
    Output: (batch, latent_dim)

    Temporal dimension is halved at each stage (stride=2), so rep_length
    must be divisible by 8. The FC bottleneck maps the flattened feature
    map to the latent vector.
    """
    def __init__(self, rep_length: int, latent_dim: int):
        super().__init__()
        self.rep_length = rep_length

        self.conv = nn.Sequential(
            # Stage 1: 1 → 16 channels, length / 2
            nn.Conv1d(1,  16, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # Stage 2: 16 → 32 channels, length / 4
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # Stage 3: 32 → 64 channels, length / 8
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )
        self._flat_dim = 64 * (rep_length // 8)
        self.fc = nn.Linear(self._flat_dim, latent_dim)

    def forward(self, x):
        # x: (batch, 1, rep_length)
        h = self.conv(x)                    # (batch, 64, rep_length//8)
        h = h.view(h.size(0), -1)           # (batch, flat_dim)
        return self.fc(h)                   # (batch, latent_dim)


class RepDecoder(nn.Module):
    """
    Mirror of RepEncoder using transposed convolutions.

    Input : (batch, latent_dim)
    Output: (batch, 1, rep_length)
    """
    def __init__(self, rep_length: int, latent_dim: int):
        super().__init__()
        self.rep_length  = rep_length
        self._flat_dim   = 64 * (rep_length // 8)
        self._inner_len  = rep_length // 8

        self.fc = nn.Linear(latent_dim, self._flat_dim)

        self.deconv = nn.Sequential(
            # Stage 3 reverse: length * 2
            nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            # Stage 2 reverse: length * 4
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2,
                               padding=2, output_padding=1),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            # Stage 1 reverse: length * 8 = rep_length
            nn.ConvTranspose1d(16,  1, kernel_size=7, stride=2,
                               padding=3, output_padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        # z: (batch, latent_dim)
        h = self.fc(z)                                  # (batch, flat_dim)
        h = h.view(h.size(0), 64, self._inner_len)      # (batch, 64, L//8)
        return self.deconv(h)                           # (batch, 1, rep_length)


class RepAutoencoder(nn.Module):
    def __init__(self, rep_length: int = 128, latent_dim: int = 16):
        super().__init__()
        self.encoder = RepEncoder(rep_length, latent_dim)
        self.decoder = RepDecoder(rep_length, latent_dim)

    def forward(self, x):
        z    = self.encoder(x)
        recon = self.decoder(z)
        return recon, z


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_autoencoder(good_reps: np.ndarray,
                      rep_length: int,
                      latent_dim: int,
                      epochs: int,
                      lr: float,
                      batch_size: int,
                      device: torch.device) -> RepAutoencoder:
    """
    Train a RepAutoencoder on good-form reps.

    Parameters
    ----------
    good_reps  : (N, rep_length) array of normalised, resampled reps
    rep_length : fixed rep length (must be divisible by 8)
    latent_dim : bottleneck dimension
    epochs     : training epochs
    lr         : Adam learning rate
    batch_size : mini-batch size
    device     : torch device

    Returns
    -------
    Trained RepAutoencoder in eval mode.
    """
    X = torch.tensor(good_reps[:, None, :], dtype=torch.float32)
    loader = DataLoader(TensorDataset(X), batch_size=batch_size, shuffle=True)

    model = RepAutoencoder(rep_length=rep_length, latent_dim=latent_dim).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = loss_fn(recon, batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * len(batch)
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"      epoch {epoch+1:4d}/{epochs}  "
                  f"loss={epoch_loss/len(good_reps):.6f}")

    model.eval()
    return model


# ---------------------------------------------------------------------------
# Encoding and scoring
# ---------------------------------------------------------------------------

def encode_reps(model: RepAutoencoder,
                reps: np.ndarray,
                device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """
    Encode reps through the trained autoencoder.

    Returns
    -------
    latents : (N, latent_dim) array of latent vectors
    recon_errors : (N,) MSE between input and reconstruction
    """
    X = torch.tensor(reps[:, None, :], dtype=torch.float32).to(device)
    with torch.no_grad():
        recon, z = model(X)
    latents      = z.cpu().numpy()
    recon_errors = ((X - recon) ** 2).mean(dim=(1, 2)).cpu().numpy()
    return latents, recon_errors


def score_reps_autoencoder(latents: np.ndarray,
                            n_good: int,
                            threshold_sigma: float) -> tuple[np.ndarray,
                                                              float,
                                                              np.ndarray]:
    """
    Compute anomaly scores as Euclidean distance in latent space to the
    centroid of the first n_good encodings.

    The threshold is set at mean + threshold_sigma * std of the good-form
    distances. This is person/session-adaptive: the centroid is computed
    from the actual first N reps of the test set, not from training data.

    Parameters
    ----------
    latents         : (N, latent_dim) array
    n_good          : number of reps to treat as good-form anchor
    threshold_sigma : standard deviations above good-form mean for threshold

    Returns
    -------
    distances  : (N,) Euclidean distances to centroid
    threshold  : scalar anomaly threshold
    anomalous  : (N,) boolean array
    """
    n_anchor    = min(n_good, len(latents))
    centroid    = latents[:n_anchor].mean(axis=0)
    distances   = np.linalg.norm(latents - centroid, axis=1)

    good_dists  = distances[:n_anchor]
    mu, sigma   = good_dists.mean(), good_dists.std()
    sigma       = max(sigma, 1e-6)
    threshold   = mu + threshold_sigma * sigma

    anomalous   = distances > threshold
    return distances, threshold, anomalous


# ---------------------------------------------------------------------------
# Data pipeline
# ---------------------------------------------------------------------------

def resample_rep(signal: np.ndarray, rep_length: int) -> np.ndarray:
    """Resample a 1-D signal to rep_length using linear interpolation."""
    if len(signal) == rep_length:
        return signal.copy()
    x_old = np.linspace(0, 1, len(signal))
    x_new = np.linspace(0, 1, rep_length)
    return interp1d(x_old, signal, kind='linear')(x_new)


def normalize_recording(mag_f: np.ndarray,
                         segments: list,
                         n_good: int) -> tuple[np.ndarray, float, float]:
    """
    Min-max normalise mag_f to [0, 1] using the value range of the
    good-form reps (first n_good segments) only.

    Using only good-form reps as the normalisation reference means that
    amplitude differences within a recording are preserved relative to
    the good-form baseline — a shallow rep will still appear shallow
    after normalisation.

    Returns (normalised_mag_f, min_val, max_val).
    """
    good_segs = segments[:min(n_good, len(segments))]
    if not good_segs:
        # Fallback: normalise over full signal
        mn, mx = mag_f.min(), mag_f.max()
    else:
        good_samples = np.concatenate([mag_f[s:e] for s, e in good_segs])
        mn, mx = good_samples.min(), good_samples.max()

    rng = mx - mn
    if rng < 1e-8:
        return mag_f - mn, mn, mx

    return (mag_f - mn) / rng, mn, mx


def process_recording(rec_dir: Path,
                       primary_device: str,
                       lowpass_hz: float,
                       prominence_factor: float,
                       trim_margin_s: float,
                       rep_length: int,
                       n_good: int,
                       rep_boundaries=None) -> dict | None:
    """
    Load, segment, normalise, and resample all reps from one recording.

    Parameters
    ----------
    rep_boundaries : list of (start_s, end_s) pairs from load_rep_boundaries,
                     or None to use automatic ACF-based valley detection.
                     Times are seconds relative to the trimmed window start.

    Returns a dict with keys:
        reps        : (N, rep_length) float32 array
        is_good     : (N,) bool array  (True = good form)
        mag_f       : full normalised low-pass magnitude
        times       : normalised time array (seconds from window start)
        segments    : list of (start_idx, end_idx) tuples
        n_reps      : total reps found
    or None on failure.
    """
    signals = load_recording(rec_dir)
    if not signals:
        return None

    try:
        t_start, t_end = compute_sync_window(signals, trim_margin_s)
    except ValueError as e:
        print(f"      [skip] {e}")
        return None

    # Select primary accel signal for segmentation
    primary_key = (primary_device, 'accel')
    if primary_key not in signals:
        accel_keys = [k for k in signals if k[1] == 'accel']
        if not accel_keys:
            print("      [skip] no accelerometer signal found")
            return None
        primary_key = accel_keys[0]
        print(f"      [warn] primary {(primary_device,'accel')} not found, "
              f"using {primary_key}")

    d   = signals[primary_key]
    ts, xs, ys, zs = trim_to_window(
        d['ts'], d['xs'], d['ys'], d['zs'], t_start, t_end)
    if len(ts) < 20:
        print("      [skip] too few samples after trim")
        return None

    times = ts - ts[0]

    if rep_boundaries is not None:
        # Convert time-boundary pairs → sample-index pairs
        print(f"      Using {len(rep_boundaries)} pre-computed rep boundaries")
        segments = []
        for start_s, end_s in rep_boundaries:
            mask = (times >= start_s) & (times < end_s)
            idx  = np.where(mask)[0]
            if len(idx) >= 2:
                segments.append((int(idx[0]), int(idx[-1]) + 1))
            else:
                print(f"      [warn] rep [{start_s:.2f}s – {end_s:.2f}s] "
                      f"has no samples, skipping")
        # Still need mag_f for normalisation and resampling
        mag   = np.sqrt(xs**2 + ys**2 + zs**2)
        mag_f = lowpass_filter(mag, compute_fs(times), lowpass_hz)
    else:
        valley_idx, mag_f, min_sep = detect_valleys(
            times, xs, ys, zs,
            lowpass_hz=lowpass_hz,
            prominence_factor=prominence_factor,
        )
        segments = segment_reps(valley_idx)
        print(f"      {len(segments)} reps  (min_sep={min_sep:.2f}s, "
              f"primary={primary_key})")

    if len(segments) < 2:
        print(f"      [skip] only {len(segments)} rep(s) detected/loaded")
        return None

    # Normalise magnitude by the good-form rep range
    mag_norm, mn, mx = normalize_recording(mag_f, segments, n_good)

    # Resample each segment to fixed length
    reps = np.stack([
        resample_rep(mag_norm[s:e], rep_length).astype(np.float32)
        for s, e in segments
    ])

    is_good = np.array([i < n_good for i in range(len(segments))],
                       dtype=bool)

    return {
        'reps':     reps,
        'is_good':  is_good,
        'mag_f':    mag_norm,
        'times':    times,
        'segments': segments,
        'n_reps':   len(segments),
    }


# ---------------------------------------------------------------------------
# Directory walker
# ---------------------------------------------------------------------------

def load_all_recordings(data_dir: Path,
                         primary_device: str,
                         lowpass_hz: float,
                         prominence_factor: float,
                         trim_margin_s: float,
                         rep_length: int,
                         n_good: int,
                         boundaries_db: dict | None = None) -> dict:
    """
    Walk data_dir/{exercise}/{subject}/{recording_id}/ and process every
    recording.

    Returns
    -------
    data : nested dict
        data[exercise][subject] = list of recording dicts, each containing:
            'rec_id'   : str  (exercise/subject/recording_id)
            'reps'     : (N, rep_length) array
            'is_good'  : (N,) bool array
            'mag_f'    : normalised magnitude array
            'times'    : time array
            'segments' : list of (start, end) tuples
    """
    data_dir = Path(data_dir)
    data     = defaultdict(lambda: defaultdict(list))

    for ex_dir in sorted(p for p in data_dir.iterdir() if p.is_dir()):
        exercise = ex_dir.name
        print(f"\n[{exercise}]")

        for subj_dir in sorted(p for p in ex_dir.iterdir() if p.is_dir()):
            subject = subj_dir.name
            print(f"  Subject: {subject}")

            for rec_dir in sorted(p for p in subj_dir.iterdir()
                                  if p.is_dir()):
                rec_id = f"{exercise}/{subject}/{rec_dir.name}"
                print(f"    Recording: {rec_dir.name}")

                rec_boundaries = (
                    match_recording_to_boundaries(rec_dir, boundaries_db)
                    if boundaries_db is not None else None
                )

                result = process_recording(
                    rec_dir, primary_device,
                    lowpass_hz, prominence_factor, trim_margin_s,
                    rep_length, n_good,
                    rep_boundaries=rec_boundaries,
                )
                if result is not None:
                    result['rec_id'] = rec_id
                    data[exercise][subject].append(result)

    return data


# ---------------------------------------------------------------------------
# LOSO cross-validation
# ---------------------------------------------------------------------------

def run_loso(data: dict,
             rep_length: int,
             latent_dim: int,
             epochs: int,
             lr: float,
             batch_size: int,
             n_good: int,
             threshold_sigma: float,
             device: torch.device) -> dict:
    """
    Run leave-one-subject-out CV for every exercise independently.

    Returns
    -------
    results : dict keyed by exercise → list of fold dicts, each with:
        'subject'    : held-out subject name
        'y_true'     : (N,) int array  (0=good, 1=bad)
        'distances'  : (N,) float array
        'anomalous'  : (N,) bool array
        'threshold'  : float
        'latents'    : (N, latent_dim) array
        'recon_errs' : (N,) float array
        'rec_ids'    : list of recording ID strings (one per rep)
        'is_good_per_rep': (N,) bool (good-form anchor label)
    """
    results = {}

    for exercise, subject_data in data.items():
        subjects = sorted(subject_data.keys())
        if len(subjects) < 2:
            print(f"\n[{exercise}] only 1 subject — skipping LOSO")
            continue

        print(f"\n[{exercise}] LOSO ({len(subjects)} subjects)")
        results[exercise] = []

        for held_out in subjects:
            print(f"  Held-out: {held_out}")

            # ---- Collect training reps (good form only, other subjects) ----
            train_reps = []
            for subj, recs in subject_data.items():
                if subj == held_out:
                    continue
                for rec in recs:
                    good = rec['reps'][rec['is_good']]
                    if len(good) > 0:
                        train_reps.append(good)

            if not train_reps:
                print("    [skip] no training reps available")
                continue

            train_arr = np.concatenate(train_reps, axis=0)
            print(f"    Training on {len(train_arr)} good-form reps "
                  f"from {len(subjects)-1} subjects")

            # ---- Train autoencoder ----
            model = train_autoencoder(
                train_arr, rep_length, latent_dim,
                epochs, lr, batch_size, device)

            # ---- Encode all reps from held-out subject ----
            all_reps, all_is_good, all_rec_ids = [], [], []
            for rec in subject_data[held_out]:
                all_reps.append(rec['reps'])
                all_is_good.append(rec['is_good'])
                all_rec_ids.extend([rec['rec_id']] * rec['n_reps'])

            all_reps    = np.concatenate(all_reps, axis=0)
            all_is_good = np.concatenate(all_is_good, axis=0)

            latents, recon_errs = encode_reps(model, all_reps, device)

            # ---- Score: distance to good-form centroid within this subject ----
            # Use the combined good-form reps across recordings of this
            # subject as the centroid anchor (concatenated in order, so the
            # first n_good reps across all their recordings form the anchor).
            distances, threshold, anomalous = score_reps_autoencoder(
                latents, n_good=int(all_is_good.sum()),
                threshold_sigma=threshold_sigma)

            y_true = (~all_is_good).astype(int)   # 1 = bad form

            results[exercise].append({
                'subject':       held_out,
                'y_true':        y_true,
                'distances':     distances,
                'anomalous':     anomalous,
                'threshold':     threshold,
                'latents':       latents,
                'recon_errs':    recon_errs,
                'rec_ids':       all_rec_ids,
                'is_good_per_rep': all_is_good,
            })

            # Quick per-fold summary
            if y_true.sum() > 0 and (~anomalous).sum() > 0:
                p, r, f, _ = precision_recall_fscore_support(
                    y_true, anomalous.astype(int),
                    average='binary', zero_division=0)
                print(f"    P={p:.2f}  R={r:.2f}  F1={f:.2f}  "
                      f"flagged={anomalous.sum()}/{len(anomalous)}")

    return results


# ---------------------------------------------------------------------------
# Evaluation summary
# ---------------------------------------------------------------------------

def print_summary(results: dict):
    print("\n" + "=" * 60)
    print("LOSO EVALUATION SUMMARY")
    print("=" * 60)

    for exercise, folds in results.items():
        print(f"\n[{exercise}]")
        all_y, all_pred = [], []
        for fold in folds:
            all_y.extend(fold['y_true'].tolist())
            all_pred.extend(fold['anomalous'].astype(int).tolist())

        all_y    = np.array(all_y)
        all_pred = np.array(all_pred)

        if all_y.sum() == 0 or (1 - all_y).sum() == 0:
            print("  Insufficient class variety for metrics")
            continue

        print(classification_report(all_y, all_pred,
                                    target_names=['good', 'bad'],
                                    zero_division=0))
        try:
            all_dist = np.concatenate([f['distances'] for f in folds])
            auc = roc_auc_score(all_y, all_dist)
            print(f"  AUC-ROC (latent distance): {auc:.3f}")
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_per_set_scores(results: dict, save_dir: Path):
    """
    One figure per subject fold showing the anomaly score (latent distance)
    per rep as a bar chart, coloured by ground-truth form label.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    for exercise, folds in results.items():
        for fold in folds:
            subject   = fold['subject']
            distances = fold['distances']
            threshold = fold['threshold']
            is_good   = fold['is_good_per_rep']
            rec_ids   = fold['rec_ids']

            n = len(distances)
            rep_nums = np.arange(1, n + 1)

            # Colour: green = good form, red = bad form (ground truth)
            colors = ['#2ecc71' if g else '#e74c3c' for g in is_good]

            fig, ax = plt.subplots(figsize=(max(8, n * 0.5), 4))
            ax.bar(rep_nums, distances, color=colors,
                   edgecolor='white', linewidth=0.4, zorder=3)
            ax.axhline(threshold, color='#c0392b', linewidth=1.5,
                       linestyle='--', zorder=4,
                       label=f'Threshold ({threshold:.3f})')

            ax.set_xlabel("Rep number", fontsize=11)
            ax.set_ylabel("Latent distance\nto good-form centroid", fontsize=9)
            ax.set_title(
                f"{exercise} — subject: {subject}\n"
                f"green=good form  red=bad form  (ground truth labels)",
                fontsize=11)
            ax.set_xticks(rep_nums)
            ax.legend(fontsize=9)
            ax.grid(True, axis='y', linestyle='--', alpha=0.5)
            ax.set_xlim(0.5, n + 0.5)
            ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

            plt.tight_layout()
            fname = save_dir / f"{exercise}_{subject}_scores.png"
            fig.savefig(fname, dpi=150)
            plt.close(fig)
            print(f"  Saved: {fname}")


def plot_pca_projections(results: dict, save_dir: Path):
    """
    2D PCA projection of all test-set latent vectors per exercise,
    coloured by ground-truth form label.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    for exercise, folds in results.items():
        all_latents = np.concatenate([f['latents']  for f in folds], axis=0)
        all_is_good = np.concatenate([f['is_good_per_rep'] for f in folds])
        subjects    = [f['subject'] for f in folds]
        subject_ids = np.concatenate([
            np.full(len(f['distances']), i)
            for i, f in enumerate(folds)
        ])

        if all_latents.shape[0] < 3:
            continue

        n_components = min(2, all_latents.shape[1], all_latents.shape[0])
        pca      = PCA(n_components=n_components)
        proj     = pca.fit_transform(all_latents)
        var_expl = pca.explained_variance_ratio_

        fig, ax = plt.subplots(figsize=(8, 6))

        good_mask = all_is_good
        bad_mask  = ~all_is_good

        ax.scatter(proj[good_mask, 0],
                   proj[good_mask, 1] if proj.shape[1] > 1 else
                   np.zeros(good_mask.sum()),
                   c='#2ecc71', s=40, alpha=0.7, label='Good form',
                   edgecolors='white', linewidths=0.3, zorder=3)
        ax.scatter(proj[bad_mask,  0],
                   proj[bad_mask,  1] if proj.shape[1] > 1 else
                   np.zeros(bad_mask.sum()),
                   c='#e74c3c', s=40, alpha=0.7, label='Bad form',
                   edgecolors='white', linewidths=0.3, zorder=3)

        ax.set_xlabel(
            f"PC1 ({var_expl[0]*100:.1f}% var)", fontsize=10)
        if proj.shape[1] > 1:
            ax.set_ylabel(
                f"PC2 ({var_expl[1]*100:.1f}% var)", fontsize=10)
        ax.set_title(
            f"{exercise} — latent space PCA\n"
            f"({len(subjects)} subjects, all LOSO test reps)",
            fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, linestyle='--', alpha=0.4)

        plt.tight_layout()
        fname = save_dir / f"{exercise}_pca.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Saved: {fname}")


def plot_loso_f1(results: dict, save_dir: Path):
    """
    Bar chart of F1 score per subject fold, one figure per exercise.
    """
    save_dir.mkdir(parents=True, exist_ok=True)

    for exercise, folds in results.items():
        subjects, f1s = [], []
        for fold in folds:
            y_true = fold['y_true']
            y_pred = fold['anomalous'].astype(int)
            _, _, f1, _ = precision_recall_fscore_support(
                y_true, y_pred, average='binary', zero_division=0)
            subjects.append(fold['subject'])
            f1s.append(f1)

        if not f1s:
            continue

        x = np.arange(len(subjects))
        fig, ax = plt.subplots(figsize=(max(6, len(subjects) * 1.2), 4))
        bars = ax.bar(x, f1s, color='#3498db', edgecolor='white',
                      linewidth=0.4, zorder=3)
        ax.axhline(np.mean(f1s), color='#e67e22', linewidth=1.5,
                   linestyle='--', zorder=4,
                   label=f'Mean F1 = {np.mean(f1s):.2f}')

        ax.set_xticks(x)
        ax.set_xticklabels(subjects, rotation=30, ha='right', fontsize=9)
        ax.set_ylabel("F1 (bad-form detection)", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{exercise} — LOSO F1 per held-out subject",
                     fontsize=11)
        ax.legend(fontsize=9)
        ax.grid(True, axis='y', linestyle='--', alpha=0.5)

        # Annotate bars
        for rect, val in zip(bars, f1s):
            ax.text(rect.get_x() + rect.get_width() / 2,
                    val + 0.02, f'{val:.2f}',
                    ha='center', va='bottom', fontsize=8)

        plt.tight_layout()
        fname = save_dir / f"{exercise}_loso_f1.png"
        fig.savefig(fname, dpi=150)
        plt.close(fig)
        print(f"  Saved: {fname}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Train a 1D CNN autoencoder per exercise and evaluate '
                    'form deviation detection with LOSO cross-validation.')
    parser.add_argument('data_dir',
                        help='Root directory: '
                             '<data_dir>/<exercise>/<subject>/<recording_id>/')
    parser.add_argument('--primary-device', default='watch',
                        choices=['watch', 'headphones', 'phone'],
                        help='Device used for rep segmentation (default: watch)')
    parser.add_argument('--n-good', type=int, default=10,
                        help='Number of reps per set treated as good form '
                             '(default: 10)')
    parser.add_argument('--rep-length', type=int, default=128,
                        help='Fixed length to resample each rep to. '
                             'Must be divisible by 8. (default: 128)')
    parser.add_argument('--latent-dim', type=int, default=16,
                        help='Autoencoder bottleneck dimension (default: 16)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Training epochs per fold (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Adam learning rate (default: 1e-3)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Mini-batch size (default: 32)')
    parser.add_argument('--threshold-sigma', type=float, default=2.0,
                        help='Anomaly threshold: mean + k*std of good-form '
                             'distances (default: 2.0)')
    parser.add_argument('--lowpass-hz', type=float, default=5.0,
                        help='Low-pass filter cutoff in Hz (default: 5.0)')
    parser.add_argument('--prominence', type=float, default=0.5,
                        help='Valley prominence factor x IQR (default: 0.5)')
    parser.add_argument('--trim-margin', type=float, default=2.0,
                        help='Seconds to trim from each end of the sync '
                             'window to remove startup/shutdown noise '
                             '(default: 2.0)')
    parser.add_argument('--rep-boundaries', metavar='CSV', default=None,
                        help='CSV file of pre-computed rep boundaries. When '
                             'provided, ACF-based segmentation is skipped. '
                             'Expected columns: relative_path, rep_index, '
                             'start_s, end_s.')
    parser.add_argument('--save-dir', default='./results',
                        help='Directory for output plots and models '
                             '(default: ./results)')
    args = parser.parse_args()

    if args.rep_length % 8 != 0:
        parser.error(f'--rep-length must be divisible by 8, '
                     f'got {args.rep_length}')

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load rep boundaries if provided
    boundaries_db = None
    if args.rep_boundaries:
        boundaries_db = load_rep_boundaries(args.rep_boundaries)
        print(f"Loaded rep boundaries for {len(boundaries_db)} recording(s) "
              f"from {args.rep_boundaries}")

    # ------------------------------------------------------------------
    # Load and process all recordings
    # ------------------------------------------------------------------
    print("\nLoading recordings ...")
    data = load_all_recordings(
        data_dir          = Path(args.data_dir),
        primary_device    = args.primary_device,
        lowpass_hz        = args.lowpass_hz,
        prominence_factor = args.prominence,
        trim_margin_s     = args.trim_margin,
        rep_length        = args.rep_length,
        n_good            = args.n_good,
        boundaries_db     = boundaries_db,
    )

    if not data:
        print("\nNo recordings loaded. Check directory structure and "
              "file naming.")
        return

    # Print dataset summary
    print("\nDataset summary:")
    for ex, subjects in data.items():
        total_reps  = sum(r['n_reps'] for recs in subjects.values()
                          for r in recs)
        total_good  = sum(r['is_good'].sum() for recs in subjects.values()
                          for r in recs)
        print(f"  {ex}: {len(subjects)} subjects, "
              f"{sum(len(r) for r in subjects.values())} recordings, "
              f"{total_reps} reps ({total_good} good / "
              f"{total_reps - total_good} bad)")

    # ------------------------------------------------------------------
    # LOSO cross-validation
    # ------------------------------------------------------------------
    print("\nRunning LOSO cross-validation ...")
    results = run_loso(
        data            = data,
        rep_length      = args.rep_length,
        latent_dim      = args.latent_dim,
        epochs          = args.epochs,
        lr              = args.lr,
        batch_size      = args.batch_size,
        n_good          = args.n_good,
        threshold_sigma = args.threshold_sigma,
        device          = device,
    )

    if not results:
        print("\nNo LOSO results produced (need >= 2 subjects per exercise).")
        return

    # ------------------------------------------------------------------
    # Evaluation summary
    # ------------------------------------------------------------------
    print_summary(results)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\nGenerating plots ...")
    plot_per_set_scores(results, save_dir / 'per_set')
    plot_pca_projections(results, save_dir / 'pca')
    plot_loso_f1(results, save_dir / 'loso_f1')

    print(f"\nAll outputs saved to: {save_dir}")


if __name__ == '__main__':
    main()
