"""Audio utilities + causal feature extraction for the EOT assignment.

Causality rule: for a pause at `pause_start`, every feature below is computed
from audio[0 : pause_start] only (via `speech_before`), or from *other pauses
in the same turn that already ended before this pause started* (their
pause_end <= this pause_start). Nothing from pause_end/pause_end-onward of
the CURRENT pause, and nothing from later pauses, is ever touched.
"""
import numpy as np
import soundfile as sf

FRAME_MS = 25
HOP_MS = 10


def load_wav(path):
    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    return x, sr


def speech_before(x, sr, pause_start, window_s=1.5):
    """The last `window_s` seconds of audio strictly before the pause."""
    end = int(pause_start * sr)
    start = max(0, end - int(window_s * sr))
    return x[start:end]


def frames(x, sr, frame_ms=FRAME_MS, hop_ms=HOP_MS):
    fl = int(sr * frame_ms / 1000)
    hp = int(sr * hop_ms / 1000)
    if len(x) < fl:
        return np.empty((0, fl), dtype=np.float32)
    n = 1 + (len(x) - fl) // hp
    idx = np.arange(fl)[None, :] + hp * np.arange(n)[:, None]
    return x[idx]


def frame_energy_db(x, sr):
    """Short-time energy per frame, in dB."""
    fr = frames(x, sr)
    rms = np.sqrt(np.mean(fr ** 2, axis=1) + 1e-12)
    return 20 * np.log10(rms + 1e-12)


def autocorr_f0(frame, sr, fmin=60.0, fmax=400.0, voicing_thresh=0.30):
    """Fundamental frequency of one frame via autocorrelation.

    Returns 0.0 for unvoiced/silent frames.
    """
    frame = frame - np.mean(frame)
    if np.max(np.abs(frame)) < 1e-4:
        return 0.0
    ac = np.correlate(frame, frame, mode="full")[len(frame) - 1:]
    if ac[0] <= 0:
        return 0.0
    ac = ac / ac[0]
    lo = int(sr / fmax)
    hi = min(int(sr / fmin), len(ac) - 1)
    if hi <= lo:
        return 0.0
    lag = lo + int(np.argmax(ac[lo:hi]))
    if ac[lag] < voicing_thresh:
        return 0.0
    return float(sr / lag)


def f0_contour(x, sr, frame_ms=40, hop_ms=HOP_MS):
    """Per-frame F0 (Hz), 0.0 where unvoiced. Longer frames help pitch."""
    fr = frames(x, sr, frame_ms=frame_ms, hop_ms=hop_ms)
    return np.array([autocorr_f0(f, sr) for f in fr], dtype=np.float32)


def _slope(y, hop_s):
    """Least-squares slope of y vs time (units/sec). 0.0 if <2 points."""
    n = len(y)
    if n < 2:
        return 0.0
    t = np.arange(n) * hop_s
    A = np.vstack([t, np.ones(n)]).T
    m, _ = np.linalg.lstsq(A, y, rcond=None)[0]
    return float(m)


def _voiced_runs(voiced_mask, hop_s):
    """Durations (s) of contiguous True runs in voiced_mask."""
    runs = []
    cur = 0
    for v in voiced_mask:
        if v:
            cur += 1
        elif cur:
            runs.append(cur * hop_s)
            cur = 0
    if cur:
        runs.append(cur * hop_s)
    return runs


FEATURE_NAMES = [
    "e_last", "e_slope_500ms", "e_decay_window",
    "f0_last", "f0_slope", "f0_rel_to_mean",
    "voiced_frac_last500", "voiced_frac_window",
    "final_voiced_run_ratio", "n_voiced_runs_per_sec",
    "pause_index", "time_since_turn_start",
    "mean_prior_pause_dur", "energy_zscore",
]


def extract_features(x, sr, pause_start, pause_index=0, prior_pause_durs=None):
    """Causal prosodic feature vector for the pause starting at pause_start.

    prior_pause_durs: durations (s) of earlier pauses in THIS turn whose
    pause_end <= pause_start (i.e. already resolved in the past). Safe.
    """
    seg = speech_before(x, sr, pause_start, window_s=2.5)
    short = speech_before(x, sr, pause_start, window_s=0.5)

    if len(seg) < sr // 10:
        return np.zeros(len(FEATURE_NAMES), dtype=np.float32)

    e = frame_energy_db(seg, sr)
    e_short = frame_energy_db(short, sr) if len(short) >= sr * FRAME_MS / 1000 else e[-3:]
    hop_s = HOP_MS / 1000.0

    e_last = float(e_short[-5:].mean()) if len(e_short) else float(e[-1])
    n500 = max(1, int(0.5 / hop_s))
    e_slope_500 = _slope(e[-n500:], hop_s)
    e_decay = float(e[:len(e) // 2].mean() - e[-max(1, len(e) // 4):].mean()) if len(e) > 4 else 0.0

    f0 = f0_contour(seg, sr)
    voiced_mask = f0 > 0
    voiced = f0[voiced_mask]
    f0_last = float(voiced[-3:].mean()) if len(voiced) >= 3 else 0.0
    f0_mean_all = float(voiced.mean()) if len(voiced) else 0.0
    f0_rel = f0_last - f0_mean_all if f0_last and f0_mean_all else 0.0

    hop_f0_s = 0.010
    n_voiced_tail = max(1, int(0.5 / hop_f0_s))
    tail_voiced = f0[-n_voiced_tail:] if len(f0) >= n_voiced_tail else f0
    voiced_idx = np.where(tail_voiced > 0)[0]
    f0_slope = _slope(tail_voiced[voiced_idx], hop_f0_s) if len(voiced_idx) >= 2 else 0.0

    voiced_frac_last500 = float(np.mean(tail_voiced > 0)) if len(tail_voiced) else 0.0
    voiced_frac_window = float(np.mean(voiced_mask)) if len(voiced_mask) else 0.0

    runs = _voiced_runs(voiced_mask, hop_f0_s)
    final_run = runs[-1] if runs else 0.0
    mean_run = float(np.mean(runs)) if runs else 0.0
    final_voiced_run_ratio = (final_run / mean_run) if mean_run > 0 else 0.0
    n_voiced_runs_per_sec = (len(runs) / (len(seg) / sr)) if len(seg) else 0.0

    mean_prior = float(np.mean(prior_pause_durs)) if prior_pause_durs else 0.0

    turn_e = frame_energy_db(seg, sr)
    energy_zscore = 0.0
    if len(turn_e) > 4:
        mu, sd = turn_e.mean(), turn_e.std() + 1e-6
        energy_zscore = float((e_last - mu) / sd)

    return np.array([
        e_last, e_slope_500, e_decay,
        f0_last, f0_slope, f0_rel,
        voiced_frac_last500, voiced_frac_window,
        final_voiced_run_ratio, n_voiced_runs_per_sec,
        float(pause_index), float(pause_start),
        mean_prior, energy_zscore,
    ], dtype=np.float32)
