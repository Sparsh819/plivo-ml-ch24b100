"""Extended EOT features: Groups B, C, D, E, F, G from the reviewed feature
spec (EOT_FEATURE_PROMPT.md), plus A4/A6. All windowed strictly before
pause_start via features.speech_before(), same causality guarantee as
features.py.

STATUS: implemented and tested (RUNLOG.md Run 7), NOT currently wired into
predict.py/train_model.py. Concatenating these 37 features onto the
existing 15 (52 total) measurably hurt every model type tried on this
~496-pause dataset: GBDT overfits harder in-fold with more features per
training sample, and a regularized linear model that survived CV instead
underfits on the full refit (in-sample AUC 0.80 vs the shipped model's
0.91+). Kept in the repo for transparency and as a starting point if a
larger training set or per-feature selection becomes available - see
RUNLOG.md for the full comparison.

Deliberately EXCLUDED:
- A1 pause_duration_ms: uses pause_end directly - a live agent does not
  know a pause's duration until it's already over, and that's exactly the
  thing being predicted. Using it as an input is training on the answer.
- A3 pause_position_ratio's total_audio_duration term: the file's full
  length is fixed by when the true end-of-turn happens, so it leaks how
  much turn remains at an early pause even though we "have the file on
  disk" for offline dev - a live agent doesn't know the call's eventual
  length while it's in progress.

Deliberately uses this repo's own fast autocorrelation pitch/voicing
detector (features.f0_contour) instead of librosa.pyin for every feature
that needs "is this frame voiced" or "what's the F0 here" (Groups C, E,
and the voicing term in F). Measured: librosa.pyin costs ~1.1s per 1.5s
window after JIT warmup. At ~496 pauses that's fine for a one-time
training pass (~9 min), but predict.py must also run it at inference -
and the entire point of this task is beating a <=1.6s response budget.
An ~1.1s-per-decision feature-extraction cost is architecturally
disqualifying for a live agent even though score.py's static CSV grading
wouldn't itself penalize it. librosa's non-pitch-tracking calls (rms, zcr,
spectral_*, mfcc) are plain STFT-based, not Viterbi search, and measured
at ~0.1s per window after warmup - fast enough to keep.
"""
import numpy as np
import librosa

from features import speech_before, f0_contour, _voiced_runs, _slope, HOP_MS, FRAME_MS

FEATURE_NAMES_V2 = [
    "A4_speech_duration_so_far", "A6_speech_rate_so_far",
    "B1_rms_mean", "B2_rms_max", "B3_rms_std", "B4_rms_slope_half",
    "B5_energy_decay_ratio", "B6_energy_final_ratio", "B7_energy_linreg_slope",
    "C1_f0_mean", "C2_f0_std", "C3_f0_slope_half", "C4_f0_linreg_slope",
    "C5_f0_final_500ms_slope", "C6_f0_range", "C7_f0_final_relative", "C8_voiced_ratio",
    "D1_zcr_ratio_end_start", "D2_spectral_flatness_final", "D3_spectral_rolloff_final",
    "D4_spectral_centroid_slope", "D5_spectral_bandwidth_mean",
    "E1_final_seg_duration", "E2_final_lengthening_ratio",
    "E3_syllable_rate_first_half", "E4_syllable_rate_second_half",
    "E5_rate_slowdown_ratio", "E6_rhythm_cv",
    "F1_breath_event_400ms", "F2_breath_event_600ms", "F3_breath_frame_count",
    "G1_mfcc1_mean", "G2_mfcc2_mean", "G3_mfcc1_slope", "G4_mfcc2_slope",
    "G5_mfcc3_slope", "G6_delta_mfcc_energy",
]


def _half_split_mean_diff(arr):
    if len(arr) < 2:
        return 0.0
    h = len(arr) // 2
    first, second = arr[:h], arr[h:]
    if len(first) == 0 or len(second) == 0:
        return 0.0
    return float(np.mean(second) - np.mean(first))


def extract_features_v2(x, sr, pause_start, prior_pause_durs=None):
    hop_len = max(1, int(sr * HOP_MS / 1000))
    frame_len = max(hop_len, int(sr * FRAME_MS / 1000))
    hop_s = hop_len / sr

    seg = speech_before(x, sr, pause_start, window_s=1.5)
    if len(seg) < sr // 10:
        return np.zeros(len(FEATURE_NAMES_V2), dtype=np.float32)

    # --- A4 / A6: turn-so-far context (own fast tracker, not full audio window) ---
    prior_total = float(np.sum(prior_pause_durs)) if prior_pause_durs else 0.0
    speech_duration_so_far = max(0.0, pause_start - prior_total)
    full_hist = x[:int(pause_start * sr)]
    if len(full_hist) >= sr // 10:
        f0_hist = f0_contour(full_hist, sr, frame_ms=40, hop_ms=HOP_MS)
        hist_segs = _voiced_runs(f0_hist > 0, HOP_MS / 1000.0)
        speech_rate_so_far = (len(hist_segs) / speech_duration_so_far) if speech_duration_so_far > 0 else 0.0
    else:
        speech_rate_so_far = 0.0

    # --- Group B: RMS energy (librosa, fast) ---
    rms = librosa.feature.rms(y=seg, frame_length=frame_len, hop_length=hop_len)[0]
    n200 = max(1, int(0.2 / hop_s))
    n100 = max(1, int(0.1 / hop_s))
    rms_mean = float(rms.mean())
    rms_max = float(rms.max())
    rms_std = float(rms.std())
    rms_slope_half = _half_split_mean_diff(rms)
    first200 = rms[:n200].mean() if len(rms) >= n200 else rms.mean()
    last200 = rms[-n200:].mean()
    energy_decay_ratio = float(last200 / first200) if first200 > 1e-9 else 0.0
    energy_final_ratio = float(rms[-n100:].mean() / rms_mean) if rms_mean > 1e-9 else 0.0
    energy_linreg_slope = _slope(rms, hop_s)

    # --- Group C: pitch (own fast autocorrelation tracker) ---
    f0 = f0_contour(seg, sr, frame_ms=40, hop_ms=HOP_MS)
    voiced_mask = f0 > 0
    voiced = f0[voiced_mask]
    f0_mean = float(voiced.mean()) if len(voiced) else 0.0
    f0_std = float(voiced.std()) if len(voiced) else 0.0
    h = len(f0) // 2
    first_voiced = f0[:h][f0[:h] > 0]
    second_voiced = f0[h:][f0[h:] > 0]
    f0_slope_half = float(second_voiced.mean() - first_voiced.mean()) if len(first_voiced) and len(second_voiced) else 0.0
    voiced_idx_all = np.where(voiced_mask)[0]
    f0_linreg_slope = _slope(f0[voiced_idx_all], hop_s) if len(voiced_idx_all) >= 2 else 0.0
    n500 = max(1, int(0.5 / hop_s))
    tail_f0 = f0[-n500:] if len(f0) >= n500 else f0
    tail_voiced_idx = np.where(tail_f0 > 0)[0]
    f0_final_500ms_slope = _slope(tail_f0[tail_voiced_idx], hop_s) if len(tail_voiced_idx) >= 2 else 0.0
    f0_range = float(voiced.max() - voiced.min()) if len(voiced) else 0.0
    last_voiced_val = voiced[-1] if len(voiced) else 0.0
    f0_final_relative = float(last_voiced_val / f0_mean) if f0_mean > 1e-9 else 0.0
    voiced_ratio = float(np.mean(voiced_mask)) if len(voiced_mask) else 0.0

    # --- Group D: vocal quality (librosa, fast) ---
    zcr = librosa.feature.zero_crossing_rate(seg, frame_length=frame_len, hop_length=hop_len)[0]
    zcr_first200 = zcr[:n200].mean() if len(zcr) >= n200 else zcr.mean()
    zcr_last200 = zcr[-n200:].mean()
    zcr_ratio_end_start = float(zcr_last200 / zcr_first200) if zcr_first200 > 1e-9 else 0.0

    flatness = librosa.feature.spectral_flatness(y=seg, hop_length=hop_len)[0]
    spectral_flatness_final = float(flatness[-n200:].mean()) if len(flatness) else 0.0

    rolloff = librosa.feature.spectral_rolloff(y=seg, sr=sr, hop_length=hop_len, roll_percent=0.85)[0]
    spectral_rolloff_final = float(rolloff[-n200:].mean()) if len(rolloff) else 0.0

    centroid = librosa.feature.spectral_centroid(y=seg, sr=sr, hop_length=hop_len)[0]
    spectral_centroid_slope = _half_split_mean_diff(centroid)

    bandwidth = librosa.feature.spectral_bandwidth(y=seg, sr=sr, hop_length=hop_len)[0]
    spectral_bandwidth_mean = float(bandwidth.mean()) if len(bandwidth) else 0.0

    # --- Group E: rhythm / syllable proxy (own fast tracker) ---
    segs = _voiced_runs(voiced_mask, hop_s)
    final_seg_duration = segs[-1] if segs else 0.0
    mean_seg = float(np.mean(segs)) if segs else 0.0
    final_lengthening_ratio = (final_seg_duration / mean_seg) if mean_seg > 0 else 0.0

    first_half_mask, second_half_mask = voiced_mask[:h], voiced_mask[h:]
    segs_first = _voiced_runs(first_half_mask, hop_s)
    segs_second = _voiced_runs(second_half_mask, hop_s)
    dur_first = len(first_half_mask) * hop_s
    dur_second = len(second_half_mask) * hop_s
    syllable_rate_first_half = (len(segs_first) / dur_first) if dur_first > 0 else 0.0
    syllable_rate_second_half = (len(segs_second) / dur_second) if dur_second > 0 else 0.0
    rate_slowdown_ratio = (syllable_rate_second_half / syllable_rate_first_half) if syllable_rate_first_half > 1e-9 else 0.0
    rhythm_cv = (np.std(segs) / np.mean(segs)) if len(segs) > 1 and np.mean(segs) > 0 else 0.0

    # --- Group F: breath detection (rms/flatness from librosa, voicing from own tracker) ---
    min_len = min(len(rms), len(flatness), len(voiced_mask))
    rms_a, flat_a, voiced_a = rms[:min_len], flatness[:min_len], voiced_mask[:min_len]
    breath_candidate = (rms_a > 0.003) & (rms_a < 0.03) & (flat_a > 0.005) & (~voiced_a)
    n400 = max(1, int(0.4 / hop_s))
    n600 = max(1, int(0.6 / hop_s))
    breath_event_400ms = float(np.any(breath_candidate[-n400:])) if min_len else 0.0
    breath_event_600ms = float(np.any(breath_candidate[-n600:])) if min_len else 0.0
    breath_frame_count = float(np.sum(breath_candidate[-n600:])) if min_len else 0.0

    # --- Group G: MFCC spectral shape (librosa, fast) ---
    mfcc = librosa.feature.mfcc(y=seg, sr=sr, n_mfcc=13, hop_length=hop_len)
    if mfcc.shape[1] == 0:
        return np.zeros(len(FEATURE_NAMES_V2), dtype=np.float32)
    delta = librosa.feature.delta(mfcc) if mfcc.shape[1] >= 9 else np.zeros_like(mfcc)
    mfcc1_mean = float(mfcc[0].mean())
    mfcc2_mean = float(mfcc[1].mean())
    mfcc1_slope = _slope(mfcc[0], hop_s)
    mfcc2_slope = _slope(mfcc[1], hop_s)
    mfcc3_slope = _slope(mfcc[2], hop_s)
    delta_mfcc_energy = float(np.mean(np.abs(delta)))

    return np.array([
        speech_duration_so_far, speech_rate_so_far,
        rms_mean, rms_max, rms_std, rms_slope_half,
        energy_decay_ratio, energy_final_ratio, energy_linreg_slope,
        f0_mean, f0_std, f0_slope_half, f0_linreg_slope,
        f0_final_500ms_slope, f0_range, f0_final_relative, voiced_ratio,
        zcr_ratio_end_start, spectral_flatness_final, spectral_rolloff_final,
        spectral_centroid_slope, spectral_bandwidth_mean,
        final_seg_duration, final_lengthening_ratio,
        syllable_rate_first_half, syllable_rate_second_half,
        rate_slowdown_ratio, rhythm_cv,
        breath_event_400ms, breath_event_600ms, breath_frame_count,
        mfcc1_mean, mfcc2_mean, mfcc1_slope, mfcc2_slope, mfcc3_slope, delta_mfcc_energy,
    ], dtype=np.float32)
