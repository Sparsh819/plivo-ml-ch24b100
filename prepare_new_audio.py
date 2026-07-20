"""Turn ANY raw audio file (wav/mp3/ogg/flac) into a data_dir predict.py can
run on. Not part of the graded deliverables - a dev convenience so you can
try the model on audio you find yourself. predict.py only knows how to
score pauses it's told about; a fresh file has no labels.csv yet, so this
script finds candidate pauses (>=100ms of low energy) with a simple
energy-threshold VAD and writes the labels.csv/audio/ layout predict.py
expects.

    python prepare_new_audio.py --audio path/to/file.mp3 --out_dir my_test --turn_id t1
    python predict.py --data_dir my_test --out my_test_predictions.csv
"""
import argparse
import csv
import os
import shutil

from features import load_wav, frame_energy_db, HOP_MS


def find_pauses(x, sr, min_pause_ms=100, silence_db=-40.0):
    e = frame_energy_db(x, sr)
    hop_s = HOP_MS / 1000.0
    is_silent = e < silence_db
    pauses = []
    i, n = 0, len(is_silent)
    while i < n:
        if is_silent[i]:
            j = i
            while j < n and is_silent[j]:
                j += 1
            dur_ms = (j - i) * hop_s * 1000
            if dur_ms >= min_pause_ms:
                pauses.append((i * hop_s, j * hop_s))
            i = j
        else:
            i += 1
    return pauses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, help="any wav/mp3/ogg/flac file")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--turn_id", default="turn_000")
    ap.add_argument("--silence_db", type=float, default=-40.0,
                     help="frames quieter than this (dB) count as silence; "
                          "raise (e.g. -30) if it finds too few pauses, "
                          "lower (e.g. -50) if it finds too many")
    args = ap.parse_args()

    x, sr = load_wav(args.audio)
    print(f"loaded {args.audio}: {len(x)/sr:.2f}s @ {sr}Hz")
    if sr != 16000:
        print(f"NOTE: dataset audio was 16kHz; this file is {sr}Hz. "
              f"Features are Hz/dB based so it should still work, but "
              f"results are less certain off-distribution.")

    pauses = find_pauses(x, sr, silence_db=args.silence_db)
    if not pauses:
        raise SystemExit("No pauses >=100ms found - try --silence_db -30 (stricter) "
                          "or -50 (looser) depending on background noise.")
    print(f"found {len(pauses)} candidate pause(s)")

    audio_dir = os.path.join(args.out_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)
    fname = f"{args.turn_id}{os.path.splitext(args.audio)[1] or '.wav'}"
    shutil.copy(args.audio, os.path.join(audio_dir, fname))

    rows = [{"turn_id": args.turn_id, "audio_file": f"audio/{fname}",
             "pause_index": i, "pause_start": f"{ps:.3f}", "pause_end": f"{pe:.3f}",
             "label": "unknown"}
            for i, (ps, pe) in enumerate(pauses)]
    with open(os.path.join(args.out_dir, "labels.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["turn_id", "audio_file", "pause_index",
                                           "pause_start", "pause_end", "label"])
        w.writeheader()
        w.writerows(rows)

    print(f"wrote -> {args.out_dir}/labels.csv")
    print(f"now run: python predict.py --data_dir {args.out_dir} --out {args.out_dir}_predictions.csv")


if __name__ == "__main__":
    main()
