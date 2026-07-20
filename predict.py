"""Final EOT predictor. Loads a SAVED model (model.joblib) and scores
unseen turns - it never refits on the data it's predicting.

    python predict.py --data_dir eot_data/english --out predictions.csv

Works on any folder matching the documented schema (audio/ + labels.csv
with turn_id,audio_file,pause_index,pause_start,pause_end[,label]) - the
label column is optional at inference time and is never read.
"""
import argparse
import csv
import os
import pickle

from features import load_wav, extract_features


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--model", default=os.path.join(os.path.dirname(__file__), "model.joblib"))
    args = ap.parse_args()

    with open(args.model, "rb") as f:
        bundle = pickle.load(f)
    model, scaler = bundle["model"], bundle["scaler"]

    rows = list(csv.DictReader(open(os.path.join(args.data_dir, "labels.csv"))))
    by_turn = {}
    for r in rows:
        by_turn.setdefault(r["turn_id"], []).append(r)
    for t in by_turn:
        by_turn[t].sort(key=lambda r: int(r["pause_index"]))

    cache = {}
    out_rows = []
    for turn_id, turn_rows in by_turn.items():
        prior_durs = []
        for r in turn_rows:
            path = os.path.join(args.data_dir, r["audio_file"])
            if path not in cache:
                cache[path] = load_wav(path)
            x, sr = cache[path]
            pi = int(r["pause_index"])
            ps = float(r["pause_start"])
            feat = extract_features(x, sr, ps, pause_index=pi,
                                     prior_pause_durs=list(prior_durs))
            p = float(model.predict_proba(scaler.transform(feat.reshape(1, -1)))[0, 1])
            out_rows.append({"turn_id": turn_id, "pause_index": pi, "p_eot": f"{p:.4f}"})
            # pause_end is only used to build context for LATER pauses in
            # this same turn (already-resolved past information at that
            # point) - never used for the CURRENT pause's own features.
            prior_durs.append(float(r["pause_end"]) - ps)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["turn_id", "pause_index", "p_eot"])
        w.writeheader()
        w.writerows(out_rows)
    print(f"wrote {len(out_rows)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
