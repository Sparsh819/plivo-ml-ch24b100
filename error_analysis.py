"""Dev-only: inspect worst misclassified pauses to guide feature iteration."""
import csv
import os
import pickle

import numpy as np

from features import load_wav, extract_features, FEATURE_NAMES


def analyze(data_dir, model_path="model.joblib", topn=8):
    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    model, scaler = bundle["model"], bundle["scaler"]

    rows = list(csv.DictReader(open(os.path.join(data_dir, "labels.csv"))))
    by_turn = {}
    for r in rows:
        by_turn.setdefault(r["turn_id"], []).append(r)
    for t in by_turn:
        by_turn[t].sort(key=lambda r: int(r["pause_index"]))

    cache = {}
    results = []
    for turn_id, turn_rows in by_turn.items():
        prior_durs = []
        for r in turn_rows:
            path = os.path.join(data_dir, r["audio_file"])
            if path not in cache:
                cache[path] = load_wav(path)
            x, sr = cache[path]
            pi, ps = int(r["pause_index"]), float(r["pause_start"])
            feat = extract_features(x, sr, ps, pause_index=pi, prior_pause_durs=list(prior_durs))
            p = float(model.predict_proba(scaler.transform(feat.reshape(1, -1)))[0, 1])
            y = 1 if r["label"] == "eot" else 0
            results.append((abs(p - y), turn_id, pi, r["label"], p, feat,
                             float(r["pause_end"]) - ps))
            prior_durs.append(float(r["pause_end"]) - ps)

    results.sort(key=lambda t: -t[0])
    print(f"=== {data_dir}: worst {topn} errors ===")
    for err, tid, pi, lab, p, feat, dur in results[:topn]:
        print(f"{tid} pause#{pi} true={lab:5s} pred_p={p:.2f} dur={dur:.2f}s")
        for name, val in zip(FEATURE_NAMES, feat):
            print(f"    {name:24s} {val:8.2f}")


if __name__ == "__main__":
    for lang in ["english", "hindi"]:
        analyze(f"../eot_handout/eot_data/eot_data/{lang}")
