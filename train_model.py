"""Train the EOT classifier and save it to model.joblib.

Trains on BOTH provided language folders jointly (the features are
prosodic/universal, not lexical, and the hidden test set is "mostly
Hindi" so the model should not overfit to one language). Model selection
uses the actual competition metric (mean response delay @ <=5% cutoff
turns), evaluated with grouped (by turn) cross-validation, not accuracy.

    python train_model.py --data_dirs eot_data/english eot_data/hindi --out model.joblib
"""
import argparse
import csv
import os
import pickle

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from features import load_wav, extract_features, FEATURE_NAMES
from score import evaluate as score_evaluate


def load_split(data_dir, lang_tag):
    rows = list(csv.DictReader(open(os.path.join(data_dir, "labels.csv"))))
    by_turn = {}
    for r in rows:
        by_turn.setdefault(r["turn_id"], []).append(r)
    for t in by_turn:
        by_turn[t].sort(key=lambda r: int(r["pause_index"]))

    cache = {}
    X, y, groups, keys, durs = [], [], [], [], []
    for turn_id, turn_rows in by_turn.items():
        prior_durs = []
        for r in turn_rows:
            path = os.path.join(data_dir, r["audio_file"])
            if path not in cache:
                cache[path] = load_wav(path)
            x, sr = cache[path]
            pi = int(r["pause_index"])
            ps = float(r["pause_start"])
            feat = extract_features(x, sr, ps, pause_index=pi,
                                     prior_pause_durs=list(prior_durs))
            X.append(feat)
            y.append(1 if r["label"] == "eot" else 0)
            groups.append(f"{lang_tag}:{turn_id}")
            keys.append((turn_id, pi))
            durs.append(float(r["pause_end"]) - ps)
            prior_durs.append(float(r["pause_end"]) - ps)
    return np.array(X), np.array(y), groups, keys, durs


def cv_delay_score(X, y, groups, durs, make_model, n_splits=5):
    """Grouped CV using the ACTUAL scorer metric (mean delay @ <=5% cutoff)."""
    gkf = GroupKFold(n_splits=n_splits)
    delays, aucs = [], []
    groups_arr = np.array(groups)
    for tr, te in gkf.split(X, y, groups_arr):
        scaler = StandardScaler().fit(X[tr])
        model = make_model()
        model.fit(scaler.transform(X[tr]), y[tr])
        p = model.predict_proba(scaler.transform(X[te]))[:, 1]
        durs_te = np.array(durs)[te]
        pauses = [{"turn_id": groups_arr[te][i], "dur": float(durs_te[i]),
                    "label": "eot" if y[te][i] else "hold", "p": float(p[i])}
                   for i in range(len(te))]
        best = None
        for t in np.round(np.arange(0.05, 1.0, 0.05), 3):
            for d in np.round(np.arange(0.10, 1.65, 0.05), 3):
                cut, lat = score_evaluate(pauses, t, d)
                if cut <= 0.05 and (best is None or lat < best):
                    best = lat
        delays.append(best if best is not None else 1.6)
    return float(np.mean(delays))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dirs", nargs="+", required=True)
    ap.add_argument("--out", default="model.joblib")
    args = ap.parse_args()

    Xs, ys, gs, ds = [], [], [], []
    for d in args.data_dirs:
        tag = os.path.basename(os.path.normpath(d))
        X, y, groups, keys, durs = load_split(d, tag)
        print(f"{d}: {len(X)} pauses, {len(set(groups))} turns")
        Xs.append(X); ys.append(y); gs.extend(groups); ds.extend(durs)
    X = np.vstack(Xs)
    y = np.concatenate(ys)

    candidates = {
        "logreg": lambda: LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
        "logreg_strong_reg": lambda: LogisticRegression(max_iter=2000, C=0.3, class_weight="balanced"),
        "gbdt_shallow": lambda: GradientBoostingClassifier(
            n_estimators=80, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0),
        "gbdt_shallow_more_trees": lambda: GradientBoostingClassifier(
            n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=0),
        "gbdt_depth3": lambda: GradientBoostingClassifier(
            n_estimators=80, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0),
    }
    results = {}
    for name, factory in candidates.items():
        d = cv_delay_score(X, y, gs, ds, factory)
        results[name] = d
        print(f"  CV mean-delay[{name}] = {d*1000:.0f} ms")

    best_name = min(results, key=results.get)
    print(f"selected model: {best_name} (CV mean-delay = {results[best_name]*1000:.0f} ms)")

    scaler = StandardScaler().fit(X)
    final_model = candidates[best_name]()
    final_model.fit(scaler.transform(X), y)

    with open(args.out, "wb") as f:
        pickle.dump({"model": final_model, "scaler": scaler,
                     "feature_names": FEATURE_NAMES, "model_name": best_name}, f)
    print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
