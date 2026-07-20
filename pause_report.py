"""Dev-only diagnostic report: for every pause in every audio file, show the
true label, predicted p_eot, and exactly what the live-agent simulation
would do at the operating point score.py actually selected - so results can
be eyeballed and verified pause-by-pause, not just as one aggregate number.

Not a graded deliverable - a verification tool.

    python pause_report.py --data_dir ../eot_handout/eot_data/eot_data/english --pred predictions_english.csv --out english_pause_report.csv
    python pause_report.py --data_dir ../eot_handout/eot_data/eot_data/hindi   --pred predictions_hindi.csv   --out hindi_pause_report.csv
"""
import argparse
import csv
import os

from score import load, score as score_fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    labels_csv = os.path.join(args.data_dir, "labels.csv")
    best = score_fn(labels_csv, args.pred)
    threshold, delay = best["threshold"], best["delay"]
    print(f"operating point: threshold={threshold}  delay={delay*1000:.0f}ms  "
          f"(mean delay {best['latency']*1000:.0f}ms @ {best['cutoff']*100:.1f}% cutoff, AUC {best['auc']:.3f})")

    pauses = load(labels_csv, args.pred)
    turn_ids = list(dict.fromkeys(p["turn_id"] for p in pauses))  # preserve order

    rows = []
    turn_cut = {t: False for t in turn_ids}
    for pz in pauses:
        fires = pz["p"] >= threshold
        if pz["label"] == "hold":
            false_cutoff = fires and delay < pz["dur"]
            if false_cutoff:
                turn_cut[pz["turn_id"]] = True
            outcome = "FALSE CUTOFF" if false_cutoff else (
                "fired but pause ended first (no cutoff)" if fires else "correctly waited")
            response_ms = ""
        else:
            outcome = f"fired early, responds at {delay*1000:.0f}ms" if fires else "missed, falls back to 1600ms timeout"
            response_ms = f"{delay*1000:.0f}" if fires else "1600"
        rows.append({
            "turn_id": pz["turn_id"],
            "true_label": pz["label"],
            "pause_duration_ms": f"{pz['dur']*1000:.0f}",
            "predicted_p_eot": f"{pz['p']:.4f}",
            "fires_at_threshold": "yes" if fires else "no",
            "response_time_ms": response_ms,
            "outcome": outcome,
        })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["turn_id", "true_label", "pause_duration_ms",
                                           "predicted_p_eot", "fires_at_threshold",
                                           "response_time_ms", "outcome"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote per-pause detail -> {args.out}")

    turn_out = args.out.replace(".csv", "_by_turn.csv")
    turn_rows = []
    for t in turn_ids:
        eot_rows = [p for p in pauses if p["turn_id"] == t and p["label"] == "eot"]
        if eot_rows:
            eot = eot_rows[0]
            fires = eot["p"] >= threshold
            final_delay_ms = f"{delay*1000:.0f}" if fires else "1600"
        else:
            final_delay_ms = "n/a"
        turn_rows.append({
            "turn_id": t,
            "n_pauses": sum(1 for p in pauses if p["turn_id"] == t),
            "had_false_cutoff": "yes" if turn_cut[t] else "no",
            "final_response_ms": final_delay_ms,
        })
    with open(turn_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["turn_id", "n_pauses", "had_false_cutoff", "final_response_ms"])
        w.writeheader()
        w.writerows(turn_rows)
    print(f"wrote per-turn summary -> {turn_out}")

    n_cut = sum(1 for v in turn_cut.values() if v)
    n_missed = sum(1 for r in turn_rows if r["final_response_ms"] == "1600")
    print(f"summary: {len(turn_ids)} turns, {n_cut} with a false cutoff, "
          f"{n_missed} true-eot pauses that missed threshold and hit the 1600ms timeout")


if __name__ == "__main__":
    main()
