"""Silence-only baseline: every pause looks like end-of-turn (p_eot = 1).

The agent then relies purely on the swept action delay - i.e., a silence
timer. This is the number our model must beat.

    python baseline.py --data_dir eot_data/english --out base.csv

(Unmodified copy of the baseline shipped in the handout.)
"""
import argparse
import csv
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out", default="predictions.csv")
    args = ap.parse_args()

    rows = []
    with open(os.path.join(args.data_dir, "labels.csv")) as f:
        for r in csv.DictReader(f):
            rows.append({"turn_id": r["turn_id"],
                         "pause_index": r["pause_index"],
                         "p_eot": 1.0})
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["turn_id", "pause_index", "p_eot"])
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {len(rows)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
