# plivo-ml-ch24b100 — End-of-Turn Detection

Plivo AI/ML campus assignment (STT track). See `SUMMARY.html` for the full
writeup, `RUNLOG.md` for every scoring run, `NOTES.md` for the model
discussion.

## Reproduce

```
pip install numpy scipy scikit-learn pandas soundfile joblib

python baseline.py    --data_dir eot_data/english --out base.csv
python score.py       --data_dir eot_data/english --pred base.csv

python train_model.py --data_dirs eot_data/english eot_data/hindi --out model.joblib
python predict.py     --data_dir eot_data/english --out predictions_english.csv
python score.py       --data_dir eot_data/english --pred predictions_english.csv
python predict.py     --data_dir eot_data/hindi   --out predictions_hindi.csv
python score.py       --data_dir eot_data/hindi   --pred predictions_hindi.csv
```

`eot_data/` (the provided audio + labels) is not committed — point
`--data_dir` at wherever you extracted `eot_handout.zip`'s `eot_data/`.

## Files

- `features.py` — causal audio utilities + `extract_features()` (the actual
  feature engineering work; only ever touches `audio[0:pause_start]` plus
  already-resolved earlier pauses in the same turn).
- `train_model.py` — builds features for English+Hindi jointly, selects a
  model via grouped (by-turn) CV scored with the real competition metric,
  saves `model.joblib`.
- `predict.py` — required interface: `python predict.py --data_dir <folder>
  --out predictions.csv`. Loads the saved model, never refits.
- `baseline.py`, `score.py` — unmodified copies of the provided baseline and
  official scorer, kept for reproducibility.
- `error_analysis.py` — dev tool used to find the model's worst errors.
- `predictions_english.csv`, `predictions_hindi.csv` — final predictions for
  both provided folders.
- `model.joblib` — the trained model + fitted scaler.
