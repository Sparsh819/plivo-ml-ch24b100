# RUNLOG — End-of-Turn Detection

Every entry below is a real `score.py` run. Format: score, then what changed
and why.

## Run 1 — baseline, English
```
python baseline.py --data_dir eot_data/english --out base_en.csv
python score.py    --data_dir eot_data/english --pred base_en.csv
```
**mean delay = 1600 ms, cutoff = 0.0%, AUC = 0.514.**
Silence-only (`p_eot=1` always) degenerates to "wait for the 1.6 s timeout
every time" on English — there's no signal to fire early on safely, so the
sweep picks threshold=1.0. This is the number to beat.

## Run 2 — baseline, Hindi
```
python baseline.py --data_dir eot_data/hindi --out base_hi.csv
python score.py    --data_dir eot_data/hindi --pred base_hi.csv
```
**mean delay = 850 ms, cutoff = 5.0%, AUC = 0.501.**
Hindi hold-pauses in this set are shorter on average, so "always fire" can
sit at a low threshold/short delay and still stay under the 5% cutoff
budget by luck, not by understanding anything. AUC ~0.50 confirms it's
literally guessing — the lower baseline number is an artifact of this
dataset's pause-duration distribution, not a stronger baseline.

## Exploratory pass (not scored — informs features)
Ran `explore.py`: compared hold vs eot pauses on raw energy/pitch stats.
- eot pauses are longer on average (English 1.34s vs 0.62s; Hindi 0.94s vs
  0.38s) — expected, but not usable as a *feature* (pause duration is the
  thing we're trying to predict before it's known).
- **Final pitch (F0) is lower before eot than before hold**, in both
  languages (English 213.8 Hz vs 235.8 Hz; Hindi 192.4 Hz vs 232.9 Hz).
  Classic statement-final pitch fall vs continuation-rise. This is the
  strongest single cue found and drives the F0 features below.
- Trailing energy alone was a weak, mixed signal — not reliable on its own.

## Run 3 — first prosodic feature set + model selection
Built 14 causal features in `features.py`: trailing energy level/slope/decay,
F0 level/slope/relative-to-mean, voicing fraction (window + last 500ms),
final-voiced-run-length ratio (final-syllable lengthening), voiced-run rate
(speaking rate proxy), pause position in turn, elapsed turn time, mean prior
pause duration in this turn, and an energy z-score. Trained on English+Hindi
**jointly** (prosodic cues are largely language-agnostic, and the hidden
test set is "mostly Hindi" so the model shouldn't overfit to one language).

Model selection used **grouped (by turn) cross-validation scored with the
actual competition metric** (mean delay @ ≤5% cutoff), not accuracy —
accuracy on a ~60/40 class split is a misleading proxy for this metric.
```
CV mean-delay: logreg=1049ms, logreg_strong_reg=1046ms, gbdt_shallow=1008ms
selected: gbdt_shallow
```
In-sample check (`predict.py` run on the same data used to fit — optimistic,
see Run 5 for the honest number):
- English: **970 ms** (was 1600), cutoff 5.0%, AUC 0.857
- Hindi: **764 ms** (was 850), cutoff 4.0%, AUC 0.877

## Run 4 — error analysis
`error_analysis.py` ranks pauses by |p_eot − label|. Worst errors clustered
almost entirely at **pause_index=0** (short, single-pause turns) where the
true label is `eot` but F0 was *rising* into the pause — the opposite of
the general falling-pitch-at-EOT pattern the model learned from longer,
multi-pause turns. Likely cause: turn-final confirmation questions
("...for seven, right?") carry rising intonation even though the turn is
genuinely over. Rather than hand-craft a special case with ~250 training
pauses (overfitting risk), widened the GBDT search so trees have room to
learn the pause_index × f0_slope interaction directly.

## Run 5 — expanded model grid, final model
```
python train_model.py --data_dirs eot_data/english eot_data/hindi --out model.joblib
```
```
CV mean-delay: gbdt_shallow=1008ms, gbdt_shallow_more_trees=986ms, gbdt_depth3=1015ms
selected: gbdt_shallow_more_trees (150 trees, depth 2, lr 0.05)
```
**This 986 ms grouped-CV number is the honest, held-out estimate** — every
turn in the validation fold was excluded from that fold's training data.
It's the number that should best predict hidden-test-set behavior.

Final `predict.py` run (loads the saved model, does not refit):
```
python predict.py --data_dir eot_data/english --out predictions_english.csv
python score.py   --data_dir eot_data/english --pred predictions_english.csv
python predict.py --data_dir eot_data/hindi   --out predictions_hindi.csv
python score.py   --data_dir eot_data/hindi   --pred predictions_hindi.csv
```
- **English: 730 ms** (baseline 1600 ms, −54%), cutoff 5.0%, AUC 0.901
- **Hindi: 655 ms** (baseline 850 ms, −23%), cutoff 5.0%, AUC 0.923

These two are in-sample (the final model is refit on all available data
before shipping, standard practice once model selection is done via CV) —
treat **986 ms held-out CV** as the more realistic expectation for unseen
turns, including the hidden mostly-Hindi test set.

## Summary table

| stage                          | English delay | Hindi delay | held-out CV (both) |
|---------------------------------|---------------|-------------|---------------------|
| silence-only baseline            | 1600 ms       | 850 ms      | —                   |
| final model (in-sample)          | 730 ms        | 655 ms      | —                   |
| final model (grouped CV, honest) | —             | —           | 986 ms              |
