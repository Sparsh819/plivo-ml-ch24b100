# NOTES

The model uses causal prosodic features from the 2.5s of audio before each
pause: trailing F0 level and slope (falling pitch → end-of-turn, rising or
level pitch → continuation), trailing energy level/slope/decay, voicing
fraction, final-voiced-run-length ratio (final-syllable lengthening), a
speaking-rate proxy, and turn-context features (pause position, elapsed
time, mean duration of this turn's earlier pauses). It's a gradient-boosted
tree classifier (150 shallow trees) trained jointly on English and Hindi, so
it leans on language-agnostic prosody rather than lexical content, and
model selection used grouped cross-validation scored with the real
competition metric, not accuracy. It still fails mostly on short,
single-pause turns with rising final intonation — turn-final confirmation
questions ("...for seven, right?") — where F0 rises even though the turn is
genuinely over, which contradicts the falling-pitch pattern learned from
longer multi-pause turns; this is the largest error cluster found in
`error_analysis.py`. With one more day I'd add a dedicated
question-intonation feature (e.g. F0 range/rise magnitude over the whole
utterance, not just the trailing slope) and get more than 100 turns per
language so the pause_index=0 regime isn't so sparse. I'd also add a
speaker-normalization pass (per-turn z-scoring of F0/energy is only
partially done today) since absolute pitch varies a lot across speakers and
that's currently diluting the model's confidence on some turns.
