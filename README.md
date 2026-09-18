# Pitch Nets: can a graph attention network read a pass better than a gradient-boosted tree?

**Live app: https://igorazevedo.com/pitch-nets/**

**Question.** Given a StatsBomb 360 freeze frame (every visible player) and a pass's start and end,
how likely is the pass to be completed: LightGBM on hand-crafted frame features, or a graph attention
network that reads the players as a set?

**Answer.** They tie within noise and the tree edges it. Test log loss 0.1748 vs 0.1787; the paired
bootstrap over test matches puts the tree ahead by 0.004 with 95% CI [-0.0006, 0.0088], and the GAT
is ahead in only 4.5% of resamples. Twenty geometric features distil almost everything a single
freeze frame says about a single pass.

## Comparison (15,186 open-play passes, 18 held-out test matches)

| Model | Log loss [95% CI] | Brier [95% CI] | ROC AUC [95% CI] | Completed-pass error per match [95% CI] | Params | Train (s) |
|---|---|---|---|---|---|---|
| Constant rate | 0.3632 [0.3463, 0.3814] | 0.1041 [0.0975, 0.1112] | 0.500 | 14.95 [11.30, 18.57] | 0 | 0 |
| Length x third lookup | 0.3115 [0.2939, 0.3301] | 0.0912 [0.0851, 0.0975] | 0.766 [0.747, 0.780] | 15.56 [12.02, 19.02] | 0 | 0 |
| LightGBM | **0.1748** [0.1603, 0.1896] | **0.0528** [0.0483, 0.0576] | **0.947** [0.940, 0.954] | 6.80 [5.05, 8.95] | 37,170 | 5 |
| Graph attention net | 0.1787 [0.1649, 0.1935] | 0.0541 [0.0497, 0.0585] | 0.945 [0.940, 0.951] | **6.46** [4.77, 8.51] | 106,637 | 90 |

CIs are percentile bootstraps over test matches (500 draws). Completed-pass error is the absolute
difference between predicted and actual completed passes per match, in passes. LightGBM parameters
are trees x leaves; its train time excludes an 82 s random search of 25 configurations. The GAT
was given a 12-minute wall clock and stopped itself after 26 epochs (90 s on Apple MPS).

Where the GAT is better: short passes under 10 units (+0.009 log loss) and the edge of the box.
Where the tree is better: long balls over 30 units (0.036), the defensive third, and touchline and
corner cells with a few dozen test passes each. The live app maps this on a pitch.

## Method

108,034 open-play passes with a 360 frame from 127 club matches (Bayer Leverkusen 2023/24,
Barcelona 2020/21, PSG 2021/22 and 2022/23). Set pieces excluded. Inputs known before the ball is
kicked only: start, end, passer under pressure, and the frame (location, teammate and keeper flag of
every visible player). Splits are by match, chronological within each competition-season, 70/15/15.
Two trivial baselines were fixed before any model was trained. LightGBM sees 30 scalar features
(length, angle, location, opponent and teammate distances and counts around the target and along
the pass lane). The GAT is three blocks of four-head attention over player tokens plus one pass
token, with a learned attention bias from each pair's relative offset (dx, dy, distance), read out
from the pass token; AdamW, plateau schedule, early stopping on validation log loss. Every model is
scored on identical rows by one function with one bootstrap.

## How to run

```
uv sync
uv run python scripts/build_dataset.py   # needs cache/ populated, see below
uv run python scripts/features.py
uv run python scripts/train_baselines.py
uv run python scripts/train_tree.py
uv run python scripts/deep.py
uv run python scripts/evaluate.py
uv run python scripts/surfaces.py
uv run pytest
uv run marimo edit --sandbox app.py
```

Data is not in the repo. Fetch `matches/{9/281,11/90,7/108,7/235}.json`, and the `events/` and
`three-sixty/` files for each match with `match_status_360 == "available"`, from
[statsbomb/open-data](https://github.com/statsbomb/open-data) into `cache/`. All inference runs
offline; the app only reads the 70 kB of parquet in `public/data/`.

## Leakage and split checks

- The label is `pass.outcome` being absent. Pass height, body part and recipient are never read by
  either model: `features.LEAK_COLS` lists them and the feature builders assert none is present.
- Both models see the same 108,034 rows, the same split labels and, for the surface renderer, the
  same feature code path as training.
- `build_dataset.py` asserts every match and every (match, possession) sits in exactly one split,
  that every team attacks left to right (own keeper mean x 8.7, opposing keeper 115.3), and that the
  frame's actor stands on the event location (median offset 0.0).
- `evaluate.py` asserts all prediction files cover identical (pass, match, label) rows before scoring.

## Limitations

- 360 frames only contain players in the broadcast picture; receivers are sometimes out of frame.
- End location is the observed end location, so for incomplete passes it is where the ball went,
  not where the passer aimed. Both models share this.
- One seed, one architecture size; the GAT overfits from epoch 10 and would likely gain from
  dropout or more data, but the brief was a head-to-head, not a tuning contest.
- Four teams' seasons dominate the passing distribution (Leverkusen, Barcelona, PSG).

## Data

StatsBomb open data, used under the
[StatsBomb public data licence](https://github.com/statsbomb/open-data/blob/master/LICENSE.pdf).
Raw data is not redistributed here; only small derived tables are committed.
