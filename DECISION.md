# Research decision

Written at T+0:08. Three candidates were scored against the brief's criteria
(1 structural reason for the deep model, 2 labels from open data, 3 trainable in
under 40 min, 4 pitch-renderable, 5 two trivial baselines, 6 useful to an analyst).
The choice is final and was not revisited.

## Candidates

### A. Pass completion from StatsBomb 360 freeze frames (CHOSEN)

Given an open-play pass (start, intended end, whether the passer is under
pressure) and the 360 freeze frame at the moment of the pass (every visible
player as a node: location, teammate, keeper), predict whether the pass is
completed. Club data with 360 frames: Bayer Leverkusen 2023/24 (Bundesliga),
Barcelona 2020/21 (La Liga), PSG 2021/22 and 2022/23 (Ligue 1). 127 matches,
roughly 100k passes with a frame.

| Criterion | Score | Note |
|---|---|---|
| 1 structure | strong | Completion depends on the *configuration* of defenders around the lane and the receiver. Hand-crafted features (nearest opponent, count in corridor) flatten a relational set into scalars. A GNN reads the set directly. Not a win by construction: xPass literature shows engineered features do very well, so the head-to-head is genuinely open. |
| 2 labels | yes | `pass.outcome` absent means completed. |
| 3 time | yes | ~100k graphs of ~15 nodes; a small attention network trains in minutes on CPU/MPS. |
| 4 pitch render | excellent | For one real frame, score every candidate end location on a grid and draw a completion-probability surface for each model, side by side. |
| 5 baselines | yes | (i) constant train completion rate; (ii) lookup table of completion rate by pass length bin and end-location third. |
| 6 analyst use | high | "Which passing options were safe here" is a standard pre- and post-match question. |

### B. Next-event zone from the last 10 events of a possession (brief's fallback)

12x8 grid, XGBoost on lag features vs a small decoder-only transformer over
event tokens. Passes all criteria but is weaker on 1 and 6: lag features
already capture most of a 10-event window, and "where does the next event
happen" is rarely a question an analyst asks. Rejected in favour of A.

### C. Expected goals from shot freeze frames (GNN vs trees)

Every StatsBomb shot carries a freeze frame, so no 360 needed and all open
data is usable. Rejected on criterion 1: with ~1 shot per 40 passes the
sample is small, and shot-freeze-frame xG is known to be captured well by a
handful of engineered features (goalkeeper position, defenders in cone), so
trees win close to by construction.

## Architecture

The deep model is a graph attention network over the fully connected player
graph, implemented as a transformer encoder over player tokens with a learned
edge bias from pairwise offsets, plus one pass token carrying start and end
location; this is the smallest model that reads the freeze frame as a set
instead of through pre-flattened features. Multi-head attention lets the model
weight opponents near the pass lane and near the receiver differently, which
is exactly the relational structure hand-crafted features approximate with
fixed radii. It trains in minutes on 100k small graphs, so it fits the
12-minute wall clock with early stopping.

## Fixed protocol (identical for every model)

- Target: open-play pass completed (1) or not (0). Set pieces excluded.
- Inputs known before the ball is kicked only: start, end location, under
  pressure flag, freeze frame. Pass height and body part are excluded because
  they are only known once the pass is played.
- Split by match, chronologically within each competition-season: first 70%
  of matches train, next 15% validation, last 15% test.
- Metrics: log loss, Brier, ROC AUC, and completed-passes error per match
  (absolute difference between predicted and actual completed passes, in
  passes). Bootstrap CIs over test matches.
