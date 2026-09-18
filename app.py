# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "polars",
#     "altair",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import io
    import sys
    from pathlib import Path

    import altair as alt
    import marimo as mo
    import polars as pl

    _ = alt.data_transformers.disable_max_rows()
    return Path, alt, io, mo, pl, sys


@app.cell
async def _(Path, io, mo, pl, sys):
    async def load(name):
        # In the browser the notebook directory is a URL, so fetch the static file next to the page.
        # Locally it is a plain path. No model runs here: every number was computed offline.
        loc = mo.notebook_location()
        if "pyodide" in sys.modules:
            from pyodide.http import pyfetch

            r = await pyfetch(str(loc / "public" / "data" / name))
            return pl.read_parquet(io.BytesIO(await r.bytes()))
        return pl.read_parquet(Path(loc) / "public" / "data" / name)

    comparison = await load("comparison.parquet")
    examples = await load("examples.parquet")
    players = await load("example_players.parquet")
    surfaces = await load("surfaces.parquet")
    error_map = await load("error_map.parquet")
    breakdown = await load("breakdown.parquet")
    return breakdown, comparison, error_map, examples, players, surfaces


@app.cell
def _(mo):
    mo.md(r"""
    # Can a graph attention network read a pass better than a gradient-boosted tree?

    **Question.** Given a StatsBomb 360 freeze frame (every visible player) and a pass's start and end
    location, how likely is the pass to be completed? A LightGBM model on hand-crafted frame features
    against a graph attention network that reads the players as a set. Same passes, same match-level
    chronological splits, same metrics.

    **Answer.** They tie within noise and the tree edges it: test log loss 0.1748 vs 0.1787, a gap of
    0.0040 with 95% CI [-0.0006, 0.0088] under a paired bootstrap over test matches. Twenty geometric
    features distil almost everything the frame has to say about a single pass.

    *Data: [StatsBomb open data](https://github.com/statsbomb/open-data), 127 club matches with 360 frames
    (Bayer Leverkusen 2023/24, Barcelona 2020/21, PSG 2021/22 and 2022/23), used under the
    StatsBomb public-data licence and not redistributed. Only derived tables are served here.*
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 0. What we are trying to achieve, and how each approach works

    **The target.** For every open-play pass we ask: given only what is knowable the instant the ball
    leaves the passer's foot, what is the probability the pass reaches a teammate? Clubs call this an
    xPass or pass-completion model. It is used to price the risk of a pass, to credit players who
    complete passes harder than expected, and in match review to ask "what were the options here".
    The label is simply whether StatsBomb recorded the pass as completed.

    **The real question is about representation, not accuracy.** A StatsBomb 360 freeze frame is a
    set of players with positions and team labels. There are two ways to hand that set to a model:
    compress it into a handful of numbers a football person would choose (distance to the nearest
    opponent, opponents in the passing lane, and so on), or hand the model the players themselves and
    let it learn which relationships matter. The first is what gradient-boosted trees need. The second
    is what a graph neural network is for. Everything else is held fixed: same passes, same match-level
    chronological splits, same inputs known before the kick (pass height and body part are excluded
    because they are only observed after it), same metrics, same bootstrap. Only the representation and
    the model class differ, so the gap between the two rows is the value of reading the frame as a set.

    **What counts as a win.** Not the point estimate. The two models are scored pass-for-pass on the same
    18 test matches, and a paired bootstrap over those matches gives a confidence interval on the
    difference. A win is that interval excluding zero. Two trivial baselines, fixed before any model was
    trained, set the floor that any model claiming to read the frame must clear.

    | Approach | What it sees | Inductive bias | Parameters | Test log loss |
    |---|---|---|---:|---:|
    | **Constant rate** | Nothing. One number for every pass. | The label's base rate. | 0 | 0.3632 |
    | **Length x third lookup** | Pass length (6 bins) and which third of the pitch it ends in. | Long passes and passes forward fail more. | 18 | 0.3115 |
    | **LightGBM** | 30 hand-made numbers summarising the pass and the frame. | Axis-aligned thresholds on fixed-radius counts and distances. | 37,170 | 0.1748 |
    | **Graph attention net** | The frame itself: every visible player as a token, plus a pass token. | Learned pairwise attention, biased by relative offset. | 106,637 | 0.1787 |

    **Constant rate.** Predicts the training-set completion rate, 0.872, for every pass. It answers
    "how much does the label explain by itself" and is the floor for log loss.

    **Length x third lookup.** An 18-cell table: completion rate in training data by pass length bin
    (under 10, 10 to 20, ..., 50 and over) crossed with the third of the pitch the pass ends in. It
    captures the two things everyone already knows, that long passes and passes into the final third
    fail more, with no learning beyond counting.

    **LightGBM.** The freeze frame is reduced to 30 scalars before the model ever sees it: pass geometry
    (length, angle, start and end, distance to goal, whether the target is in the box); what is around the
    target (nearest opponent, opponents within 3, 6 and 10 units, nearest teammate, teammates within 6);
    what is along the lane (opponents within 2, 4 and 8 units of the segment, the closest one); pressure
    on the passer and the nearest opponent to him; and the opposing keeper's depth. A 590-tree ensemble
    with 63 leaves splits on thresholds of these numbers, which is why its surface in chart 1 is blocky.
    Its strength is that every feature is a sensible football quantity, it trains in five seconds, and
    it degrades gracefully. Its weakness is that the radii are guesses and the identity of *which*
    defender is where is lost in the counting.

    **Graph attention net.** The frame stays a set. Every visible player becomes a token of 13 numbers:
    position, teammate and keeper flags, offset from the pass start and from the pass end, distance to
    the lane and where along it. A fourteenth token carries the pass itself. Three rounds of four-head
    attention let every token look at every other token, with a small network turning each pair's
    relative offset into an attention bias, so "who is close to whom" enters the model directly rather
    than through a chosen radius. The prediction is read from the pass token. In principle it can learn
    any configuration pattern: a defender between two teammates, a keeper covering a channel, a marker
    on the receiver's blind side. In practice its 107k parameters must learn pitch geometry from 76k
    training passes, and it starts to overfit after about ten epochs.
    """)
    return


@app.cell
def _(alt, pl):
    def pitch_layers(stroke="#f4f4f4", width=1.2):
        boxes = pl.DataFrame(
            {
                "x": [0.0, 0.0, 102.0, 0.0, 114.0],
                "y": [0.0, 18.0, 18.0, 30.0, 30.0],
                "x2": [120.0, 18.0, 120.0, 6.0, 120.0],
                "y2": [80.0, 62.0, 62.0, 50.0, 50.0],
            }
        )
        rect = alt.Chart(boxes).mark_rect(fill=None, stroke=stroke, strokeWidth=width).encode(
            x=alt.X("x:Q", scale=alt.Scale(domain=[0, 120]), axis=None),
            y=alt.Y("y:Q", scale=alt.Scale(domain=[80, 0]), axis=None),
            x2="x2:Q",
            y2="y2:Q",
        )
        half = alt.Chart(pl.DataFrame({"x": [60.0], "y": [0.0], "y2": [80.0]})).mark_rule(stroke=stroke, strokeWidth=width).encode(x="x:Q", y="y:Q", y2="y2:Q")
        circle = alt.Chart(pl.DataFrame({"x": [60], "y": [40]})).mark_point(size=2600, stroke=stroke, strokeWidth=width, fill=None).encode(x="x:Q", y="y:Q")
        spots = alt.Chart(pl.DataFrame({"x": [12, 60, 108], "y": [40, 40, 40]})).mark_point(size=12, fill=stroke, stroke=None).encode(x="x:Q", y="y:Q")
        return [rect, half, circle, spots]

    return (pitch_layers,)


@app.cell
def _(mo):
    mo.md(r"""
    ## 1. Every passing option, scored by both models

    Pick a real test-set pass. The colour is each model's completion probability for a pass from the same spot
    to every 4x4 cell of the pitch, given the players in the frame at that instant. The dashed line is the pass
    actually played. Attack runs left to right. Only players in the broadcast frame are drawn.
    """)
    return


@app.cell
def _(examples, mo):
    labels = {
        int(r["example"]): f"{r['player']} ({r['team']}), {r['minute']}', {r['home']} v {r['away']}, "
        f"{'completed' if r['completed'] else 'incomplete'}"
        for r in examples.iter_rows(named=True)
    }
    pick = mo.ui.dropdown(options={v: k for k, v in labels.items()}, value=list(labels.values())[0], label="Situation")
    pick
    return (pick,)


@app.cell
def _(examples, pick, pl):
    ex = examples.filter(pl.col("example") == pick.value).row(0, named=True)
    return (ex,)


@app.cell
def _(alt, ex, mo, pick, pitch_layers, pl, players, surfaces):
    ROLES = ["passer", "teammate", "opponent", "opponent keeper"]
    ppl = players.filter(pl.col("example") == pick.value).with_columns(
        role=pl.when("actor").then(pl.lit("passer")).when("teammate").then(pl.lit("teammate")).when("keeper").then(pl.lit("opponent keeper")).otherwise(pl.lit("opponent"))
    )
    surf = surfaces.filter(pl.col("example") == pick.value).with_columns(
        x1=pl.col("end_x") - 2, x2=pl.col("end_x") + 2, y1=pl.col("end_y") - 2, y2=pl.col("end_y") + 2
    )
    pass_line = pl.DataFrame({"x": [ex["x"]], "y": [ex["y"]], "x2": [ex["end_x"]], "y2": [ex["end_y"]]})

    def panel(model, p_actual):
        heat = (
            alt.Chart(surf.filter(pl.col("model") == model))
            .mark_rect(opacity=0.92)
            .encode(
                x=alt.X("x1:Q", scale=alt.Scale(domain=[0, 120]), axis=None),
                x2="x2:Q",
                y=alt.Y("y1:Q", scale=alt.Scale(domain=[80, 0]), axis=None),
                y2="y2:Q",
                color=alt.Color("p:Q", scale=alt.Scale(scheme="viridis", domain=[0, 1]), title="P(complete)"),
                tooltip=[alt.Tooltip("end_x:Q", title="end x"), alt.Tooltip("end_y:Q", title="end y"), alt.Tooltip("p:Q", format=".2f", title="P(complete)")],
            )
        )
        line = alt.Chart(pass_line).mark_rule(stroke="white", strokeWidth=2.5, strokeDash=[6, 3]).encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q")
        target = alt.Chart(pass_line).mark_point(shape="cross", size=160, stroke="white", strokeWidth=2.5).encode(x="x2:Q", y="y2:Q")
        dots = (
            alt.Chart(ppl)
            .mark_point(size=110, filled=True, stroke="white", strokeWidth=1)
            .encode(
                x="px:Q",
                y="py:Q",
                color=alt.Color("role:N", scale=alt.Scale(domain=ROLES, range=["#ffffff", "#4cc9f0", "#f72585", "#ffb703"]), legend=alt.Legend(title=None, orient="bottom")),
                shape=alt.Shape("role:N", scale=alt.Scale(domain=ROLES, range=["diamond", "circle", "circle", "square"]), legend=None),
                tooltip=["role:N"],
            )
        )
        return alt.layer(heat, *pitch_layers(), line, target, dots).properties(
            width=440, height=300, title=alt.Title(f"{model}: P(complete) = {p_actual:.2f}", anchor="start", fontSize=14)
        )

    chart1 = (
        alt.hconcat(panel("LightGBM", ex["p_tree"]), panel("Graph attention net", ex["p_gat"]))
        .resolve_scale(color="independent")
        .configure_view(fill="#2a6f3a", stroke=None)
        .configure(background="#fafafa")
    )
    mo.ui.altair_chart(chart1)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 2. The head-to-head
    """)
    return


@app.cell
def _(comparison, mo, pl):
    def ci(col, digits):
        return pl.format("{} [{}, {}]", pl.col(col).round(digits), pl.col(f"{col}_lo").round(digits), pl.col(f"{col}_hi").round(digits))

    table = comparison.select(
        pl.col("model").alias("Model"),
        ci("log_loss", 4).alias("Log loss [95% CI]"),
        ci("brier", 4).alias("Brier [95% CI]"),
        ci("auc", 3).alias("ROC AUC [95% CI]"),
        ci("passes_err_per_match", 2).alias("Completed-pass error per match [95% CI]"),
        pl.col("n_params").alias("Parameters"),
        pl.col("train_s").round(1).alias("Train time (s)"),
    )
    mo.ui.table(table, selection=None, show_column_summaries=False, pagination=False)
    return


@app.cell
def _(mo):
    mo.md(r"""
    *15,186 open-play passes across 18 held-out test matches, the last 15% of each competition-season.
    CIs are percentile bootstraps over test matches. Completed-pass error is |predicted minus actual completed passes| per match, in passes.
    LightGBM parameters are trees x leaves; its time excludes an 82 s random search.*
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## 3. Where each model is better

    Mean test log loss of the tree minus the graph attention net, by where the pass ended, in 10x10 cells.
    Blue: the GAT is better there. Red: the tree is better. Hover a cell for counts and completion rates.
    """)
    return


@app.cell
def _(error_map, pl):
    em = error_map.with_columns(x1=pl.col("cx") * 10, x2=pl.col("cx") * 10 + 10, y1=pl.col("cy") * 10, y2=pl.col("cy") * 10 + 10)
    best = em.sort("diff", descending=True).head(1).row(0, named=True)
    worst = em.sort("diff").head(1).row(0, named=True)
    return best, em, worst


@app.cell
def _(alt, best, em, mo, pitch_layers, pl, worst):
    LIM = 0.08  # clamp: two corner cells with 16-37 passes sit at -0.3 and would wash out the interior
    heat = (
        alt.Chart(em)
        .mark_rect(stroke="#fafafa", strokeWidth=0.5)
        .encode(
            x=alt.X("x1:Q", scale=alt.Scale(domain=[0, 120]), axis=None),
            x2="x2:Q",
            y=alt.Y("y1:Q", scale=alt.Scale(domain=[80, 0]), axis=None),
            y2="y2:Q",
            color=alt.Color("diff:Q", scale=alt.Scale(scheme="redblue", domain=[-LIM, LIM], clamp=True), title=["Tree log loss", "minus GAT log loss", "(clamped at ±0.08)"]),
            tooltip=[
                alt.Tooltip("n:Q", title="test passes"),
                alt.Tooltip("completion_rate:Q", format=".2f", title="completion rate"),
                alt.Tooltip("ll_tree:Q", format=".3f", title="tree log loss"),
                alt.Tooltip("ll_gat:Q", format=".3f", title="GAT log loss"),
                alt.Tooltip("diff:Q", format="+.3f", title="difference"),
            ],
        )
    )
    notes = pl.DataFrame({"x": [best["x1"] + 5, worst["x1"] + 5, 110], "y": [best["y1"] + 5, worst["y1"] + 5, 40], "label": ["A", "B", "C"]})
    marks = alt.Chart(notes).mark_point(size=520, stroke="#111", strokeWidth=1.5, fill="#fafafa", opacity=0.9).encode(x="x:Q", y="y:Q")
    text = alt.Chart(notes).mark_text(fontSize=13, fontWeight="bold", color="#111").encode(x="x:Q", y="y:Q", text="label:N")
    chart2 = alt.layer(heat, *pitch_layers(stroke="#555", width=1), marks, text).properties(width=720, height=480).configure_view(stroke=None).configure(background="#fafafa")
    mo.ui.altair_chart(chart2)
    return


@app.cell
def _(mo):
    mo.md(r"""
    - **A** (x 90-100, y 50-60, 177 passes): the GAT's largest edge, +0.047 log loss (+11%). Passes arriving at the
      edge of the box, where the receiver is marked and the exact arrangement of the two or three nearest defenders
      decides the outcome; this is what a set-based model is for, and it shows up, but only here.
    - **B** (x 0-10, y 70-80, 16 passes): the tree's largest edge, -0.294. Corner and touchline cells hold a few dozen
      test passes each and the broadcast frame is often cropped there; the GAT extrapolates badly at the boundary while
      the tree's bounded features degrade gracefully.
    - **C** the penalty area (x over 100, central): passes into the box net +0.001 for the GAT over 908 passes, against
      -0.014 in the defensive third over 2,746. Where passes are easy the tree's twenty features are sufficient statistics
      and the GAT only adds variance; the small GAT gains sit where passes are hard, and the long-ball losses (30+ units,
      see table) outweigh them.
    """)
    return


@app.cell
def _(breakdown, mo, pl):
    slices = (
        breakdown.pivot(on="model", index=["group", "bin", "n"], values="log_loss")
        .with_columns(edge=pl.col("LightGBM") - pl.col("Graph attention net"))
        .sort("group", "bin")
        .select(
            pl.col("group").alias("Slice"),
            pl.col("bin").alias("Bin"),
            pl.col("n").alias("Test passes"),
            pl.col("LightGBM").round(4).alias("Tree log loss"),
            pl.col("Graph attention net").round(4).alias("GAT log loss"),
            pl.col("edge").round(4).alias("Tree minus GAT"),
        )
    )
    mo.ui.table(slices, selection=None, show_column_summaries=False, pagination=False)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Reproduce

    108,034 open-play passes with a 360 frame from 127 club matches, split 70/15/15 by match and chronologically within each
    competition-season; an assertion checks that no match or possession straddles a split. Training lives in `train.py`, a marimo
    notebook with one cell per model. Code, data instructions and leakage checks:
    [github.com/igor17400/pass-prediction-analysis](https://github.com/igor17400/pass-prediction-analysis).
    """)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
