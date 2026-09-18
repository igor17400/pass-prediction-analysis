# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "polars",
#     "altair",
#     "pyarrow",
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

    return Path, alt, io, mo, pl, sys


@app.cell
async def _(Path, io, mo, pl, sys):
    async def load(name):
        # In the browser the notebook directory is a URL: fetch the static file that sits next to the page.
        # Locally it is a plain path. No model runs here either way; every number was computed offline in scripts/.
        loc = mo.notebook_location()
        if "pyodide" in sys.modules:
            from pyodide.http import pyfetch

            r = await pyfetch(str(loc / "public" / "data" / name))
            return pl.read_parquet(io.BytesIO(await r.bytes()))
        return pl.read_parquet(Path(loc) / "public" / "data" / name)

    comparison = await load("comparison.parquet")
    delta = (await load("delta.parquet")).row(0, named=True)
    examples = await load("examples.parquet")
    players = await load("example_players.parquet")
    surfaces = await load("surfaces.parquet")
    error_map = await load("error_map.parquet")
    breakdown = await load("breakdown.parquet")
    return breakdown, comparison, delta, error_map, examples, players, surfaces


@app.cell
def _(comparison, delta, mo):
    _tree = comparison.row(2, named=True)
    _gat = comparison.row(3, named=True)
    mo.md(
        f"""
# Can a graph attention network read a pass better than a gradient-boosted tree?

**Question.** Given a StatsBomb 360 freeze frame (every visible player) and a pass's start and end
location, how likely is the pass to be completed? A LightGBM model on hand-crafted frame features
against a graph attention network that reads the players as a set. Same passes, same match-level
chronological splits, same metrics.

**Answer.** They tie within noise and the tree edges it: test log loss {_tree["log_loss"]:.4f} vs
{_gat["log_loss"]:.4f}, a gap of {-delta["delta_log_loss"]:.4f} with 95% CI
[{-delta["delta_hi"]:.4f}, {-delta["delta_lo"]:.4f}] under a paired bootstrap over test matches.
Twenty geometric features distil almost everything the frame has to say about a single pass.

*Data: [StatsBomb open data](https://github.com/statsbomb/open-data), 127 club matches with 360 frames
(Bayer Leverkusen 2023/24, Barcelona 2020/21, PSG 2021/22 and 2022/23), used under the
StatsBomb public-data licence and not redistributed. Only derived tables are served here.*
"""
    )
    return


@app.cell
def _(alt, examples, mo):
    _labels = {
        int(r["example"]): f"{r['player']} ({r['team']}), {r['minute']}', {r['home']} v {r['away']}, "
        f"{'completed' if r['completed'] else 'incomplete'}"
        for r in examples.iter_rows(named=True)
    }
    pick = mo.ui.dropdown(options={v: k for k, v in _labels.items()}, value=list(_labels.values())[0], label="Situation")
    _ = alt.data_transformers.disable_max_rows()
    return (pick,)


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
def _(alt, examples, mo, pick, pitch_layers, pl, players, surfaces):
    _ex = examples.filter(pl.col("example") == pick.value).row(0, named=True)
    _ppl = players.filter(pl.col("example") == pick.value).with_columns(
        role=pl.when("actor").then(pl.lit("passer")).when("teammate").then(pl.lit("teammate")).when("keeper").then(pl.lit("opponent keeper")).otherwise(pl.lit("opponent"))
    )
    _surf = surfaces.filter(pl.col("example") == pick.value).with_columns(x1=pl.col("end_x") - 2, x2=pl.col("end_x") + 2, y1=pl.col("end_y") - 2, y2=pl.col("end_y") + 2)
    _pass = pl.DataFrame({"x": [_ex["x"]], "y": [_ex["y"]], "x2": [_ex["end_x"]], "y2": [_ex["end_y"]]})

    def _panel(model, p_actual):
        heat = (
            alt.Chart(_surf.filter(pl.col("model") == model))
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
        line = alt.Chart(_pass).mark_rule(stroke="white", strokeWidth=2.5, strokeDash=[6, 3]).encode(x="x:Q", y="y:Q", x2="x2:Q", y2="y2:Q")
        target = alt.Chart(_pass).mark_point(shape="cross", size=160, stroke="white", strokeWidth=2.5).encode(x="x2:Q", y="y2:Q")
        dots = (
            alt.Chart(_ppl)
            .mark_point(size=110, filled=True, stroke="white", strokeWidth=1)
            .encode(
                x="px:Q",
                y="py:Q",
                color=alt.Color(
                    "role:N",
                    scale=alt.Scale(domain=["passer", "teammate", "opponent", "opponent keeper"], range=["#ffffff", "#4cc9f0", "#f72585", "#ffb703"]),
                    legend=alt.Legend(title=None, orient="bottom"),
                ),
                shape=alt.Shape("role:N", scale=alt.Scale(domain=["passer", "teammate", "opponent", "opponent keeper"], range=["diamond", "circle", "circle", "square"]), legend=None),
                tooltip=["role:N"],
            )
        )
        return alt.layer(heat, *pitch_layers(), line, target, dots).properties(width=440, height=300, title=alt.Title(f"{model}: P(complete) = {p_actual:.2f}", anchor="start", fontSize=14))

    chart1 = alt.hconcat(_panel("LightGBM", _ex["p_tree"]), _panel("Graph attention net", _ex["p_gat"])).resolve_scale(color="independent").configure_view(fill="#2a6f3a", stroke=None).configure(background="#fafafa")
    _outcome = "completed" if _ex["completed"] else "was not completed"
    mo.vstack(
        [
            mo.md("## 1. Every passing option, scored by both models"),
            mo.md(
                "Pick a real test-set pass. The colour is each model's completion probability for a pass from the same spot "
                "to every 4x4 cell of the pitch, given the players in the frame at that instant. The dashed line is the pass actually played. "
                "Attack runs left to right."
            ),
            pick,
            mo.ui.altair_chart(chart1),
            mo.md(
                f"*{_ex['player']} ({_ex['team']}), {_ex['comp']}, {_ex['home']} v {_ex['away']}, minute {_ex['minute']}. "
                f"The pass {_outcome}{', played under pressure' if _ex['under_pressure'] else ''}. Only players in the broadcast frame are drawn.*"
            ),
        ]
    )
    return


@app.cell
def _(comparison, mo, pl):
    _t = comparison.select(
        pl.col("model").alias("Model"),
        pl.format("{} [{}, {}]", pl.col("log_loss").round(4), pl.col("log_loss_lo").round(4), pl.col("log_loss_hi").round(4)).alias("Log loss [95% CI]"),
        pl.format("{} [{}, {}]", pl.col("brier").round(4), pl.col("brier_lo").round(4), pl.col("brier_hi").round(4)).alias("Brier [95% CI]"),
        pl.format("{} [{}, {}]", pl.col("auc").round(3), pl.col("auc_lo").round(3), pl.col("auc_hi").round(3)).alias("ROC AUC [95% CI]"),
        pl.format("{} [{}, {}]", pl.col("passes_err_per_match").round(2), pl.col("passes_err_per_match_lo").round(2), pl.col("passes_err_per_match_hi").round(2)).alias("Completed-pass error per match [95% CI]"),
        pl.col("n_params").alias("Parameters"),
        pl.col("train_s").round(1).alias("Train time (s)"),
    )
    _n = comparison.row(0, named=True)
    mo.vstack(
        [
            mo.md("## 2. The head-to-head"),
            mo.ui.table(_t, selection=None, show_column_summaries=False, pagination=False),
            mo.md(
                f"*{_n['n_test']:,} open-play passes across {_n['n_test_matches']} held-out test matches, the last 15% of each competition-season. "
                "CIs are percentile bootstraps over test matches. Completed-pass error is |predicted minus actual completed passes| per match, in passes. "
                "LightGBM parameters are trees x leaves; its time excludes an 82 s random search.*"
            ),
        ]
    )
    return


@app.cell
def _(alt, breakdown, error_map, mo, pitch_layers, pl):
    _em = error_map.with_columns(x1=pl.col("cx") * 10, x2=pl.col("cx") * 10 + 10, y1=pl.col("cy") * 10, y2=pl.col("cy") * 10 + 10, pct=100 * pl.col("diff") / pl.col("ll_tree"))
    _lim = 0.08  # clamp: two corner cells with 16-37 passes sit at -0.3 and would wash out the interior
    _heat = (
        alt.Chart(_em)
        .mark_rect(stroke="#fafafa", strokeWidth=0.5)
        .encode(
            x=alt.X("x1:Q", scale=alt.Scale(domain=[0, 120]), axis=None),
            x2="x2:Q",
            y=alt.Y("y1:Q", scale=alt.Scale(domain=[80, 0]), axis=None),
            y2="y2:Q",
            color=alt.Color("diff:Q", scale=alt.Scale(scheme="redblue", domain=[-_lim, _lim], clamp=True), title=["Tree log loss", "minus GAT log loss", "(clamped at ±0.08)"]),
            tooltip=[
                alt.Tooltip("n:Q", title="test passes"),
                alt.Tooltip("completion_rate:Q", format=".2f", title="completion rate"),
                alt.Tooltip("ll_tree:Q", format=".3f", title="tree log loss"),
                alt.Tooltip("ll_gat:Q", format=".3f", title="GAT log loss"),
                alt.Tooltip("diff:Q", format="+.3f", title="difference"),
            ],
        )
    )
    _best = _em.sort("diff", descending=True).head(1).row(0, named=True)
    _worst = _em.sort("diff").head(1).row(0, named=True)
    _box = _em.filter(pl.col("x1") >= 100, pl.col("y1").is_between(20, 50))
    _box_d = float((_box["diff"] * _box["n"]).sum() / _box["n"].sum())
    _own = _em.filter(pl.col("x1") < 40)
    _own_d = float((_own["diff"] * _own["n"]).sum() / _own["n"].sum())
    _notes = pl.DataFrame(
        {
            "x": [_best["x1"] + 5, _worst["x1"] + 5, 110],
            "y": [_best["y1"] + 5, _worst["y1"] + 5, 40],
            "label": ["A", "B", "C"],
        }
    )
    _marks = alt.Chart(_notes).mark_point(size=520, stroke="#111", strokeWidth=1.5, fill="#fafafa", opacity=0.9).encode(x="x:Q", y="y:Q")
    _text = alt.Chart(_notes).mark_text(fontSize=13, fontWeight="bold", color="#111").encode(x="x:Q", y="y:Q", text="label:N")
    chart2 = alt.layer(_heat, *pitch_layers(stroke="#555", width=1), _marks, _text).properties(width=720, height=480).configure_view(stroke=None).configure(background="#fafafa")
    _bd = (
        breakdown.pivot(on="model", index=["group", "bin", "n"], values="log_loss")
        .with_columns(edge=(pl.col("LightGBM") - pl.col("Graph attention net")))
        .sort("group", "bin")
        .select(pl.col("group").alias("Slice"), pl.col("bin").alias("Bin"), pl.col("n").alias("Test passes"), pl.col("LightGBM").round(4).alias("Tree log loss"), pl.col("Graph attention net").round(4).alias("GAT log loss"), pl.col("edge").round(4).alias("Tree minus GAT"))
    )
    mo.vstack(
        [
            mo.md("## 3. Where each model is better"),
            mo.md(
                "Mean test log loss of the tree minus the graph attention net, by where the pass ended, in 10x10 cells. "
                "Blue: the GAT is better there. Red: the tree is better. Hover a cell for counts and completion rates."
            ),
            mo.ui.altair_chart(chart2),
            mo.md(
                f"""
- **A** (x {_best['x1']:.0f}-{_best['x2']:.0f}, y {_best['y1']:.0f}-{_best['y2']:.0f}, {_best['n']} passes): the GAT's largest edge, {_best['diff']:+.3f} log loss ({_best['pct']:+.0f}%). Passes arriving at the edge of the box, where the receiver is marked and the exact arrangement of the two or three nearest defenders decides the outcome; this is what a set-based model is for, and it shows up, but only here.
- **B** (x {_worst['x1']:.0f}-{_worst['x2']:.0f}, y {_worst['y1']:.0f}-{_worst['y2']:.0f}, {_worst['n']} passes): the tree's largest edge, {_worst['diff']:+.3f}. Corner and touchline cells hold a few dozen test passes each and the broadcast frame is often cropped there; the GAT extrapolates badly at the boundary while the tree's bounded features degrade gracefully.
- **C** the penalty area (x over 100, central): passes into the box net {_box_d:+.3f} for the GAT over {int(_box['n'].sum())} passes, against {_own_d:+.3f} in the defensive third over {int(_own['n'].sum())}. Where passes are easy the tree's twenty features are sufficient statistics and the GAT only adds variance; the small GAT gains sit where passes are hard, and the long-ball losses (30+ units, see table) outweigh them.
"""
            ),
            mo.ui.table(_bd, selection=None, show_column_summaries=False, pagination=False),
        ]
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
## Method in short

108,034 open-play passes with a 360 frame from 127 club matches. Set pieces excluded. Inputs known before the ball is
kicked only: start, end, passer under pressure, and the frame (location, teammate, keeper flag of every visible player).
Pass height and body part are excluded because they are only observed once the pass is played. Splits are by match and
chronological within each competition-season (70/15/15); an assertion checks that no match or possession straddles a split.
Two trivial baselines (constant rate; completion rate by length bin and end third) were fixed before any model was trained.
LightGBM: 25-trial random search on validation log loss, early stopping. GAT: 3 blocks of 4-head attention over player tokens
plus a pass token, with a learned attention bias from each pair's relative offset; AdamW, early stopping on validation log loss,
12-minute wall-clock cap (it stopped itself after 90 s). Code and reproduction steps: [github.com/igor17400/pitch-nets](https://github.com/igor17400/pitch-nets).
"""
    )
    return


if __name__ == "__main__":
    app.run()
