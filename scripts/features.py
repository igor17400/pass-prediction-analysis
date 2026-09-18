"""Feature construction shared by the tree model, the deep model and the surface renderer.

`passes` needs columns pid, event_id, x, y, end_x, end_y, under_pressure.
`frames` needs columns event_id, node, px, py, teammate, actor, keeper.
The frame is joined by event_id, so the same frame can be scored for many candidate
end locations (different pid) which is how the pitch surfaces are made.
Nothing here may read the label: see LEAK_COLS in the test.
"""
import polars as pl

GOAL = (120.0, 40.0)
LEAK_COLS = {"completed", "outcome", "height", "body_part", "recipient"}


def _geometry(passes: pl.DataFrame, frames: pl.DataFrame) -> pl.DataFrame:
    """One row per (pass, visible player) with distances to the pass start, end and lane."""
    j = passes.select("pid", "event_id", "x", "y", "end_x", "end_y").join(frames, on="event_id")
    dx, dy = pl.col("end_x") - pl.col("x"), pl.col("end_y") - pl.col("y")
    seg2 = (dx**2 + dy**2).clip(lower_bound=1e-6)
    t = (((pl.col("px") - pl.col("x")) * dx + (pl.col("py") - pl.col("y")) * dy) / seg2).clip(0, 1)
    return j.with_columns(
        d_start=((pl.col("px") - pl.col("x")) ** 2 + (pl.col("py") - pl.col("y")) ** 2).sqrt(),
        d_end=((pl.col("px") - pl.col("end_x")) ** 2 + (pl.col("py") - pl.col("end_y")) ** 2).sqrt(),
        t_lane=t,
        d_lane=((pl.col("px") - (pl.col("x") + t * dx)) ** 2 + (pl.col("py") - (pl.col("y") + t * dy)) ** 2).sqrt(),
    )


def tree_features(passes: pl.DataFrame, frames: pl.DataFrame) -> pl.DataFrame:
    """Hand-crafted scalar summary of the freeze frame, one row per pid."""
    g = _geometry(passes, frames)
    opp, tm = ~pl.col("teammate"), pl.col("teammate") & ~pl.col("actor")
    big = 60.0
    agg = g.group_by("pid").agg(
        n_visible=pl.len(),
        n_opp=opp.sum(),
        n_tm=tm.sum(),
        opp_min_end=pl.when(opp).then("d_end").min().fill_null(big),
        opp_n3_end=(opp & (pl.col("d_end") < 3)).sum(),
        opp_n6_end=(opp & (pl.col("d_end") < 6)).sum(),
        opp_n10_end=(opp & (pl.col("d_end") < 10)).sum(),
        tm_min_end=pl.when(tm).then("d_end").min().fill_null(big),
        tm_n6_end=(tm & (pl.col("d_end") < 6)).sum(),
        opp_min_start=pl.when(opp).then("d_start").min().fill_null(big),
        opp_n3_start=(opp & (pl.col("d_start") < 3)).sum(),
        opp_min_lane=pl.when(opp).then("d_lane").min().fill_null(big),
        opp_n2_lane=(opp & (pl.col("d_lane") < 2)).sum(),
        opp_n4_lane=(opp & (pl.col("d_lane") < 4)).sum(),
        opp_n8_lane=(opp & (pl.col("d_lane") < 8)).sum(),
        opp_ahead=(opp & (pl.col("px") > pl.col("x")) & (pl.col("px") < pl.col("end_x"))).sum(),
        opp_gk_x=pl.when(opp & pl.col("keeper")).then("px").max().fill_null(120.0),
        tm_min_end_minus_opp=(pl.when(tm).then("d_end").min().fill_null(big) - pl.when(opp).then("d_end").min().fill_null(big)),
    )
    dx, dy = pl.col("end_x") - pl.col("x"), pl.col("end_y") - pl.col("y")
    base = passes.select("pid", "x", "y", "end_x", "end_y", pl.col("under_pressure").cast(pl.Int8)).with_columns(
        length=(dx**2 + dy**2).sqrt(),
        angle=pl.arctan2(dy, dx),
        forward=dx,
        end_dist_goal=((pl.col("end_x") - GOAL[0]) ** 2 + (pl.col("end_y") - GOAL[1]) ** 2).sqrt(),
        start_dist_goal=((pl.col("x") - GOAL[0]) ** 2 + (pl.col("y") - GOAL[1]) ** 2).sqrt(),
        end_in_box=((pl.col("end_x") > 102) & pl.col("end_y").is_between(18, 62)).cast(pl.Int8),
        end_wide=(pl.col("end_y") - 40).abs(),
    )
    return base.join(agg, on="pid", how="left")


NODE_COLS = ["px_n", "py_n", "teammate", "actor", "keeper", "rel_sx", "rel_sy", "rel_ex", "rel_ey", "d_start_n", "d_end_n", "d_lane_n", "t_lane"]
PASS_COLS = ["sx_n", "sy_n", "ex_n", "ey_n", "len_n", "cos_a", "sin_a", "under_pressure"]


def node_features(passes: pl.DataFrame, frames: pl.DataFrame) -> pl.DataFrame:
    """Per-player token features for the attention model, one row per (pid, node)."""
    g = _geometry(passes, frames)
    return g.select(
        "pid",
        "node",
        px_n=pl.col("px") / 120,
        py_n=pl.col("py") / 80,
        teammate=pl.col("teammate").cast(pl.Float32),
        actor=pl.col("actor").cast(pl.Float32),
        keeper=pl.col("keeper").cast(pl.Float32),
        rel_sx=(pl.col("px") - pl.col("x")) / 40,
        rel_sy=(pl.col("py") - pl.col("y")) / 40,
        rel_ex=(pl.col("px") - pl.col("end_x")) / 40,
        rel_ey=(pl.col("py") - pl.col("end_y")) / 40,
        d_start_n=pl.col("d_start") / 40,
        d_end_n=pl.col("d_end") / 40,
        d_lane_n=pl.col("d_lane") / 40,
        t_lane=pl.col("t_lane"),
    )


def pass_features(passes: pl.DataFrame) -> pl.DataFrame:
    dx, dy = pl.col("end_x") - pl.col("x"), pl.col("end_y") - pl.col("y")
    return passes.select(
        "pid",
        sx_n=pl.col("x") / 120,
        sy_n=pl.col("y") / 80,
        ex_n=pl.col("end_x") / 120,
        ey_n=pl.col("end_y") / 80,
        len_n=(dx**2 + dy**2).sqrt() / 40,
        cos_a=dx / (dx**2 + dy**2).sqrt().clip(lower_bound=1e-6),
        sin_a=dy / (dx**2 + dy**2).sqrt().clip(lower_bound=1e-6),
        under_pressure=pl.col("under_pressure").cast(pl.Float32),
    )


if __name__ == "__main__":
    from pathlib import Path

    cache = Path(__file__).resolve().parents[1] / "cache"
    passes = pl.read_parquet(cache / "passes.parquet").with_columns(pid=pl.col("event_id"))
    frames = pl.read_parquet(cache / "frames.parquet")
    meta = passes.select("pid", "match_id", "possession", "comp", "split", "completed")
    tf = tree_features(passes, frames)
    assert not (set(tf.columns) & LEAK_COLS)
    tf.join(meta, on="pid").write_parquet(cache / "tree_features.parquet")
    node_features(passes, frames).write_parquet(cache / "node_features.parquet")
    pass_features(passes).join(meta, on="pid").write_parquet(cache / "pass_features.parquet")
    print(tf.describe())
