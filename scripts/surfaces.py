"""Offline inference artifacts for the app (public/data/). No model runs in the browser.

examples.parquet   players + pass line for a handful of real test passes
surfaces.parquet   completion probability of every candidate end location, per model, per example
error_map.parquet  12x8 grid of mean test log loss per model by pass end location
breakdown.parquet  log loss by pass length bin and by pressure, per model
"""
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import node_features, pass_features, tree_features  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CACHE, DATA = ROOT / "cache", ROOT / "public" / "data"
GX, GY = np.arange(2, 120, 4.0), np.arange(2, 80, 4.0)  # 30 x 20 candidate end locations
N_EXAMPLES = 6


def pick_examples(passes: pl.DataFrame) -> pl.DataFrame:
    """Test passes into the final third with a well populated frame; where the two models disagree most."""
    lg = pl.read_parquet(CACHE / "preds_lightgbm.parquet").select("pid", p_tree="p")
    gt = pl.read_parquet(CACHE / "preds_gat.parquet").select("pid", p_gat="p")
    n = pl.read_parquet(CACHE / "frames.parquet").group_by("event_id").agg(pl.len().alias("n_vis"))
    c = (
        passes.filter(pl.col("split") == "test", pl.col("end_x") > 80, pl.col("end_x") - pl.col("x") > 10)
        .join(lg, on="pid").join(gt, on="pid").join(n, on="event_id")
        .filter(pl.col("n_vis") >= 16)
        .with_columns(gap=(pl.col("p_tree") - pl.col("p_gat")).abs())
        .sort("gap", descending=True)
    )
    # two per outcome type at least, then fill by disagreement
    return pl.concat([c.filter("completed").head(3), c.filter(~pl.col("completed")).head(3)]).head(N_EXAMPLES).with_row_index("example")


def main() -> None:
    passes = pl.read_parquet(CACHE / "passes.parquet").with_columns(pid=pl.col("event_id"))
    frames = pl.read_parquet(CACHE / "frames.parquet")
    ex = pick_examples(passes)
    ex_frames = frames.join(ex.select("event_id", "example"), on="event_id")

    # candidate passes: same frame and start, every grid cell as the end location
    grid = pl.DataFrame({"end_x": np.repeat(GX, len(GY)), "end_y": np.tile(GY, len(GX))})
    cand = ex.select("example", "event_id", "x", "y", "under_pressure").join(grid, how="cross").with_columns(
        pid=pl.format("{}_{}_{}", "example", "end_x", "end_y")
    )
    booster = lgb.Booster(model_file=str(CACHE / "lgbm.txt"))
    tf = tree_features(cand, frames)
    p_tree = booster.predict(tf.drop("pid").to_numpy().astype(np.float32), num_threads=1)  # single thread: torch and lightgbm openmp runtimes deadlock otherwise
    # torch is imported only after LightGBM has finished: loading both OpenMP runtimes first deadlocks on macOS
    import torch

    torch.set_num_threads(1)
    from deep import PlayerGAT, build_tensors, predict

    model = PlayerGAT()
    model.load_state_dict(torch.load(CACHE / "gat.pt", map_location="cpu"))
    p_gat = predict(model, *build_tensors(node_features(cand, frames), pass_features(cand)), torch.device("cpu"))
    surfaces = pl.concat(
        [
            cand.select("example", "end_x", "end_y").with_columns(model=pl.lit("LightGBM"), p=pl.Series(p_tree)),
            cand.select("example", "end_x", "end_y").with_columns(model=pl.lit("Graph attention net"), p=pl.Series(p_gat.astype(np.float64))),
        ]
    )
    surfaces.write_parquet(DATA / "surfaces.parquet")
    ex.select("example", "comp", "match_date", "home", "away", "team", "player", "minute", "x", "y", "end_x", "end_y", "completed", "under_pressure", "p_tree", "p_gat").write_parquet(DATA / "examples.parquet")
    ex_frames.select("example", "px", "py", "teammate", "actor", "keeper").write_parquet(DATA / "example_players.parquet")

    # spatial error map on the test set, by pass end location, 12 x 8 cells of 10 x 10 units
    preds = pl.concat(
        [pl.read_parquet(CACHE / f"preds_{k}.parquet").filter(pl.col("split") == "test").with_columns(model=pl.lit(n)) for k, n in (("lightgbm", "LightGBM"), ("gat", "Graph attention net"))]
    ).join(passes.select("pid", "x", "y", "end_x", "end_y", "under_pressure"), on="pid")
    y, p = pl.col("completed").cast(pl.Float64), pl.col("p").clip(1e-6, 1 - 1e-6)
    preds = preds.with_columns(ll=-(y * p.log() + (1 - y) * (1 - p).log()), length=((pl.col("end_x") - pl.col("x")) ** 2 + (pl.col("end_y") - pl.col("y")) ** 2).sqrt())
    emap = preds.with_columns(cx=(pl.col("end_x") / 10).floor().clip(0, 11), cy=(pl.col("end_y") / 10).floor().clip(0, 7)).group_by("cx", "cy", "model").agg(
        pl.len().alias("n"), pl.col("ll").mean().alias("log_loss"), pl.col("completed").mean().alias("completion_rate")
    )
    emap.pivot(on="model", index=["cx", "cy", "n", "completion_rate"], values="log_loss").rename({"LightGBM": "ll_tree", "Graph attention net": "ll_gat"}).with_columns(
        diff=pl.col("ll_tree") - pl.col("ll_gat")
    ).write_parquet(DATA / "error_map.parquet")
    breakdown = pl.concat(
        [
            preds.with_columns(group=pl.lit("length"), bin=pl.when(pl.col("length") < 10).then(pl.lit("<10")).when(pl.col("length") < 20).then(pl.lit("10-20")).when(pl.col("length") < 30).then(pl.lit("20-30")).otherwise(pl.lit("30+"))),
            preds.with_columns(group=pl.lit("pressure"), bin=pl.when("under_pressure").then(pl.lit("under pressure")).otherwise(pl.lit("no pressure"))),
            preds.with_columns(group=pl.lit("end zone"), bin=pl.when(pl.col("end_x") < 40).then(pl.lit("own third")).when(pl.col("end_x") < 80).then(pl.lit("middle third")).otherwise(pl.lit("final third"))),
        ]
    ).group_by("group", "bin", "model").agg(pl.len().alias("n"), pl.col("ll").mean().alias("log_loss"), pl.col("completed").mean().alias("completion_rate"))
    breakdown.write_parquet(DATA / "breakdown.parquet")
    print(ex.select("example", "player", "minute", "completed", "p_tree", "p_gat"))
    print(f"sizes: {sum(f.stat().st_size for f in DATA.glob('*.parquet')) / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
