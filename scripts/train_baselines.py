"""Two trivial baselines, stated before anything was trained:
B0 constant: the train-set completion rate.
B1 lookup: completion rate by (pass length bin x end-location third), fitted on train.
"""
from pathlib import Path

import polars as pl

CACHE = Path(__file__).resolve().parents[1] / "cache"


def main() -> None:
    df = pl.read_parquet(CACHE / "tree_features.parquet").with_columns(
        len_bin=(pl.col("length") / 10).floor().clip(0, 5),
        third=(pl.col("end_x") / 40).floor().clip(0, 2),
    )
    train = df.filter(pl.col("split") == "train")
    rate = train["completed"].mean()
    table = train.group_by("len_bin", "third").agg(pl.col("completed").mean().alias("p"))
    keep = ["pid", "match_id", "completed", "split"]
    df.select(keep).with_columns(p=pl.lit(rate)).write_parquet(CACHE / "preds_constant.parquet")
    df.join(table, on=["len_bin", "third"], how="left").with_columns(pl.col("p").fill_null(rate)).select(keep + ["p"]).write_parquet(CACHE / "preds_lookup.parquet")
    print(f"constant rate {rate:.3f}; lookup table {len(table)} cells")


if __name__ == "__main__":
    main()
