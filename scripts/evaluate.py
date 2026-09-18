"""Score every prediction file on the test matches with the same metrics and CIs. Writes data/comparison.parquet."""
import json
from pathlib import Path

import polars as pl

from metrics import evaluate, paired_delta

ROOT = Path(__file__).resolve().parents[1]
CACHE, DATA = ROOT / "cache", ROOT / "public" / "data"
MODELS = {"constant": "Constant rate", "lookup": "Length x third lookup", "lightgbm": "LightGBM", "gat": "Graph attention net"}


def main() -> None:
    lg = json.load(open(CACHE / "lgbm_info.json"))
    gat = json.load(open(CACHE / "gat_info.json"))
    extra = {"lightgbm": (lg["n_params"], lg["train_s"], lg["tune_s"]), "gat": (gat["n_params"], gat["train_s"], 0.0)}
    rows, ref = [], None
    for key, name in MODELS.items():
        df = pl.read_parquet(CACHE / f"preds_{key}.parquet").filter(pl.col("split") == "test")
        if ref is None:
            ref = df.select("pid", "match_id", "completed")
        assert df.select("pid", "match_id", "completed").sort("pid").equals(ref.sort("pid")), "models scored on different rows"
        n_params, train_s, tune_s = extra.get(key, (0, 0.0, 0.0))
        rows.append(dict(model=name, **evaluate(df), n_params=n_params, train_s=train_s, tune_s=tune_s, n_test=len(df), n_test_matches=df["match_id"].n_unique()))
    out = pl.DataFrame(rows)
    out.write_parquet(DATA / "comparison.parquet")
    # paired bootstrap of the tree-minus-GAT gap over test matches: the honest test of "who won"
    a = pl.read_parquet(CACHE / "preds_lightgbm.parquet").filter(pl.col("split") == "test").select("pid", "match_id", "completed", p_tree="p")
    b = pl.read_parquet(CACHE / "preds_gat.parquet").filter(pl.col("split") == "test").select("pid", p_gat="p")
    paired = a.join(b, on="pid")
    delta = paired_delta(paired)
    pl.DataFrame([delta]).write_parquet(DATA / "delta.parquet")
    print("tree minus GAT log loss:", {k: round(v, 4) for k, v in delta.items()})
    with pl.Config(tbl_cols=-1, tbl_width_chars=200, float_precision=4):
        print(out.select("model", "log_loss", "log_loss_lo", "log_loss_hi", "brier", "auc", "auc_lo", "auc_hi", "passes_err_per_match", "n_params", "train_s"))


if __name__ == "__main__":
    main()
