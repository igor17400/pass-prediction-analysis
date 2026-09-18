"""Metrics with bootstrap CIs over test matches. Same function for every model."""
import numpy as np
import polars as pl
from sklearn.metrics import roc_auc_score


def _point(df: pl.DataFrame) -> dict:
    y, p = df["completed"].to_numpy().astype(float), df["p"].to_numpy().clip(1e-6, 1 - 1e-6)
    per_match = df.group_by("match_id").agg((pl.col("p").sum() - pl.col("completed").sum()).abs().alias("err"))
    return dict(
        log_loss=float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))),
        brier=float(np.mean((y - p) ** 2)),
        auc=float(roc_auc_score(y, p)) if 0 < y.mean() < 1 else float("nan"),
        passes_err_per_match=float(per_match["err"].mean()),
    )


def evaluate(df: pl.DataFrame, n_boot: int = 500, seed: int = 0) -> dict:
    """df has pid, match_id, completed, p. Bootstrap resamples matches with replacement."""
    point = _point(df)
    matches = df["match_id"].unique().to_numpy()
    rng = np.random.default_rng(seed)
    parts = df.partition_by("match_id", as_dict=True)
    boots = []
    for _ in range(n_boot):
        pick = rng.choice(matches, size=len(matches), replace=True)
        boots.append(_point(pl.concat([parts[(m,)] for m in pick])))
    out = {}
    for k, v in point.items():
        lo, hi = np.nanpercentile([b[k] for b in boots], [2.5, 97.5])
        out[k] = v
        out[f"{k}_lo"], out[f"{k}_hi"] = float(lo), float(hi)
    return out
