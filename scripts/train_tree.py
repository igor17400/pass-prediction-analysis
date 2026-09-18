"""LightGBM on the hand-crafted freeze-frame features. Small random search on val, capped by time."""
import json
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl

CACHE = Path(__file__).resolve().parents[1] / "cache"
META = ["pid", "match_id", "possession", "comp", "split", "completed"]
TUNE_BUDGET_S = 480


def main() -> None:
    df = pl.read_parquet(CACHE / "tree_features.parquet")
    feats = [c for c in df.columns if c not in META]
    assert "completed" not in feats and "outcome" not in feats
    parts = {s: df.filter(pl.col("split") == s) for s in ("train", "val", "test")}
    X = {s: parts[s].select(feats).to_numpy().astype(np.float32) for s in parts}
    y = {s: parts[s]["completed"].to_numpy().astype(int) for s in parts}

    rng = np.random.default_rng(0)
    grid = dict(num_leaves=[15, 31, 63, 127], learning_rate=[0.02, 0.05, 0.1], min_child_samples=[20, 50, 200], feature_fraction=[0.6, 0.8, 1.0], lambda_l2=[0.0, 1.0, 10.0])
    best, t0, trials = None, time.time(), 0
    while time.time() - t0 < TUNE_BUDGET_S and trials < 25:
        params = {k: rng.choice(v).item() for k, v in grid.items()} | dict(objective="binary", verbose=-1, seed=trials, num_threads=8)
        m = lgb.train(params, lgb.Dataset(X["train"], y["train"]), 2000, valid_sets=[lgb.Dataset(X["val"], y["val"])], callbacks=[lgb.early_stopping(50, verbose=False)])
        score = m.best_score["valid_0"]["binary_logloss"]
        trials += 1
        if best is None or score < best[0]:
            best = (score, params, m.best_iteration)
    score, params, n_iter = best
    t1 = time.time()
    model = lgb.train(params, lgb.Dataset(X["train"], y["train"]), n_iter)
    train_s = time.time() - t1
    model.save_model(str(CACHE / "lgbm.txt"))
    preds = pl.concat([parts[s].select("pid", "match_id", "completed", "split").with_columns(p=pl.Series(model.predict(X[s]))) for s in parts])
    preds.write_parquet(CACHE / "preds_lightgbm.parquet")
    info = dict(params=params, n_iter=n_iter, val_logloss=score, trials=trials, tune_s=t1 - t0, train_s=train_s, n_params=int(model.num_trees()) * int(params["num_leaves"]), features=feats)
    json.dump(info, open(CACHE / "lgbm_info.json", "w"), indent=1)
    imp = sorted(zip(model.feature_importance("gain"), feats), reverse=True)[:10]
    print(f"{trials} trials in {t1 - t0:.0f}s, best val logloss {score:.4f}, {n_iter} trees x {params['num_leaves']} leaves; refit {train_s:.1f}s")
    print("top gain:", [(f, round(g)) for g, f in imp])


if __name__ == "__main__":
    main()
