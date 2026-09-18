# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "marimo",
#     "polars",
#     "numpy",
#     "lightgbm",
#     "torch",
#     "scikit-learn",
# ]
# ///

import marimo

__generated_with = "0.24.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import json
    import sys
    import time
    from pathlib import Path

    import lightgbm as lgb
    import marimo as mo
    import numpy as np
    import polars as pl
    import torch

    ROOT = Path(__file__).resolve().parent
    CACHE = ROOT / "cache"
    DATA = ROOT / "public" / "data"
    sys.path.insert(0, str(ROOT / "scripts"))
    from deep import PlayerGAT, build_tensors, device, predict
    from features import LEAK_COLS, node_features, pass_features, tree_features
    from metrics import evaluate, paired_delta

    # LightGBM and torch ship separate OpenMP runtimes; on macOS a multi-threaded torch forward
    # after a LightGBM call deadlocks. Training runs on MPS/CUDA so this costs nothing there.
    torch.set_num_threads(1)
    return (
        CACHE,
        DATA,
        LEAK_COLS,
        PlayerGAT,
        build_tensors,
        device,
        evaluate,
        json,
        lgb,
        mo,
        node_features,
        np,
        paired_delta,
        pass_features,
        pl,
        predict,
        time,
        torch,
        tree_features,
    )


@app.cell
def _(mo):
    mo.md(r"""
    # Training: pass completion from 360 freeze frames

    Runs locally, never in the browser. Reads `cache/passes.parquet` and `cache/frames.parquet`
    (built by `scripts/build_dataset.py` from the StatsBomb JSON), trains every model, and writes
    the small artifacts the app reads into `public/data/`.

    Splits were assigned in the dataset builder: by match, chronologically within each
    competition-season, 70/15/15. Every model below is fitted on `train`, selected on `val`,
    and scored once on `test`.
    """)
    return


@app.cell
def _(CACHE, LEAK_COLS, node_features, pass_features, pl, tree_features):
    passes = pl.read_parquet(CACHE / "passes.parquet").with_columns(pid=pl.col("event_id"))
    frames = pl.read_parquet(CACHE / "frames.parquet")
    meta = passes.select("pid", "match_id", "possession", "comp", "split", "completed")

    tree_df = tree_features(passes, frames).join(meta, on="pid")
    node_df = node_features(passes, frames)
    pass_df = pass_features(passes).join(meta, on="pid")

    FEATURES = [c for c in tree_df.columns if c not in meta.columns]
    assert not (set(FEATURES) & LEAK_COLS), "a label-derived column leaked into the features"

    splits = {s: tree_df.filter(pl.col("split") == s) for s in ("train", "val", "test")}
    X = {s: splits[s].select(FEATURES).to_numpy().astype("float32") for s in splits}
    y = {s: splits[s]["completed"].to_numpy().astype(int) for s in splits}
    {s: (X[s].shape, round(float(y[s].mean()), 3)) for s in splits}
    return FEATURES, X, frames, node_df, pass_df, passes, splits, tree_df, y


@app.cell
def _(mo):
    mo.md(r"""
    ## Baselines

    Fixed before any model was trained. Constant: the train completion rate. Lookup: completion
    rate by pass length bin (10-unit bins, capped at 50+) crossed with the third of the pitch the
    pass ends in, 18 cells.
    """)
    return


@app.cell
def _(CACHE, pl, tree_df):
    binned = tree_df.with_columns(
        len_bin=(pl.col("length") / 10).floor().clip(0, 5),
        third=(pl.col("end_x") / 40).floor().clip(0, 2),
    )
    train_rate = binned.filter(pl.col("split") == "train")["completed"].mean()
    lookup = binned.filter(pl.col("split") == "train").group_by("len_bin", "third").agg(pl.col("completed").mean().alias("p"))

    PRED_COLS = ["pid", "match_id", "completed", "split"]
    binned.select(PRED_COLS).with_columns(p=pl.lit(train_rate)).write_parquet(CACHE / "preds_constant.parquet")
    binned.join(lookup, on=["len_bin", "third"], how="left").with_columns(pl.col("p").fill_null(train_rate)).select(PRED_COLS + ["p"]).write_parquet(
        CACHE / "preds_lookup.parquet"
    )
    print(f"constant rate {train_rate:.3f}, lookup table {len(lookup)} cells")
    return (PRED_COLS,)


@app.cell
def _(mo):
    mo.md(r"""
    ## LightGBM

    Random search over leaves, learning rate, minimum child samples, feature fraction and L2, 25
    trials or 8 minutes, each with early stopping on validation log loss. The best configuration is
    refitted once on train at its early-stopped round count.
    """)
    return


@app.cell
def _(CACHE, FEATURES, PRED_COLS, X, json, lgb, np, pl, splits, time, y):
    GRID = dict(
        num_leaves=[15, 31, 63, 127],
        learning_rate=[0.02, 0.05, 0.1],
        min_child_samples=[20, 50, 200],
        feature_fraction=[0.6, 0.8, 1.0],
        lambda_l2=[0.0, 1.0, 10.0],
    )
    rng = np.random.default_rng(0)
    best = None
    t_search = time.time()
    for trial in range(25):
        if time.time() - t_search > 480:
            break
        params = {k: rng.choice(v).item() for k, v in GRID.items()} | dict(objective="binary", verbose=-1, seed=trial, num_threads=8)
        booster = lgb.train(
            params,
            lgb.Dataset(X["train"], y["train"]),
            num_boost_round=2000,
            valid_sets=[lgb.Dataset(X["val"], y["val"])],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        score = booster.best_score["valid_0"]["binary_logloss"]
        if best is None or score < best[0]:
            best = (score, params, booster.best_iteration)
    search_s = time.time() - t_search

    lgb_val, lgb_params, lgb_rounds = best
    t_fit = time.time()
    lgbm = lgb.train(lgb_params, lgb.Dataset(X["train"], y["train"]), num_boost_round=lgb_rounds)
    lgb_train_s = time.time() - t_fit
    lgbm.save_model(str(CACHE / "lgbm.txt"))

    pl.concat([splits[s].select(PRED_COLS).with_columns(p=pl.Series(lgbm.predict(X[s]))) for s in splits]).write_parquet(CACHE / "preds_lightgbm.parquet")
    lgb_info = dict(
        params=lgb_params,
        n_iter=lgb_rounds,
        val_logloss=lgb_val,
        trials=trial + 1,
        tune_s=search_s,
        train_s=lgb_train_s,
        n_params=lgbm.num_trees() * lgb_params["num_leaves"],
        features=FEATURES,
    )
    json.dump(lgb_info, open(CACHE / "lgbm_info.json", "w"), indent=1)

    gain = sorted(zip(lgbm.feature_importance("gain"), FEATURES), reverse=True)[:8]
    print(f"{trial + 1} trials in {search_s:.0f}s, best val log loss {lgb_val:.4f}: {lgb_rounds} trees x {lgb_params['num_leaves']} leaves, refit {lgb_train_s:.1f}s")
    print("top gain:", [f for _, f in gain])
    return lgb_info, lgbm


@app.cell
def _(mo):
    mo.md(r"""
    ## Graph attention net

    Every visible player is a token of 13 features, plus one pass token; three blocks of four-head
    attention with a learned bias from each pair's relative offset (see `scripts/deep.py`).
    AdamW, batch 512, plateau schedule, early stopping on validation log loss, 12-minute wall clock.
    """)
    return


@app.cell
def _(
    CACHE,
    PRED_COLS,
    PlayerGAT,
    build_tensors,
    device,
    json,
    node_df,
    np,
    pass_df,
    pl,
    predict,
    time,
    torch,
):
    WALL_S = 12 * 60
    torch.manual_seed(0)
    dev = device()

    gat_splits = {s: pass_df.filter(pl.col("split") == s) for s in ("train", "val", "test")}
    tensors = {s: build_tensors(node_df, gat_splits[s]) for s in gat_splits}
    labels = {s: torch.from_numpy(gat_splits[s]["completed"].to_numpy().astype(np.float32)) for s in gat_splits}

    gat = PlayerGAT().to(dev)
    n_params = sum(p.numel() for p in gat.parameters())
    opt = torch.optim.AdamW(gat.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    x_tr, pad_tr, p_tr = tensors["train"]
    y_val = labels["val"].numpy()
    best_val, best_state, bad, epoch = 9.0, None, 0, 0
    t0 = time.time()
    while time.time() - t0 < WALL_S and bad < 8:
        gat.train()
        perm = torch.randperm(len(x_tr))
        for i in range(0, len(x_tr), 512):
            idx = perm[i : i + 512]
            loss = loss_fn(gat(x_tr[idx].to(dev), pad_tr[idx].to(dev), p_tr[idx].to(dev)), labels["train"][idx].to(dev))
            opt.zero_grad()
            loss.backward()
            opt.step()
        pv = predict(gat, *tensors["val"], dev)
        val = float(-np.mean(y_val * np.log(pv + 1e-6) + (1 - y_val) * np.log(1 - pv + 1e-6)))
        epoch += 1
        sched.step(val)
        print(f"epoch {epoch:2d}  val log loss {val:.4f}  ({time.time() - t0:.0f}s)")
        if val < best_val:
            best_val, bad, best_state = val, 0, {k: v.detach().clone() for k, v in gat.state_dict().items()}
        else:
            bad += 1
    gat_train_s = time.time() - t0

    gat.load_state_dict(best_state)
    torch.save(best_state, CACHE / "gat.pt")
    pl.concat(
        [gat_splits[s].select(PRED_COLS).with_columns(p=pl.Series(predict(gat, *tensors[s], dev).astype(np.float64))) for s in gat_splits]
    ).write_parquet(CACHE / "preds_gat.parquet")
    gat_info = dict(n_params=n_params, train_s=gat_train_s, epochs=epoch, best_val_logloss=best_val, device=str(dev))
    json.dump(gat_info, open(CACHE / "gat_info.json", "w"), indent=1)
    print(f"{n_params} params, {epoch} epochs, {gat_train_s:.0f}s on {dev}, best val {best_val:.4f}")
    return dev, gat, gat_info


@app.cell
def _(mo):
    mo.md(r"""
    ## Evaluation

    One function scores every prediction file on the test matches: log loss, Brier, ROC AUC and
    completed-pass error per match, each with a percentile bootstrap over test matches. Then a
    paired bootstrap of the tree-minus-GAT log loss gap, which is the actual test of who won.
    """)
    return


@app.cell
def _(CACHE, DATA, evaluate, gat_info, lgb_info, mo, paired_delta, pl):
    MODELS = {"constant": "Constant rate", "lookup": "Length x third lookup", "lightgbm": "LightGBM", "gat": "Graph attention net"}
    cost = {"lightgbm": (lgb_info["n_params"], lgb_info["train_s"], lgb_info["tune_s"]), "gat": (gat_info["n_params"], gat_info["train_s"], 0.0)}

    test_preds = {k: pl.read_parquet(CACHE / f"preds_{k}.parquet").filter(pl.col("split") == "test") for k in MODELS}
    ref = test_preds["constant"].select("pid", "match_id", "completed").sort("pid")
    for k, df in test_preds.items():
        assert df.select("pid", "match_id", "completed").sort("pid").equals(ref), f"{k} scored on different rows"

    rows = []
    for k, name in MODELS.items():
        params_, train_s_, tune_s_ = cost.get(k, (0, 0.0, 0.0))
        rows.append(dict(model=name, **evaluate(test_preds[k]), n_params=params_, train_s=train_s_, tune_s=tune_s_, n_test=len(ref), n_test_matches=ref["match_id"].n_unique()))
    comparison = pl.DataFrame(rows)
    comparison.write_parquet(DATA / "comparison.parquet")

    paired = test_preds["lightgbm"].select("pid", "match_id", "completed", p_tree="p").join(test_preds["gat"].select("pid", p_gat="p"), on="pid")
    delta = paired_delta(paired)
    pl.DataFrame([delta]).write_parquet(DATA / "delta.parquet")

    print("tree minus GAT log loss:", {k: round(v, 4) for k, v in delta.items()})
    mo.ui.table(comparison.select("model", "log_loss", "log_loss_lo", "log_loss_hi", "brier", "auc", "passes_err_per_match", "n_params", "train_s"), selection=None)
    return (test_preds,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Artifacts for the app

    Six real test passes where the two models disagree most, each frame scored by both models for a
    pass to every 4x4 cell; the 12x8 spatial error map on the test set; and log loss by length,
    pressure and end zone. Everything the deployed page shows comes from these files.
    """)
    return


@app.cell
def _(
    DATA,
    build_tensors,
    frames,
    gat,
    lgbm,
    node_features,
    np,
    pass_features,
    passes,
    pl,
    predict,
    test_preds,
    torch,
    tree_features,
):
    GX, GY = np.arange(2, 120, 4.0), np.arange(2, 80, 4.0)

    visible = frames.group_by("event_id").agg(pl.len().alias("n_vis"))
    candidates = (
        passes.filter(pl.col("split") == "test", pl.col("end_x") > 80, pl.col("end_x") - pl.col("x") > 10)
        .join(test_preds["lightgbm"].select("pid", p_tree="p"), on="pid")
        .join(test_preds["gat"].select("pid", p_gat="p"), on="pid")
        .join(visible, on="event_id")
        .filter(pl.col("n_vis") >= 16)
        .with_columns(gap=(pl.col("p_tree") - pl.col("p_gat")).abs())
        .sort("gap", descending=True)
    )
    examples = pl.concat([candidates.filter("completed").head(3), candidates.filter(~pl.col("completed")).head(3)]).with_row_index("example")

    grid = pl.DataFrame({"end_x": np.repeat(GX, len(GY)), "end_y": np.tile(GY, len(GX))})
    cand = examples.select("example", "event_id", "x", "y", "under_pressure").join(grid, how="cross").with_columns(pid=pl.format("{}_{}_{}", "example", "end_x", "end_y"))
    p_tree_grid = lgbm.predict(tree_features(cand, frames).drop("pid").to_numpy().astype(np.float32), num_threads=1)
    p_gat_grid = predict(gat, *build_tensors(node_features(cand, frames), pass_features(cand)), dev).astype(np.float64)
    pl.concat(
        [
            cand.select("example", "end_x", "end_y").with_columns(model=pl.lit("LightGBM"), p=pl.Series(p_tree_grid)),
            cand.select("example", "end_x", "end_y").with_columns(model=pl.lit("Graph attention net"), p=pl.Series(p_gat_grid)),
        ]
    ).write_parquet(DATA / "surfaces.parquet")
    examples.select("example", "comp", "match_date", "home", "away", "team", "player", "minute", "x", "y", "end_x", "end_y", "completed", "under_pressure", "p_tree", "p_gat").write_parquet(
        DATA / "examples.parquet"
    )
    frames.join(examples.select("event_id", "example"), on="event_id").select("example", "px", "py", "teammate", "actor", "keeper").write_parquet(DATA / "example_players.parquet")

    scored = pl.concat(
        [test_preds["lightgbm"].with_columns(model=pl.lit("LightGBM")), test_preds["gat"].with_columns(model=pl.lit("Graph attention net"))]
    ).join(passes.select("pid", "x", "y", "end_x", "end_y", "under_pressure"), on="pid")
    yy, pp = pl.col("completed").cast(pl.Float64), pl.col("p").clip(1e-6, 1 - 1e-6)
    scored = scored.with_columns(
        ll=-(yy * pp.log() + (1 - yy) * (1 - pp).log()),
        length=((pl.col("end_x") - pl.col("x")) ** 2 + (pl.col("end_y") - pl.col("y")) ** 2).sqrt(),
    )
    (
        scored.with_columns(cx=(pl.col("end_x") / 10).floor().clip(0, 11), cy=(pl.col("end_y") / 10).floor().clip(0, 7))
        .group_by("cx", "cy", "model")
        .agg(pl.len().alias("n"), pl.col("ll").mean().alias("log_loss"), pl.col("completed").mean().alias("completion_rate"))
        .pivot(on="model", index=["cx", "cy", "n", "completion_rate"], values="log_loss")
        .rename({"LightGBM": "ll_tree", "Graph attention net": "ll_gat"})
        .with_columns(diff=pl.col("ll_tree") - pl.col("ll_gat"))
        .write_parquet(DATA / "error_map.parquet")
    )
    length_bin = pl.when(pl.col("length") < 10).then(pl.lit("<10")).when(pl.col("length") < 20).then(pl.lit("10-20")).when(pl.col("length") < 30).then(pl.lit("20-30")).otherwise(pl.lit("30+"))
    zone_bin = pl.when(pl.col("end_x") < 40).then(pl.lit("own third")).when(pl.col("end_x") < 80).then(pl.lit("middle third")).otherwise(pl.lit("final third"))
    pressure_bin = pl.when("under_pressure").then(pl.lit("under pressure")).otherwise(pl.lit("no pressure"))
    pl.concat(
        [
            scored.with_columns(group=pl.lit("length"), bin=length_bin),
            scored.with_columns(group=pl.lit("pressure"), bin=pressure_bin),
            scored.with_columns(group=pl.lit("end zone"), bin=zone_bin),
        ]
    ).group_by("group", "bin", "model").agg(pl.len().alias("n"), pl.col("ll").mean().alias("log_loss"), pl.col("completed").mean().alias("completion_rate")).write_parquet(DATA / "breakdown.parquet")

    examples.select("example", "player", "minute", "completed", "p_tree", "p_gat")
    return


if __name__ == "__main__":
    app.run()
