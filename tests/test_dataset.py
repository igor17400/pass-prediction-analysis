"""Dataset construction and target alignment checks on a tiny synthetic frame."""
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_dataset import assign_splits  # noqa: E402
from features import LEAK_COLS, NODE_COLS, PASS_COLS, node_features, pass_features, tree_features  # noqa: E402

PASSES = pl.DataFrame(dict(pid=["a", "b"], event_id=["e", "e"], x=[10.0, 10.0], y=[40.0, 40.0], end_x=[30.0, 10.0], end_y=[40.0, 60.0], under_pressure=[False, True]))
FRAMES = pl.DataFrame(
    dict(
        event_id=["e"] * 4,
        node=[0, 1, 2, 3],
        px=[10.0, 20.0, 30.0, 10.0],
        py=[40.0, 41.0, 40.0, 60.0],
        teammate=[True, False, True, False],
        actor=[True, False, False, False],
        keeper=[False, False, False, False],
    )
)


def test_tree_features_geometry():
    f = tree_features(PASSES, FRAMES).sort("pid")
    a, b = f.row(0, named=True), f.row(1, named=True)
    assert a["length"] == 20.0 and b["length"] == 20.0
    assert a["opp_n2_lane"] == 1 and a["opp_min_lane"] == 1.0  # opponent 1 unit off the lane to (30,40)
    assert b["opp_min_end"] == 0.0 and b["opp_n3_end"] == 1  # opponent standing on the target of pass b
    assert a["tm_min_end"] == 0.0  # teammate waiting on the target of pass a
    assert b["under_pressure"] == 1 and a["under_pressure"] == 0
    assert not (set(f.columns) & LEAK_COLS)


def test_node_and_pass_features_align():
    n = node_features(PASSES, FRAMES)
    assert n.shape == (8, 2 + len(NODE_COLS))
    actor = n.filter(pl.col("actor") == 1)
    assert (actor["d_start_n"] == 0).all()
    p = pass_features(PASSES).sort("pid")
    assert p.columns == ["pid"] + PASS_COLS
    assert abs(p["cos_a"][0] - 1.0) < 1e-9 and abs(p["sin_a"][1] - 1.0) < 1e-9


def test_splits_are_chronological_and_by_match():
    m = pl.DataFrame(dict(match_id=list(range(20)), comp=["c"] * 20, match_date=[f"2024-01-{i + 1:02d}" for i in range(20)]))
    s = assign_splits(m).sort("match_date")
    assert s["split"].to_list() == ["train"] * 14 + ["val"] * 3 + ["test"] * 3
