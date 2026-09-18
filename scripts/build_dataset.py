"""Parse StatsBomb events + 360 frames into two parquet tables in cache/.

passes.parquet  one row per open-play pass that has a 360 frame
frames.parquet  one row per visible player in that pass's freeze frame
Splits are assigned here, by match, chronologically within each competition-season.
"""
import json
from pathlib import Path

import polars as pl

CACHE = Path(__file__).resolve().parents[1] / "cache"


def load_matches() -> pl.DataFrame:
    rows = []
    for f in CACHE.glob("matches/*/*.json"):
        for m in json.load(open(f)):
            if m.get("match_status_360") != "available" or m["competition"]["competition_id"] == 44:
                continue
            rows.append(
                dict(
                    match_id=m["match_id"],
                    comp=f"{m['competition']['competition_name']} {m['season']['season_name']}",
                    match_date=m["match_date"],
                    home=m["home_team"]["home_team_name"],
                    away=m["away_team"]["away_team_name"],
                )
            )
    return pl.DataFrame(rows)


def parse_match(match_id: int) -> tuple[list[dict], list[dict]]:
    events = json.load(open(CACHE / f"events/{match_id}.json"))
    frames = {f["event_uuid"]: f["freeze_frame"] for f in json.load(open(CACHE / f"three-sixty/{match_id}.json"))}
    passes, nodes = [], []
    for e in events:
        if e["type"]["name"] != "Pass" or e["id"] not in frames:
            continue
        p = e["pass"]
        if "type" in p:  # set piece (throw-in, corner, free kick, goal kick, kick-off, recovery...)
            continue
        outcome = p.get("outcome", {}).get("name")
        passes.append(
            dict(
                event_id=e["id"],
                match_id=match_id,
                index=e["index"],
                period=e["period"],
                minute=e["minute"],
                second=e["second"],
                possession=e["possession"],
                team=e["team"]["name"],
                player=e["player"]["name"],
                x=e["location"][0],
                y=e["location"][1],
                end_x=p["end_location"][0],
                end_y=p["end_location"][1],
                under_pressure=bool(e.get("under_pressure", False)),
                outcome=outcome or "Complete",
                completed=outcome is None,
                height=p["height"]["name"],
                body_part=p.get("body_part", {}).get("name"),
            )
        )
        for i, n in enumerate(frames[e["id"]]):
            nodes.append(
                dict(
                    event_id=e["id"],
                    node=i,
                    px=n["location"][0],
                    py=n["location"][1],
                    teammate=n["teammate"],
                    actor=n["actor"],
                    keeper=n["keeper"],
                )
            )
    return passes, nodes


def assign_splits(matches: pl.DataFrame) -> pl.DataFrame:
    """First 70% of each competition-season's matches by date -> train, next 15% val, last 15% test."""
    return matches.sort("comp", "match_date", "match_id").with_columns(
        frac=(pl.int_range(pl.len()).over("comp") + 0.5) / pl.len().over("comp")
    ).with_columns(
        split=pl.when(pl.col("frac") < 0.70).then(pl.lit("train")).when(pl.col("frac") < 0.85).then(pl.lit("val")).otherwise(pl.lit("test"))
    ).drop("frac")


def main() -> None:
    matches = assign_splits(load_matches())
    passes, nodes = [], []
    for mid in matches["match_id"].to_list():
        p, n = parse_match(mid)
        passes += p
        nodes += n
    passes = pl.DataFrame(passes).join(matches, on="match_id")
    frames = pl.DataFrame(nodes)

    # --- empirical pitch-convention checks -------------------------------------
    assert passes.select(pl.col("x").is_between(0, 120).all(), pl.col("y").is_between(0, 80).all()).row(0) == (True, True)
    # 360 player positions sit up to ~10 units outside the pitch (players off-pitch, projection noise): allow that, no more
    assert frames.select(pl.col("px").is_between(-15, 135).all(), pl.col("py").is_between(-15, 95).all()).row(0) == (True, True)
    assert frames.select((pl.col("px").is_between(0, 120) & pl.col("py").is_between(0, 80)).mean()).item() > 0.98
    keepers = frames.filter("keeper").group_by("teammate").agg(pl.col("px").mean())
    own_gk = keepers.filter("teammate")["px"][0]
    opp_gk = keepers.filter(~pl.col("teammate"))["px"][0]
    assert own_gk < 30 and opp_gk > 90, (own_gk, opp_gk)  # every team attacks left -> right
    # actor is in every frame, exactly once, and stands where the event says
    n_actor = frames.group_by("event_id").agg(pl.col("actor").sum())
    ok = n_actor.filter(pl.col("actor") == 1)["event_id"]
    passes = passes.filter(pl.col("event_id").is_in(ok.implode()))
    frames = frames.filter(pl.col("event_id").is_in(ok.implode()))
    actor_d = (
        frames.filter("actor").join(passes.select("event_id", "x", "y"), on="event_id")
        .select(((pl.col("px") - pl.col("x")) ** 2 + (pl.col("py") - pl.col("y")) ** 2).sqrt().median())
        .item()
    )
    assert actor_d < 2.0, actor_d
    # splits: a match (and hence a possession) belongs to exactly one split
    assert passes.group_by("match_id").agg(pl.col("split").n_unique()).select(pl.col("split").max()).item() == 1
    assert passes.group_by("match_id", "possession").agg(pl.col("split").n_unique()).select(pl.col("split").max()).item() == 1

    passes.write_parquet(CACHE / "passes.parquet")
    frames.write_parquet(CACHE / "frames.parquet")
    print(passes.group_by("comp", "split").agg(pl.len(), pl.col("match_id").n_unique().alias("matches"), pl.col("completed").mean()).sort("comp", "split"))
    print(f"{len(passes)} passes, {len(frames)} nodes, gk own/opp mean x = {own_gk:.1f}/{opp_gk:.1f}, actor offset median {actor_d:.2f}")


if __name__ == "__main__":
    main()
