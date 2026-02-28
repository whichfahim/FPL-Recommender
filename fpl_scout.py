"""
FPL Scout — Player Recommendation Algorithm
Fetches live data from the Fantasy Premier League API
and scores players using a weighted multi-feature model.

Install deps:  pip install requests pandas scikit-learn
"""

import requests
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

BASE_URL = "https://fantasy.premierleague.com/api/"

# ── 1. FETCH DATA ──────────────────────────────────────────────
def fetch_bootstrap():
    """Main endpoint — all players, teams, gameweek info."""
    r = requests.get(BASE_URL + "bootstrap-static/", timeout=15)
    r.raise_for_status()
    return r.json()

def fetch_fixtures():
    """All fixtures with FDR ratings."""
    r = requests.get(BASE_URL + "fixtures/", timeout=15)
    r.raise_for_status()
    return r.json()

def fetch_player_history(player_id):
    """Per-player historical GW data + upcoming fixtures."""
    r = requests.get(f"{BASE_URL}element-summary/{player_id}/", timeout=15)
    r.raise_for_status()
    return r.json()

# ── 2. BUILD PLAYER DATAFRAME ──────────────────────────────────
def build_dataframe(bootstrap, fixtures):
    elements = bootstrap["elements"]
    teams    = {t["id"]: t for t in bootstrap["teams"]}
    events   = bootstrap["events"]
    pos_map  = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

    # Find next gameweek
    current_gw = next(
        (e["id"] for e in events if e["is_next"]),
        events[-1]["id"]
    )
    print(f"Analysing for GW {current_gw}")

    # Map fixture difficulty for next GW
    next_fdr = {}
    for f in fixtures:
        if f["event"] == current_gw and not f["finished"]:
            next_fdr[f["team_h"]] = f["team_h_difficulty"]
            next_fdr[f["team_a"]] = f["team_a_difficulty"]

    rows = []
    for p in elements:
        if p["status"] not in ("a", "u"):  # skip injured/suspended
            continue

        team   = teams.get(p["team"], {})
        fdr    = next_fdr.get(p["team"], 3)
        played = max((current_gw - 1) * 90, 1)

        rows.append({
            "id":            p["id"],
            "name":          p["web_name"],
            "team":          team.get("short_name", "???"),
            "position":      pos_map.get(p["element_type"], "MID"),
            "price":         p["now_cost"] / 10,
            "form":          float(p["form"] or 0),
            "ppg":           float(p["points_per_game"] or 0),
            "ownership":     float(p["selected_by_percent"] or 0),
            "minutes_pct":   p["minutes"] / played,
            "ict_index":     float(p["ict_index"] or 0),
            "bonus":         p["bonus"],
            "xgi_per90":     float(p.get("expected_goal_involvements_per_90", 0) or 0),
            "fdr":           fdr,
            "fixture_ease":  6 - fdr,  # invert FDR
            "status":        p["status"],
        })

    return pd.DataFrame(rows), current_gw

# ── 3. SCORE PLAYERS ───────────────────────────────────────────
WEIGHTS = {
    "form":         0.30,
    "ppg":          0.25,
    "fixture_ease": 0.20,
    "xgi_per90":    0.15,
    "ict_index":    0.10,
    "bonus":        0.08,
    "ownership":   -0.05,  # negative = differential bonus
}

def score_players(df):
    features = list(WEIGHTS.keys())
    scaler   = MinMaxScaler()
    normed   = scaler.fit_transform(df[features])
    normed_df = pd.DataFrame(normed, columns=features, index=df.index)

    # Invert ownership (high ownership → lower score)
    normed_df["ownership"] = 1 - normed_df["ownership"]

    # Weighted sum
    df["score"] = sum(
        normed_df[f] * w for f, w in WEIGHTS.items()
    )

    # Rotation risk penalty
    df.loc[df["minutes_pct"] < 0.6, "score"] *= 0.7

    # Scale 0–100
    df["score"] = (df["score"] * 100).round(1)
    return df.sort_values("score", ascending=False).reset_index(drop=True)

# ── 4. FILTER & DISPLAY ────────────────────────────────────────
def get_recommendations(df, position=None, max_price=None, top_n=20):
    out = df.copy()
    if position:
        out = out[out["position"] == position]
    if max_price:
        out = out[out["price"] <= max_price]
    cols = ["name", "team", "position", "price", "score",
            "ppg", "form", "fdr", "ownership", "xgi_per90"]
    return out[cols].head(top_n).to_string(index=False)

# ── 5. MAIN ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Fetching FPL data...")
    bootstrap = fetch_bootstrap()
    fixtures  = fetch_fixtures()

    df, gw = build_dataframe(bootstrap, fixtures)
    df     = score_players(df)

    print(f"{'='*60}")
    print(f"  FPL SCOUT — GW {gw} Recommendations")
    print('='*60)

    for pos in ["GK", "DEF", "MID", "FWD"]:
        print(f"── TOP {pos}s ──")
        print(get_recommendations(df, position=pos, top_n=5))

    print("── TOP OVERALL ──")
    print(get_recommendations(df, top_n=10))

    # Export to CSV
    df.to_csv(f"fpl_scout_gw{gw}.csv", index=False)
    print(f"Saved: fpl_scout_gw{gw}.csv")