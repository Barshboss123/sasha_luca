"""
Feature engineering for tennis match prediction.

For each match we compute rolling statistics for both players BEFORE the match date,
then take the difference (player1 - player2) as model inputs so the model learns
relative strength rather than absolute values.

Features included:
  - Elo rating (overall + surface-specific)
  - Win rate (overall, surface, last-N matches)
  - H2H win rate
  - Average stats: ace%, df%, 1stIn%, 1stWon%, 2ndWon%, bpSaved%, bpFaced%
  - Rank and rank points
  - Days since last match (rest/fatigue proxy)
  - Surface one-hot encoding
"""

import numpy as np
import pandas as pd
from typing import Tuple

SURFACES = ["Hard", "Clay", "Grass", "Carpet"]
ROLLING_WINDOWS = [10, 30]  # last N matches for rolling stats

# ------------------------------------------------------------------
# Elo helpers
# ------------------------------------------------------------------
K = 32
INITIAL_ELO = 1500


def expected_score(ra: float, rb: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rb - ra) / 400))


def update_elo(ra: float, rb: float, won: int) -> Tuple[float, float]:
    ea = expected_score(ra, rb)
    new_ra = ra + K * (won - ea)
    new_rb = rb + K * ((1 - won) - (1 - ea))
    return new_ra, new_rb


# ------------------------------------------------------------------
# Build player-level rolling state
# ------------------------------------------------------------------

def build_player_histories(df: pd.DataFrame) -> dict:
    """
    Walk through all matches chronologically and maintain a per-player dict of:
      elo, elo_surface, wins, losses, surface wins/losses,
      rolling stats lists, h2h dict, last match date
    """
    df = df.sort_values("tourney_date").reset_index(drop=True)

    players: dict = {}

    def get_player(pid):
        if pid not in players:
            players[pid] = {
                "elo": INITIAL_ELO,
                "elo_surface": {s: INITIAL_ELO for s in SURFACES},
                "wins": 0,
                "losses": 0,
                "surface_wins": {s: 0 for s in SURFACES},
                "surface_losses": {s: 0 for s in SURFACES},
                "recent_results": [],   # list of (won, surface)
                "recent_stats": [],     # list of stat dicts
                "h2h": {},              # opponent_id -> (wins, losses)
                "last_match_date": None,
            }
        return players[pid]

    rows = []

    for _, row in df.iterrows():
        wid = row.get("winner_id")
        lid = row.get("loser_id")
        surface = row.get("surface", "Hard")
        if surface not in SURFACES:
            surface = "Hard"
        date = row.get("tourney_date")

        wp = get_player(wid)
        lp = get_player(lid)

        # Capture PRE-match features
        def player_features(p, opponent_id, pid):
            n = len(p["recent_results"])
            last_n = {w: p["recent_results"][-w:] for w in ROLLING_WINDOWS}
            feats = {
                "elo": p["elo"],
                f"elo_{surface.lower()}": p["elo_surface"].get(surface, INITIAL_ELO),
                "overall_win_rate": p["wins"] / (p["wins"] + p["losses"]) if (p["wins"] + p["losses"]) > 0 else 0.5,
                f"surface_win_rate_{surface.lower()}": (
                    p["surface_wins"].get(surface, 0) /
                    max(p["surface_wins"].get(surface, 0) + p["surface_losses"].get(surface, 0), 1)
                ),
                "rank": row.get(f"{'winner' if pid == wid else 'loser'}_rank", 300),
                "rank_pts": row.get(f"{'winner' if pid == wid else 'loser'}_rank_points", 0),
            }
            for w in ROLLING_WINDOWS:
                recent = last_n[w]
                feats[f"win_rate_last{w}"] = np.mean([r[0] for r in recent]) if recent else 0.5
                feats[f"surface_win_rate_last{w}_{surface.lower()}"] = (
                    np.mean([r[0] for r in recent if r[1] == surface])
                    if any(r[1] == surface for r in recent) else 0.5
                )

            # H2H
            h2h = p["h2h"].get(opponent_id, (0, 0))
            feats["h2h_win_rate"] = h2h[0] / max(sum(h2h), 1)
            feats["h2h_matches"] = sum(h2h)

            # Rolling service/return stats
            if p["recent_stats"]:
                stats_window = p["recent_stats"][-20:]
                for stat in ["ace_pct", "df_pct", "first_in_pct", "first_won_pct",
                             "second_won_pct", "bp_saved_pct", "bp_faced_per_game"]:
                    vals = [s[stat] for s in stats_window if stat in s and s[stat] is not None]
                    feats[f"avg_{stat}"] = np.mean(vals) if vals else np.nan
            else:
                for stat in ["ace_pct", "df_pct", "first_in_pct", "first_won_pct",
                             "second_won_pct", "bp_saved_pct", "bp_faced_per_game"]:
                    feats[f"avg_{stat}"] = np.nan

            # Rest days
            if p["last_match_date"] is not None and date is not None:
                try:
                    d1 = pd.to_datetime(str(date), format="%Y%m%d")
                    d2 = pd.to_datetime(str(p["last_match_date"]), format="%Y%m%d")
                    feats["days_since_last_match"] = (d1 - d2).days
                except Exception:
                    feats["days_since_last_match"] = 7
            else:
                feats["days_since_last_match"] = 7

            return feats

        wf = player_features(wp, lid, wid)
        lf = player_features(lp, wid, lid)

        rows.append({
            "tourney_date": date,
            "tourney_name": row.get("tourney_name"),
            "surface": surface,
            "round": row.get("round"),
            "winner_id": wid,
            "loser_id": lid,
            **{f"w_{k}": v for k, v in wf.items()},
            **{f"l_{k}": v for k, v in lf.items()},
            "label": 1,  # winner always wins (will be symmetrized later)
        })

        # ---- Update state ----
        # Elo
        new_we, new_le = update_elo(wp["elo"], lp["elo"], 1)
        wp["elo"], lp["elo"] = new_we, new_le
        new_wse, new_lse = update_elo(
            wp["elo_surface"][surface], lp["elo_surface"][surface], 1
        )
        wp["elo_surface"][surface] = new_wse
        lp["elo_surface"][surface] = new_lse

        # Win/loss counts
        wp["wins"] += 1
        lp["losses"] += 1
        wp["surface_wins"][surface] = wp["surface_wins"].get(surface, 0) + 1
        lp["surface_losses"][surface] = lp["surface_losses"].get(surface, 0) + 1

        # Recent results
        wp["recent_results"].append((1, surface))
        lp["recent_results"].append((0, surface))

        # H2H
        wh = wp["h2h"].get(lid, (0, 0))
        wp["h2h"][lid] = (wh[0] + 1, wh[1])
        lh = lp["h2h"].get(wid, (0, 0))
        lp["h2h"][wid] = (lh[0], lh[1] + 1)

        # Service stats
        def extract_stats(prefix, r):
            svpt = r.get(f"{prefix}_svpt", 0) or 0
            if svpt == 0:
                return {}
            aces = r.get(f"{prefix}_ace", 0) or 0
            dfs = r.get(f"{prefix}_df", 0) or 0
            first_in = r.get(f"{prefix}_1stIn", 0) or 0
            first_won = r.get(f"{prefix}_1stWon", 0) or 0
            second_won = r.get(f"{prefix}_2ndWon", 0) or 0
            bpSaved = r.get(f"{prefix}_bpSaved", 0) or 0
            bpFaced = r.get(f"{prefix}_bpFaced", 0) or 0
            games = r.get(f"{prefix}_SvGms", 1) or 1
            return {
                "ace_pct": aces / svpt,
                "df_pct": dfs / svpt,
                "first_in_pct": first_in / svpt,
                "first_won_pct": first_won / first_in if first_in > 0 else None,
                "second_won_pct": second_won / max(svpt - first_in, 1),
                "bp_saved_pct": bpSaved / bpFaced if bpFaced > 0 else 1.0,
                "bp_faced_per_game": bpFaced / games,
            }

        wp["recent_stats"].append(extract_stats("w", row))
        lp["recent_stats"].append(extract_stats("l", row))

        # Last match date
        wp["last_match_date"] = date
        lp["last_match_date"] = date

    return pd.DataFrame(rows), players


def symmetrize(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each match create two rows: one where player1=winner (label=1)
    and one where player1=loser (label=0). This removes ordering bias.
    """
    rows_flipped = df.copy()
    # Swap w_ and l_ columns
    w_cols = [c for c in df.columns if c.startswith("w_")]
    l_cols = [c for c in df.columns if c.startswith("l_")]
    rename_w = {c: c.replace("w_", "p2_", 1) for c in w_cols}
    rename_l = {c: c.replace("l_", "p1_", 1) for c in l_cols}
    rows_flipped = rows_flipped.rename(columns={**rename_w, **rename_l})
    rows_flipped["label"] = 0

    orig = df.copy()
    rename_w2 = {c: c.replace("w_", "p1_", 1) for c in w_cols}
    rename_l2 = {c: c.replace("l_", "p2_", 1) for c in l_cols}
    orig = orig.rename(columns={**rename_w2, **rename_l2})
    orig["label"] = 1

    combined = pd.concat([orig, rows_flipped], ignore_index=True)
    return combined


def build_diff_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute p1 - p2 differences for all numeric features.
    Also keep surface one-hot.
    """
    p1_cols = [c for c in df.columns if c.startswith("p1_")]
    p2_cols = [c for c in df.columns if c.startswith("p2_")]

    # Match p1/p2 pairs
    p1_map = {c[3:]: c for c in p1_cols}
    p2_map = {c[3:]: c for c in p2_cols}
    common = set(p1_map) & set(p2_map)

    diff_df = pd.DataFrame(index=df.index)
    for feat in sorted(common):
        diff_df[f"diff_{feat}"] = pd.to_numeric(df[p1_map[feat]], errors="coerce") - \
                                   pd.to_numeric(df[p2_map[feat]], errors="coerce")

    # Surface one-hot
    for s in SURFACES:
        diff_df[f"surface_{s.lower()}"] = (df["surface"] == s).astype(int)

    diff_df["label"] = df["label"].values
    diff_df["tourney_date"] = df["tourney_date"].values
    diff_df["tourney_name"] = df["tourney_name"].values
    diff_df["surface"] = df["surface"].values
    diff_df["round"] = df["round"].values

    return diff_df
