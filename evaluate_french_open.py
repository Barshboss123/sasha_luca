"""
Evaluate the model on French Open matches (Roland Garros).

For each RG match in the test set, the model outputs:
  P(player1 wins) and P(player2 wins)

Results are saved to results/french_open_predictions.csv and a summary
plot is generated.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
from sklearn.calibration import calibration_curve

from data_loader import load_atp_matches
from features import build_player_histories, symmetrize, build_diff_features
from model import train, load_model, get_feature_cols, predict_proba

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
TRAIN_CUTOFF = 20260101   # Train on 2000–2025, test on 2026

RG_NAME_PATTERNS = ["Roland Garros", "French Open"]


def is_rg(name):
    if pd.isna(name):
        return False
    return any(p.lower() in str(name).lower() for p in RG_NAME_PATTERNS)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load raw data
    # ------------------------------------------------------------------
    print("=" * 60)
    print("Loading ATP match data (2000–2024)...")
    raw = load_atp_matches(2000, 2026)
    print(f"Total matches: {len(raw):,}")

    # Clean
    raw = raw.dropna(subset=["winner_id", "loser_id", "tourney_date"])
    raw["tourney_date"] = pd.to_numeric(raw["tourney_date"], errors="coerce")
    raw = raw.dropna(subset=["tourney_date"])
    raw["tourney_date"] = raw["tourney_date"].astype(int)
    print(f"After cleaning: {len(raw):,} matches")

    # ------------------------------------------------------------------
    # 2. Build features (walking forward through time)
    # ------------------------------------------------------------------
    print("\nBuilding player features (this takes a minute)...")
    match_df, _ = build_player_histories(raw)
    sym_df = symmetrize(match_df)
    feat_df = build_diff_features(sym_df)
    feat_df["tourney_date"] = pd.to_numeric(feat_df["tourney_date"], errors="coerce")

    print(f"Feature matrix: {feat_df.shape}")

    # ------------------------------------------------------------------
    # 3. Train / test split
    # ------------------------------------------------------------------
    train_mask = feat_df["tourney_date"] < TRAIN_CUTOFF
    test_mask  = feat_df["tourney_date"] >= TRAIN_CUTOFF

    train_df = feat_df[train_mask].copy()
    test_df  = feat_df[test_mask].copy()

    print(f"\nTrain samples: {len(train_df):,}")
    print(f"Test  samples: {len(test_df):,}")

    # ------------------------------------------------------------------
    # 4. Train model
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    model, feat_cols = train(train_df, val_df=test_df)

    # ------------------------------------------------------------------
    # 5. French Open specific evaluation
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("French Open (Roland Garros) evaluation...")

    rg_mask = test_df["tourney_name"].apply(is_rg)
    rg_df   = test_df[rg_mask].copy()
    print(f"RG matches in test set: {len(rg_df)}")

    if len(rg_df) == 0:
        print("No RG matches found in test set — using all clay matches as proxy.")
        rg_df = test_df[test_df["surface"] == "Clay"].copy()
        print(f"Clay matches: {len(rg_df)}")

    probs = predict_proba(model, feat_cols, rg_df)
    rg_df = rg_df.copy()
    rg_df["p1_win_prob"] = probs
    rg_df["p2_win_prob"] = 1 - probs
    rg_df["predicted_winner"] = (probs > 0.5).astype(int)
    rg_df["correct"] = (rg_df["predicted_winner"] == rg_df["label"]).astype(int)

    y_true = rg_df["label"].values
    print(f"\nFrench Open metrics:")
    print(f"  Accuracy:    {rg_df['correct'].mean():.4f}")
    print(f"  Log-loss:    {log_loss(y_true, probs):.4f}")
    print(f"  Brier score: {brier_score_loss(y_true, probs):.4f}")
    print(f"  ROC-AUC:     {roc_auc_score(y_true, probs):.4f}")

    # ------------------------------------------------------------------
    # 6. Save predictions
    # ------------------------------------------------------------------
    out_cols = ["tourney_date", "tourney_name", "surface", "round",
                "p1_win_prob", "p2_win_prob", "label", "correct"]
    available = [c for c in out_cols if c in rg_df.columns]
    rg_df[available].to_csv(
        os.path.join(RESULTS_DIR, "french_open_predictions.csv"), index=False
    )
    print(f"\nPredictions saved to results/french_open_predictions.csv")

    # ------------------------------------------------------------------
    # 7. Plots
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Tennis Match Prediction — French Open / Clay Court Results",
                 fontsize=14, fontweight="bold")

    # (a) Probability histogram
    ax = axes[0, 0]
    ax.hist(probs[y_true == 1], bins=25, alpha=0.6, color="steelblue", label="p1 wins")
    ax.hist(probs[y_true == 0], bins=25, alpha=0.6, color="tomato",    label="p2 wins")
    ax.set_xlabel("Predicted P(player1 wins)")
    ax.set_ylabel("Count")
    ax.set_title("Predicted probability distribution")
    ax.legend()

    # (b) Calibration curve
    ax = axes[0, 1]
    frac_pos, mean_pred = calibration_curve(y_true, probs, n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac_pos, "o-", color="steelblue", label="Model")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Fraction of positives")
    ax.set_title("Calibration curve")
    ax.legend()

    # (c) Accuracy by probability bucket
    ax = axes[1, 0]
    bins = np.linspace(0, 1, 11)
    bucket = np.digitize(probs, bins) - 1
    bucket = np.clip(bucket, 0, 9)
    bucket_acc  = [rg_df["correct"].values[bucket == b].mean()
                   if (bucket == b).sum() > 0 else np.nan for b in range(10)]
    bucket_cnt  = [(bucket == b).sum() for b in range(10)]
    centers = (bins[:-1] + bins[1:]) / 2
    bars = ax.bar(centers, bucket_acc, width=0.09, color="steelblue", alpha=0.8)
    for bar, cnt in zip(bars, bucket_cnt):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                str(cnt), ha="center", va="bottom", fontsize=7)
    ax.axhline(0.5, color="red", linestyle="--", label="Random")
    ax.set_xlabel("Predicted probability bucket")
    ax.set_ylabel("Actual win rate")
    ax.set_title("Accuracy by confidence bucket (n labels)")
    ax.set_ylim(0, 1.05)
    ax.legend()

    # (d) Round-by-round accuracy
    ax = axes[1, 1]
    round_order = ["R128", "R64", "R32", "R16", "QF", "SF", "F"]
    if "round" in rg_df.columns:
        round_acc = rg_df.groupby("round")["correct"].mean().reindex(round_order).dropna()
        round_cnt = rg_df.groupby("round")["correct"].count().reindex(round_order).dropna()
        ax.bar(round_acc.index, round_acc.values, color="steelblue", alpha=0.8)
        for i, (r, v) in enumerate(round_acc.items()):
            ax.text(i, v + 0.01, f"n={round_cnt[r]}", ha="center", fontsize=8)
        ax.axhline(0.5, color="red", linestyle="--")
        ax.set_ylabel("Accuracy")
        ax.set_title("Accuracy by tournament round")
        ax.set_ylim(0, 1.05)
    else:
        ax.text(0.5, 0.5, "Round data unavailable", ha="center", va="center")

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "french_open_analysis.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {plot_path}")

    # ------------------------------------------------------------------
    # 8. Pretty-print sample predictions
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Sample predictions (first 20 RG matches):")
    print(f"{'Date':<10} {'Round':<6} {'P1 Win%':>8} {'P2 Win%':>8} {'Actual':>8} {'Correct':>8}")
    print("-" * 58)
    sample = rg_df.head(20)
    for _, r in sample.iterrows():
        actual = "P1" if r["label"] == 1 else "P2"
        correct = "✓" if r["correct"] == 1 else "✗"
        rnd = str(r.get("round", "?"))[:5]
        print(f"{str(r['tourney_date']):<10} {rnd:<6} "
              f"{r['p1_win_prob']:>7.1%} {r['p2_win_prob']:>8.1%} "
              f"{actual:>8} {correct:>8}")


if __name__ == "__main__":
    main()
