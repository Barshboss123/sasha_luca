"""
Ensemble model for tennis match prediction.

Uses a stacked ensemble of:
  1. Logistic Regression (baseline + calibration)
  2. XGBoost
  3. LightGBM
  4. Random Forest

Final prediction is a probability-calibrated weighted average.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import log_loss, brier_score_loss, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import pickle, os

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def get_feature_cols(df: pd.DataFrame) -> list[str]:
    exclude = {"label", "tourney_date", "tourney_name", "surface", "round",
               "winner_id", "loser_id"}
    return [c for c in df.columns if c not in exclude and df[c].dtype != object]


def build_ensemble():
    lr = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=1000, C=0.1)),
    ])

    xgb_clf = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )

    lgb_clf = lgb.LGBMClassifier(
        n_estimators=500,
        num_leaves=63,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
    )

    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=10,
        random_state=42,
        n_jobs=-1,
    )

    # Calibrate tree models
    xgb_cal = CalibratedClassifierCV(xgb_clf, cv=3, method="isotonic")
    lgb_cal = CalibratedClassifierCV(lgb_clf, cv=3, method="isotonic")
    rf_cal  = CalibratedClassifierCV(rf,      cv=3, method="isotonic")

    ensemble = VotingClassifier(
        estimators=[
            ("lr",  lr),
            ("xgb", xgb_cal),
            ("lgb", lgb_cal),
            ("rf",  rf_cal),
        ],
        voting="soft",
        weights=[1, 3, 3, 2],  # tree models get more weight
    )

    # Wrap with imputer for tree models that don't have their own
    full_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("ensemble", ensemble),
    ])

    return full_pipeline


def train(train_df: pd.DataFrame, val_df: pd.DataFrame | None = None):
    feat_cols = get_feature_cols(train_df)
    X_train = train_df[feat_cols].values
    y_train = train_df["label"].values

    model = build_ensemble()
    print(f"Training on {len(train_df):,} samples with {len(feat_cols)} features...")
    model.fit(X_train, y_train)

    if val_df is not None and len(val_df) > 0:
        X_val = val_df[feat_cols].values
        y_val = val_df["label"].values
        probs = model.predict_proba(X_val)[:, 1]
        print(f"\nValidation metrics:")
        print(f"  Log-loss:    {log_loss(y_val, probs):.4f}")
        print(f"  Brier score: {brier_score_loss(y_val, probs):.4f}")
        print(f"  ROC-AUC:     {roc_auc_score(y_val, probs):.4f}")
        acc = ((probs > 0.5) == y_val).mean()
        print(f"  Accuracy:    {acc:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "ensemble.pkl")
    feat_path  = os.path.join(MODELS_DIR, "feature_cols.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(feat_path, "wb") as f:
        pickle.dump(feat_cols, f)
    print(f"\nModel saved to {model_path}")
    return model, feat_cols


def load_model():
    model_path = os.path.join(MODELS_DIR, "ensemble.pkl")
    feat_path  = os.path.join(MODELS_DIR, "feature_cols.pkl")
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(feat_path, "rb") as f:
        feat_cols = pickle.load(f)
    return model, feat_cols


def predict_proba(model, feat_cols, match_features: pd.DataFrame) -> np.ndarray:
    X = match_features[feat_cols].values
    return model.predict_proba(X)[:, 1]
