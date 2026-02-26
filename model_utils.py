"""
model_utils.py
--------------
Backend utility module for Tourism Experience Analytics Streamlit App.
Handles: data loading, preprocessing, model training, and recommendations.
All heavy operations are cached using @st.cache_resource for performance.
"""

from __future__ import annotations  # BUG 6 FIX: enables tuple[...] hints on Python 3.8

import os
import warnings
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, classification_report,
    mean_absolute_error, mean_squared_error, r2_score,
)
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────────────────────

BASE_PATH = os.path.join(os.path.dirname(__file__), "Tourism Dataset", "")

VISIT_MODE_MAP = {
    1: "Business",
    2: "Couples",
    3: "Family",
    4: "Friends",
    5: "Solo",
}

SEASON_MAP = {
    0: "Autumn",
    1: "Spring",
    2: "Summer",
    3: "Winter",
}

REG_FEATURES = [
    "VisitYear", "VisitMonth", "VisitMode", "ContinentId", "RegionId",
    "CountryId", "CityId", "AttractionId", "AttractionCityId",
    "AttractionTypeId", "Season",
]

CLF_FEATURES = [
    "VisitYear", "VisitMonth", "ContinentId", "RegionId",
    "CountryId", "CityId", "AttractionId", "AttractionCityId",
    "AttractionTypeId", "Season", "Rating",
]


# ─────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────────────────────

def get_season_label(month: int) -> str:
    """Map month number to season string."""
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Spring"
    elif month in [6, 7, 8]:
        return "Summer"
    return "Autumn"


def month_to_season_code(month: int) -> int:
    """Convert month to the label-encoded season integer used in models."""
    label = get_season_label(month)
    # LabelEncoder sorts alphabetically: Autumn=0, Spring=1, Summer=2, Winter=3
    code_map = {"Autumn": 0, "Spring": 1, "Summer": 2, "Winter": 3}
    return code_map[label]


# ─────────────────────────────────────────────────────────────
# DATA LOADING  (cached — runs once per session)
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="⏳ Loading datasets…")
def load_and_preprocess() -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all 9 Excel files from Tourism Dataset folder,
    merge them into a master DataFrame, and return both the
    raw merged `df` and the model-ready `df_model`.
    """
    city        = pd.read_excel(f"{BASE_PATH}City.xlsx")
    continent   = pd.read_excel(f"{BASE_PATH}Continent.xlsx")
    country     = pd.read_excel(f"{BASE_PATH}Country.xlsx")
    item        = pd.read_excel(f"{BASE_PATH}Item.xlsx")
    mode        = pd.read_excel(f"{BASE_PATH}Mode.xlsx")
    region      = pd.read_excel(f"{BASE_PATH}Region.xlsx")
    transaction = pd.read_excel(f"{BASE_PATH}Transaction.xlsx")
    type_df     = pd.read_excel(f"{BASE_PATH}Type.xlsx")
    user        = pd.read_excel(f"{BASE_PATH}User.xlsx")

    # Merge into master df
    df = pd.merge(transaction, user, on="UserId", how="left")
    df = pd.merge(df, item, on="AttractionId", how="left")

    # Clean
    # BUG 7 FIX: Also drop NaN on Rating and VisitMode (ML targets), not just CityId
    df = df.dropna(subset=["CityId", "Rating", "VisitMode"])
    df["CityId"]     = df["CityId"].astype(int)
    df["VisitYear"]  = df["VisitYear"].astype(int)
    df["VisitMonth"] = df["VisitMonth"].astype(int)
    df["VisitMode"]  = df["VisitMode"].astype(int)
    df["Rating"]     = df["Rating"].astype(int)

    # Feature engineering – Season column on main df
    df["Season"] = df["VisitMonth"].apply(get_season_label)

    # Build df_model (model-ready copy)
    df_model = df.drop(columns=["TransactionId", "AttractionAddress"], errors="ignore")

    # Label encode Season
    le = LabelEncoder()
    df_model["Season"] = le.fit_transform(df_model["Season"])

    # Drop text Attraction col for model (keep in df for EDA)
    df_model_clean = df_model.drop(columns=["Attraction"], errors="ignore")

    return df, df_model_clean


# ─────────────────────────────────────────────────────────────
# MODEL TRAINING  (cached — runs once per session)
# ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="🤖 Training ML models (first time only)…")
def train_all_models(df_model: pd.DataFrame):
    """
    Train:
      1. RandomForestRegressor  → predict Rating
      2. RandomForestClassifier → predict VisitMode
      3. Content-Based Recommendation System (cosine similarity)

    Returns a dict with all trained objects.
    """
    # ── Regression ─────────────────────────────────────────
    X_reg = df_model[REG_FEATURES]
    y_reg = df_model["Rating"]

    scaler_reg = StandardScaler()
    X_reg_scaled = scaler_reg.fit_transform(X_reg)

    X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
        X_reg_scaled, y_reg, test_size=0.2, random_state=42
    )

    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_reg.fit(X_train_reg, y_train_reg)

    y_pred_reg = rf_reg.predict(X_test_reg)
    reg_metrics = {
        "R2":   round(r2_score(y_test_reg, y_pred_reg), 4),
        "RMSE": round(np.sqrt(mean_squared_error(y_test_reg, y_pred_reg)), 4),
        "MAE":  round(mean_absolute_error(y_test_reg, y_pred_reg), 4),
        "MSE":  round(mean_squared_error(y_test_reg, y_pred_reg), 4),
    }

    # ── Classification ──────────────────────────────────────
    X_clf = df_model[CLF_FEATURES]
    y_clf = df_model["VisitMode"]

    scaler_clf = StandardScaler()
    X_clf_scaled = scaler_clf.fit_transform(X_clf)

    X_train_clf, X_test_clf, y_train_clf, y_test_clf = train_test_split(
        X_clf_scaled, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )

    rf_clf = RandomForestClassifier(
        n_estimators=100, class_weight="balanced", random_state=42, n_jobs=-1
    )
    rf_clf.fit(X_train_clf, y_train_clf)

    y_pred_clf = rf_clf.predict(X_test_clf)
    clf_metrics = {
        "Accuracy": round(accuracy_score(y_test_clf, y_pred_clf), 4),
        "Report":   classification_report(y_test_clf, y_pred_clf, output_dict=True),
    }

    # ── Recommendation System ──────────────────────────────
    # Build attraction profile
    # BUG 8 FIX: Changed VisitCount to use ("Rating", "count") instead of ("UserId", "count")
    # UserId is not guaranteed to exist in df_model; Rating always does.
    attraction_profile = df_model.groupby("AttractionId").agg(
        AttractionTypeId=("AttractionTypeId", "first"),
        AttractionCityId=("AttractionCityId", "first"),
        AvgRating=("Rating", "mean"),
        VisitCount=("Rating", "count"),
    ).reset_index()

    feat_cols = ["AttractionTypeId", "AttractionCityId", "AvgRating", "VisitCount"]
    scaler_rec = MinMaxScaler()
    attr_features = scaler_rec.fit_transform(attraction_profile[feat_cols])

    cosine_sim = cosine_similarity(attr_features)

    return {
        "rf_reg":            rf_reg,
        "scaler_reg":        scaler_reg,
        "reg_metrics":       reg_metrics,
        "reg_features":      REG_FEATURES,
        "rf_clf":            rf_clf,
        "scaler_clf":        scaler_clf,
        "clf_metrics":       clf_metrics,
        "clf_features":      CLF_FEATURES,
        "cosine_sim":        cosine_sim,
        "attraction_profile": attraction_profile,
        "scaler_rec":        scaler_rec,
    }


# ─────────────────────────────────────────────────────────────
# PREDICTION HELPERS
# ─────────────────────────────────────────────────────────────

def predict_rating(models: dict, user_input: dict) -> float:
    """Predict attraction rating from user input dict."""
    row = pd.DataFrame([user_input])[models["reg_features"]]
    scaled = models["scaler_reg"].transform(row)
    pred = models["rf_reg"].predict(scaled)[0]
    return float(np.clip(pred, 1, 5))


def predict_visit_mode(models: dict, user_input: dict) -> tuple[int, str]:
    """Predict visit mode from user input dict. Returns (code, label)."""
    row = pd.DataFrame([user_input])[models["clf_features"]]
    scaled = models["scaler_clf"].transform(row)
    code = int(models["rf_clf"].predict(scaled)[0])
    label = VISIT_MODE_MAP.get(code, f"Mode {code}")
    return code, label


def get_recommendations(models: dict, attraction_id: int, top_n: int = 5) -> pd.DataFrame:
    """Return top-N similar attractions for a given AttractionId."""
    profile = models["attraction_profile"]
    cosine_sim = models["cosine_sim"]

    matches = profile[profile["AttractionId"] == attraction_id].index
    if len(matches) == 0:
        return pd.DataFrame(columns=["AttractionId", "AttractionTypeId", "AvgRating", "Similarity"])

    idx = matches[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1 : top_n + 1]
    rec_indices = [i[0] for i in sim_scores]

    result = profile.iloc[rec_indices][["AttractionId", "AttractionTypeId", "AttractionCityId", "AvgRating"]].copy()
    result["Similarity Score"] = [round(s[1], 4) for s in sim_scores]
    result["AvgRating"] = result["AvgRating"].round(2)
    result = result.reset_index(drop=True)
    result.index += 1  # 1-based rank
    return result
