from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

mpl.use("Agg")


def detect_suspicious_trips(df: pd.DataFrame) -> pd.DataFrame:
    suspicious = pd.DataFrame()

    conditions = [
        (df["trip_duration"] < 5) & (df["fare_amount"] > 50),
        (df["trip_duration"] > 180),
        (df["trip_distance"] < 1) & (df["fare_amount"] > 100),
        (df["fare_amount"] <= 0),
        (df["avg_speed_mph"] < 1),
        (df["avg_speed_mph"] > 60),
    ]

    reasons = [
        "Short & Extremely Expensive",
        "Unrealistically Long (3+ hours)",
        "Short Trip, Extreme Fare",
        "Invalid or Zero Fare",
        "Unrealistically Slow (<1 mph)",
        "Impossibly Fast (>60 mph)",
    ]

    for i, condition in enumerate(conditions):
        trips = df[condition].copy()
        if not trips.empty:
            trips["reason"] = reasons[i]
            suspicious = pd.concat([suspicious, trips], ignore_index=True)

    suspicious.drop_duplicates(
        subset=["tpep_pickup_datetime", "tpep_dropoff_datetime"], inplace=True
    )
    suspicious["year"] = suspicious["year"].astype(int)

    return suspicious


def run_isolation_forest(df: pd.DataFrame) -> pd.DataFrame:
    features = df[["trip_duration", "trip_distance", "fare_amount", "avg_speed_mph"]]
    clf = IsolationForest(n_estimators=50, contamination=0.01, random_state=42)
    df["anomaly_flag"] = clf.fit_predict(features)
    return df[df["anomaly_flag"] == -1]


def train_simple_model(df: pd.DataFrame, year: int) -> None:
    df["label"] = df["severity_score"] > 0

    if df["label"].nunique() < 2:
        print(
            f"Error: Only one class found in 'label' for {year}. Model needs both True and False."
        )
        return

    features = df[["trip_duration", "trip_distance", "fare_amount", "avg_speed_mph"]]
    model = LogisticRegression()
    model.fit(features, df["label"])

    print(f"\nLogistic Regression Model Coefficients ({year}):")
    for feature, coef in zip(features.columns, model.coef_[0]):
        print(f"  {feature}: {coef:.4f}")

    probs = model.predict_proba(features)[:, 1]
    plt.figure(figsize=(8, 5))
    sns.histplot(probs, bins=30, kde=True, color="skyblue")
    plt.title(f"Prediction Confidence for Suspicious Trips ({year})")
    plt.xlabel("Predicted Probability")
    plt.ylabel("Number of Trips")
    plt.tight_layout()
    plt.savefig(
        Path(__file__).parent
        / "output_anomaly_comparison"
        / f"prediction_confidence_histogram_{year}.png"
    )
    plt.close()
