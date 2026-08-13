"""
Generate Fusion Detector Predictions CSV (for dashboard)
------------------------------------------------------------
Runs the ML (Isolation Forest) + Sequence-check fusion detector on a
single representative dataset and saves a full per-row CSV with
predictions, in the format the dashboard expects.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import numpy as np

from telemetry_simulator_v3 import generate_dataset

BASE_FEATURES = [
    "altitude_km", "velocity_kms", "battery_voltage",
    "thermal_c", "rf_signal_dbm", "attitude_roll_deg"
]


def engineer_ml_features(df):
    df = df.copy()
    for col in BASE_FEATURES:
        df[f"{col}_delta"] = df[col].diff().fillna(0).abs()
    return df


def ml_predict_with_scores(df, contamination=0.015):
    feats = BASE_FEATURES + [f"{c}_delta" for c in BASE_FEATURES]
    X = df[feats].values
    X_scaled = StandardScaler().fit_transform(X)
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    model.fit(X_scaled)
    preds = (model.predict(X_scaled) == -1).astype(int)
    scores = -model.decision_function(X_scaled)
    return preds, scores


def sequence_predict(df):
    flags = np.zeros(len(df), dtype=int)
    last_seen = -1
    for i, seq in enumerate(df["sequence_number"].values):
        if seq <= last_seen:
            flags[i] = 1
        else:
            last_seen = seq
    return flags


if __name__ == "__main__":
    df = generate_dataset(data_seed=42, attack_seed=7)
    df = engineer_ml_features(df)

    ml_pred, anomaly_score = ml_predict_with_scores(df)
    seq_pred = sequence_predict(df)
    fusion_pred = np.maximum(ml_pred, seq_pred)

    df["is_anomaly_true"] = (df["label"] != "normal").astype(int)
    df["is_anomaly_pred"] = fusion_pred
    df["ml_only_pred"] = ml_pred
    df["sequence_only_pred"] = seq_pred
    df["anomaly_score"] = anomaly_score

    out_path = "../data/telemetry_with_predictions_fusion.csv"
    df.to_csv(out_path, index=False)

    print(f"Saved {len(df)} rows to {out_path}")
    print(f"\nFusion detector caught {df['is_anomaly_pred'].sum()} / "
          f"{df['is_anomaly_true'].sum()} true anomalies")

    print("\n=== Detection Rate by Attack Type (Fusion) ===")
    for atype in sorted(df[df["label"] != "normal"]["label"].unique()):
        subset = df[df["label"] == atype]
        caught = subset["is_anomaly_pred"].sum()
        total = len(subset)
        print(f"  {atype}: {caught}/{total} caught ({100*caught/total:.0f}%)")
