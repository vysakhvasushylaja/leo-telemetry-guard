"""
Fusion Anomaly Detector: ML (Isolation Forest) + Sequence-Freshness Check
------------------------------------------------------------------------------
Evaluates three detection strategies across 5 independent synthetic
datasets: ML-only, Sequence-only, and Fusion (OR of both).
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

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


def ml_only_predict(df, contamination=0.015):
    feats = BASE_FEATURES + [f"{c}_delta" for c in BASE_FEATURES]
    X = df[feats].values
    X_scaled = StandardScaler().fit_transform(X)
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    model.fit(X_scaled)
    preds = model.predict(X_scaled)
    return (preds == -1).astype(int)


def sequence_only_predict(df):
    flags = np.zeros(len(df), dtype=int)
    last_seen = -1
    for i, seq in enumerate(df["sequence_number"].values):
        if seq <= last_seen:
            flags[i] = 1
        else:
            last_seen = seq
    return flags


def evaluate(y_true, y_pred, label):
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {"method": label, "precision": precision, "recall": recall, "f1": f1}


def evaluate_replay_only(df, y_pred):
    replay_mask = df["label"] == "anomaly_replay"
    if replay_mask.sum() == 0:
        return 0.0
    return y_pred[replay_mask.values].mean()


def run_single_seed(seed):
    df = engineer_ml_features(generate_dataset(data_seed=42, attack_seed=seed))
    y_true = (df["label"] != "normal").astype(int).values

    ml_pred = ml_only_predict(df)
    seq_pred = sequence_only_predict(df)
    fusion_pred = np.maximum(ml_pred, seq_pred)

    results = []
    for pred, name in [(ml_pred, "ML-only"), (seq_pred, "Sequence-only"), (fusion_pred, "Fusion")]:
        r = evaluate(y_true, pred, name)
        r["replay_recall"] = evaluate_replay_only(df, pred)
        r["seed"] = seed
        results.append(r)
    return results


if __name__ == "__main__":
    SEEDS = [7, 17, 27, 37, 47]
    all_results = []

    print(f"Running fusion evaluation across {len(SEEDS)} independent synthetic datasets...\n")
    for seed in SEEDS:
        all_results.extend(run_single_seed(seed))

    results_df = pd.DataFrame(all_results)

    print("=== Per-Seed Results ===")
    print(results_df[["seed", "method", "precision", "recall", "f1", "replay_recall"]]
          .to_string(index=False))

    print("\n=== Aggregated Results (mean +/- std across 5 seeds) ===\n")
    summary = results_df.groupby("method").agg(
        precision_mean=("precision", "mean"), precision_std=("precision", "std"),
        recall_mean=("recall", "mean"), recall_std=("recall", "std"),
        f1_mean=("f1", "mean"), f1_std=("f1", "std"),
        replay_recall_mean=("replay_recall", "mean"), replay_recall_std=("replay_recall", "std"),
    ).round(3)
    print(summary.to_string())

    results_df.to_csv("../data/fusion_evaluation_results.csv", index=False)
    print("\nFull results saved to ../data/fusion_evaluation_results.csv")
