"""
Real-Data Validation: Isolation Forest on NASA SMAP/MSL Telemetry
------------------------------------------------------------------------
Validates the Isolation Forest detector (same design as the synthetic
v2 detector) against real, published NASA spacecraft telemetry
(Hundman et al., 2018), using the expert-labeled ground-truth anomalies.
"""

import ast
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score

DATA_DIR = "../data/smap_msl/data/data"
LABELS_PATH = "../data/smap_msl/labeled_anomalies.csv"


def load_channel(chan_id, split="test"):
    return np.load(f"{DATA_DIR}/{split}/{chan_id}.npy")


def build_ground_truth(chan_id, n_timesteps, labels_df):
    row = labels_df[labels_df["chan_id"] == chan_id].iloc[0]
    sequences = ast.literal_eval(row["anomaly_sequences"])
    y_true = np.zeros(n_timesteps, dtype=int)
    for start, end in sequences:
        end = min(end, n_timesteps - 1)
        if start < n_timesteps:
            y_true[start:end + 1] = 1
    return y_true


def engineer_features(X):
    primary = X[:, 0:1]
    delta = np.abs(np.diff(primary, axis=0, prepend=primary[0:1]))
    return np.hstack([primary, delta])


def evaluate_channel(chan_id, labels_df, contamination=0.05):
    X_test = load_channel(chan_id, "test")
    y_true = build_ground_truth(chan_id, len(X_test), labels_df)

    if y_true.sum() == 0:
        return None

    feats = engineer_features(X_test)
    X_scaled = StandardScaler().fit_transform(feats)

    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=42)
    preds = model.fit_predict(X_scaled)
    y_pred = (preds == -1).astype(int)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    return {
        "channel": chan_id,
        "spacecraft": labels_df[labels_df["chan_id"] == chan_id].iloc[0]["spacecraft"],
        "n_timesteps": len(X_test),
        "n_anomalous_timesteps": int(y_true.sum()),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


if __name__ == "__main__":
    labels_df = pd.read_csv(LABELS_PATH)
    print(f"Loaded labels for {len(labels_df)} channels "
          f"({(labels_df['spacecraft']=='SMAP').sum()} SMAP, "
          f"{(labels_df['spacecraft']=='MSL').sum()} MSL)\n")

    results = []
    for chan_id in labels_df["chan_id"]:
        try:
            r = evaluate_channel(chan_id, labels_df)
            if r is not None:
                results.append(r)
                print(f"  {chan_id:6s} ({r['spacecraft']}): "
                      f"P={r['precision']:.2f} R={r['recall']:.2f} F1={r['f1']:.2f}")
        except FileNotFoundError:
            continue

    results_df = pd.DataFrame(results)
    results_df.to_csv("../data/smap_msl_validation_results.csv", index=False)

    print(f"\n=== Aggregated Results Across {len(results_df)} Real NASA Channels ===\n")
    print(f"Mean Precision: {results_df['precision'].mean():.3f} +/- {results_df['precision'].std():.3f}")
    print(f"Mean Recall:    {results_df['recall'].mean():.3f} +/- {results_df['recall'].std():.3f}")
    print(f"Mean F1:        {results_df['f1'].mean():.3f} +/- {results_df['f1'].std():.3f}")

    print("\n=== By Spacecraft ===\n")
    print(results_df.groupby("spacecraft")[["precision", "recall", "f1"]].mean().round(3))

    print(f"\nFull results saved to ../data/smap_msl_validation_results.csv")
