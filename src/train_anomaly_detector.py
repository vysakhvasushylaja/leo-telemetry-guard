"""
Anomaly Detection - Isolation Forest
--------------------------------------
Loads the synthetic telemetry dataset and trains an Isolation Forest
to flag anomalous packets.
"""

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import joblib

FEATURES = [
    "altitude_km", "velocity_kms", "battery_voltage",
    "thermal_c", "rf_signal_dbm", "attitude_roll_deg"
]


def load_data(path="../data/telemetry_dataset.csv"):
    df = pd.read_csv(path)
    df["is_anomaly_true"] = (df["label"] != "normal").astype(int)
    return df


def train_isolation_forest(df, contamination=0.01):
    X = df[FEATURES].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42
    )
    model.fit(X_scaled)

    preds = model.predict(X_scaled)
    df["is_anomaly_pred"] = (preds == -1).astype(int)
    df["anomaly_score"] = -model.decision_function(X_scaled)

    return model, scaler, df


def evaluate(df):
    print("\n=== Confusion Matrix ===")
    print(confusion_matrix(df["is_anomaly_true"], df["is_anomaly_pred"]))

    print("\n=== Classification Report ===")
    print(classification_report(
        df["is_anomaly_true"], df["is_anomaly_pred"],
        target_names=["normal", "anomaly"]
    ))


def plot_results(df):
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    axes[0].plot(df.index, df["altitude_km"], label="Altitude (km)", color="steelblue")
    axes[0].scatter(
        df[df["is_anomaly_pred"] == 1].index,
        df[df["is_anomaly_pred"] == 1]["altitude_km"],
        color="red", label="Detected anomaly", zorder=5, s=15
    )
    axes[0].set_ylabel("Altitude (km)")
    axes[0].legend()
    axes[0].set_title("Telemetry Stream with Detected Anomalies")

    axes[1].plot(df.index, df["anomaly_score"], color="darkorange", label="Anomaly score")
    axes[1].set_ylabel("Anomaly score")
    axes[1].set_xlabel("Sample index (time)")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("../data/anomaly_detection_plot.png", dpi=150)
    print("\nPlot saved to ../data/anomaly_detection_plot.png")


if __name__ == "__main__":
    print("Loading dataset...")
    df = load_data()

    print("Training Isolation Forest...")
    model, scaler, df = train_isolation_forest(df, contamination=0.01)

    evaluate(df)
    plot_results(df)

    joblib.dump(model, "../models/isolation_forest.pkl")
    joblib.dump(scaler, "../models/scaler.pkl")
    df.to_csv("../data/telemetry_with_predictions.csv", index=False)

    print("\nModel saved to ../models/isolation_forest.pkl")
