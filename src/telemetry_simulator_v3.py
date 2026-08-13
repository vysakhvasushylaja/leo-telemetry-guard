"""
LEO Satellite Telemetry Simulator v3
--------------------------------------
Adds sequence numbers to enable replay-attack detection via
sequence-freshness checking, in addition to the existing
rate-of-change feature engineering.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone


def generate_normal_telemetry(n_samples=5000, seed=42):
    rng = np.random.default_rng(seed)
    t = np.arange(n_samples)

    altitude = 550 + 5 * np.sin(t / 200) + rng.normal(0, 0.5, n_samples)
    velocity = 7.6 + 0.02 * np.sin(t / 200 + 1) + rng.normal(0, 0.01, n_samples)
    battery_voltage = 28 + 1.5 * np.sin(t / 300) + rng.normal(0, 0.1, n_samples)
    thermal = 15 + 20 * np.sin(t / 300 + 0.5) + rng.normal(0, 1, n_samples)
    rf_signal = -80 + rng.normal(0, 2, n_samples)
    attitude_roll = rng.normal(0, 0.3, n_samples).cumsum() * 0.01
    attitude_roll = np.clip(attitude_roll, -5, 5)

    timestamps = [datetime.now(timezone.utc) + timedelta(seconds=int(i)) for i in t]

    df = pd.DataFrame({
        "timestamp": timestamps,
        "sequence_number": t,
        "altitude_km": altitude,
        "velocity_kms": velocity,
        "battery_voltage": battery_voltage,
        "thermal_c": thermal,
        "rf_signal_dbm": rf_signal,
        "attitude_roll_deg": attitude_roll,
        "label": "normal"
    })
    return df


def inject_anomalies(df, n_anomalies=50, seed=7):
    rng = np.random.default_rng(seed)
    df = df.copy()
    n = len(df)
    anomaly_types = ["spoofing", "replay", "jamming", "thermal_fault", "power_drain"]

    chosen_idx = rng.choice(n, size=n_anomalies, replace=False)

    for idx in chosen_idx:
        atype = rng.choice(anomaly_types)

        if atype == "spoofing":
            df.loc[idx, "altitude_km"] += rng.uniform(20, 50) * rng.choice([-1, 1])
            df.loc[idx, "velocity_kms"] += rng.uniform(0.5, 1.5) * rng.choice([-1, 1])

        elif atype == "replay":
            if idx > 100:
                src = idx - rng.integers(50, 100)
                cols = ["altitude_km", "velocity_kms", "battery_voltage",
                        "thermal_c", "rf_signal_dbm"]
                df.loc[idx, cols] = df.loc[src, cols].values
                df.loc[idx, "sequence_number"] = df.loc[src, "sequence_number"]

        elif atype == "jamming":
            df.loc[idx, "rf_signal_dbm"] -= rng.uniform(20, 40)

        elif atype == "thermal_fault":
            df.loc[idx, "thermal_c"] += rng.uniform(30, 60)

        elif atype == "power_drain":
            df.loc[idx, "battery_voltage"] -= rng.uniform(5, 10)

        df.loc[idx, "label"] = f"anomaly_{atype}"

    return df


def generate_dataset(n_samples=5000, n_anomalies=50, data_seed=42, attack_seed=7):
    df = generate_normal_telemetry(n_samples=n_samples, seed=data_seed)
    df = inject_anomalies(df, n_anomalies=n_anomalies, seed=attack_seed)
    return df


if __name__ == "__main__":
    df = generate_dataset()
    out_path = "../data/telemetry_dataset_v3.csv"
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")
    print(df['label'].value_counts())
