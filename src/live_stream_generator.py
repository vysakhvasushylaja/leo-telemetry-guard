"""
Live Streaming Telemetry Generator
--------------------------------------
Continuously generates telemetry packets, one at a time, appending
each to a CSV file. Detection runs inline using sequence-check +
rolling z-score check.
"""

import time
import os
import numpy as np
import pandas as pd
from datetime import datetime, timezone
from collections import deque

BASE_FEATURES = [
    "altitude_km", "velocity_kms", "battery_voltage",
    "thermal_c", "rf_signal_dbm", "attitude_roll_deg"
]

OUT_PATH = "../data/live_telemetry_stream.csv"
ANOMALY_RATE = 0.03
WINDOW_SIZE = 100
Z_THRESHOLD = 3.5


def generate_normal_values(t, rng, last_attitude):
    altitude = 550 + 5 * np.sin(t / 200) + rng.normal(0, 0.5)
    velocity = 7.6 + 0.02 * np.sin(t / 200 + 1) + rng.normal(0, 0.01)
    battery_voltage = 28 + 1.5 * np.sin(t / 300) + rng.normal(0, 0.1)
    thermal = 15 + 20 * np.sin(t / 300 + 0.5) + rng.normal(0, 1)
    rf_signal = -80 + rng.normal(0, 2)
    attitude_roll = float(np.clip(last_attitude + rng.normal(0, 0.3) * 0.01, -5, 5))
    return {
        "altitude_km": altitude, "velocity_kms": velocity,
        "battery_voltage": battery_voltage, "thermal_c": thermal,
        "rf_signal_dbm": rf_signal, "attitude_roll_deg": attitude_roll,
    }


def apply_attack(values, atype, rng):
    v = dict(values)
    if atype == "spoofing":
        v["altitude_km"] += rng.uniform(20, 50) * rng.choice([-1, 1])
        v["velocity_kms"] += rng.uniform(0.5, 1.5) * rng.choice([-1, 1])
    elif atype == "jamming":
        v["rf_signal_dbm"] -= rng.uniform(20, 40)
    elif atype == "thermal_fault":
        v["thermal_c"] += rng.uniform(30, 60)
    elif atype == "power_drain":
        v["battery_voltage"] -= rng.uniform(5, 10)
    return v


def zscore_flag(values, window):
    if len(window) < 20:
        return 0
    arr = np.array([[w[f] for f in BASE_FEATURES] for w in window])
    means = arr.mean(axis=0)
    stds = arr.std(axis=0) + 1e-6
    current = np.array([values[f] for f in BASE_FEATURES])
    z = np.abs((current - means) / stds)
    return int(np.any(z > Z_THRESHOLD))


def run_stream(n_packets=100000, interval_seconds=1.0):
    rng = np.random.default_rng()
    window = deque(maxlen=WINDOW_SIZE)
    history_seq = deque(maxlen=200)
    last_seen_seq = -1
    last_attitude = 0.0

    if os.path.exists(OUT_PATH):
        os.remove(OUT_PATH)

    print("Starting live telemetry stream... (Ctrl+C to stop)")
    print(f"Writing to {OUT_PATH}\n")

    for t in range(n_packets):
        values = generate_normal_values(t, rng, last_attitude)
        last_attitude = values["attitude_roll_deg"]
        seq_number = t
        label = "normal"

        roll = rng.random()
        if roll < ANOMALY_RATE * 0.2 and len(history_seq) > 50:
            idx = rng.integers(0, len(history_seq))
            src_values, src_seq = history_seq[idx]
            values = dict(src_values)
            seq_number = src_seq
            label = "anomaly_replay"
        elif roll < ANOMALY_RATE:
            atype = rng.choice(["spoofing", "jamming", "thermal_fault", "power_drain"])
            values = apply_attack(values, atype, rng)
            label = f"anomaly_{atype}"

        seq_flag = int(seq_number <= last_seen_seq)
        z_flag = zscore_flag(values, window)
        fusion_flag = int(bool(seq_flag) or bool(z_flag))

        if seq_number > last_seen_seq:
            last_seen_seq = seq_number
        if label == "normal":
            window.append(values)
        history_seq.append((values, t))

        packet = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "sequence_number": seq_number,
            **values,
            "label": label,
            "is_anomaly_true": int(label != "normal"),
            "is_anomaly_pred": fusion_flag,
            "seq_flag": seq_flag,
            "zscore_flag": z_flag,
        }

        df_row = pd.DataFrame([packet])
        write_header = not os.path.exists(OUT_PATH)
        df_row.to_csv(OUT_PATH, mode="a", header=write_header, index=False)

        if t % 10 == 0:
            marker = " <-- FLAGGED" if fusion_flag else ""
            print(f"[{t:5d}] {label:22s} alt={values['altitude_km']:6.1f}  "
                  f"batt={values['battery_voltage']:5.1f}{marker}")

        time.sleep(interval_seconds)


if __name__ == "__main__":
    run_stream()
