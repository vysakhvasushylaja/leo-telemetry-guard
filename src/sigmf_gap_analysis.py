"""
Real IQ Data — Second Satellite (ASTROBIO2)
Load fully (small file), find window with actual signal (skip silence).
"""

import numpy as np


def load_iq_sigmf_full(filepath):
    with open(filepath, 'rb') as f:
        raw_bytes = f.read()
    raw = np.frombuffer(raw_bytes, dtype='<i2')
    I = raw[0::2].astype(np.float32)
    Q = raw[1::2].astype(np.float32)
    return I, Q


def find_active_window(I, Q, window_size=20000):
    amplitude = I**2 + Q**2
    n_windows = len(I) // window_size
    energies = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        energies.append(np.mean(amplitude[start:end]))
    best_window = int(np.argmax(energies))
    start = best_window * window_size
    return I[start:start + window_size], Q[start:start + window_size], best_window, energies[best_window]


def extract_rf_features(I, Q):
    I = I / (np.max(np.abs(I)) + 1e-9)
    Q = Q / (np.max(np.abs(Q)) + 1e-9)
    phase = np.unwrap(np.arctan2(Q, I))
    phase_diff = np.diff(phase)
    frequency_offset = float(np.mean(phase_diff) / (2 * np.pi))
    phase_noise = float(np.std(phase_diff))
    amplitude = np.sqrt(I**2 + Q**2)
    amplitude_variation = float(np.std(amplitude))
    return {
        "frequency_offset": frequency_offset,
        "phase_noise": phase_noise,
        "amplitude_variation": amplitude_variation,
    }


if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else "../data/real_iq_samples/ASTROBIO2_2022-07-24T19_25_49.sigmf-data"

    print(f"Loading SATELLITE 2 (ASTROBIO2) IQ recording (full file): {filepath}")
    I_full, Q_full = load_iq_sigmf_full(filepath)
    print(f"Total samples: {len(I_full)}")

    print("Finding active (non-silent) signal window...")
    I, Q, window_idx, energy = find_active_window(I_full, Q_full)
    print(f"Best window: #{window_idx}, energy: {energy:.2f}\n")

    print("=== ASTROBIO2 Features (from active window) ===")
    features = extract_rf_features(I, Q)
    for k, v in features.items():
        print(f"  {k}: {v:.6f}")

    print("\n=== COMPARISON with Satellite 1 (BY70-2) ===")
    print("BY70-2 results (from earlier run):")
    print("  frequency_offset: -0.049650")
    print("  phase_noise: 0.250741")
    print("  amplitude_variation: 0.099046")
    print()
    print(f"ASTROBIO2 vs BY70-2 — Difference:")
    print(f"  frequency_offset: {abs(features['frequency_offset'] - (-0.049650)):.6f}")
    print(f"  phase_noise: {abs(features['phase_noise'] - 0.250741):.6f}")
    print(f"  amplitude_variation: {abs(features['amplitude_variation'] - 0.099046):.6f}")
