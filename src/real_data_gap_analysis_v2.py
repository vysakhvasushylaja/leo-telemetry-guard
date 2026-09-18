"""
Real IQ Data — Gap Analysis Script (Memory-Efficient Version)
------------------------------------------------------------
Reads only a LIMITED CHUNK of the large .raw file (not the
entire 500+MB file into memory).
"""

import numpy as np


def load_iq_raw_chunk(filepath, header_bytes=41, n_samples=500000):
    bytes_per_sample_pair = 4
    bytes_to_read = n_samples * bytes_per_sample_pair

    with open(filepath, 'rb') as f:
        f.seek(header_bytes)
        raw_bytes = f.read(bytes_to_read)

    raw = np.frombuffer(raw_bytes, dtype='<i2')
    I = raw[0::2].astype(np.float32)
    Q = raw[1::2].astype(np.float32)
    I = I / (np.max(np.abs(I)) + 1e-9)
    Q = Q / (np.max(np.abs(Q)) + 1e-9)
    return I, Q


def extract_rf_features(I, Q):
    phase = np.unwrap(np.arctan2(Q, I))
    phase_diff = np.diff(phase)
    frequency_offset = np.mean(phase_diff) / (2 * np.pi)
    phase_noise = np.std(phase_diff)
    amplitude = np.sqrt(I**2 + Q**2)
    amplitude_variation = np.std(amplitude)
    return {
        "frequency_offset": float(frequency_offset),
        "phase_noise": float(phase_noise),
        "amplitude_variation": float(amplitude_variation),
    }


def analyze_in_windows(I, Q, window_size=10000, n_windows=10):
    results = []
    total_len = len(I)
    step = max(1, (total_len - window_size) // max(1, n_windows - 1))
    for i in range(n_windows):
        start = i * step
        end = start + window_size
        if end > total_len:
            break
        feats = extract_rf_features(I[start:end], Q[start:end])
        feats["window"] = i
        results.append(feats)
    return results


if __name__ == "__main__":
    import sys
    filepath = sys.argv[1] if len(sys.argv) > 1 else "../data/real_iq_samples/BY02_2020-07-12T11:39:50_436152kHz_192ksps.raw"
    sample_rate = 192000

    print(f"Loading real IQ recording (limited chunk): {filepath}")
    I, Q = load_iq_raw_chunk(filepath, n_samples=500000)
    print(f"Sample rate: {sample_rate} Hz, Samples loaded: {len(I)}, Duration: {len(I)/sample_rate:.2f}s\n")

    print("=== Full-chunk features ===")
    full_features = extract_rf_features(I, Q)
    for k, v in full_features.items():
        print(f"  {k}: {v:.6f}")

    print("\n=== Consistency check across 10 windows ===")
    print("(A real fingerprint should be relatively STABLE across windows)\n")
    window_results = analyze_in_windows(I, Q)

    for r in window_results:
        print(f"  Window {r['window']}: freq_offset={r['frequency_offset']:.6f}  "
              f"phase_noise={r['phase_noise']:.6f}  amp_var={r['amplitude_variation']:.6f}")

    freq_offsets = [r["frequency_offset"] for r in window_results]
    phase_noises = [r["phase_noise"] for r in window_results]
    amp_vars = [r["amplitude_variation"] for r in window_results]

    print("\n=== Consistency Summary (lower std = more stable fingerprint) ===")
    print(f"  frequency_offset: mean={np.mean(freq_offsets):.6f}, std={np.std(freq_offsets):.6f}")
    print(f"  phase_noise:      mean={np.mean(phase_noises):.6f}, std={np.std(phase_noises):.6f}")
    print(f"  amplitude_var:    mean={np.mean(amp_vars):.6f}, std={np.std(amp_vars):.6f}")
