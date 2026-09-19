import numpy as np
from scipy.io import wavfile

def extract_rf_features(I, Q):
    I = I / (np.max(np.abs(I)) + 1e-9)
    Q = Q / (np.max(np.abs(Q)) + 1e-9)
    phase = np.unwrap(np.arctan2(Q, I))
    phase_diff = np.diff(phase)
    frequency_offset = float(np.mean(phase_diff) / (2 * np.pi))
    phase_noise = float(np.std(phase_diff))
    amplitude = np.sqrt(I**2 + Q**2)
    amplitude_variation = float(np.std(amplitude))
    return {"frequency_offset": frequency_offset, "phase_noise": phase_noise, "amplitude_variation": amplitude_variation}

def find_locked_window(filepath, window_size=20000, n_chunks=20, chunk_size=2_000_000):
    sr, data = wavfile.read(filepath, mmap=True)
    total_len = data.shape[0]
    step = max(1, (total_len - chunk_size) // n_chunks)
    candidates = []
    for c in range(n_chunks):
        start = c * step
        end = min(start + chunk_size, total_len)
        chunk = np.array(data[start:end])
        I_c = chunk[:, 0].astype(np.float32)
        Q_c = chunk[:, 1].astype(np.float32)
        n_windows = len(I_c) // window_size
        amp_sq = I_c**2 + Q_c**2
        for w in range(n_windows):
            ws, we = w * window_size, (w + 1) * window_size
            energy = np.mean(amp_sq[ws:we])
            if energy < 1e-3:
                continue
            feats = extract_rf_features(I_c[ws:we], Q_c[ws:we])
            candidates.append((start + ws, energy, feats['phase_noise'], feats))
    if not candidates:
        return None
    energies = [c[1] for c in candidates]
    threshold = np.percentile(energies, 80)
    high_energy = [c for c in candidates if c[1] >= threshold]
    high_energy.sort(key=lambda c: c[2])
    return high_energy[0] if high_energy else None

if __name__ == "__main__":
    import sys
    filepath = sys.argv[1]
    print(f"Scanning {filepath} for locked-signal windows...")
    result = find_locked_window(filepath)
    if result is None:
        print("No usable signal found.")
    else:
        sample_idx, energy, phase_noise, feats = result
        print(f"Best window at sample #{sample_idx}, energy={energy:.2f}")
        print("Features:")
        for k, v in feats.items():
            print(f"  {k}: {v:.6f}")
