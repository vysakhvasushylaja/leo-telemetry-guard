import numpy as np

def load_iq_sigmf_full(filepath):
    with open(filepath, 'rb') as f:
        raw_bytes = f.read()
    raw = np.frombuffer(raw_bytes, dtype='<i2')
    I = raw[0::2].astype(np.float32)
    Q = raw[1::2].astype(np.float32)
    return I, Q

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

if __name__ == "__main__":
    import sys
    filepath = sys.argv[1]
    window_idx = int(sys.argv[2])
    window_size = 20000
    I, Q = load_iq_sigmf_full(filepath)
    start = window_idx * window_size
    I_w, Q_w = I[start:start+window_size], Q[start:start+window_size]
    feats = extract_rf_features(I_w, Q_w)
    print(f"Features from window #{window_idx}:")
    for k, v in feats.items():
        print(f"  {k}: {v:.6f}")
