import numpy as np

def load_iq_sigmf_full(filepath):
    with open(filepath, 'rb') as f:
        raw_bytes = f.read()
    raw = np.frombuffer(raw_bytes, dtype='<i2')
    I = raw[0::2].astype(np.float32)
    Q = raw[1::2].astype(np.float32)
    return I, Q

def scan_windows(I, Q, window_size=20000):
    amplitude_sq = I**2 + Q**2
    n_windows = len(I) // window_size
    results = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        Iw, Qw = I[start:end], Q[start:end]
        energy = np.mean(amplitude_sq[start:end])
        if energy < 1e-6:
            continue
        Iw_n = Iw / (np.max(np.abs(Iw)) + 1e-9)
        Qw_n = Qw / (np.max(np.abs(Qw)) + 1e-9)
        phase = np.unwrap(np.arctan2(Qw_n, Iw_n))
        phase_diff = np.diff(phase)
        phase_noise = np.std(phase_diff)
        results.append((i, energy, phase_noise))
    return results

if __name__ == "__main__":
    import sys
    filepath = sys.argv[1]
    I, Q = load_iq_sigmf_full(filepath)
    results = scan_windows(I, Q)
    energies = [r[1] for r in results]
    energy_threshold = np.percentile(energies, 70)
    candidates = [r for r in results if r[1] >= energy_threshold]
    candidates.sort(key=lambda r: r[2])
    print(f"Total windows scanned: {len(results)}")
    print(f"Candidates (top 30% energy): {len(candidates)}")
    print("Top 5 LOWEST phase-noise windows (among high-energy ones):")
    for idx, energy, pn in candidates[:5]:
        print(f"  Window #{idx}: energy={energy:.2f}, phase_noise={pn:.4f}")
