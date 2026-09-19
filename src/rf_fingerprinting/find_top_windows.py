import numpy as np

def load_iq_sigmf_full(filepath):
    with open(filepath, 'rb') as f:
        raw_bytes = f.read()
    raw = np.frombuffer(raw_bytes, dtype='<i2')
    I = raw[0::2].astype(np.float32)
    Q = raw[1::2].astype(np.float32)
    return I, Q

def top_n_windows(I, Q, window_size=20000, n=5):
    amplitude = I**2 + Q**2
    n_windows = len(I) // window_size
    energies = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        energies.append(np.mean(amplitude[start:end]))
    energies = np.array(energies)
    top_idx = np.argsort(energies)[::-1][:n]
    return [(int(idx), float(energies[idx])) for idx in top_idx]

if __name__ == "__main__":
    import sys
    filepath = sys.argv[1]
    I, Q = load_iq_sigmf_full(filepath)
    print(f"Total samples: {len(I)}")
    print("Top 5 highest-energy windows:")
    for idx, energy in top_n_windows(I, Q):
        print(f"  Window #{idx}: energy={energy:.2f}")
