"""
RF Fingerprinting — Isolation Forest Classification Demo
------------------------------------------------------------
Trains on BY70-2's genuine pattern, tests whether the model
correctly flags other satellites as "unknown/different" —
demonstrating lightweight, GPU-free RF fingerprinting.
"""

from sklearn.ensemble import IsolationForest

# BY70-2's 10-window features (from earlier consistency test)
by70_2_windows = [
    [-0.049664, 0.229161, 0.091633],
    [-0.049666, 0.248235, 0.098591],
    [-0.049670, 0.254556, 0.100556],
    [-0.049659, 0.254767, 0.100207],
    [-0.049658, 0.260737, 0.102725],
    [-0.049663, 0.254308, 0.099975],
    [-0.049670, 0.251946, 0.099320],
    [-0.049669, 0.250115, 0.097951],
    [-0.049674, 0.255936, 0.100609],
    [-0.049503, 0.247649, 0.098894],
]

print("=== Training Isolation Forest on BY70-2's genuine pattern ===")
model = IsolationForest(contamination=0.1, random_state=42)
model.fit(by70_2_windows)
print("Model trained on 10 windows of real BY70-2 satellite data.\n")

test_cases = {
    "BY70-2 (genuine)": [-0.049650, 0.250741, 0.099046],
    "ASTROBIO2 (different satellite)": [0.018842, 1.525757, 0.157169],
    "CELESTA (different satellite)": [0.000000, 0.034403, 0.035793],
}

print("=== Classification Results ===")
correct = 0
for name, features in test_cases.items():
    pred = model.predict([features])
    result = "GENUINE" if pred[0] == 1 else "ANOMALY/UNKNOWN"
    expected = "GENUINE" if "genuine" in name else "ANOMALY/UNKNOWN"
    is_correct = result == expected
    correct += is_correct
    print(f"  {name}: {result} {'✓' if is_correct else '✗'}")

print(f"\nAccuracy: {correct}/{len(test_cases)} ({100*correct/len(test_cases):.0f}%)")
