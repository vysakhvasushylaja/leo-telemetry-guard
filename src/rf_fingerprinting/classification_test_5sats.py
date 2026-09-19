from sklearn.ensemble import IsolationForest

by70_2_windows = [
    [-0.049664, 0.229161, 0.091633],[-0.049666, 0.248235, 0.098591],
    [-0.049670, 0.254556, 0.100556],[-0.049659, 0.254767, 0.100207],
    [-0.049658, 0.260737, 0.102725],[-0.049663, 0.254308, 0.099975],
    [-0.049670, 0.251946, 0.099320],[-0.049669, 0.250115, 0.097951],
    [-0.049674, 0.255936, 0.100609],[-0.049503, 0.247649, 0.098894],
]

model = IsolationForest(contamination=0.1, random_state=42)
model.fit(by70_2_windows)

test_cases = {
    "BY70-2 (genuine)": [-0.049650, 0.250741, 0.099046],
    "NOAA-15": [-0.086265, 0.418075, 0.144928],
    "CELESTA": [0.003718, 1.528385, 0.137166],
    "FENGYUN-4A": [0.001075, 1.276613, 0.170733],
    "UHF-FO": [-0.005674, 1.369510, 0.159344],
}

correct = 0
for name, feats in test_cases.items():
    pred = model.predict([feats])
    result = "GENUINE" if pred[0]==1 else "UNKNOWN"
    expected = "GENUINE" if "genuine" in name else "UNKNOWN"
    ok = result == expected
    correct += ok
    print(f"{name}: {result} {'✓' if ok else '✗'}")
print(f"\nAccuracy: {correct}/{len(test_cases)} ({100*correct/len(test_cases):.0f}%)")
