import numpy as np
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

print("=== Attack Scenario 1: Naive Attacker (random guess) ===")
naive_fake = [0.02, 0.8, 0.15]
pred = model.predict([naive_fake])
print(f"Naive fake: {'FOOLED' if pred[0]==1 else 'CORRECTLY REJECTED'}")

print("\n=== Attack Scenario 2: Sophisticated Attacker (exact BY70-2 match) ===")
sophisticated_fake = [-0.049650, 0.250741, 0.099046]
pred = model.predict([sophisticated_fake])
print(f"Exact feature match: {'FOOLED' if pred[0]==1 else 'REJECTED'}")

print("\n=== Attack Scenario 3: Near-Perfect Spoofing (small noise) ===")
rng = np.random.default_rng(1)
results = []
for trial in range(20):
    near_perfect = [
        -0.049650 + rng.normal(0, 0.0001),
        0.250741 + rng.normal(0, 0.01),
        0.099046 + rng.normal(0, 0.003),
    ]
    pred = model.predict([near_perfect])
    results.append(pred[0] == 1)
fooled_rate = sum(results) / len(results) * 100
print(f"Near-perfect spoofing fooled model: {fooled_rate:.0f}% of 20 trials")

print("\n=== Attack Scenario 4: Cross-Satellite Confusion (real NOAA-15 vs trained BY70-2) ===")
noaa15_as_attack = [-0.086265, 0.418075, 0.144928]
pred = model.predict([noaa15_as_attack])
print(f"NOAA-15 signal presented as BY70-2: {'FOOLED' if pred[0]==1 else 'REJECTED'}")
