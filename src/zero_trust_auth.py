"""
Zero-Trust Authentication Layer
---------------------------------
Simulates per-packet zero-trust verification for LEO telemetry.
"""

import hashlib
import hmac
import json
import random
import pandas as pd


NODE_SECRETS = {
    "SAT-01": "sat01-secret-key-demo",
    "GROUND-STATION-A": "ground-a-secret-key-demo",
}

TRUST_THRESHOLD = 40
TRUST_DECAY_ON_FAIL = 25
TRUST_DECAY_ON_ANOMALY = 10
TRUST_RECOVERY_ON_OK = 2


def sign_packet(packet: dict, node_id: str) -> str:
    secret = NODE_SECRETS[node_id].encode()
    payload = json.dumps(packet, sort_keys=True).encode()
    return hmac.new(secret, payload, hashlib.sha256).hexdigest()


def verify_packet(packet: dict, node_id: str, signature: str) -> bool:
    if node_id not in NODE_SECRETS:
        return False
    expected = sign_packet(packet, node_id)
    return hmac.compare_digest(expected, signature)


class TrustManager:
    def __init__(self):
        self.scores = {node: 100 for node in NODE_SECRETS}
        self.quarantined = set()

    def record_result(self, node_id: str, sig_valid: bool, is_anomaly: bool):
        score = self.scores.get(node_id, 100)

        if not sig_valid:
            score -= TRUST_DECAY_ON_FAIL
        elif is_anomaly:
            score -= TRUST_DECAY_ON_ANOMALY
        else:
            score = min(100, score + TRUST_RECOVERY_ON_OK)

        score = max(0, score)
        self.scores[node_id] = score

        if score < TRUST_THRESHOLD:
            self.quarantined.add(node_id)

        return score

    def is_quarantined(self, node_id: str) -> bool:
        return node_id in self.quarantined

    def reset(self, node_id: str):
        self.scores[node_id] = 100
        self.quarantined.discard(node_id)


def simulate_zero_trust_pipeline(csv_path="../data/telemetry_with_predictions_v2.csv",
                                  tamper_fraction=0.05, seed=99):
    rng = random.Random(seed)
    df = pd.read_csv(csv_path)

    trust_mgr = TrustManager()
    results = []

    for idx, row in df.iterrows():
        node_id = "SAT-01"

        packet = {
            "altitude_km": round(float(row["altitude_km"]), 3),
            "velocity_kms": round(float(row["velocity_kms"]), 3),
            "battery_voltage": round(float(row["battery_voltage"]), 3),
            "thermal_c": round(float(row["thermal_c"]), 3),
        }

        signature = sign_packet(packet, node_id)

        tampered = rng.random() < tamper_fraction
        if tampered:
            packet["altitude_km"] += rng.uniform(1, 5)

        sig_valid = verify_packet(packet, node_id, signature)
        is_anomaly = bool(row.get("is_anomaly_pred", 0))

        trust_score = trust_mgr.record_result(node_id, sig_valid, is_anomaly)
        quarantined = trust_mgr.is_quarantined(node_id)

        results.append({
            "index": idx,
            "node_id": node_id,
            "tampered_in_transit": tampered,
            "sig_valid": sig_valid,
            "is_anomaly_pred": is_anomaly,
            "trust_score": trust_score,
            "quarantined": quarantined,
        })

        if quarantined and idx % 500 == 0:
            print(f"[!] Node {node_id} quarantined at packet {idx} "
                  f"(trust score: {trust_score})")

    out_df = pd.DataFrame(results)
    out_df.to_csv("../data/zero_trust_results.csv", index=False)

    print("\n=== Zero-Trust Summary ===")
    print(f"Total packets processed: {len(out_df)}")
    print(f"Packets tampered in transit: {out_df['tampered_in_transit'].sum()}")
    print(f"Signature failures caught: {(~out_df['sig_valid']).sum()}")
    print(f"Final trust score for SAT-01: {trust_mgr.scores['SAT-01']}")
    print(f"Node quarantined by end of run: {trust_mgr.is_quarantined('SAT-01')}")

    caught = out_df[out_df["tampered_in_transit"] & (~out_df["sig_valid"])]
    total_tampered = out_df["tampered_in_transit"].sum()
    if total_tampered:
        print(f"Tamper detection rate: {len(caught)}/{total_tampered} "
              f"({100*len(caught)/total_tampered:.0f}%)")

    return out_df


if __name__ == "__main__":
    simulate_zero_trust_pipeline()
