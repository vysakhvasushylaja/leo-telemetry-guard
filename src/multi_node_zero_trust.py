"""
Multi-Node Zero-Trust Authentication Layer
---------------------------------------------
Extends the single-node zero-trust model to a small constellation:
multiple satellites + a ground station, each with independently
tracked trust scores.
"""

import hashlib
import hmac
import json
import random
import pandas as pd


NODE_SECRETS = {
    "SAT-01": "sat01-secret-key-demo",
    "SAT-02": "sat02-secret-key-demo",
    "GROUND-STATION-A": "ground-a-secret-key-demo",
}

NODE_TAMPER_RATES = {
    "SAT-01": 0.05,
    "SAT-02": 0.01,
    "GROUND-STATION-A": 0.08,
}

TRUST_THRESHOLD = 40
TRUST_DECAY_ON_FAIL = 25
TRUST_DECAY_ON_ANOMALY = 10
TRUST_RECOVERY_ON_OK = 2
CLEAN_STREAK_TO_REAUTH = 20
REAUTH_TRUST_RESET = 60


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
    def __init__(self, node_ids):
        self.scores = {node: 100 for node in node_ids}
        self.quarantined = set()
        self.clean_streak = {node: 0 for node in node_ids}
        self.log = []

    def record_result(self, node_id: str, sig_valid: bool, is_anomaly: bool, packet_idx: int):
        was_quarantined = node_id in self.quarantined
        score = self.scores[node_id]
        is_clean = sig_valid and not is_anomaly

        if not sig_valid:
            score -= TRUST_DECAY_ON_FAIL
            self.clean_streak[node_id] = 0
        elif is_anomaly:
            score -= TRUST_DECAY_ON_ANOMALY
            self.clean_streak[node_id] = 0
        else:
            score = min(100, score + TRUST_RECOVERY_ON_OK)
            self.clean_streak[node_id] += 1

        score = max(0, score)
        self.scores[node_id] = score

        if not was_quarantined and score < TRUST_THRESHOLD:
            self.quarantined.add(node_id)
            self.clean_streak[node_id] = 0
            self.log.append((packet_idx, node_id, "QUARANTINED", score))
        elif was_quarantined:
            if is_clean and self.clean_streak[node_id] >= CLEAN_STREAK_TO_REAUTH:
                self.quarantined.discard(node_id)
                self.scores[node_id] = REAUTH_TRUST_RESET
                score = REAUTH_TRUST_RESET
                self.log.append((packet_idx, node_id, "RE-AUTHENTICATED", score))

        return score

    def is_quarantined(self, node_id: str) -> bool:
        return node_id in self.quarantined


def simulate_constellation(csv_path="../data/telemetry_with_predictions_v2.csv", seed=99):
    rng = random.Random(seed)
    df = pd.read_csv(csv_path)

    node_ids = list(NODE_SECRETS.keys())
    trust_mgr = TrustManager(node_ids)
    results = []

    for idx, row in df.iterrows():
        node_id = rng.choice(node_ids)
        tamper_fraction = NODE_TAMPER_RATES[node_id]

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

        currently_quarantined = trust_mgr.is_quarantined(node_id)
        trust_score = trust_mgr.record_result(node_id, sig_valid, is_anomaly, idx)

        results.append({
            "index": idx,
            "node_id": node_id,
            "tampered_in_transit": tampered,
            "sig_valid": sig_valid,
            "is_anomaly_pred": is_anomaly,
            "trust_score": trust_score,
            "quarantined_at_time_of_packet": currently_quarantined,
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv("../data/multi_node_zero_trust_results.csv", index=False)

    print("=== State Transition Log (last 15 events) ===")
    for packet_idx, node_id, event, score in trust_mgr.log[-15:]:
        print(f"  packet {packet_idx}: {node_id} -> {event} (trust score: {score})")

    print(f"\n(Total state transitions: {len(trust_mgr.log)})")

    print("\n=== Per-Node Summary ===")
    for node_id in node_ids:
        node_df = out_df[out_df["node_id"] == node_id]
        tampered_count = node_df["tampered_in_transit"].sum()
        caught_count = (node_df["tampered_in_transit"] & (~node_df["sig_valid"])).sum()
        quarantine_time = node_df["quarantined_at_time_of_packet"].sum()

        print(f"\n  {node_id}:")
        print(f"    Packets handled: {len(node_df)}")
        if tampered_count:
            print(f"    Tampered: {tampered_count}, Caught: {caught_count} "
                  f"({100*caught_count/tampered_count:.0f}%)")
        else:
            print(f"    Tampered: 0")
        print(f"    Final trust score: {trust_mgr.scores[node_id]}")
        print(f"    Currently quarantined: {trust_mgr.is_quarantined(node_id)}")
        print(f"    Time spent quarantined: {quarantine_time}/{len(node_df)} packets "
              f"({100*quarantine_time/len(node_df):.0f}%)")

    return out_df, trust_mgr


if __name__ == "__main__":
    simulate_constellation()
