# AI-Driven Zero-Trust Telemetry Guard

Anomaly detection system for Low Earth Orbit (LEO) satellite communication networks — combining unsupervised machine learning, cryptographic zero-trust verification, and real-time mobile alerting.

## Methodology & Data

Real LEO satellite telemetry is not publicly accessible (proprietary/classified). This project uses a synthetic telemetry generator built around realistic LEO orbital parameters (altitude ~550km, velocity ~7.6km/s) to demonstrate the detection architecture and methodology. The goal is to validate the pipeline design — not to claim results on real-world attack data.

## Pipeline

Telemetry Simulator -> ML Anomaly Detection -> Zero-Trust Verification -> Real-Time Alerting -> Dashboard

## Project Evolution — v1 to v3

This project was built iteratively, with each version targeting a specific, identified weakness rather than being built all at once.

| Version | What Changed | Result |
|---|---|---|
| v1 | Isolation Forest baseline, raw sensor features | F1: 0.51 |
| v2 | Added rate-of-change (delta) features | F1: 0.69, but only 17% replay-attack detection |
| v3 | Added sequence-number fusion detector to fix the replay gap | F1: 0.717 +/- 0.028, replay detection 100% +/- 0% |

## Results Summary

| Component | Result |
|---|---|
| ML detection (Isolation Forest v2) | F1-score 0.69 (up from 0.51 baseline) |
| Detection: spoofing / jamming / power drain | 100% each |
| Detection: thermal fault | 71% |
| Fusion detector (v3) | F1: 0.717 +/- 0.028, replay detection 100% +/- 0% (5-seed validated) |
| Zero-trust tamper detection | 100% (237/237 tampered packets caught) |
| Multi-node constellation | 3 nodes, micro-segmentation confirmed |
| Real-time alerting | Confirmed on web + mobile via ntfy |

## v3 Extension — Closing the Replay-Detection Gap

The 17% replay-detection limitation identified in v2 was traced to a structural cause: Isolation Forest evaluates each packet independently, so a replayed packet's genuinely-normal values look statistically fine — the anomaly is in the timing, not the data.

Fix: a lightweight sequence-freshness check was added. Every packet now carries a monotonically increasing sequence number, and a fusion detector flags a packet if either the ML model or the sequence check fires.

Validated across 5 independent synthetic datasets (mean +/- std, not a single run):

| Method | Precision | Recall | F1-score | Replay Recall |
|---|---|---|---|---|
| ML-only (baseline) | 0.539 +/- 0.049 | 0.808 +/- 0.073 | 0.646 +/- 0.058 | 7% +/- 7% |
| Sequence-only | 1.000 +/- 0.000 | 0.148 +/- 0.070 | 0.253 +/- 0.107 | 100% +/- 0% |
| Fusion | 0.578 +/- 0.030 | 0.944 +/- 0.022 | 0.717 +/- 0.028 | 100% +/- 0% |

The sequence-only check achieves perfect precision but low recall on its own — it only catches replay, nothing else — confirming it's a precise, narrow complement to the ML detector rather than a replacement for it.

## Setup (Ubuntu)

cd leo-telemetry-guard
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

## Run the Pipeline

cd src

python3 telemetry_simulator.py
python3 train_anomaly_detector.py
python3 train_anomaly_detector_v2.py
python3 multi_node_zero_trust_v2.py

python3 telemetry_simulator_v3.py
python3 fusion_detector.py
python3 generate_fusion_predictions.py

python3 ntfy_alerts.py
streamlit run dashboard.py

## Key Findings

1. Feature engineering beat threshold tuning: adding rate-of-change features (v1 to v2) improved F1 more than any amount of contamination-parameter tuning alone.

2. Isolation Forest is structurally blind to replay attacks, not just under-tuned: a replayed packet contains genuinely normal values, so the anomaly only exists in the packet's sequence/timing, which a point-anomaly detector cannot see by design.

3. Diagnosing the root cause led to a targeted, lightweight fix: rather than a heavier model, a simple sequence-number check, fused with the existing ML detector, closed the replay-detection gap completely (17% to 100%, validated across 5 independent runs).

4. Zero-trust micro-segmentation works: in the multi-node simulation, a heavily-attacked ground station (142 tamper attempts, 54% quarantine time) had zero effect on an unrelated satellite node (0% quarantine time) — trust is tracked independently per node.

## Roadmap

- Synthetic telemetry generator (done)
- Isolation Forest baseline + rate-of-change feature engineering, v1 to v2 (done)
- Zero-trust auth layer with trust scoring, quarantine, re-authentication (done)
- Multi-node constellation simulation (done)
- Real-time ntfy alerting (done)
- Interactive dashboard (done)
- Sequence-aware fusion detector for replay-attack detection, v3: 17% to 100% (done)
- Validation against real telemetry, RTL-SDR capture of public satellite signals (planned)
- LSTM Autoencoder for broader sequence-anomaly detection (planned)
