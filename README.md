# AI-Driven Zero-Trust Telemetry Guard

Anomaly detection system for Low Earth Orbit (LEO) satellite communication networks — combining unsupervised machine learning, cryptographic zero-trust verification, and real-time mobile alerting.

## Methodology & Data

Real LEO satellite telemetry is not publicly accessible (proprietary/classified). This project uses a synthetic telemetry generator built around realistic LEO orbital parameters (altitude ~550km, velocity ~7.6km/s) to demonstrate the detection architecture and methodology. The goal is to validate the pipeline design — not to claim results on real-world attack data. See the full report for a detailed discussion of this scope.

## Pipeline

Telemetry Simulator -> ML Anomaly Detection -> Zero-Trust Verification -> Real-Time Alerting -> Dashboard

## Results Summary

| Component | Result |
|---|---|
| ML detection (Isolation Forest v2) | F1-score 0.69 (up from 0.51 baseline) |
| Detection: spoofing / jamming / power drain | 100% each |
| Detection: replay attacks | 17% (documented limitation) |
| Zero-trust tamper detection | 100% (237/237 tampered packets caught) |
| Multi-node constellation | 3 nodes, micro-segmentation confirmed |
| Real-time alerting | Confirmed on web + mobile via ntfy |

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
python3 ntfy_alerts.py
streamlit run dashboard.py

## Key Finding

Isolation Forest achieves 100% detection on attacks that manifest as sudden statistical deviations (spoofing, jamming, power drain), but only 17% on replay attacks, because a replayed packet contains genuinely normal values — the anomaly is in the timing, not the data. This motivates future work on a sequence-aware model (LSTM Autoencoder).

## Roadmap

- Synthetic telemetry generator (done)
- Isolation Forest baseline + rate-of-change feature engineering (done)
- Zero-trust auth layer with trust scoring, quarantine, re-authentication (done)
- Multi-node constellation simulation (done)
- Real-time ntfy alerting (done)
- Interactive dashboard (done)
- LSTM Autoencoder for replay-attack detection (planned)
- Validation against real/higher-fidelity telemetry (planned)
