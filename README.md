# AI-Driven Zero-Trust Telemetry Guard

Anomaly detection system for Low Earth Orbit (LEO) satellite communication networks — combining unsupervised machine learning, cryptographic zero-trust verification, and real-time mobile alerting.

## Methodology & Data

Real LEO satellite telemetry is not publicly accessible (proprietary/classified). This project uses a synthetic telemetry generator built around realistic LEO orbital parameters (altitude ~550km, velocity ~7.6km/s) to demonstrate the detection architecture and methodology, and additionally validates the same detector against real, published NASA spacecraft telemetry.

## Pipeline

Telemetry Simulator -> ML Anomaly Detection -> Zero-Trust Verification -> Real-Time Alerting -> Dashboard

## Project Evolution — v1 to v4

This project was built iteratively, with each version targeting a specific, identified weakness or open question rather than being built all at once.

| Version | What Changed | Result |
|---|---|---|
| v1 | Isolation Forest baseline, raw sensor features | F1: 0.51 |
| v2 | Added rate-of-change (delta) features | F1: 0.69, but only 17% replay-attack detection |
| v3 | Added sequence-number fusion detector to fix the replay gap | F1: 0.717 +/- 0.028, replay detection 100% +/- 0% |
| v4 | Validated on real NASA telemetry, then iteratively improved | F1: 0.174 -> 0.457 across 3 diagnosed fixes (see below) |

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
| Real NASA data validation (v4, best) | F1: 0.457 +/- 0.257, up from 0.174 naive baseline (+163%) |

## v3 Extension — Closing the Replay-Detection Gap

The 17% replay-detection limitation identified in v2 was traced to a structural cause: Isolation Forest evaluates each packet independently, so a replayed packet's genuinely-normal values look statistically fine — the anomaly is in the timing, not the data.

Fix: a lightweight sequence-freshness check was added. Every packet now carries a monotonically increasing sequence number, and a fusion detector flags a packet if either the ML model or the sequence check fires.

Validated across 5 independent synthetic datasets (mean +/- std, not a single run):

| Method | Precision | Recall | F1-score | Replay Recall |
|---|---|---|---|---|
| ML-only (baseline) | 0.539 +/- 0.049 | 0.808 +/- 0.073 | 0.646 +/- 0.058 | 7% +/- 7% |
| Sequence-only | 1.000 +/- 0.000 | 0.148 +/- 0.070 | 0.253 +/- 0.107 | 100% +/- 0% |
| Fusion | 0.578 +/- 0.030 | 0.944 +/- 0.022 | 0.717 +/- 0.028 | 100% +/- 0% |

## v4 Extension — Real NASA Data Validation (Iterative Improvement)

Every result above was trained and evaluated on synthetic data. To test whether it generalises, the same Isolation Forest detector was run against NASA's SMAP and MSL spacecraft telemetry dataset — 82 real, expert-labeled channels from the Soil Moisture Active Passive satellite and the Mars Curiosity rover, published alongside Hundman et al. (2018).

A naive first application produced a substantial drop: F1-score 0.174 +/- 0.173, versus 0.69 on synthetic data. Rather than hide this, it was used as a diagnostic starting point, and three specific causes were identified and fixed in turn:

| Iteration | Approach | Mean F1 |
|---|---|---|
| Naive | Fixed contamination, minimal features | 0.174 +/- 0.173 |
| +1 | Adaptive per-channel contamination | 0.296 +/- 0.259 |
| +2 | Rolling-window features (mean, std, deviation) | 0.368 +/- 0.265 |
| +3 | Multi-scale windows (5/20/50) + robust statistics | 0.457 +/- 0.257 |

Three targeted, diagnosed fixes improved real-data F1-score by 163% (0.174 to 0.457), closing 66% of the gap to synthetic-data performance (0.69). The remaining gap is attributed to Isolation Forest's point-anomaly design rather than further feature engineering, and motivates a sequence-aware model (e.g. an LSTM Autoencoder) as future work.

Run it yourself:
```bash
python3 validate_on_smap_msl.py       # naive baseline (F1: 0.174)
python3 validate_on_smap_msl_v2.py    # + adaptive contamination (F1: 0.296)
python3 validate_on_smap_msl_v3.py    # + rolling-window features (F1: 0.368)
python3 validate_on_smap_msl_v4.py    # + multi-scale + robust stats (F1: 0.457)
```

To download the dataset (requires a free Kaggle account and API key):
```bash
pip install kaggle
kaggle datasets download -d patrickfleith/nasa-anomaly-detection-dataset-smap-msl
unzip nasa-anomaly-detection-dataset-smap-msl.zip -d data/smap_msl
```

## Live Streaming Dashboard

In addition to the batch/offline pipeline above, a live streaming version demonstrates the same detection logic running in real time: telemetry packets are generated one at a time (simulating a satellite pass), and each is checked inline using a sequence-freshness check plus a rolling z-score anomaly check, with the dashboard auto-refreshing every 3 seconds to show detections as they happen.

```bash
# Terminal 1 - generates the live stream
python3 live_stream_generator.py

# Terminal 2 - auto-refreshing dashboard
pip install streamlit-autorefresh
streamlit run live_dashboard.py
```

## Setup (Ubuntu)

```bash
cd leo-telemetry-guard
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Run the Pipeline

```bash
cd src

# Core pipeline (v1/v2)
python3 telemetry_simulator.py
python3 train_anomaly_detector.py
python3 train_anomaly_detector_v2.py
python3 multi_node_zero_trust_v2.py

# v3 fusion detector (replay-attack fix)
python3 telemetry_simulator_v3.py
python3 fusion_detector.py
python3 generate_fusion_predictions.py

# v4 real-data validation, iterative (requires SMAP/MSL dataset download, see above)
python3 validate_on_smap_msl.py
python3 validate_on_smap_msl_v2.py
python3 validate_on_smap_msl_v3.py
python3 validate_on_smap_msl_v4.py

# Alerting + dashboards
python3 ntfy_alerts.py
streamlit run dashboard.py            # batch dashboard
streamlit run live_dashboard.py       # live streaming dashboard (run live_stream_generator.py first)
```

## Project Structure

## Key Findings

1. Feature engineering beat threshold tuning: adding rate-of-change features (v1 to v2) improved F1 more than any amount of contamination-parameter tuning alone.

2. Isolation Forest is structurally blind to replay attacks, not just under-tuned: a replayed packet contains genuinely normal values, so the anomaly only exists in the packet's sequence/timing, which a point-anomaly detector cannot see by design.

3. Diagnosing the root cause led to a targeted, lightweight fix: rather than a heavier model, a simple sequence-number check, fused with the existing ML detector, closed the replay-detection gap completely (17% to 100%, validated across 5 independent runs).

4. Zero-trust micro-segmentation works: in the multi-node simulation, a heavily-attacked ground station (142 tamper attempts, 54% quarantine time) had zero effect on an unrelated satellite node (0% quarantine time) — trust is tracked independently per node.

5. Synthetic-data performance does not automatically transfer to real data: the same detector scoring F1 0.69 on synthetic data scored only F1 0.174 on real NASA spacecraft telemetry with a naive approach.

6. Diagnosed, iterative fixes closed most of the synthetic-to-real gap: three targeted improvements (adaptive per-channel contamination, rolling-window features, multi-scale + robust statistics) raised real-data F1 from 0.174 to 0.457 — a 163% improvement, closing 66% of the gap to synthetic performance, with the remainder attributed to the detector's point-anomaly design rather than further feature tuning.

## Roadmap

- [x] Synthetic telemetry generator
- [x] Isolation Forest baseline + rate-of-change feature engineering (v1 to v2)
- [x] Zero-trust auth layer (HMAC signing, trust scoring, quarantine, re-authentication)
- [x] Multi-node constellation simulation
- [x] Real-time ntfy alerting
- [x] Interactive dashboard (batch + live streaming versions)
- [x] Sequence-aware fusion detector for replay-attack detection (v3: 17% to 100%)
- [x] Validation against real telemetry - NASA SMAP/MSL dataset (v4: F1 0.174 naive baseline)
- [x] Per-channel adaptive contamination (v4: F1 0.174 to 0.296)
- [x] Rolling-window and multi-scale features (v4: F1 0.296 to 0.457)
- [ ] Validation against real LEO communication-satellite signals (RTL-SDR capture, in progress)
- [ ] LSTM Autoencoder to close the remaining synthetic-to-real gap
