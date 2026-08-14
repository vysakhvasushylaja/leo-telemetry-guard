## Key Findings

1. Feature engineering beat threshold tuning: adding rate-of-change features (v1 to v2) improved F1 more than any amount of contamination-parameter tuning alone.

2. Isolation Forest is structurally blind to replay attacks, not just under-tuned: a replayed packet contains genuinely normal values, so the anomaly only exists in the packet's sequence/timing, which a point-anomaly detector cannot see by design.

3. Diagnosing the root cause led to a targeted, lightweight fix: rather than a heavier model, a simple sequence-number check, fused with the existing ML detector, closed the replay-detection gap completely (17% to 100%, validated across 5 independent runs).

4. Zero-trust micro-segmentation works: in the multi-node simulation, a heavily-attacked ground station (142 tamper attempts, 54% quarantine time) had zero effect on an unrelated satellite node (0% quarantine time) — trust is tracked independently per node.

5. Synthetic-data performance does not automatically transfer to real data: the same detector scoring F1 0.69 on synthetic data scored only F1 0.174 on real NASA spacecraft telemetry, most likely due to a single fixed sensitivity setting not matching each channel's true anomaly rate. This result is reported honestly rather than hidden, and per-channel adaptive calibration is the planned fix.

## Roadmap

- [x] Synthetic telemetry generator
- [x] Isolation Forest baseline + rate-of-change feature engineering (v1 to v2)
- [x] Zero-trust auth layer (HMAC signing, trust scoring, quarantine, re-authentication)
- [x] Multi-node constellation simulation
- [x] Real-time ntfy alerting
- [x] Interactive dashboard (batch + live streaming versions)
- [x] Sequence-aware fusion detector for replay-attack detection (v3: 17% to 100%)
- [x] Validation against real telemetry - NASA SMAP/MSL dataset (v4: F1 0.174, honestly reported)
- [ ] Per-channel adaptive contamination to close the synthetic-to-real F1 gap
- [ ] Validation against real LEO communication-satellite signals (RTL-SDR capture, in progress)
- [ ] LSTM Autoencoder for broader sequence-anomaly detection
