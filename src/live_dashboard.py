"""
LEO Telemetry Guard - LIVE Dashboard
------------------------------------------------
Run alongside live_stream_generator.py:
    Terminal 1: python3 live_stream_generator.py
    Terminal 2: streamlit run live_dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import os
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="LEO Telemetry Guard - LIVE", page_icon="🛰️", layout="wide")

st_autorefresh(interval=3000, key="live_refresh")

st.title("🛰️ AI-Driven Zero-Trust Telemetry Guard — LIVE")
st.caption("Real-time telemetry stream, updating automatically every 3 seconds")

DATA_PATH = "../data/live_telemetry_stream.csv"

if not os.path.exists(DATA_PATH):
    st.error("No live stream data found yet. Start the generator first:\n\n`python3 live_stream_generator.py` (in a separate terminal)")
    st.stop()

try:
    df = pd.read_csv(DATA_PATH)
except Exception as e:
    st.warning(f"Waiting for data... ({e})")
    st.stop()

if len(df) == 0:
    st.warning("Stream started but no packets received yet. Waiting...")
    st.stop()

last_update = df["timestamp"].iloc[-1]
st.success(f"🟢 LIVE — {len(df):,} packets received — last packet at {last_update}")

col1, col2, col3, col4 = st.columns(4)
total_anomalies = df["is_anomaly_true"].sum()
detected = df[df["is_anomaly_true"] == 1]["is_anomaly_pred"].sum() if total_anomalies > 0 else 0
recall = (detected / total_anomalies * 100) if total_anomalies > 0 else 0
false_positives = df[(df["is_anomaly_true"] == 0) & (df["is_anomaly_pred"] == 1)].shape[0]

with col1:
    st.metric("Packets Received (Live)", f"{len(df):,}")
with col2:
    st.metric("Anomalies Detected", f"{int(df['is_anomaly_pred'].sum())}")
with col3:
    st.metric("Detection Recall", f"{recall:.0f}%")
with col4:
    st.metric("False Positives", f"{false_positives}")

st.divider()
st.subheader("📡 Live Telemetry Stream")

window = df.tail(300).reset_index(drop=True)
fig1 = go.Figure()
fig1.add_trace(go.Scatter(x=window.index, y=window["altitude_km"], mode="lines", name="Altitude (km)", line=dict(color="steelblue", width=1.5)))
flagged = window[window["is_anomaly_pred"] == 1]
fig1.add_trace(go.Scatter(x=flagged.index, y=flagged["altitude_km"], mode="markers", name="Flagged (Live)", marker=dict(color="red", size=9, symbol="x")))
fig1.update_layout(xaxis_title="Recent packets", yaxis_title="Altitude (km)", height=350, hovermode="x unified")
st.plotly_chart(fig1, use_container_width=True)

st.subheader("🚨 Recent Flagged Events")
recent_flags = df[df["is_anomaly_pred"] == 1].tail(10).iloc[::-1]
if len(recent_flags) > 0:
    display_cols = ["timestamp", "sequence_number", "label", "seq_flag", "zscore_flag"]
    st.dataframe(recent_flags[display_cols], use_container_width=True, hide_index=True)
else:
    st.info("No anomalies flagged yet — stream is clean so far.")

st.divider()
st.subheader("🎯 Detection Rate by Attack Type (So Far)")
attack_types = df[df["label"] != "normal"]["label"].unique()
if len(attack_types) > 0:
    rates = []
    for atype in sorted(attack_types):
        subset = df[df["label"] == atype]
        caught = subset["is_anomaly_pred"].sum()
        rates.append({"Attack Type": atype.replace("anomaly_", ""), "Detection Rate (%)": 100 * caught / len(subset), "Count": len(subset)})
    rates_df = pd.DataFrame(rates)
    fig2 = px.bar(rates_df, x="Attack Type", y="Detection Rate (%)", color="Detection Rate (%)", color_continuous_scale="RdYlGn", text="Detection Rate (%)", range_color=[0, 100], hover_data=["Count"])
    fig2.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
    fig2.update_layout(height=300, showlegend=False)
    st.plotly_chart(fig2, use_container_width=True)
else:
    st.info("No attacks observed yet in this run.")

st.caption(f"Auto-refreshing every 3 seconds — {len(df):,} packets processed so far")
