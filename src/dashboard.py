"""
LEO Telemetry Guard - Interactive Dashboard
------------------------------------------------
Run with: streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="LEO Telemetry Guard",
    page_icon="🛰️",
    layout="wide"
)

st.title("🛰️ AI-Driven Zero-Trust Telemetry Guard")
st.caption("LEO Satellite Anomaly Detection & Zero-Trust Security Dashboard")

st.info(
    "⚠️ **Note on data**: This dashboard visualizes a synthetic telemetry "
    "dataset generated to demonstrate the detection architecture and "
    "methodology, since real LEO telemetry is not publicly accessible. "
    "See README for details.",
    icon="ℹ️"
)

@st.cache_data
def load_telemetry():
    return pd.read_csv("../data/telemetry_with_predictions_fusion.csv")

@st.cache_data
def load_zero_trust():
    return pd.read_csv("../data/multi_node_zero_trust_results.csv")

try:
    telemetry_df = load_telemetry()
    zt_df = load_zero_trust()
except FileNotFoundError as e:
    st.error(f"Missing data file: {e}. Run the pipeline scripts first.")
    st.stop()

col1, col2, col3, col4 = st.columns(4)

total_anomalies = (telemetry_df["label"] != "normal").sum()
detected = telemetry_df["is_anomaly_pred"].sum()
recall = (telemetry_df[telemetry_df["label"] != "normal"]["is_anomaly_pred"].sum()
          / total_anomalies * 100) if total_anomalies else 0

tampered = zt_df["tampered_in_transit"].sum()
caught = (zt_df["tampered_in_transit"] & (~zt_df["sig_valid"])).sum()
tamper_rate = (caught / tampered * 100) if tampered else 0

with col1:
    st.metric("Telemetry Packets Analyzed", f"{len(telemetry_df):,}")
with col2:
    st.metric("Fusion Detector Recall", f"{recall:.0f}%")
with col3:
    st.metric("Zero-Trust Tamper Detection", f"{tamper_rate:.0f}%")
with col4:
    st.metric("Nodes in Constellation", zt_df["node_id"].nunique())

st.divider()

st.subheader("📡 Telemetry Stream with Detected Anomalies")

fig1 = go.Figure()
fig1.add_trace(go.Scatter(
    x=telemetry_df.index, y=telemetry_df["altitude_km"],
    mode="lines", name="Altitude (km)", line=dict(color="steelblue", width=1)
))
anomalies = telemetry_df[telemetry_df["is_anomaly_pred"] == 1]
fig1.add_trace(go.Scatter(
    x=anomalies.index, y=anomalies["altitude_km"],
    mode="markers", name="Detected Anomaly",
    marker=dict(color="red", size=6, symbol="x")
))
fig1.update_layout(
    xaxis_title="Packet Index (time)", yaxis_title="Altitude (km)",
    height=400, hovermode="x unified"
)
st.plotly_chart(fig1, use_container_width=True)

st.subheader("🎯 Detection Rate by Attack Type")

attack_types = telemetry_df[telemetry_df["label"] != "normal"]["label"].unique()
rates = []
for atype in sorted(attack_types):
    subset = telemetry_df[telemetry_df["label"] == atype]
    caught_n = subset["is_anomaly_pred"].sum()
    rates.append({"Attack Type": atype.replace("anomaly_", ""),
                   "Detection Rate (%)": 100 * caught_n / len(subset),
                   "Total": len(subset), "Caught": caught_n})

rates_df = pd.DataFrame(rates)
fig2 = px.bar(rates_df, x="Attack Type", y="Detection Rate (%)",
              color="Detection Rate (%)", color_continuous_scale="RdYlGn",
              text="Detection Rate (%)", range_color=[0, 100])
fig2.update_traces(texttemplate="%{text:.0f}%", textposition="outside")
fig2.update_layout(height=350, showlegend=False)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

st.subheader("🔐 Zero-Trust: Per-Node Trust Score Over Time")

node_choice = st.multiselect(
    "Select nodes to display",
    options=zt_df["node_id"].unique().tolist(),
    default=zt_df["node_id"].unique().tolist()
)

fig3 = go.Figure()
for node in node_choice:
    node_data = zt_df[zt_df["node_id"] == node].sort_values("index")
    fig3.add_trace(go.Scatter(
        x=node_data["index"], y=node_data["trust_score"],
        mode="lines", name=node
    ))
fig3.add_hline(y=40, line_dash="dash", line_color="red",
                annotation_text="Quarantine threshold")
fig3.update_layout(
    xaxis_title="Packet Index (time)", yaxis_title="Trust Score",
    height=400, hovermode="x unified"
)
st.plotly_chart(fig3, use_container_width=True)

st.subheader("📊 Per-Node Security Summary")

summary_rows = []
for node in zt_df["node_id"].unique():
    node_df = zt_df[zt_df["node_id"] == node]
    tampered_n = node_df["tampered_in_transit"].sum()
    caught_n = (node_df["tampered_in_transit"] & (~node_df["sig_valid"])).sum()
    quarantine_pct = 100 * node_df["quarantined_at_time_of_packet"].sum() / len(node_df)

    summary_rows.append({
        "Node": node,
        "Packets Handled": len(node_df),
        "Tamper Attempts": int(tampered_n),
        "Caught": int(caught_n),
        "Detection Rate": f"{100*caught_n/tampered_n:.0f}%" if tampered_n else "N/A",
        "Final Trust Score": node_df["trust_score"].iloc[-1],
        "Time Quarantined": f"{quarantine_pct:.0f}%"
    })

st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

st.divider()
st.caption("LEO Telemetry Guard — AI-Driven Zero-Trust Anomaly Detection for LEO Satellite Networks")
