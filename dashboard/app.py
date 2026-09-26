"""
Step 9 - Streamlit + Plotly doctor dashboard (Member 3's surface).

Run with the API already up:

    streamlit run dashboard/app.py

This is the "official" stack the team chose. The FastAPI-served web app at /
is a functionally equivalent fallback that needs no Streamlit install - both read
the same /api/* endpoints and the same SQLite file, so use whichever is handy.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

API = os.environ.get("VITALGUARD_API", "http://localhost:8000")
st.set_page_config(page_title="VitalGuard AI", page_icon="🏥", layout="wide")

C_TIER = {"critical": "#ff4d5e", "high": "#ff9f2e", "moderate": "#ffd83d", "low": "#5ad1a5", "minimal": "#4aa3ff"}


@st.cache_data(ttl=6)
def get(path: str):
    try:
        r = requests.get(f"{API}{path}", timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"API unreachable at {API} — is `python -m vitalguard.server` running?\n\n`{exc}`")
        return None


@st.cache_data(ttl=6)
def post(path: str, body: dict):
    try:
        r = requests.post(f"{API}{path}", json=body, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as exc:
        st.error(f"{path} failed: {exc}")
        return None


# ---------------------------------------------------------------- header ---
snap = get("/api/snapshot")
left, right = st.columns([3, 1])
with left:
    st.title("🏥 VitalGuard AI")
    st.caption("Detect deterioration before it becomes critical — camera PPG → heart rate → risk score → explanation → alert")
with right:
    if st.button("↻ refresh now"):
        st.cache_data.clear()
        snap = get("/api/snapshot")

if not snap:
    st.stop()

s = snap["stats"]
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("patients", s["patients"])
c2.metric("readings stored", s["readings"])
c3.metric("open alerts", s["open_alerts"], delta=None if not s["open_alerts"] else "needs review", delta_color="inverse")
c4.metric("mean risk", "—" if s["mean_risk"] is None else f"{s['mean_risk'] * 100:.1f}%")
c5.metric("mean SpO₂", "—" if s["mean_spo2"] is None else f"{s['mean_spo2']:.1f}%")
st.caption(f"SQLite: `{Path(s['db_path']).name}` · API `{API}` · {datetime.now():%H:%M:%S}")

if s["patients"] == 0:
    st.info("Database is empty.")
    if st.button("Load demo ward (14 patients)"):
        post("/api/seed", {})
        st.cache_data.clear()
        st.rerun()
    st.stop()

# ---------------------------------------------------------------- patients -
df = pd.DataFrame(snap["patients"])
df["risk_pct"] = (pd.to_numeric(df["risk_score"], errors="coerce") * 100).round(1)
cols = ["name", "mrn", "ward", "hr", "spo2", "rr", "sbp", "dbp", "temp_c", "news2",
        "risk_pct", "tier", "open_alerts", "signal_quality"]
cols = [c for c in cols if c in df.columns]
names = {"name": "patient", "mrn": "MRN", "ward": "ward", "hr": "HR", "spo2": "SpO₂",
         "rr": "RR", "sbp": "SBP", "dbp": "DBP", "temp_c": "temp", "news2": "NEWS2",
         "risk_pct": "risk %", "tier": "tier", "open_alerts": "alerts", "signal_quality": "PPG q"}

# st.dataframe serialises through Arrow, which refuses a numeric column that also
# contains "—". So: format numerics as strings first, then label the gaps.
show = df[cols].copy()
for c in cols:
    if pd.api.types.is_numeric_dtype(show[c]):
        show[c] = show[c].map(lambda v: "" if pd.isna(v) else (f"{v:g}" if isinstance(v, float) else str(v)))
    else:
        show[c] = show[c].astype(object).map(lambda v: "" if (v is None or (isinstance(v, float) and pd.isna(v))) else str(v))
show = show.replace({"": "—"}).fillna("—")
show.columns = [names[c] for c in cols]


def style(row):
    tier = row.get("tier", "minimal")
    return [f"background-color:{C_TIER.get(tier, '#4aa3ff')}22"] * len(row)


st.subheader("Ward board")
# Styler.map is element-wise in pandas >=2.1; it takes func + subset only,
# so do NOT forward axis= (it reaches the lambda as a kwarg and raises).
def _tier_style(v):
    return f"color:{C_TIER[v]};font-weight:600" if v in C_TIER else ""


show = show.style.map(_tier_style)
st.dataframe(show, width='stretch', hide_index=True)

sel = st.selectbox("Patient", ["—"] + list(df["name"]))
if sel == "—":
    st.stop()
pid = int(df.loc[df["name"] == sel, "id"].iloc[0])
hist = get(f"/api/patients/{pid}/history?limit=40") or []
pat = df.loc[df["id"] == pid].iloc[0]

# ---------------------------------------------------------------- trends ---
st.subheader(sel)
a, b = st.columns([2, 1])
if hist:
    h = pd.DataFrame(hist)[["ts", "hr", "spo2", "rr", "sbp", "dbp", "temp_c", "risk_score", "news2", "signal_quality"]].iloc[::-1]
    h["ts"] = pd.to_datetime(h["ts"])
    with a:
        fig = go.Figure()
        # a plain go.Figure has no secondary_y kwarg on traces: the right axis is a
        # second y-axis declared in the layout and selected per-trace with yaxis="y2"
        fig.add_trace(go.Scatter(x=h.ts, y=h.hr, name="HR (PPG)", yaxis="y",
                                 line=dict(color="#ff4d5e", width=2)))
        fig.add_trace(go.Scatter(x=h.ts, y=h.rr, name="Resp /min", yaxis="y",
                                 line=dict(color="#ffd83d", width=2, dash="dot")))
        fig.add_trace(go.Scatter(x=h.ts, y=h.spo2, name="SpO₂ %", yaxis="y2",
                                 line=dict(color="#4aa3ff", width=2)))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=28, b=0),
                          legend=dict(orientation="h", y=1.12),
                          paper_bgcolor="#0d1524", plot_bgcolor="#0a1424", font_color="#cfe0ff",
                          yaxis=dict(title_text="BPM  /  breaths per min", range=[30, 190],
                                     title_font_color="#ff8b96", gridcolor="#16233a", zeroline=False),
                          yaxis2=dict(title_text="SpO₂ %", overlaying="y", side="right",
                                      range=[80, 100], title_font_color="#8cc4ff",
                                      showgrid=False, zeroline=False))
        st.plotly_chart(fig, width='stretch')

        fig2 = go.Figure()
        fig2.add_trace(go.Bar(x=h.ts, y=h.risk_score * 100, name="deterioration risk %",
                              marker_color=[C_TIER.get(t, "#4aa3ff") for t in h["tier"]] if "tier" in h else "#4aa3ff"))
        fig2.add_trace(go.Scatter(x=h.ts, y=h.news2.astype(float), name="NEWS2 points",
                                 line=dict(color="#e6efff", width=1, shape="spline")))
        fig2.add_hline(y=7, line_dash="dash", line_color="#ff9f2e", annotation_text="NEWS2 escalation ≥7", annotation_font_color="#ff9f2e")
        fig2.update_layout(height=250, margin=dict(l=0, r=0, t=28, b=0), barmode="group",
                           paper_bgcolor="#0d1524", plot_bgcolor="#0a1424", font_color="#cfe0ff",
                           legend=dict(orientation="h", y=1.14))
        st.plotly_chart(fig2, width='stretch')

    with b:
        last = hist[0]
        ex = last.get("explanation") or {}
        if isinstance(ex, str):
            ex = json.loads(ex or "{}")
        tier = ex.get("tier", "minimal")
        st.markdown(f"### {ex.get('risk_percent', 0)}%\n**{tier.upper()}**")
        st.progress(min(1.0, (last["risk_score"] or 0)))
        st.metric("NEWS2 (reference)", ex.get("news2", "—"), delta="escalation line is 5-7", delta_color="off")
        st.metric("PPG signal quality", last.get("signal_quality", "—"), help="step 5 gate; below 60 the reading is not scored")
        st.metric("HR from camera", f"{last.get('hr')} BPM")
        ad = ex.get("advice") or {}
        st.info(f"**{ad.get('action', '')}**\n\n{ad.get('detail', '')}")
        if ad.get("conflict"):
            st.warning(ad["conflict"])

        st.markdown("**Why — SHAP (log-odds)**")
        contribs = ex.get("top_factors") or []
        if contribs:
            cf = pd.DataFrame([{"factor": f"{c['label']} = {c['value_text']}", "shap": c["shap"]} for c in contribs]).sort_values("shap")
            fig3 = go.Figure(go.Bar(x=cf.shap, y=cf.factor, orientation="h",
                                    marker_color=["#ff4d5e" if v > 0 else "#5ad1a5" for v in cf.shap]))
            fig3.update_layout(height=40 + 26 * len(cf), margin=dict(l=0, r=0, t=6, b=20),
                               paper_bgcolor="#0d1524", plot_bgcolor="#0a1424", font_color="#cfe0ff",
                               xaxis_title="contribution to log-odds", xaxis_zeroline=True, xaxis_zerolinecolor="#3d5a8a")
            st.plotly_chart(fig3, width='stretch')
            st.caption(ex.get("shap_method", ""))
        for f in ex.get("red_flags") or []:
            st.error(f["text"])

    # ------------------------------------------------- clinician actions ---
    st.divider()
    t1, t2, t3 = st.columns([1, 1, 2])
    with t1:
        st.number_input("SpO₂ %", 50.0, 100.0, float(last.get("spo2") or 97), key="spo2i")
        st.number_input("Systolic", 40.0, 260.0, float(last.get("sbp") or 120), key="sbpi")
    with t2:
        st.number_input("Resp /min", 4.0, 60.0, float(last.get("rr") or 16), key="rri")
        st.number_input("Temp °C", 30.0, 43.0, float(last.get("temp_c") or 36.8), step=0.1, key="ti")
    with t3:
        st.markdown("**Round**")
        st.write(f"{len(hist)} readings · last at {str(last['ts'])[:19]}")
        if st.button("Re-score with these vitals", type="primary"):
            r = post(f"/api/patients/{pid}/score", {"spo2": st.session_state.spo2i, "rr": st.session_state.rri,
                                                    "sbp": st.session_state.sbpi, "dbp": last.get("dbp"),
                                                    "temp_c": st.session_state.ti, "source": "dashboard"})
            st.cache_data.clear()
            if r:
                st.success(f"risk {r['risk']['risk_percent']}% · {r['risk']['tier']} · reading #{r.get('saved_reading_id')}")

    with st.expander("reading log"):
        st.dataframe(pd.DataFrame(hist)[["ts", "source", "hr", "spo2", "rr", "sbp", "temp_c", "signal_quality", "risk_score", "tier"]],
                     width='stretch', hide_index=True)

# ---------------------------------------------------------------- alerts ---
st.divider()
st.subheader("Alert queue")
al = pd.DataFrame(get("/api/alerts?limit=60") or [])
if al.empty:
    st.success("No alerts.")
else:
    al["ts"] = al["ts"].str.slice(5, 19).str.replace("T", " ")
    st.dataframe(al[["ts", "name", "mrn", "ward", "tier", "message", "acknowledged_at"]], width='stretch', hide_index=True)
    aid = st.selectbox("acknowledge alert id", [0] + list(al["id"]), format_func=lambda v: "—" if v == 0 else f"#{v}")
    if aid:
        if st.button("Mark acknowledged"):
            post(f"/api/alerts/{aid}/ack", {})
            st.cache_data.clear()
            st.rerun()
