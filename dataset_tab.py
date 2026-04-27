"""
dataset_tab.py — Admin-only dataset panel for SolarInspect AI.
Shows only real scans from users. No synthetic data, no line chart.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os

DATASET_PATH = os.path.join("data", "solar_data.csv")

@st.cache_data(show_spinner=False, ttl=30)
def load_solar_data():
    if not os.path.exists(DATASET_PATH):
        return None
    df = pd.read_csv(DATASET_PATH)
    if "source"     not in df.columns: df["source"]     = "synthetic"
    if "confidence" not in df.columns: df["confidence"] = None
    if "severity"   not in df.columns: df["severity"]   = None
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    return df


def render_dataset_tab(TXT, TXT_M, TXT_S, BG_CARD, BORDER, BAR_BG, IS_AR, DM):

    def t(en, ar):
        return ar if IS_AR else en

    def section(label):
        st.markdown(
            f'<div style="font-family:Space Mono,monospace;font-size:0.72rem;'
            f'color:{TXT_M};letter-spacing:3px;margin:22px 0 12px;text-transform:uppercase;">'
            f'{label}</div>',
            unsafe_allow_html=True,
        )

    def kpi(col, label, value, color):
        with col:
            st.markdown(
                f'<div style="background:{BG_CARD};border:1px solid {BORDER};border-radius:12px;'
                f'padding:16px;text-align:center;">'
                f'<div style="font-family:Space Mono,monospace;font-size:0.68rem;color:{TXT_M};'
                f'letter-spacing:2px;margin-bottom:6px;">{label}</div>'
                f'<div style="font-size:1.7rem;font-weight:800;color:{color};">{value}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    df = load_solar_data()

    # ── No data yet
    if df is None or df.empty:
        st.markdown(
            f'<div style="text-align:center;color:{TXT_M};padding:48px 0;font-size:0.95rem;">'
            f'📂 No scans recorded yet. Scans will appear here after users upload images.'
            f'</div>',
            unsafe_allow_html=True
        )
        return

    # ── Only real scans
    real = df[df["source"] == "scan"].copy()

    if real.empty:
        st.markdown(
            f'<div style="text-align:center;color:{TXT_M};padding:48px 0;font-size:0.95rem;">'
            f'📂 No real scans yet. Scans will appear here after users upload images.'
            f'</div>',
            unsafe_allow_html=True
        )
        return

    # ── KPIs
    section("REAL SCANS OVERVIEW")
    total        = len(real)
    unique_users = real["panel_id"].nunique()
    top_defect   = real["defect_type"].value_counts().index[0]
    critical_n   = len(real[real["severity"] == "critical"]) if "severity" in real.columns else 0

    k1, k2, k3, k4 = st.columns(4)
    kpi(k1, "TOTAL SCANS",   str(total),        "#f5a623")
    kpi(k2, "UNIQUE USERS",  str(unique_users),  "#2ecc71")
    kpi(k3, "MOST COMMON",   top_defect,         "#3498db")
    kpi(k4, "CRITICAL",      str(critical_n),    "#e74c3c")

    # ── Defect bar chart
    section("DEFECT BREAKDOWN")
    counts = real["defect_type"].value_counts()
    COLORS = {
        "Clean":"#2ecc71","Dusty":"#3498db","Bird-drop":"#f5a623",
        "Electrical-damage":"#e74c3c","Physical-damage":"#c0392b","Snow-covered":"#9b59b6",
    }
    fig = go.Figure(go.Bar(
        x=counts.index.tolist(),
        y=counts.values.tolist(),
        marker_color=[COLORS.get(d,"#aaa") for d in counts.index],
        text=counts.values.tolist(),
        textposition="outside",
        textfont=dict(color=TXT_M, size=11),
    ))
    fig.update_layout(
        height=240, paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
        font=dict(color=TXT_M),
        xaxis=dict(gridcolor=BORDER),
        yaxis=dict(gridcolor=BORDER),
        margin=dict(l=20, r=20, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True, key="dst_bar")

    # ── Filter by user
    section("SCAN RECORDS")
    users  = ["All Users"] + sorted(real["panel_id"].dropna().unique().tolist())
    sel_u  = st.selectbox(t("Filter by user","تصفية حسب المستخدم"), users, key="dst_user")
    freal  = real if sel_u == "All Users" else real[real["panel_id"] == sel_u]

    # ── Show columns that exist and have data
    show = [c for c in ["timestamp","panel_id","defect_type","confidence","severity",
                         "irradiation","ambient_temp_c","module_temp_c","ac_power_kw"]
            if c in freal.columns]
    disp = freal[show].rename(columns={"panel_id":"user_email"}).sort_values("timestamp", ascending=False)
    st.dataframe(disp, use_container_width=True, hide_index=True)
    st.caption(t(f"{len(disp)} records", f"{len(disp)} سجل"))

    # ── Download
    csv = disp.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=t("⬇️ Download CSV","⬇️ تحميل CSV"),
        data=csv,
        file_name=f"real_scans_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        key="dst_dl"
    )
