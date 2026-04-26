"""
dataset_tab.py
──────────────
Standalone dataset module for SolarInspect AI.
Import and call render_dataset_tab() inside a `with tab5:` block in app.py.

Dataset columns (data/solar_data.csv):
  timestamp        — datetime string "YYYY-MM-DD HH:MM"
  date             — date string "YYYY-MM-DD"
  hour             — integer 6-19
  panel_id         — string "P-001" … "P-005"
  irradiation      — float 0-1.1 (W/m²/1000)
  ambient_temp_c   — float (°C)
  module_temp_c    — float (°C)
  dc_power_kw      — float (kW)
  ac_power_kw      — float (kW)
  defect_type      — string (one of 6 defect classes)
  efficiency_pct   — float 0-100
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import os
import sqlite3

DATASET_PATH = os.path.join("data", "solar_data.csv")

@st.cache_data(show_spinner=False, ttl=30)
# ttl=30 means cache refreshes every 30 seconds so new scans appear quickly
def load_solar_data() -> pd.DataFrame | None:
    """
    Load the dataset CSV.
    The file contains both synthetic rows (source='synthetic' or no source col)
    and real scan rows (source='scan') appended by db_save_scan().
    Returns None if file doesn't exist yet.
    """
    if not os.path.exists(DATASET_PATH):
        return None
    df = pd.read_csv(DATASET_PATH)
    # Ensure source column exists
    if "source" not in df.columns:
        df["source"] = "synthetic"
    # Ensure confidence column exists
    if "confidence" not in df.columns:
        df["confidence"] = None
    if "severity" not in df.columns:
        df["severity"] = None
    # Parse dates safely
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"]      = pd.to_datetime(df["date"],      errors="coerce")
    return df


def render_dataset_tab(TXT, TXT_M, TXT_S, BG_CARD, BORDER, BAR_BG, IS_AR, DM):

    def t(en, ar):
        return ar if IS_AR else en

    def section(label):
        st.markdown(
            f'<div style="font-family:Space Mono,monospace;font-size:0.72rem;'
            f'color:{TXT_M};letter-spacing:3px;margin:26px 0 14px;'
            f'text-transform:uppercase;">{label}</div>',
            unsafe_allow_html=True,
        )

    df = load_solar_data()

    if df is None:
        st.info(t(
            "No dataset yet. The dataset will be created automatically once "
            "users start scanning solar panels.",
            "لا توجد بيانات بعد. ستُنشأ تلقائياً عند بدء المستخدمين الفحص."
        ))
        return

    # ── Split real scans vs synthetic data
    real_scans = df[df["source"] == "scan"].copy()
    synthetic  = df[df["source"] != "scan"].copy()

    # ── REAL SCANS SUMMARY — shown first, most important
    section("REAL SCANS FROM USERS")

    if real_scans.empty:
        st.markdown(
            f'<div style="color:{TXT_M};font-size:0.9rem;padding:12px 0;">'
            f'No real scans yet — scans will appear here automatically.</div>',
            unsafe_allow_html=True
        )
    else:
        total_scans  = len(real_scans)
        unique_users = real_scans["panel_id"].nunique()
        defect_counts = real_scans["defect_type"].value_counts()

        k1, k2, k3 = st.columns(3)
        for col, lbl, val, color in [
            (k1, "TOTAL SCANS",   str(total_scans),  "#f5a623"),
            (k2, "UNIQUE USERS",  str(unique_users),  "#2ecc71"),
            (k3, "MOST COMMON",   defect_counts.index[0] if not defect_counts.empty else "—", "#3498db"),
        ]:
            with col:
                st.markdown(
                    f'<div style="background:{BG_CARD};border:1px solid {BORDER};'
                    f'border-radius:12px;padding:16px;text-align:center;">'
                    f'<div style="font-family:Space Mono,monospace;font-size:0.7rem;'
                    f'color:{TXT_M};letter-spacing:2px;margin-bottom:6px;">{lbl}</div>'
                    f'<div style="font-size:1.6rem;font-weight:800;color:{color};">{val}</div>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # ── Defect breakdown from real scans
        st.markdown("<br>", unsafe_allow_html=True)
        fig_bar = go.Figure(go.Bar(
            x=defect_counts.index.tolist(),
            y=defect_counts.values.tolist(),
            marker_color="#f5a623",
            text=defect_counts.values.tolist(),
            textposition="outside",
            textfont=dict(color=TXT_M, size=11),
        ))
        fig_bar.update_layout(
            title=dict(text="Real Scan Results by Defect Type", font=dict(color=TXT, size=14)),
            height=260, paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
            font=dict(color=TXT_M),
            xaxis=dict(gridcolor=BORDER),
            yaxis=dict(gridcolor=BORDER),
            margin=dict(l=30, r=20, t=40, b=30),
        )
        st.plotly_chart(fig_bar, use_container_width=True, key="dst_fig_bar")

        # ── Real scans table
        section("REAL SCAN RECORDS")
        show_cols = ["timestamp", "panel_id", "defect_type", "confidence", "severity"]
        available = [c for c in show_cols if c in real_scans.columns]
        real_display = real_scans[available].rename(columns={"panel_id": "user_email"})
        real_display = real_display.sort_values("timestamp", ascending=False)
        st.dataframe(real_display, use_container_width=True, hide_index=True)

        # ── Download real scans
        csv_real = real_display.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=t("⬇️ Download Real Scans CSV", "⬇️ تحميل بيانات الفحوصات"),
            data=csv_real,
            file_name=f"real_scans_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="dl_real"
        )

    # ── SYNTHETIC / FULL DATASET section
    section("FULL DATASET (Synthetic + Real)")

    # Filters
    f1, f2, f3 = st.columns(3)
    with f1:
        panels  = ["All"] + sorted(df["panel_id"].dropna().unique().tolist())
        sel_panel = st.selectbox(t("User / Panel", "المستخدم"), panels, key="dst_panel")
    with f2:
        defects = ["All"] + sorted(df["defect_type"].dropna().unique().tolist())
        sel_defect = st.selectbox(t("Defect Type", "نوع العيب"), defects, key="dst_defect")
    with f3:
        sources = ["All", "Real Scans", "Synthetic"]
        sel_source = st.selectbox(t("Source", "المصدر"), sources, key="dst_source")

    fdf = df.copy()
    if sel_panel  != "All":    fdf = fdf[fdf["panel_id"]   == sel_panel]
    if sel_defect != "All":    fdf = fdf[fdf["defect_type"] == sel_defect]
    if sel_source == "Real Scans": fdf = fdf[fdf["source"] == "scan"]
    if sel_source == "Synthetic":  fdf = fdf[fdf["source"] != "scan"]

    # KPI cards
    section("KEY METRICS")
    k1, k2, k3, k4 = st.columns(4)
    ac_col = "ac_power_kw"
    for col, lbl, val, color in [
        (k1, "TOTAL ROWS",    f"{len(fdf):,}",                                        "#f5a623"),
        (k2, "REAL SCANS",    str(len(fdf[fdf["source"]=="scan"])),                   "#2ecc71"),
        (k3, "SYNTHETIC",     str(len(fdf[fdf["source"]!="scan"])),                   "#3498db"),
        (k4, "DEFECT TYPES",  str(fdf["defect_type"].nunique()),                      "#e74c3c"),
    ]:
        with col:
            st.markdown(
                f'<div style="background:{BG_CARD};border:1px solid {BORDER};'
                f'border-radius:12px;padding:16px;text-align:center;">'
                f'<div style="font-family:Space Mono,monospace;font-size:0.7rem;'
                f'color:{TXT_M};letter-spacing:2px;margin-bottom:6px;">{lbl}</div>'
                f'<div style="font-size:1.6rem;font-weight:800;color:{color};">{val}</div>'
                f'</div>',
                unsafe_allow_html=True
            )

    # Defect distribution
    section("DEFECT DISTRIBUTION")
    defect_counts_all = fdf["defect_type"].value_counts().reset_index()
    defect_counts_all.columns = ["defect", "count"]
    COLORS = {
        "Clean":"#2ecc71","Dusty":"#3498db","Bird-drop":"#f5a623",
        "Electrical-damage":"#e74c3c","Physical-damage":"#c0392b","Snow-covered":"#9b59b6",
    }
    fig_pie = go.Figure(go.Pie(
        labels=defect_counts_all["defect"],
        values=defect_counts_all["count"],
        marker_colors=[COLORS.get(d, "#aaa") for d in defect_counts_all["defect"]],
        hole=0.45,
        textinfo="percent+label",
        textfont=dict(color=TXT, size=11),
    ))
    fig_pie.update_layout(
        height=280, paper_bgcolor=BG_CARD,
        font=dict(color=TXT_M), showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10),
    )
    st.plotly_chart(fig_pie, use_container_width=True, key="dst_fig_pie")

    # Paginated raw table
    section("RAW DATA TABLE")
    disp_cols = [c for c in ["timestamp","panel_id","defect_type","source","confidence","severity","ac_power_kw"] if c in fdf.columns]
    display_df = fdf[disp_cols].sort_values("timestamp", ascending=False)
    rows_per_page = 20
    total_pages   = max(1, (len(display_df) - 1) // rows_per_page + 1)
    page = st.number_input(
        t(f"Page (1–{total_pages})", f"الصفحة"), min_value=1,
        max_value=total_pages, value=1, step=1, key="dst_page"
    )
    start = (page - 1) * rows_per_page
    st.dataframe(display_df.iloc[start:start + rows_per_page], use_container_width=True, hide_index=True)
    st.caption(t(
        f"Rows {start+1}–{min(start+rows_per_page, len(display_df))} of {len(display_df):,}",
        f"صفوف {start+1}–{min(start+rows_per_page, len(display_df))} من {len(display_df):,}"
    ))

    # Download filtered
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=t("⬇️ Download Filtered CSV", "⬇️ تحميل البيانات"),
        data=csv_bytes,
        file_name=f"solar_dataset_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        key="dl_full"
    )
    """
    Render the full Dataset & Insights tab.
    All colour variables come from the caller (app.py) so the tab respects
    the current dark/light theme automatically.
    """

    def t(en, ar):
        return ar if IS_AR else en

    # ── Section title helper
    def section(label_en, label_ar=""):
        label = label_ar if IS_AR and label_ar else label_en
        st.markdown(
            f'<div style="font-family:Space Mono,monospace;font-size:0.72rem;'
            f'color:{TXT_M};letter-spacing:3px;margin:26px 0 14px;'
            f'text-transform:uppercase;">{label}</div>',
            unsafe_allow_html=True,
        )

    # ────────────────────────────────────────────────────────────────
    # Load data
    # ────────────────────────────────────────────────────────────────
    df = load_solar_data()

    if df is None:
        st.info(
            t(
                "Dataset not found. Run `python generate_sample_data.py` "
                "to create data/solar_data.csv, then restart the app.",
                "الملف غير موجود. شغّل generate_sample_data.py أولاً.",
            )
        )
        return

    # ────────────────────────────────────────────────────────────────
    # FILTERS sidebar-style row
    # ────────────────────────────────────────────────────────────────
    section("DATASET FILTERS", "فلاتر البيانات")
    f1, f2, f3 = st.columns(3)

    with f1:
        panels = ["All"] + sorted(df["panel_id"].unique().tolist())
        sel_panel = st.selectbox(
            t("Panel ID", "معرّف اللوح"), panels, key="ds_panel"
        )
    with f2:
        defects = ["All"] + sorted(df["defect_type"].unique().tolist())
        sel_defect = st.selectbox(
            t("Defect Type", "نوع العيب"), defects, key="ds_defect"
        )
    with f3:
        date_min = df["date"].min().date()
        date_max = df["date"].max().date()
        sel_dates = st.date_input(
            t("Date Range", "نطاق التاريخ"),
            value=(date_min, date_max),
            min_value=date_min,
            max_value=date_max,
            key="ds_dates",
        )

    # Apply filters
    fdf = df.copy()
    if sel_panel != "All":
        fdf = fdf[fdf["panel_id"] == sel_panel]
    if sel_defect != "All":
        fdf = fdf[fdf["defect_type"] == sel_defect]
    if len(sel_dates) == 2:
        d0 = pd.Timestamp(sel_dates[0])
        d1 = pd.Timestamp(sel_dates[1])
        fdf = fdf[(fdf["date"] >= d0) & (fdf["date"] <= d1)]

    # ────────────────────────────────────────────────────────────────
    # KPI CARDS
    # ────────────────────────────────────────────────────────────────
    section("KEY METRICS", "المؤشرات الرئيسية")

    total_energy  = fdf["ac_power_kw"].sum() * (1 / 4)   # 15-min intervals → kWh
    avg_eff       = fdf["efficiency_pct"].mean()
    peak_power    = fdf["ac_power_kw"].max()
    n_critical    = fdf[fdf["defect_type"].isin(
        ["Electrical-damage", "Physical-damage"])].shape[0]

    k1, k2, k3, k4 = st.columns(4)
    for col, lbl_en, lbl_ar, val, unit, color in [
        (k1, "TOTAL ENERGY",   "إجمالي الطاقة",  f"{total_energy:,.0f}", "kWh", "#f5a623"),
        (k2, "AVG EFFICIENCY", "متوسط الكفاءة",  f"{avg_eff:.1f}",       "%",   "#2ecc71"),
        (k3, "PEAK POWER",     "ذروة الطاقة",     f"{peak_power:.2f}",   "kW",  "#3498db"),
        (k4, "CRITICAL ROWS",  "صفوف حرجة",       str(n_critical),       "rows","#e74c3c"),
    ]:
        with col:
            lbl = lbl_ar if IS_AR else lbl_en
            st.markdown(
                f'<div style="background:{BG_CARD};border:1px solid {BORDER};'
                f'border-radius:12px;padding:18px 22px;margin-bottom:12px;text-align:center;">'
                f'<div style="font-family:Space Mono,monospace;font-size:0.7rem;'
                f'color:{TXT_M};letter-spacing:2px;margin-bottom:6px;">{lbl}</div>'
                f'<div style="font-size:1.9rem;font-weight:800;color:{color};line-height:1;">'
                f'{val}</div>'
                f'<div style="font-size:0.78rem;color:{TXT_M};margin-top:4px;">{unit}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ────────────────────────────────────────────────────────────────
    # CHART 1 — Daily AC power over time
    # ────────────────────────────────────────────────────────────────
    section("DAILY ENERGY OUTPUT", "الطاقة اليومية")

    daily = fdf.groupby("date")["ac_power_kw"].sum().reset_index()
    daily.columns = ["date", "total_ac_kwh"]

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(
        x=daily["date"], y=daily["total_ac_kwh"],
        mode="lines", fill="tozeroy",
        line=dict(color="#f5a623", width=2),
        fillcolor="rgba(245,166,35,0.12)",
        name=t("Daily Energy (kWh)", "الطاقة اليومية"),
    ))
    fig1.update_layout(
        height=280, paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
        font=dict(color=TXT_M),
        xaxis=dict(gridcolor=BORDER, title=t("Date", "التاريخ")),
        yaxis=dict(gridcolor=BORDER, title="kWh"),
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig1, use_container_width=True, key="dst_fig1")

    # ────────────────────────────────────────────────────────────────
    # CHART 2 — Defect distribution pie
    # ────────────────────────────────────────────────────────────────
    col_pie, col_bar = st.columns(2)

    with col_pie:
        section("DEFECT DISTRIBUTION", "توزيع العيوب")
        defect_counts = fdf["defect_type"].value_counts().reset_index()
        defect_counts.columns = ["defect", "count"]

        COLORS = {
            "Clean":              "#2ecc71",
            "Dusty":              "#3498db",
            "Bird-drop":          "#f5a623",
            "Electrical-damage":  "#e74c3c",
            "Physical-damage":    "#c0392b",
            "Snow-covered":       "#9b59b6",
        }
        fig2 = go.Figure(go.Pie(
            labels=defect_counts["defect"],
            values=defect_counts["count"],
            marker_colors=[COLORS.get(d, "#aaa") for d in defect_counts["defect"]],
            hole=0.45,
            textinfo="percent+label",
            textfont=dict(color=TXT, size=11),
        ))
        fig2.update_layout(
            height=280, paper_bgcolor=BG_CARD,
            font=dict(color=TXT_M),
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig2, use_container_width=True, key="dst_fig2")

    with col_bar:
        section("AVG POWER BY PANEL", "متوسط الطاقة لكل لوح")
        panel_power = fdf.groupby("panel_id")["ac_power_kw"].mean().reset_index()

        fig3 = go.Figure(go.Bar(
            x=panel_power["panel_id"],
            y=panel_power["ac_power_kw"],
            marker_color="#f5a623",
            text=[f"{v:.3f} kW" for v in panel_power["ac_power_kw"]],
            textposition="outside",
            textfont=dict(color=TXT_M, size=10),
        ))
        fig3.update_layout(
            height=280, paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
            font=dict(color=TXT_M),
            xaxis=dict(gridcolor=BORDER),
            yaxis=dict(gridcolor=BORDER, title="kW"),
            margin=dict(l=40, r=20, t=20, b=40),
        )
        st.plotly_chart(fig3, use_container_width=True, key="dst_fig3")

    # ────────────────────────────────────────────────────────────────
    # CHART 3 — Irradiation vs AC Power scatter (prediction insight)
    # ────────────────────────────────────────────────────────────────
    section("IRRADIATION vs POWER (Prediction Insight)", "الإشعاع مقابل الطاقة")

    sample = fdf.sample(min(800, len(fdf)), random_state=1)   # cap for performance
    fig4 = go.Figure(go.Scatter(
        x=sample["irradiation"],
        y=sample["ac_power_kw"],
        mode="markers",
        marker=dict(
            color=sample["efficiency_pct"],
            colorscale="RdYlGn",
            size=5,
            opacity=0.7,
            colorbar=dict(title="Efficiency %", tickfont=dict(color=TXT_M)),
        ),
        text=sample["defect_type"],
        hovertemplate=(
            "Irradiation: %{x:.3f}<br>"
            "AC Power: %{y:.3f} kW<br>"
            "Defect: %{text}<extra></extra>"
        ),
    ))
    fig4.update_layout(
        height=320, paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
        font=dict(color=TXT_M),
        xaxis=dict(gridcolor=BORDER, title=t("Irradiation (W/m²/1000)", "الإشعاع")),
        yaxis=dict(gridcolor=BORDER, title="AC Power (kW)"),
        margin=dict(l=40, r=60, t=20, b=40),
    )
    st.plotly_chart(fig4, use_container_width=True, key="dst_fig4")
    st.caption(
        t(
            "Colour = efficiency %. Green = high efficiency, Red = low. "
            "Hover over a point to see the defect type.",
            "اللون = نسبة الكفاءة. أخضر = عالي، أحمر = منخفض.",
        )
    )

    # ────────────────────────────────────────────────────────────────
    # DATA TABLE — paginated, searchable
    # ────────────────────────────────────────────────────────────────
    section("RAW DATA TABLE", "جدول البيانات الخام")

    show_cols = [
        "timestamp", "panel_id", "defect_type",
        "irradiation", "ac_power_kw", "efficiency_pct",
    ]
    display_df = fdf[show_cols].sort_values("timestamp", ascending=False)

    rows_per_page = 20
    total_pages   = max(1, (len(display_df) - 1) // rows_per_page + 1)
    page = st.number_input(
        t(f"Page (1–{total_pages})", f"الصفحة (1-{total_pages})"),
        min_value=1, max_value=total_pages, value=1, step=1, key="ds_page"
    )
    start = (page - 1) * rows_per_page
    st.dataframe(
        display_df.iloc[start : start + rows_per_page],
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        t(
            f"Showing rows {start+1}–{min(start+rows_per_page, len(display_df))} "
            f"of {len(display_df):,} filtered rows.",
            f"عرض الصفوف {start+1}–{min(start+rows_per_page, len(display_df))} "
            f"من {len(display_df):,} صفاً.",
        )
    )

    # ── CSV download of filtered data
    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label=t("⬇️ Download Filtered Data (CSV)", "⬇️ تحميل البيانات المفلترة"),
        data=csv_bytes,
        file_name=f"solar_filtered_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
    )

    # ────────────────────────────────────────────────────────────────
    # QUICK PREDICT from dataset row
    # ────────────────────────────────────────────────────────────────
    section("PREDICT FROM DATASET ROW", "التنبؤ من صف في البيانات")
    st.markdown(
        f'<div style="font-size:0.9rem;color:{TXT_S};margin-bottom:12px;">'
        + t(
            "Pick any row from the dataset and run it through the performance model.",
            "اختر أي صف من البيانات وشغّله في نموذج الأداء.",
        )
        + "</div>",
        unsafe_allow_html=True,
    )

    row_idx = st.slider(
        t("Row index", "رقم الصف"), 0, max(0, len(fdf) - 1), 0, key="ds_row"
    )
    row = fdf.iloc[row_idx]

    r1, r2, r3, r4 = st.columns(4)
    for col, lbl, val in [
        (r1, t("Irradiation", "الإشعاع"),    f"{row['irradiation']:.4f}"),
        (r2, t("Ambient °C",  "حرارة المحيط"), f"{row['ambient_temp_c']:.1f}"),
        (r3, t("Module °C",   "حرارة اللوح"),  f"{row['module_temp_c']:.1f}"),
        (r4, t("AC Power",    "طاقة AC"),      f"{row['ac_power_kw']:.3f} kW"),
    ]:
        with col:
            st.markdown(
                f'<div style="background:{BG_CARD};border:1px solid {BORDER};'
                f'border-radius:8px;padding:12px;text-align:center;">'
                f'<div style="font-size:0.72rem;color:{TXT_M};margin-bottom:4px;">{lbl}</div>'
                f'<div style="font-size:1.1rem;font-weight:700;color:#f5a623;">{val}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown(
        f'<div style="margin-top:10px;padding:10px 16px;background:{BG_CARD};'
        f'border:1px solid {BORDER};border-radius:8px;font-size:0.9rem;color:{TXT_S};">'
        f'<b style="color:#f5a623;">{t("Recorded defect:", "العيب المسجل:")}</b> '
        f'{row["defect_type"]}'
        f'</div>',
        unsafe_allow_html=True,
    )
