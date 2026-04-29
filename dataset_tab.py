"""
dataset_tab.py — Admin-only dataset panel for SolarInspect AI.
- STATIC DATASET (solar_data.csv in Supabase Storage) → analysis only
- SCAN LOG (scans table in PostgreSQL) → written on every user scan
- ADMIN MERGE → admin approves pending scans into static CSV
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import psycopg2
import psycopg2.extras
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────
# SECRETS — read from st.secrets first, fallback to os.environ
# ─────────────────────────────────────────────────────────────────────
def _get_secret(key, default=""):
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)

SUPABASE_URL    = ""   # resolved at call time via _get_secret
SUPABASE_KEY    = ""
STORAGE_BUCKET  = "solar-data"
STORAGE_CSV_KEY = "solar_data.csv"

CSV_COLUMNS = [
    "timestamp", "date", "hour", "panel_id",
    "irradiation", "ambient_temp_c", "module_temp_c",
    "dc_power_kw", "ac_power_kw", "defect_type", "efficiency_pct",
    "confidence", "severity", "source",
    "panel_capacity_kw", "panel_age_years",
]

# ─────────────────────────────────────────────────────────────────────
# DB CONNECTION
# ─────────────────────────────────────────────────────────────────────
def _db():
    conn = psycopg2.connect(
        host    = _get_secret("DB_HOST"),
        port    = int(_get_secret("DB_PORT", "5432")),
        dbname  = _get_secret("DB_NAME", "postgres"),
        user    = _get_secret("DB_USER"),
        password= _get_secret("DB_PASSWORD"),
        sslmode = "require",
    )
    return conn

# ─────────────────────────────────────────────────────────────────────
# SUPABASE STORAGE
# ─────────────────────────────────────────────────────────────────────
def _storage_headers():
    key = _get_secret("SUPABASE_SERVICE_KEY")
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
    }

@st.cache_data(show_spinner=False, ttl=5)
def load_static_dataset() -> pd.DataFrame:
    """Download solar_data.csv from Supabase Storage. Read-only for analysis."""
    url_base = _get_secret("SUPABASE_URL")
    if not url_base:
        return pd.DataFrame(columns=CSV_COLUMNS)
    try:
        import requests
        url = f"{url_base}/storage/v1/object/{STORAGE_BUCKET}/{STORAGE_CSV_KEY}"
        r = requests.get(url, headers=_storage_headers(), timeout=10)
        if r.status_code == 200:
            from io import StringIO
            df = pd.read_csv(StringIO(r.text))
            # Repair any blank/nan/none timestamps in place
            if "timestamp" in df.columns:
                df["timestamp"] = df["timestamp"].apply(
                    lambda v: pd.NaT if pd.isna(v) or str(v).strip().lower() in ("", "nan", "none", "nat", "—")
                    else v
                )
                df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            for col in ["source", "confidence", "severity"]:
                if col not in df.columns:
                    df[col] = None
            return df
        return pd.DataFrame(columns=CSV_COLUMNS)
    except Exception as e:
        st.warning(f"Could not load dataset: {e}")
        return pd.DataFrame(columns=CSV_COLUMNS)

def _save_csv_to_storage(df: pd.DataFrame) -> bool:
    url_base = _get_secret("SUPABASE_URL")
    if not url_base:
        return False
    try:
        import requests
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        url = f"{url_base}/storage/v1/object/{STORAGE_BUCKET}/{STORAGE_CSV_KEY}"
        headers = {**_storage_headers(), "Content-Type": "text/csv", "x-upsert": "true"}
        r = requests.put(url, headers=headers, data=csv_bytes, timeout=20)
        return r.status_code in (200, 201)
    except Exception as e:
        st.error(f"Storage upload failed: {e}")
        return False

# ─────────────────────────────────────────────────────────────────────
# PENDING SCANS — from PostgreSQL scans table
# ─────────────────────────────────────────────────────────────────────
def _get_pending_scans() -> pd.DataFrame:
    conn = _db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, email, scanned_at, defect_type, display_en, display_ar,
                       confidence, severity, icon,
                       irradiation, ambient_temp_c, module_temp_c,
                       dc_power_kw, ac_power_kw, efficiency_pct,
                       panel_capacity_kw, panel_age_years
                FROM scans
                WHERE merged_into_dataset = FALSE
                ORDER BY scanned_at DESC
            """)
            rows = cur.fetchall()
    finally:
        conn.close()
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["scanned_at"] = pd.to_datetime(df["scanned_at"])
    from datetime import timezone, timedelta
    cairo_now = pd.Timestamp.now(tz=None) + pd.Timedelta(hours=2)
    df["scanned_at"] = df["scanned_at"].fillna(cairo_now)
    return df

def _mark_all_merged():
    conn = _db()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE scans SET merged_into_dataset = TRUE WHERE merged_into_dataset = FALSE")
        conn.commit()
    finally:
        conn.close()

def _mark_scan_merged(scan_id: int):
    """Mark a single scan as merged/processed."""
    conn = _db()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE scans SET merged_into_dataset = TRUE WHERE id = %s", (scan_id,))
        conn.commit()
    finally:
        conn.close()

def _merge_pending_into_csv(pending_df: pd.DataFrame, static_df: pd.DataFrame) -> pd.DataFrame:
    # Compute medians from existing real data to fill missing sensor fields
    sensor_cols = ['irradiation','ambient_temp_c','module_temp_c','dc_power_kw','ac_power_kw','efficiency_pct']
    medians = {}
    for col in sensor_cols:
        if col in static_df.columns:
            vals = pd.to_numeric(static_df[col], errors='coerce').dropna()
            medians[col] = round(float(vals.median()), 3) if not vals.empty else 0.5

    new_rows = []
    for _, row in pending_df.iterrows():
        ts = row["scanned_at"]
        # Use the DB timestamp directly — it was saved with Cairo time from app.py
        try:
            if hasattr(ts, "strftime"):
                ts_str  = ts.strftime("%Y-%m-%d %H:%M:%S")
                ts_date = ts.strftime("%Y-%m-%d")
                ts_hour = ts.hour
            else:
                ts_parsed = pd.to_datetime(str(ts), errors="coerce")
                if pd.isna(ts_parsed):
                    from datetime import timedelta
                    ts_parsed = datetime.utcnow() + timedelta(hours=2)
                ts_str  = ts_parsed.strftime("%Y-%m-%d %H:%M:%S")
                ts_date = ts_parsed.strftime("%Y-%m-%d")
                ts_hour = ts_parsed.hour
        except Exception:
            from datetime import timedelta
            now = datetime.utcnow() + timedelta(hours=2)
            ts_str, ts_date, ts_hour = now.strftime("%Y-%m-%d %H:%M:%S"), now.strftime("%Y-%m-%d"), now.hour

        conf = float(row["confidence"])

        def _safe(col):
            v = row.get(col)
            if v is None or (isinstance(v, float) and pd.isna(v)):
                return medians.get(col, "")
            # Treat 0 as "not entered" for sensor fields
            try:
                if float(v) == 0.0 and col in medians:
                    return medians.get(col, "")
            except Exception:
                pass
            return v

        new_rows.append({
            "timestamp":         ts_str,
            "date":              ts_date,
            "hour":              ts_hour,
            "panel_id":          row["email"],
            "irradiation":       _safe("irradiation"),
            "ambient_temp_c":    _safe("ambient_temp_c"),
            "module_temp_c":     _safe("module_temp_c"),
            "dc_power_kw":       _safe("dc_power_kw"),
            "ac_power_kw":       _safe("ac_power_kw"),
            "defect_type":       row["defect_type"],
            "efficiency_pct":    _safe("efficiency_pct"),
            "confidence":        round(conf * 100, 1) if conf <= 1.0 else round(conf, 1),
            "severity":          row["severity"],
            "source":            "scan",
            "panel_capacity_kw": _safe("panel_capacity_kw"),
            "panel_age_years":   _safe("panel_age_years"),
        })
    new_df = pd.DataFrame(new_rows)
    for col in CSV_COLUMNS:
        if col not in static_df.columns:
            static_df[col] = ""
        if col not in new_df.columns:
            new_df[col] = ""
    merged = pd.concat([static_df[CSV_COLUMNS], new_df[CSV_COLUMNS]], ignore_index=True)
    return merged
# ─────────────────────────────────────────────────────────────────────
# MAIN RENDER
# ─────────────────────────────────────────────────────────────────────
def render_dataset_tab(TXT, TXT_M, TXT_S, BG_CARD, BORDER, BAR_BG, IS_AR, DM):

    def t(en, ar):
        return ar if IS_AR else en

    def section(en, ar=""):
        label = ar if IS_AR and ar else en
        st.markdown(
            f'<div style="font-family:Space Mono,monospace;font-size:0.72rem;'
            f'color:{TXT_M};letter-spacing:3px;margin:22px 0 12px;text-transform:uppercase;">'
            f'{label}</div>',
            unsafe_allow_html=True,
        )

    def kpi(col, label_en, label_ar, value, color):
        label = label_ar if IS_AR else label_en
        with col:
            st.markdown(
                f'<div style="background:{BG_CARD};border:1px solid {BORDER};border-radius:12px;'
                f'padding:16px;text-align:center;">'
                f'<div style="font-family:Space Mono,monospace;font-size:0.68rem;color:{TXT_M};'
                f'letter-spacing:2px;margin-bottom:6px;">{label}</div>'
                f'<div style="font-size:1.7rem;font-weight:800;color:{color};">{value}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ═══════════════════════════════════════════════════════
    # SECTION 1 — STATIC DATASET ANALYSIS
    # ═══════════════════════════════════════════════════════
    section("APPROVED DATASET — STATIC ANALYSIS", "البيانات المعتمدة — تحليل ثابت")
    st.markdown(
        f'<div style="background:{BG_CARD};border:1px solid {BORDER};border-radius:10px;'
        f'padding:12px 18px;margin-bottom:16px;font-size:0.85rem;color:{TXT_M};">'
        f'📊 {t("This dataset is","هذه البيانات")} <b style="color:#2ecc71;">{t("static","ثابتة")}</b>. '
        f'{t("Never modified by user scans. Only admin can merge scans into it below.",
             "لا تتغير تلقائياً. فقط المدير يمكنه دمج الفحوصات فيها.")}'
        f'</div>',
        unsafe_allow_html=True,
    )

    df   = load_static_dataset()
    real = df[df["source"] == "scan"].copy() if not df.empty and "source" in df.columns else pd.DataFrame()

    if df.empty or real.empty:
        st.markdown(
            f'<div style="text-align:center;color:{TXT_M};padding:32px 0;font-size:0.95rem;">'
            f'📂 {t("No approved scan data yet. Use the Merge panel below to add scans.","لا توجد بيانات معتمدة بعد. استخدم لوحة الدمج أدناه.")}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        total        = len(real)
        unique_users = real["panel_id"].nunique() if "panel_id" in real.columns else 0
        top_defect   = real["defect_type"].value_counts().index[0] if "defect_type" in real.columns else "—"
        critical_n   = len(real[real["severity"] == "critical"]) if "severity" in real.columns else 0

        k1, k2, k3, k4 = st.columns(4)
        kpi(k1, "TOTAL SCANS",  "إجمالي الفحوصات", str(total),        "#f5a623")
        kpi(k2, "UNIQUE USERS", "مستخدمون فريدون",  str(unique_users), "#2ecc71")
        kpi(k3, "MOST COMMON",  "الأكثر شيوعاً",    top_defect,        "#3498db")
        kpi(k4, "CRITICAL",     "حرج",              str(critical_n),   "#e74c3c")

        section("DEFECT BREAKDOWN", "توزيع العيوب")
        counts = real["defect_type"].value_counts()
        DEFECT_AR = {
            "Clean": "نظيف", "Dusty": "متسخ", "Bird-drop": "إفرازات طيور",
            "Electrical-damage": "تلف كهربائي", "Physical-damage": "تلف مادي", "Snow-covered": "مغطى بثلج",
        }
        COLORS = {
            "Clean": "#2ecc71", "Dusty": "#3498db", "Bird-drop": "#f5a623",
            "Electrical-damage": "#e74c3c", "Physical-damage": "#c0392b", "Snow-covered": "#9b59b6",
        }
        x_labels = [DEFECT_AR.get(d, d) if IS_AR else d for d in counts.index]
        fig = go.Figure(go.Bar(
            x=x_labels,
            y=counts.values.tolist(),
            marker_color=[COLORS.get(d, "#aaa") for d in counts.index],
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

        if "severity" in real.columns:
            section("SEVERITY DISTRIBUTION", "توزيع مستويات الخطورة")
            sev_counts = real["severity"].value_counts()
            sev_colors = {"critical": "#e74c3c", "warning": "#f5a623", "info": "#2ecc71"}
            sev_ar     = {"critical": "حرج", "warning": "تحذير", "info": "جيد"}
            sv1, sv2, sv3 = st.columns(3)
            for col_w, sk in zip([sv1, sv2, sv3], ["critical", "warning", "info"]):
                n   = sev_counts.get(sk, 0)
                pct = f"{n/total*100:.0f}%" if total > 0 else "0%"
                kpi(col_w, sk.upper(), sev_ar[sk].upper(), f"{n} ({pct})", sev_colors[sk])

        section("SCAN RECORDS", "سجلات الفحص")
        users  = [t("All Users", "جميع المستخدمين")] + sorted(real["panel_id"].dropna().unique().tolist()) if "panel_id" in real.columns else [t("All Users", "جميع المستخدمين")]
        sel_u  = st.selectbox(t("Filter by user", "تصفية حسب المستخدم"), users, key="dst_user")
        freal  = real if sel_u in ("All Users", "جميع المستخدمين") else real[real["panel_id"] == sel_u]

        # All columns — horizontal scroll via st.dataframe with wide layout
        all_show_cols = [c for c in [
            "timestamp", "panel_id", "defect_type", "severity", "confidence",
            "irradiation", "ambient_temp_c", "module_temp_c",
            "dc_power_kw", "ac_power_kw", "efficiency_pct",
            "panel_capacity_kw", "panel_age_years",
        ] if c in freal.columns]

        disp = freal[all_show_cols].copy()

        # Clean timestamps
        if "timestamp" in disp.columns:
            disp["timestamp"] = disp["timestamp"].apply(
                lambda v: "—" if pd.isna(v) or str(v).strip().lower() in ("nan","none","nat","") else str(v)[:16]
            )

        # Rename columns to friendly labels
        rename_map = {
            "timestamp":         t("Time",        "الوقت"),
            "panel_id":          t("User",         "المستخدم"),
            "defect_type":       t("Defect",       "العيب"),
            "severity":          t("Severity",     "الخطورة"),
            "confidence":        t("Conf %",       "الثقة %"),
            "irradiation":       t("Irrad.",       "إشعاع"),
            "ambient_temp_c":    t("Amb °C",       "حرارة محيط"),
            "module_temp_c":     t("Mod °C",       "حرارة لوح"),
            "dc_power_kw":       t("DC kW",        "DC kW"),
            "ac_power_kw":       t("AC kW",        "AC kW"),
            "efficiency_pct":    t("Eff %",        "كفاءة %"),
            "panel_capacity_kw": t("Cap kW",       "سعة kW"),
            "panel_age_years":   t("Age (yr)",     "العمر"),
        }
        disp = disp.rename(columns=rename_map).sort_values(t("Time","الوقت"), ascending=False)

        # Use st.dataframe with column_config for better display + horizontal scroll
        st.dataframe(
            disp,
            use_container_width=True,
            hide_index=True,
            height=400,
        )
        st.caption(t(f"{len(disp)} records shown", f"{len(disp)} سجل"))

        show_cols = [c for c in ["timestamp","panel_id","defect_type","confidence","severity"] if c in freal.columns]
        csv_bytes = freal[all_show_cols].to_csv(index=False).encode("utf-8")
        st.download_button(
            label=t("⬇️ Download Approved Dataset CSV", "⬇️ تحميل CSV"),
            data=csv_bytes,
            file_name=f"approved_dataset_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="dst_dl",
        )

        # ── SENSOR PARAMETER LINE CHARTS
        SENSOR_PARAMS = [
            ("irradiation",       t("Irradiation (W/m²/1000)", "الإشعاع الشمسي"),          "#f5a623"),
            ("ambient_temp_c",    t("Ambient Temp (°C)",        "حرارة المحيط"),             "#3498db"),
            ("module_temp_c",     t("Module Temp (°C)",         "حرارة اللوح"),              "#e74c3c"),
            ("dc_power_kw",       t("DC Power (kW)",            "طاقة DC"),                  "#2ecc71"),
            ("ac_power_kw",       t("AC Power (kW)",            "طاقة AC"),                  "#9b59b6"),
            ("efficiency_pct",    t("Efficiency (%)",           "الكفاءة %"),                "#1abc9c"),
            ("panel_capacity_kw", t("Panel Capacity (kW)",      "سعة اللوح"),                "#e67e22"),
            ("panel_age_years",   t("Panel Age (years)",        "عمر اللوح"),                "#95a5a6"),
        ]

        # Only show charts for columns that have at least some data
        available = [
            (col, label, color) for col, label, color in SENSOR_PARAMS
            if col in real.columns and real[col].replace("", float("nan")).dropna().shape[0] > 0
        ]

        if available:
            section("SENSOR PARAMETER TRENDS", "اتجاهات معاملات المستشعر")
            st.markdown(
                f'<div style="background:{BG_CARD};border:1px solid {BORDER};border-radius:10px;'
                f'padding:10px 16px;margin-bottom:16px;font-size:0.82rem;color:{TXT_M};">'
                f'📡 {t("Smoothed trends (rolling average per 50 scans). Only parameters with data are shown.","اتجاهات مسحّوبة (متوسط متحرك لكل 50 فحص). تُعرض المعاملات التي تحتوي على بيانات فقط.")}'
                f'</div>',
                unsafe_allow_html=True,
            )
            chart_df = real.copy()
            chart_df["timestamp"] = pd.to_datetime(chart_df["timestamp"], errors="coerce")
            chart_df = chart_df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

            FILL_COLORS = {
                "#f5a623": "rgba(245,166,35,0.08)",
                "#3498db": "rgba(52,152,219,0.08)",
                "#e74c3c": "rgba(231,76,60,0.08)",
                "#2ecc71": "rgba(46,204,113,0.08)",
                "#9b59b6": "rgba(155,89,182,0.08)",
                "#1abc9c": "rgba(26,188,156,0.08)",
                "#e67e22": "rgba(230,126,34,0.08)",
                "#95a5a6": "rgba(149,165,166,0.08)",
            }

            for i in range(0, len(available), 2):
                chunk = available[i:i+2]
                cols  = st.columns(len(chunk))
                for j, (col, label, color) in enumerate(chunk):
                    with cols[j]:
                        col_data = chart_df[["timestamp", col]].copy()
                        col_data[col] = pd.to_numeric(col_data[col], errors="coerce")
                        col_data = col_data.dropna(subset=[col]).reset_index(drop=True)
                        if col_data.empty:
                            continue
                        # Rolling mean — window=50 or length of data whichever is smaller
                        window = min(50, max(1, len(col_data) // 5))
                        col_data["smoothed"] = col_data[col].rolling(window=window, min_periods=1, center=True).mean()

                        fig = go.Figure()
                        # Raw data faint line
                        fig.add_trace(go.Scatter(
                            x=col_data["timestamp"], y=col_data[col],
                            mode="lines", name=t("Raw","خام"),
                            line=dict(color=color, width=1, dash="dot"),
                            opacity=0.3,
                        ))
                        # Smoothed line prominent
                        fig.add_trace(go.Scatter(
                            x=col_data["timestamp"], y=col_data["smoothed"],
                            mode="lines", name=t("Avg","متوسط"),
                            line=dict(color=color, width=2.5),
                            fill="tozeroy",
                            fillcolor=FILL_COLORS.get(color, "rgba(245,166,35,0.08)"),
                        ))
                        fig.update_layout(
                            title=dict(text=label, font=dict(color=TXT, size=13)),
                            height=220,
                            paper_bgcolor=BG_CARD,
                            plot_bgcolor=BG_CARD,
                            font=dict(color=TXT_M, size=11),
                            xaxis=dict(gridcolor=BORDER, color=TXT_M, showticklabels=True),
                            yaxis=dict(gridcolor=BORDER, color=TXT_M),
                            margin=dict(l=10, r=10, t=36, b=20),
                            showlegend=False,
                        )
                        st.plotly_chart(fig, use_container_width=True, key=f"chart_{col}")
        else:
            section("SENSOR PARAMETER TRENDS", "اتجاهات معاملات المستشعر")
            st.markdown(
                f'<div style="text-align:center;color:{TXT_M};padding:20px;background:{BG_CARD};'
                f'border:1px solid {BORDER};border-radius:10px;">'
                f'📡 {t("No sensor data yet. Users can enter sensor readings when scanning panels — the charts will appear here once data is available.","لا توجد بيانات مستشعر بعد. يمكن للمستخدمين إدخال قراءات المستشعر عند الفحص.")}'
                f'</div>',
                unsafe_allow_html=True,
            )

    # ═══════════════════════════════════════════════════════
    # SECTION 2 — ADMIN MERGE PANEL
    # ═══════════════════════════════════════════════════════
    st.markdown("<hr style='border:1px solid #2e3a50;margin:32px 0;'>", unsafe_allow_html=True)
    section("PENDING SCANS — MERGE INTO DATASET", "الفحوصات المعلقة — دمج في البيانات")

    st.markdown(
        f'<div style="background:#1e2d1e;border:1px solid #2ecc71;border-radius:10px;'
        f'padding:12px 18px;margin-bottom:16px;font-size:0.85rem;color:{TXT_S};">'
        f'🔒 <b style="color:#2ecc71;">{t("Admin Control","تحكم المدير")}:</b> '
        f'{t("New user scans accumulate here. Approve to merge into the static dataset, or discard.",  "تتراكم فحوصات المستخدمين هنا. اعتمد لدمجها في البيانات، أو تجاهلها.")}'
        f'</div>',
        unsafe_allow_html=True,
    )

    try:
        pending = _get_pending_scans()
    except Exception as e:
        st.error(f"{t('Could not load pending scans','تعذر تحميل الفحوصات المعلقة')}: {e}")
        pending = pd.DataFrame()

    if pending.empty:
        st.markdown(
            f'<div style="text-align:center;color:{TXT_M};padding:24px 0;font-size:0.92rem;">'
            f'✅ {t("No pending scans. All user scans have been reviewed.","لا توجد فحوصات معلقة. تمت مراجعة جميع الفحوصات.")}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        n_pending  = len(pending)
        n_critical = len(pending[pending["severity"] == "critical"]) if "severity" in pending.columns else 0
        p1, p2 = st.columns(2)
        kpi(p1, "PENDING SCANS",    "فحوصات معلقة",  str(n_pending),  "#f5a623")
        kpi(p2, "CRITICAL PENDING", "حرجة معلقة",    str(n_critical), "#e74c3c")

        st.markdown(f'<div style="margin-top:20px;margin-bottom:10px;font-size:0.82rem;color:{TXT_M};">{t("Review each scan and approve or discard individually:","راجع كل فحص واعتمده أو تجاهله بشكل فردي:")}</div>', unsafe_allow_html=True)

        BADGE_COLORS = {"critical":"#e74c3c","warning":"#f5a623","info":"#2ecc71"}
        DEFECT_ICONS = {
            "Bird-drop":"🐦","Clean":"✅","Dusty":"🌫️",
            "Electrical-damage":"⚡","Physical-damage":"💥","Snow-covered":"❄️",
        }
        DEFECT_AR = {
            "Bird-drop":"إفرازات الطيور","Clean":"نظيف","Dusty":"تراكم غبار",
            "Electrical-damage":"تلف كهربائي","Physical-damage":"تلف مادي","Snow-covered":"تغطية ثلج",
        }

        for idx, row in pending.iterrows():
            scan_id    = int(row["id"])
            defect     = str(row.get("defect_type",""))
            icon       = DEFECT_ICONS.get(defect,"🔍")
            disp       = DEFECT_AR.get(defect, defect) if IS_AR else row.get("display_en", defect)
            sev        = str(row.get("severity","info"))
            sev_color  = BADGE_COLORS.get(sev,"#aaa")
            conf       = float(row.get("confidence",0))
            conf_pct   = f"{conf:.0%}" if conf<=1.0 else f"{conf:.0f}%"
            email      = str(row.get("email",""))
            ts         = row.get("scanned_at")
            # Use the DB timestamp as-is (already correct Cairo time from INSERT)
            ts_str     = ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts,"strftime") else str(ts)[:16]

            card_col, btn_col = st.columns([5, 2])
            with card_col:
                st.markdown(
                    f'<div style="background:{BG_CARD};border:1px solid {sev_color}44;border-radius:10px;'
                    f'padding:12px 18px;">'
                    f'<span style="font-size:1.1rem;">{icon}</span>'
                    f'<span style="font-weight:700;margin-left:8px;color:{TXT};">{disp}</span>'
                    f'<span style="margin-left:10px;color:{sev_color};font-size:0.8rem;font-weight:700;">{sev.upper()}</span>'
                    f'<span style="margin-left:10px;color:{TXT_M};font-size:0.78rem;">{conf_pct}</span>'
                    f'<br><span style="color:{TXT_M};font-size:0.75rem;">👤 {email} &nbsp;·&nbsp; 🕐 {ts_str}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            with btn_col:
                ba, bd = st.columns(2)
                with ba:
                    if st.button("✅", key=f"merge_{scan_id}", help=t("Merge into dataset","دمج في البيانات"), use_container_width=True):
                        single = pending[pending["id"] == scan_id]
                        current_df = load_static_dataset()
                        merged_df  = _merge_pending_into_csv(single, current_df)
                        if _save_csv_to_storage(merged_df):
                            _mark_scan_merged(scan_id)
                            load_static_dataset.clear()
                            st.toast(t(f"✅ Scan merged!","✅ تم الدمج!"))
                            st.rerun()
                        else:
                            st.error(t("Upload failed.","فشل الرفع."))
                with bd:
                    if st.button("🗑", key=f"discard_{scan_id}", help=t("Discard","تجاهل"), use_container_width=True):
                        _mark_scan_merged(scan_id)
                        st.toast(t("🗑 Scan discarded.","🗑 تم التجاهل."))
                        st.rerun()

    # ═══════════════════════════════════════════════════════
    # SECTION 3 — MANUAL CSV UPLOAD
    # ═══════════════════════════════════════════════════════
    st.markdown("<hr style='border:1px solid #2e3a50;margin:32px 0;'>", unsafe_allow_html=True)
    section("MANUAL DATASET UPLOAD — REPLACE STATIC CSV", "رفع البيانات يدوياً — استبدال الملف")

    st.markdown(
        f'<div style="background:#2a1a10;border:1px solid #f5a623;border-radius:10px;'
        f'padding:12px 18px;margin-bottom:16px;font-size:0.85rem;color:{TXT_S};">'
        f'⚠️ <b style="color:#f5a623;">{t("Full Replace","استبدال كامل")}:</b> '
        f'{t("Upload a new CSV to completely replace the static dataset in Supabase Storage.",  "ارفع ملف CSV جديداً لاستبدال البيانات الثابتة في Supabase Storage.")}'
        f'</div>',
        unsafe_allow_html=True,
    )

    uploaded_csv = st.file_uploader(
        t("Upload replacement CSV (must match dataset schema)", "ارفع ملف CSV بديلاً (يجب أن يطابق هيكل البيانات)"),
        type=["csv"],
        key="dst_upload_csv",
    )

    if uploaded_csv is not None:
        try:
            preview_df = pd.read_csv(uploaded_csv)
            st.markdown(f'<div style="font-size:0.82rem;color:{TXT_M};margin-bottom:6px;">{t(f"Preview ({len(preview_df)} rows, {len(preview_df.columns)} columns):", f"معاينة ({len(preview_df)} صف، {len(preview_df.columns)} عمود):")}</div>', unsafe_allow_html=True)
            st.dataframe(preview_df.head(5), use_container_width=True, hide_index=True)
            missing_cols = [c for c in CSV_COLUMNS if c not in preview_df.columns]
            if missing_cols:
                st.warning(f'⚠️ {t("Missing columns","أعمدة مفقودة")}: {", ".join(missing_cols)}. {t("They will be added as empty.","ستُضاف فارغة.")}')
            if st.button(t("🔄 Confirm & Upload as New Static Dataset", "🔄 تأكيد ورفع كبيانات جديدة"), use_container_width=True, key="btn_upload_replace"):
                for col in CSV_COLUMNS:
                    if col not in preview_df.columns:
                        preview_df[col] = ""
                with st.spinner(t("Uploading...", "جاري الرفع...")):
                    success = _save_csv_to_storage(preview_df[CSV_COLUMNS])
                if success:
                    load_static_dataset.clear()
                    st.success(t("✅ Static dataset replaced successfully!", "✅ تم استبدال البيانات الثابتة بنجاح!"))
                    st.rerun()
                else:
                    st.error(t("Upload failed. Check SUPABASE_URL and SUPABASE_SERVICE_KEY.", "فشل الرفع. تحقق من المتغيرات."))
        except Exception as e:
            st.error(f"{t('Could not read uploaded CSV','تعذر قراءة الملف')}: {e}")
