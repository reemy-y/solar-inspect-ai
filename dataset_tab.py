"""
dataset_tab.py — Admin-only dataset panel for SolarInspect AI.

Architecture:
  - STATIC DATASET (solar_data.csv in Supabase Storage) → used for analysis only, never mutated by users
  - SCAN LOG (scans table in PostgreSQL/Supabase) → written on every user scan
  - ADMIN MERGE → admin reviews pending scans and approves merging them into the static CSV
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import os
import psycopg2
import psycopg2.extras
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────
# DATABASE CONNECTION (reuses the same DATABASE_URL as app.py)
# ─────────────────────────────────────────────────────────────────────
def _get_secret(key, default=""):
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)

SUPABASE_URL = _get_secret("SUPABASE_URL")
SUPABASE_KEY = _get_secret("SUPABASE_SERVICE_KEY")
STORAGE_BUCKET  = "solar-data"
STORAGE_CSV_KEY = "solar_data.csv"

CSV_COLUMNS = [
    "timestamp", "date", "hour", "panel_id",
    "irradiation", "ambient_temp_c", "module_temp_c",
    "dc_power_kw", "ac_power_kw", "defect_type", "efficiency_pct",
    "confidence", "severity", "source",
]


def _db():
    def _s(k, d=""):
        try:
            return st.secrets[k]
        except Exception:
            return os.environ.get(k, d)

    conn = psycopg2.connect(
        host=_s("DB_HOST"),
        port=int(_s("DB_PORT", "5432")),
        dbname=_s("DB_NAME", "postgres"),
        user=_s("DB_USER"),
        password=_s("DB_PASSWORD"),
        sslmode="require",
    )
    return conn


# ─────────────────────────────────────────────────────────────────────
# SUPABASE STORAGE — load / save static CSV
# ─────────────────────────────────────────────────────────────────────
def _storage_headers():
    return {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }


@st.cache_data(show_spinner=False, ttl=60)
def load_static_dataset() -> pd.DataFrame:
    """
    Download solar_data.csv from Supabase Storage.
    This is the approved static dataset used for analysis only.
    Returns an empty DataFrame if not found.
    """
    if not SUPABASE_URL or not SUPABASE_KEY:
        return pd.DataFrame(columns=CSV_COLUMNS)
    try:
        import requests
        url = f"{SUPABASE_URL}/storage/v1/object/{STORAGE_BUCKET}/{STORAGE_CSV_KEY}"
        r = requests.get(url, headers=_storage_headers(), timeout=10)
        if r.status_code == 200:
            from io import StringIO
            df = pd.read_csv(StringIO(r.text))
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            for col in ["source", "confidence", "severity"]:
                if col not in df.columns:
                    df[col] = None
            return df
        return pd.DataFrame(columns=CSV_COLUMNS)
    except Exception as e:
        st.warning(f"Could not load dataset from Supabase Storage: {e}")
        return pd.DataFrame(columns=CSV_COLUMNS)


def _save_csv_to_storage(df: pd.DataFrame) -> bool:
    """Upload the merged DataFrame back to Supabase Storage. Returns True on success."""
    if not SUPABASE_URL or not SUPABASE_KEY:
        return False
    try:
        import requests
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        url = f"{SUPABASE_URL}/storage/v1/object/{STORAGE_BUCKET}/{STORAGE_CSV_KEY}"
        headers = {**_storage_headers(), "Content-Type": "text/csv", "x-upsert": "true"}
        r = requests.put(url, headers=headers, data=csv_bytes, timeout=20)
        return r.status_code in (200, 201)
    except Exception as e:
        st.error(f"Storage upload failed: {e}")
        return False


# ─────────────────────────────────────────────────────────────────────
# PENDING SCANS — read from PostgreSQL
# ─────────────────────────────────────────────────────────────────────
def _get_pending_scans() -> pd.DataFrame:
    """Return all scans where merged_into_dataset = FALSE."""
    conn = _db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT id, email, scanned_at, defect_type, display_en,
                       confidence, severity, icon
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
    return df


def _mark_all_merged():
    """Mark all pending scans as merged (whether approved or discarded)."""
    conn = _db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE scans SET merged_into_dataset = TRUE WHERE merged_into_dataset = FALSE"
            )
        conn.commit()
    finally:
        conn.close()


def _merge_pending_into_csv(pending_df: pd.DataFrame, static_df: pd.DataFrame) -> pd.DataFrame:
    """Convert pending scan rows into CSV-compatible rows and concatenate with static dataset."""
    new_rows = []
    for _, row in pending_df.iterrows():
        ts = row["scanned_at"]
        new_rows.append({
            "timestamp":      str(ts)[:16],
            "date":           str(ts)[:10],
            "hour":           ts.hour if hasattr(ts, "hour") else "",
            "panel_id":       row["email"],
            "irradiation":    "",
            "ambient_temp_c": "",
            "module_temp_c":  "",
            "dc_power_kw":    "",
            "ac_power_kw":    "",
            "defect_type":    row["defect_type"],
            "efficiency_pct": "",
            "confidence":     round(float(row["confidence"]) * 100, 1),
            "severity":       row["severity"],
            "source":         "scan",
        })
    new_df = pd.DataFrame(new_rows)
    # Ensure all CSV columns exist in static_df before concat
    for col in CSV_COLUMNS:
        if col not in static_df.columns:
            static_df[col] = ""
    merged = pd.concat([static_df[CSV_COLUMNS], new_df[CSV_COLUMNS]], ignore_index=True)
    return merged


# ─────────────────────────────────────────────────────────────────────
# MAIN RENDER FUNCTION
# ─────────────────────────────────────────────────────────────────────
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
                unsafe_allow_html=True,
            )

    # ═══════════════════════════════════════════════════════
    # SECTION 1 — STATIC DATASET ANALYSIS
    # ═══════════════════════════════════════════════════════
    section("APPROVED DATASET — STATIC ANALYSIS")
    st.markdown(
        f'<div style="background:{BG_CARD};border:1px solid {BORDER};border-radius:10px;'
        f'padding:12px 18px;margin-bottom:16px;font-size:0.85rem;color:{TXT_M};">'
        f'📊 This dataset is <b style="color:#2ecc71;">static</b>. '
        f'It is never modified by user scans. Only the admin can merge new scans into it below.'
        f'</div>',
        unsafe_allow_html=True,
    )

    df = load_static_dataset()
    real = df[df["source"] == "scan"].copy() if not df.empty and "source" in df.columns else pd.DataFrame()

    if df.empty or real.empty:
        st.markdown(
            f'<div style="text-align:center;color:{TXT_M};padding:32px 0;font-size:0.95rem;">'
            f'📂 No approved scan data in the static dataset yet. Use the Merge panel below to add scans.'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        # KPIs
        total        = len(real)
        unique_users = real["panel_id"].nunique() if "panel_id" in real.columns else 0
        top_defect   = real["defect_type"].value_counts().index[0] if "defect_type" in real.columns else "—"
        critical_n   = len(real[real["severity"] == "critical"]) if "severity" in real.columns else 0

        k1, k2, k3, k4 = st.columns(4)
        kpi(k1, "TOTAL SCANS",  str(total),       "#f5a623")
        kpi(k2, "UNIQUE USERS", str(unique_users), "#2ecc71")
        kpi(k3, "MOST COMMON",  top_defect,        "#3498db")
        kpi(k4, "CRITICAL",     str(critical_n),   "#e74c3c")

        # Defect breakdown bar chart
        section("DEFECT BREAKDOWN")
        counts = real["defect_type"].value_counts()
        COLORS = {
            "Clean": "#2ecc71", "Dusty": "#3498db", "Bird-drop": "#f5a623",
            "Electrical-damage": "#e74c3c", "Physical-damage": "#c0392b", "Snow-covered": "#9b59b6",
        }
        fig = go.Figure(go.Bar(
            x=counts.index.tolist(),
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

        # Severity distribution
        if "severity" in real.columns:
            section("SEVERITY DISTRIBUTION")
            sev_counts = real["severity"].value_counts()
            sev_colors = {"critical": "#e74c3c", "warning": "#f5a623", "info": "#2ecc71"}
            sv1, sv2, sv3 = st.columns(3)
            for col_widget, sev_key in zip([sv1, sv2, sv3], ["critical", "warning", "info"]):
                n = sev_counts.get(sev_key, 0)
                pct = f"{n/total*100:.0f}%" if total > 0 else "0%"
                kpi(col_widget, sev_key.upper(), f"{n} ({pct})", sev_colors[sev_key])

        # Scan records table with user filter
        section("SCAN RECORDS")
        users = ["All Users"] + sorted(real["panel_id"].dropna().unique().tolist()) if "panel_id" in real.columns else ["All Users"]
        sel_u = st.selectbox(t("Filter by user", "تصفية حسب المستخدم"), users, key="dst_user")
        freal = real if sel_u == "All Users" else real[real["panel_id"] == sel_u]

        show = [c for c in ["timestamp", "panel_id", "defect_type", "confidence", "severity",
                             "irradiation", "ambient_temp_c", "module_temp_c", "ac_power_kw"]
                if c in freal.columns]
        disp = freal[show].rename(columns={"panel_id": "user_email"}).sort_values("timestamp", ascending=False)
        st.dataframe(disp, use_container_width=True, hide_index=True)
        st.caption(t(f"{len(disp)} records shown", f"{len(disp)} سجل"))

        # Download approved dataset
        csv_bytes = disp.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=t("⬇️ Download Approved Dataset CSV", "⬇️ تحميل CSV"),
            data=csv_bytes,
            file_name=f"approved_dataset_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            key="dst_dl",
        )

    # ═══════════════════════════════════════════════════════
    # SECTION 2 — ADMIN MERGE PANEL
    # ═══════════════════════════════════════════════════════
    st.markdown("<hr style='border:1px solid #2e3a50;margin:32px 0;'>", unsafe_allow_html=True)
    section("PENDING SCANS — MERGE INTO DATASET")

    st.markdown(
        f'<div style="background:#1e2d1e;border:1px solid #2ecc71;border-radius:10px;'
        f'padding:12px 18px;margin-bottom:16px;font-size:0.85rem;color:{TXT_S};">'
        f'🔒 <b style="color:#2ecc71;">Admin Control:</b> '
        f'New user scans accumulate here. Review them, then approve to merge into the static dataset '
        f'— or discard them. The core dataset is never changed automatically.'
        f'</div>',
        unsafe_allow_html=True,
    )

    try:
        pending = _get_pending_scans()
    except Exception as e:
        st.error(f"Could not load pending scans: {e}")
        pending = pd.DataFrame()

    if pending.empty:
        st.markdown(
            f'<div style="text-align:center;color:{TXT_M};padding:24px 0;font-size:0.92rem;">'
            f'✅ No pending scans. All user scans have been reviewed.'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        # Summary badge
        n_pending  = len(pending)
        n_critical = len(pending[pending["severity"] == "critical"]) if "severity" in pending.columns else 0
        p1, p2 = st.columns(2)
        kpi(p1, "PENDING SCANS",    str(n_pending),  "#f5a623")
        kpi(p2, "CRITICAL PENDING", str(n_critical), "#e74c3c")

        # Preview table
        st.markdown(f'<div style="margin-top:16px;margin-bottom:8px;font-size:0.82rem;color:{TXT_M};">Preview of pending scans:</div>', unsafe_allow_html=True)
        preview_cols = [c for c in ["email", "scanned_at", "defect_type", "confidence", "severity"] if c in pending.columns]
        st.dataframe(
            pending[preview_cols].rename(columns={"email": "user_email"}),
            use_container_width=True,
            hide_index=True,
        )

        # Action buttons
        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        col_approve, col_discard = st.columns(2)

        with col_approve:
            st.markdown(
                f'<div style="font-size:0.78rem;color:{TXT_M};margin-bottom:6px;">'
                f'Merges all {n_pending} scans into the approved dataset on Supabase Storage.'
                f'</div>', unsafe_allow_html=True
            )
            if st.button("✅ Approve & Merge All into Dataset", use_container_width=True, key="btn_merge"):
                with st.spinner("Merging scans into static dataset..."):
                    current_df = load_static_dataset()
                    merged_df  = _merge_pending_into_csv(pending, current_df)
                    success    = _save_csv_to_storage(merged_df)
                if success:
                    _mark_all_merged()
                    load_static_dataset.clear()   # clear cache so analysis refreshes
                    st.success(f"✅ {n_pending} scans merged into the approved dataset successfully!")
                    st.rerun()
                else:
                    st.error("Upload to Supabase Storage failed. Check SUPABASE_URL and SUPABASE_SERVICE_KEY env vars.")

        with col_discard:
            st.markdown(
                f'<div style="font-size:0.78rem;color:{TXT_M};margin-bottom:6px;">'
                f'Marks all {n_pending} scans as processed without adding them to the dataset.'
                f'</div>', unsafe_allow_html=True
            )
            if st.button("🗑 Discard All Pending Scans", use_container_width=True, key="btn_discard"):
                _mark_all_merged()
                st.warning(f"🗑 {n_pending} pending scans discarded — dataset unchanged.")
                st.rerun()

    # ═══════════════════════════════════════════════════════
    # SECTION 3 — MANUAL CSV UPLOAD (override / full replace)
    # ═══════════════════════════════════════════════════════
    st.markdown("<hr style='border:1px solid #2e3a50;margin:32px 0;'>", unsafe_allow_html=True)
    section("MANUAL DATASET UPLOAD — REPLACE STATIC CSV")

    st.markdown(
        f'<div style="background:#2a1a10;border:1px solid #f5a623;border-radius:10px;'
        f'padding:12px 18px;margin-bottom:16px;font-size:0.85rem;color:{TXT_S};">'
        f'⚠️ <b style="color:#f5a623;">Full Replace:</b> '
        f'Upload a new CSV to completely replace the static dataset in Supabase Storage. '
        f'Make sure it has the correct columns before uploading.'
        f'</div>',
        unsafe_allow_html=True,
    )

    uploaded_csv = st.file_uploader(
        "Upload replacement CSV (must match dataset schema)",
        type=["csv"],
        key="dst_upload_csv",
    )

    if uploaded_csv is not None:
        try:
            preview_df = pd.read_csv(uploaded_csv)
            st.markdown(f'<div style="font-size:0.82rem;color:{TXT_M};margin-bottom:6px;">Preview ({len(preview_df)} rows, {len(preview_df.columns)} columns):</div>', unsafe_allow_html=True)
            st.dataframe(preview_df.head(5), use_container_width=True, hide_index=True)

            missing_cols = [c for c in CSV_COLUMNS if c not in preview_df.columns]
            if missing_cols:
                st.warning(f"⚠️ Missing columns: {', '.join(missing_cols)}. They will be added as empty.")

            if st.button("🔄 Confirm & Upload as New Static Dataset", use_container_width=True, key="btn_upload_replace"):
                for col in CSV_COLUMNS:
                    if col not in preview_df.columns:
                        preview_df[col] = ""
                with st.spinner("Uploading to Supabase Storage..."):
                    success = _save_csv_to_storage(preview_df[CSV_COLUMNS])
                if success:
                    load_static_dataset.clear()
                    st.success("✅ Static dataset replaced successfully! Analysis section will refresh.")
                    st.rerun()
                else:
                    st.error("Upload failed. Check SUPABASE_URL and SUPABASE_SERVICE_KEY.")
        except Exception as e:
            st.error(f"Could not read uploaded CSV: {e}")
