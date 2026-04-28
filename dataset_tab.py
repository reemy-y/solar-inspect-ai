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

def _get_all_db_scans() -> pd.DataFrame:
    """Get ALL scans from the DB (both merged and pending) for the history view."""
    conn = _db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT email, scanned_at, defect_type, display_en, display_ar,
                       confidence, severity, icon, merged_into_dataset
                FROM scans
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
    new_rows = []
    for _, row in pending_df.iterrows():
        ts = row["scanned_at"]
        conf = float(row["confidence"])
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
            "confidence":     round(conf * 100, 1) if conf <= 1.0 else round(conf, 1),
            "severity":       row["severity"],
            "source":         "scan",
        })
    new_df = pd.DataFrame(new_rows)
    for col in CSV_COLUMNS:
        if col not in static_df.columns:
            static_df[col] = ""
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
    # SECTION 0 — ALL DB SCANS HISTORY (users + timestamps)
    # ═══════════════════════════════════════════════════════
    section("ALL USER SCANS — FROM DATABASE", "جميع الفحوصات من قاعدة البيانات")
    st.markdown(
        f'<div style="background:{BG_CARD};border:1px solid {BORDER};border-radius:10px;'
        f'padding:12px 18px;margin-bottom:16px;font-size:0.85rem;color:{TXT_M};">'
        f'🗄️ {t("Every scan submitted by users — including timestamp, defect, severity, and merge status.","كل فحص قدمه المستخدمون — يتضمن التوقيت، العيب، الخطورة، وحالة الدمج.")}'
        f'</div>',
        unsafe_allow_html=True,
    )

    try:
        all_scans_df = _get_all_db_scans()
    except Exception as e:
        st.error(f"{t('Could not load scans','تعذر تحميل الفحوصات')}: {e}")
        all_scans_df = pd.DataFrame()

    if all_scans_df.empty:
        st.markdown(
            f'<div style="text-align:center;color:{TXT_M};padding:24px 0;">'
            f'📭 {t("No scans in the database yet.","لا توجد فحوصات في قاعدة البيانات بعد.")}'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        # KPIs
        ks1, ks2, ks3, ks4 = st.columns(4)
        kpi(ks1, "TOTAL SCANS",   "إجمالي الفحوصات", str(len(all_scans_df)),                                                        "#f5a623")
        kpi(ks2, "UNIQUE USERS",  "مستخدمون فريدون",  str(all_scans_df["email"].nunique()),                                          "#2ecc71")
        kpi(ks3, "PENDING MERGE", "بانتظار الدمج",    str(len(all_scans_df[all_scans_df["merged_into_dataset"] == False])),           "#3498db")
        kpi(ks4, "CRITICAL",      "حرج",              str(len(all_scans_df[all_scans_df["severity"] == "critical"])),                 "#e74c3c")

        # User filter
        all_emails = sorted(all_scans_df["email"].dropna().unique().tolist())
        sel_hist   = st.selectbox(
            t("Filter by user", "تصفية حسب المستخدم"),
            [t("All Users", "جميع المستخدمين")] + all_emails,
            key="dst_hist_filter"
        )
        filtered_scans = all_scans_df if sel_hist in ("All Users", "جميع المستخدمين") else all_scans_df[all_scans_df["email"] == sel_hist]

        # Show as history cards — no sensor params
        BADGE_COLORS = {"critical": "#e74c3c", "warning": "#f5a623", "info": "#2ecc71"}
        SEVERITY_AR  = {"critical": "حرج", "warning": "تحذير", "info": "جيد"}

        for _, row in filtered_scans.head(50).iterrows():
            disp     = row.get("display_ar", "") if IS_AR else row.get("display_en", "")
            sev      = row.get("severity", "info")
            sev_col  = BADGE_COLORS.get(sev, "#aaa")
            sev_lbl  = SEVERITY_AR.get(sev, sev).upper() if IS_AR else sev.upper()
            conf     = float(row.get("confidence", 0))
            conf_pct = f"{conf:.0%}" if conf <= 1.0 else f"{conf:.0f}%"
            icon     = row.get("icon", "🔍")
            ts       = row.get("scanned_at")
            ts_str   = ts.strftime("%Y-%m-%d %H:%M") if hasattr(ts, "strftime") else str(ts)[:16]
            merged   = row.get("merged_into_dataset", False)
            merged_badge = f'<span style="color:#2ecc71;font-size:0.7rem;">✅ {t("merged","مدمج")}</span>' if merged else f'<span style="color:#f5a623;font-size:0.7rem;">⏳ {t("pending","معلق")}</span>'

            st.markdown(
                f'<div style="background:{BG_CARD};border:1px solid {BORDER};border-radius:10px;'
                f'padding:12px 18px;margin-bottom:6px;">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;gap:12px;">'
                f'<div style="flex:1;min-width:0;">'
                f'<span style="font-size:1.1rem;">{icon}</span>'
                f'<span style="font-weight:700;margin-left:8px;color:{TXT};">{disp}</span>'
                f'<span style="margin-left:10px;color:{TXT_M};font-size:0.78rem;">👤 {row["email"]}</span>'
                f'</div>'
                f'<div style="text-align:right;flex-shrink:0;display:flex;gap:10px;align-items:center;">'
                f'{merged_badge}'
                f'<span style="color:{sev_col};font-size:0.8rem;font-weight:700;">{sev_lbl}</span>'
                f'<span style="color:{TXT_M};font-size:0.73rem;">{conf_pct}</span>'
                f'<span style="color:{TXT_M};font-size:0.73rem;">🕐 {ts_str}</span>'
                f'</div></div></div>',
                unsafe_allow_html=True,
            )

        st.caption(t(f"Showing {len(filtered_scans.head(50))} of {len(filtered_scans)} scans",
                     f"عرض {len(filtered_scans.head(50))} من {len(filtered_scans)} فحص"))

    # ═══════════════════════════════════════════════════════
    # SECTION 1 — STATIC DATASET ANALYSIS
    # ═══════════════════════════════════════════════════════
    st.markdown("<hr style='border:1px solid #2e3a50;margin:32px 0;'>", unsafe_allow_html=True)
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

        show_cols = [c for c in ["timestamp", "panel_id", "defect_type", "confidence", "severity"] if c in freal.columns]
        disp = freal[show_cols].rename(columns={
            "panel_id": t("user_email", "البريد"),
            "defect_type": t("defect_type", "نوع العيب"),
            "confidence": t("confidence", "الثقة"),
            "severity": t("severity", "الخطورة"),
            "timestamp": t("timestamp", "التوقيت"),
        }).sort_values(t("timestamp", "التوقيت"), ascending=False)
        st.dataframe(disp, use_container_width=True, hide_index=True)
        st.caption(t(f"{len(disp)} records shown", f"{len(disp)} سجل"))

        csv_bytes = freal[show_cols].to_csv(index=False).encode("utf-8")
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

        st.markdown(f'<div style="margin-top:16px;margin-bottom:8px;font-size:0.82rem;color:{TXT_M};">{t("Preview of pending scans:","معاينة الفحوصات المعلقة:")}</div>', unsafe_allow_html=True)
        preview_cols = [c for c in ["email", "scanned_at", "defect_type", "confidence", "severity"] if c in pending.columns]
        rename_map = {
            "email":       t("email", "البريد"),
            "scanned_at":  t("scanned_at", "وقت الفحص"),
            "defect_type": t("defect_type", "نوع العيب"),
            "confidence":  t("confidence", "الثقة"),
            "severity":    t("severity", "الخطورة"),
        }
        st.dataframe(
            pending[preview_cols].rename(columns=rename_map),
            use_container_width=True,
            hide_index=True,
        )

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        col_approve, col_discard = st.columns(2)

        with col_approve:
            st.markdown(f'<div style="font-size:0.78rem;color:{TXT_M};margin-bottom:6px;">{t(f"Merges all {n_pending} scans into the approved dataset.",f"يدمج {n_pending} فحصاً في البيانات المعتمدة.")}</div>', unsafe_allow_html=True)
            if st.button(t("✅ Approve & Merge All", "✅ اعتماد ودمج الكل"), use_container_width=True, key="btn_merge"):
                with st.spinner(t("Merging scans...", "جاري الدمج...")):
                    current_df = load_static_dataset()
                    merged_df  = _merge_pending_into_csv(pending, current_df)
                    success    = _save_csv_to_storage(merged_df)
                if success:
                    _mark_all_merged()
                    load_static_dataset.clear()
                    st.success(t(f"✅ {n_pending} scans merged successfully!", f"✅ تم دمج {n_pending} فحصاً بنجاح!"))
                    st.rerun()
                else:
                    st.error(t("Upload to Supabase Storage failed. Check SUPABASE_URL and SUPABASE_SERVICE_KEY.",
                               "فشل الرفع إلى Supabase Storage. تحقق من المتغيرات."))

        with col_discard:
            st.markdown(f'<div style="font-size:0.78rem;color:{TXT_M};margin-bottom:6px;">{t(f"Discards all {n_pending} scans without adding to dataset.",f"يتجاهل {n_pending} فحصاً دون إضافة.")}</div>', unsafe_allow_html=True)
            if st.button(t("🗑 Discard All Pending", "🗑 تجاهل الكل"), use_container_width=True, key="btn_discard"):
                _mark_all_merged()
                st.warning(t(f"🗑 {n_pending} pending scans discarded — dataset unchanged.", f"🗑 تم تجاهل {n_pending} فحصاً — البيانات لم تتغير."))
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
