# ─────────────────────────────────────────────────────────────────────
# 1. IMPORTS & PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
import streamlit as st
import torch, timm, numpy as np, os, re, json, joblib, pandas as pd
import unicodedata as _ud, hashlib, psycopg2, psycopg2.extras
from PIL import Image
from datetime import datetime
from fpdf import FPDF
from dataset_tab import render_dataset_tab

st.set_page_config(
    page_title="SolarInspect AI",
    page_icon="☀️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────
# 2. AUTH & DATABASE — PostgreSQL (Supabase) + persistent sessions
# ─────────────────────────────────────────────────────────────────────
import secrets as _secrets

ADMIN_EMAIL = "reemya185@gmail.com"
TOKEN_KEY   = "solar_session_token"

def _get_secret(key, default=""):
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)

def _db():
    conn = psycopg2.connect(
        host=_get_secret("DB_HOST"),
        port=int(_get_secret("DB_PORT", "5432")),
        dbname=_get_secret("DB_NAME", "postgres"),
        user=_get_secret("DB_USER"),
        password=_get_secret("DB_PASSWORD"),
        sslmode="require",
    )
    return conn

def _hash(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def _now_cairo() -> datetime:
    """Return current datetime in Cairo timezone (UTC+2 or UTC+3 with DST)."""
    try:
        import zoneinfo
        from datetime import timezone as _tz
        cairo = zoneinfo.ZoneInfo("Africa/Cairo")
        return datetime.now(cairo).replace(tzinfo=None)
    except Exception:
        from datetime import timedelta
        return datetime.utcnow() + timedelta(hours=2)

def _create_token(email: str) -> str:
    from datetime import timedelta
    token   = _secrets.token_hex(32)
    now     = _now_cairo()
    expires = now + timedelta(days=30)
    conn = _db()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM tokens WHERE email = %s", (email,))
            cur.execute(
                "INSERT INTO tokens (token, email, created_at, expires_at) VALUES (%s, %s, %s, %s)",
                (token, email, now, expires),
            )
        conn.commit()
    finally:
        conn.close()
    return token

def _validate_token(token: str):
    if not token:
        return None
    conn = _db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT email, expires_at FROM tokens WHERE token = %s", (token,))
            row = cur.fetchone()
        if row is None:
            return None
        email, expires_at = row
        if datetime.now() > expires_at.replace(tzinfo=None):
            with conn.cursor() as cur:
                cur.execute("DELETE FROM tokens WHERE token = %s", (token,))
            conn.commit()
            return None
        return email
    finally:
        conn.close()

def _delete_token(token: str):
    conn = _db()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM tokens WHERE token = %s", (token,))
        conn.commit()
    finally:
        conn.close()

def _get_role(email: str) -> str:
    if not email:
        return "guest"
    if email.lower() == ADMIN_EMAIL.lower():
        return "admin"
    conn = _db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT role FROM users WHERE email = %s", (email,))
            row = cur.fetchone()
        return row[0] if row else "user"
    finally:
        conn.close()

def auth_signup(email: str, password: str):
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return False, "Invalid email format."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    role = "admin" if email.lower() == ADMIN_EMAIL.lower() else "user"
    conn = _db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO users (email, pw_hash, role, created_at) VALUES (%s, %s, %s, %s)",
                (email.lower().strip(), _hash(password), role, datetime.now()),
            )
        conn.commit()
        return True, "Account created! You can now log in."
    except psycopg2.errors.UniqueViolation:
        conn.rollback()
        return False, "An account with this email already exists."
    finally:
        conn.close()

def auth_login(email: str, password: str):
    conn = _db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pw_hash FROM users WHERE email = %s", (email.lower().strip(),))
            row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        return False, "No account found with this email.", None
    if row[0] != _hash(password):
        return False, "Incorrect password.", None
    token = _create_token(email.lower().strip())
    return True, "Logged in successfully.", token

def db_save_scan(email, pred_class, info, confidence,
                 irradiation=None, ambient_temp=None, module_temp=None,
                 dc_power=None, ac_power=None, efficiency_pct=None,
                 panel_capacity=None, panel_age=None):
    now = _now_cairo()
    conn = _db()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT id FROM scans
                WHERE email = %s AND defect_type = %s
                  AND scanned_at >= NOW() - INTERVAL '1 minute'
            """, (email, pred_class))
            existing = cur.fetchone()
        if existing is None:
            with conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO scans
                      (email, scanned_at, defect_type, display_en, display_ar,
                       confidence, severity, icon, merged_into_dataset,
                       irradiation, ambient_temp_c, module_temp_c,
                       dc_power_kw, ac_power_kw, efficiency_pct,
                       panel_capacity_kw, panel_age_years)
                    VALUES (%s,%s,%s,%s,%s,%s,%s,%s,FALSE,%s,%s,%s,%s,%s,%s,%s,%s)
                """, (
                    email, now, pred_class,
                    info["display_en"], info["display_ar"],
                    round(confidence, 4), info["severity"], info["icon"],
                    irradiation, ambient_temp, module_temp,
                    dc_power, ac_power, efficiency_pct,
                    panel_capacity, panel_age,
                ))
            conn.commit()
    finally:
        conn.close()

def db_get_scans(email: str, admin: bool = False) -> list:
    conn = _db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if admin:
                cur.execute("""
                    SELECT email, scanned_at, defect_type, display_en, display_ar,
                           confidence, severity, icon
                    FROM scans ORDER BY scanned_at DESC NULLS LAST
                """)
            else:
                cur.execute("""
                    SELECT email, scanned_at, defect_type, display_en, display_ar,
                           confidence, severity, icon
                    FROM scans WHERE email = %s ORDER BY scanned_at DESC NULLS LAST
                """, (email,))
            rows = cur.fetchall()
    finally:
        conn.close()

    def _fmt_time(val):
        if val is None:
            return "—"
        if hasattr(val, "strftime"):
            return val.strftime("%Y-%m-%d %H:%M")
        s = str(val)
        return s[:16] if len(s) >= 16 else s

    def _norm_conf(val):
        """Confidence in DB is stored 0–1 (e.g. 0.9956). Return as fraction for display."""
        try:
            f = float(val)
            return f if f <= 1.0 else f / 100.0
        except Exception:
            return 0.0

    return [
        {
            "email":      r["email"],
            "time":       _fmt_time(r["scanned_at"]),
            "class":      r["defect_type"],
            "display_en": r["display_en"],
            "display_ar": r["display_ar"],
            "confidence": _norm_conf(r["confidence"]),
            "severity":   r["severity"],
            "icon":       r["icon"],
            "source":     "db",
        }
        for r in rows
    ]

def db_delete_scans(email: str):
    conn = _db()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM scans WHERE email = %s", (email,))
        conn.commit()
    finally:
        conn.close()

def _get_secret(key, default=""):
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key, default)

def _load_csv_scans_as_history(email: str, admin: bool = False) -> list:
    """
    Load scans from the static solar_data.csv in Supabase Storage.
    Returns a list of dicts in the same format as db_get_scans().
    """
    DEFECT_DISPLAY = {
        "Bird-drop":          ("Bird Dropping",   "إفرازات الطيور",  "🐦"),
        "Clean":              ("Clean Panel",      "لوح نظيف",       "✅"),
        "Dusty":              ("Dust Accumulation","تراكم الغبار",   "🌫️"),
        "Electrical-damage":  ("Electrical Damage","تلف كهربائي",   "⚡"),
        "Physical-damage":    ("Physical Damage",  "تلف مادي",       "💥"),
        "Snow-covered":       ("Snow Coverage",    "تغطية الثلج",    "❄️"),
    }
    SEVERITY_MAP = {
        "Bird-drop": "warning", "Clean": "info", "Dusty": "info",
        "Electrical-damage": "critical", "Physical-damage": "critical", "Snow-covered": "warning",
    }
    url_base = _get_secret("SUPABASE_URL")
    key      = _get_secret("SUPABASE_SERVICE_KEY")
    if not url_base or not key:
        return []
    try:
        import requests
        from io import StringIO
        headers = {"apikey": key, "Authorization": f"Bearer {key}"}
        r = requests.get(
            f"{url_base}/storage/v1/object/solar-data/solar_data.csv",
            headers=headers, timeout=10
        )
        if r.status_code != 200:
            return []
        df = pd.read_csv(StringIO(r.text))
        if "source" in df.columns:
            df = df[df["source"] == "scan"]
        if not admin and "panel_id" in df.columns:
            df = df[df["panel_id"] == email]
        results = []
        for _, row in df.iterrows():
            defect      = str(row.get("defect_type", ""))
            disp_en, disp_ar, icon = DEFECT_DISPLAY.get(defect, (defect, defect, "🔍"))
            sev         = str(row.get("severity", SEVERITY_MAP.get(defect, "info")))
            conf_raw    = row.get("confidence", 0)
            try:
                conf = float(conf_raw)
                conf = conf / 100.0 if conf > 1.0 else conf
            except Exception:
                conf = 0.0
            ts_raw = row.get("timestamp", "")
            # Guard against NaN/None/empty
            try:
                if pd.isna(ts_raw) or str(ts_raw).strip().lower() in ("", "nan", "none", "nat"):
                    ts_str = "—"
                else:
                    ts_str = str(ts_raw)[:16]
            except Exception:
                ts_str = "—"
            results.append({
                "email":      str(row.get("panel_id", "")),
                "time":       ts_str,
                "class":      defect,
                "display_en": disp_en,
                "display_ar": disp_ar,
                "confidence": conf,
                "severity":   sev,
                "icon":       icon,
                "source":     "csv",
            })
        # Sort by time descending
        results.sort(key=lambda x: x["time"], reverse=True)
        return results
    except Exception:
        return []

def get_full_history(email: str, admin: bool = False) -> list:
    """Combine DB scans + CSV scans, deduplicate by (email+defect+time), sort by time desc."""
    db_scans  = db_get_scans(email, admin=admin)
    csv_scans = _load_csv_scans_as_history(email, admin=admin)
    # Deduplicate: DB scans take priority, skip CSV entries already in DB
    seen = set()
    for h in db_scans:
        seen.add((h["email"], h["class"], h["time"][:13]))  # match to hour precision
    combined = list(db_scans)
    for h in csv_scans:
        key = (h["email"], h["class"], h["time"][:13])
        if key not in seen:
            combined.append(h)
            seen.add(key)
    combined.sort(key=lambda x: x["time"], reverse=True)
    return combined

def _ensure_admin():
    admin_pw = _get_secret("ADMIN_PASSWORD", "SolarAdmin2026!")
    conn = _db()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM users WHERE email = %s", (ADMIN_EMAIL,))
            existing = cur.fetchone()
        if existing is None:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO users (email, pw_hash, role, created_at) VALUES (%s, %s, %s, %s)",
                    (ADMIN_EMAIL, _hash(admin_pw), "admin", datetime.now()),
                )
            conn.commit()
    finally:
        conn.close()

_ensure_admin()

def db_get_all_users() -> list:
    conn = _db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT email, role, created_at FROM users ORDER BY created_at DESC")
            rows = cur.fetchall()
    finally:
        conn.close()
    return [
        {
            "email":   r["email"],
            "role":    r["role"],
            "created": r["created_at"].strftime("%Y-%m-%d %H:%M") if hasattr(r["created_at"], "strftime") else str(r["created_at"]),
        }
        for r in rows
    ]

# ─────────────────────────────────────────────────────────────────────
# 3. SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────
for key, default in [
    ("lang",          "en"),
    ("history",       []),
    ("dark_mode",     True),
    ("user_email",    ""),
    ("logged_in",     False),
    ("auth_email",    ""),
    ("session_token", ""),
    ("saved_hashes",  set()),
]:
    if key not in st.session_state:
        st.session_state[key] = default

# ── Auto-login via token
if not st.session_state.logged_in and st.session_state.session_token:
    _email = _validate_token(st.session_state.session_token)
    if _email:
        st.session_state.logged_in  = True
        st.session_state.auth_email = _email
        st.session_state.user_email = _email

# ─────────────────────────────────────────────────────────────────────
# 4. THEME
# ─────────────────────────────────────────────────────────────────────
def t(en, ar):
    return ar if st.session_state.lang == "ar" else en

IS_AR    = st.session_state.lang == "ar"
RTL      = "direction:rtl;text-align:right;" if IS_AR else ""
FONT     = "'Cairo', sans-serif" if IS_AR else "'Syne', sans-serif"
DM       = True
BG       = "#1a1f2e"
BG_CARD  = "#242938"
BG_HERO  = "linear-gradient(135deg,#1f2d42 0%,#1a1f2e 60%)"
BORDER   = "#2e3a50"
TXT      = "#e8edf5"
TXT_M    = "#8fa4bd"
TXT_S    = "#b0c2d8"
URG_BG   = "#2a1520"
URG_C    = "#f07070"
TIP_BG   = "#1e2d40"
TIP_BD   = "#2e4a65"
BAR_BG   = "#2e3a50"
HERO_TC  = "#e8edf5"
INPUT_BG = "#2e3a50"

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&family=Cairo:wght@400;600;700;800&display=swap');
html, body, [class*="css"] {{
    background-color:{BG}; color:{TXT}; font-family:{FONT};
    font-size:{"19px" if IS_AR else "17px"};
    font-weight:{"600" if IS_AR else "400"};
}}
.stApp {{ background-color:{BG}; }}
.stTextInput>div>div>input,
.stNumberInput>div>div>input {{
    background-color:{INPUT_BG} !important; color:{TXT} !important;
    border-color:{BORDER} !important; border-radius:8px !important;
}}
.stNumberInput label, .stTextInput label, .stSlider label,
.stSelectbox label, [data-testid="stWidgetLabel"] p, [data-testid="stWidgetLabel"] {{
    color:{TXT_S} !important; font-size:0.92rem !important; font-weight:500 !important;
}}
.stButton>button {{
    background-color:{BG_CARD}; color:{TXT};
    border:1px solid {BORDER}; border-radius:8px;
    font-weight:600; transition:all 0.2s;
}}
.stButton>button:hover {{ border-color:#f5a623; color:#f5a623; }}
.stTabs [data-baseweb="tab"] {{ color:{TXT_M}; font-size:0.92rem; }}
.stTabs [aria-selected="true"] {{ color:{TXT} !important; border-bottom:2px solid #f5a623; }}

/* ── HEADER CONTROLS ── */
.ctrl-row {{
    display:flex; align-items:center; justify-content:flex-end;
    gap:8px; padding-top:18px; flex-wrap:nowrap;
}}
.user-chip {{
    background:{BG_CARD}; border:1px solid {BORDER}; border-radius:8px;
    padding:0 12px; height:38px; display:flex; align-items:center; gap:8px;
    font-size:0.78rem; color:{TXT_M}; white-space:nowrap; flex-shrink:1; min-width:0;
}}
.user-chip .avatar {{
    background:#f5a623; color:#000; border-radius:6px;
    width:24px; height:24px; display:flex; align-items:center;
    justify-content:center; font-weight:800; font-size:0.72rem; flex-shrink:0;
}}
.user-chip .email {{ overflow:hidden; text-overflow:ellipsis; }}

/* ── HERO ── */
.hero {{
    background:{BG_HERO}; border:1px solid {BORDER}; border-radius:16px;
    padding:36px 48px; margin-bottom:28px; position:relative; overflow:hidden; {RTL}
}}
.hero::before {{
    content:''; position:absolute; top:-60px; right:-60px;
    width:220px; height:220px;
    background:radial-gradient(circle,#f5a623 0%,transparent 70%);
    opacity:0.10; border-radius:50%;
}}
.hero-title {{ font-size:2.6rem; font-weight:800; color:{HERO_TC}; letter-spacing:-1px; margin:0; line-height:1.1; }}
.hero-title span {{ color:#f5a623; }}
.hero-sub {{ font-family:'Space Mono',monospace; color:{TXT_M}; font-size:0.85rem; margin-top:10px; letter-spacing:2px; }}

.upload-zone {{ border:2px dashed {BORDER}; border-radius:12px; padding:48px; text-align:center; background:{BG_CARD}; {RTL} }}
.metric-card {{ background:{BG_CARD}; border:1px solid {BORDER}; border-radius:12px; padding:20px 24px; margin-bottom:12px; {RTL} }}
.metric-label {{ font-family:'Space Mono',monospace; font-size:0.72rem; color:{TXT_M}; letter-spacing:2px; margin-bottom:6px; }}
.metric-value {{ font-size:2rem; font-weight:800; color:{TXT}; line-height:1; }}
.perf-card {{ background:{BG_CARD}; border:1px solid {BORDER}; border-radius:12px; padding:24px; margin-bottom:16px; {RTL} }}
.badge-critical {{ background:#c0392b22; border:1px solid #e74c3c; color:#e74c3c; border-radius:8px; padding:8px 20px; font-weight:800; font-size:1.1rem; display:inline-block; }}
.badge-warning  {{ background:#b8860b22; border:1px solid #f5a623; color:#f5a623; border-radius:8px; padding:8px 20px; font-weight:800; font-size:1.1rem; display:inline-block; }}
.badge-good     {{ background:#1a4a2a22; border:1px solid #2ecc71; color:#2ecc71; border-radius:8px; padding:8px 20px; font-weight:800; font-size:1.1rem; display:inline-block; }}
.defect-card {{ background:{BG_CARD}; border-radius:0 10px 10px 0; padding:16px 20px; margin-bottom:10px; {RTL} }}
.defect-card.critical {{ border-left:4px solid #e74c3c; }}
.defect-card.warning  {{ border-left:4px solid #f5a623; }}
.defect-card.info     {{ border-left:4px solid #3498db; }}
.defect-name   {{ font-weight:800; font-size:{"1.15rem" if IS_AR else "1.05rem"}; color:{TXT}; margin-bottom:4px; }}
.defect-desc   {{ font-size:{"1rem" if IS_AR else "0.9rem"}; color:{TXT_S}; line-height:{"1.9" if IS_AR else "1.55"}; margin-bottom:8px; }}
.defect-action {{ font-size:{"0.95rem" if IS_AR else "0.88rem"}; color:#f5a623; font-weight:700; }}
.urgency-box   {{ background:{URG_BG}; border:1px solid {URG_C}; border-radius:8px; padding:12px 16px; margin-top:10px; font-size:0.88rem; color:{URG_C}; font-weight:600; {RTL} }}
.section-title {{ font-family:'Space Mono',monospace; font-size:0.72rem; color:{TXT_M}; letter-spacing:3px; margin-bottom:14px; margin-top:26px; text-transform:uppercase; {RTL} }}
.conf-bar-bg   {{ background:{BAR_BG}; border-radius:4px; height:6px; margin-top:4px; }}
.conf-bar-fill {{ height:6px; border-radius:4px; background:linear-gradient(90deg,#f5a623,#e74c3c); }}
.history-card  {{ background:{BG_CARD}; border:1px solid {BORDER}; border-radius:10px; padding:14px 18px; margin-bottom:8px; transition:border-color 0.2s; {RTL} }}
.history-card:hover {{ border-color:#f5a623; }}
.tip-card {{ background:{TIP_BG}; border:1px solid {TIP_BD}; border-radius:10px; padding:16px; margin-bottom:8px; {RTL} }}
.tip-text {{ font-size:{"0.95rem" if IS_AR else "0.88rem"}; color:{TXT_S}; line-height:1.6; }}
.login-banner {{
    background:{BG_CARD}; border:1px solid {BORDER}; border-radius:10px;
    padding:14px 20px; margin-bottom:18px; display:flex;
    align-items:center; justify-content:space-between; gap:12px;
}}
.site-footer {{ margin-top:56px; padding:22px 0 18px 0; border-top:1px solid {BORDER}; text-align:center; }}
.site-footer .footer-brand {{ font-family:'Syne',sans-serif; font-size:0.88rem; font-weight:600; color:{TXT_M}; }}
.site-footer .footer-brand span {{ color:#f5a623; }}
.site-footer .footer-tagline {{ font-family:'Space Mono',monospace; font-size:0.70rem; color:{TXT_M}; letter-spacing:2px; margin-top:4px; opacity:0.7; }}
#MainMenu, footer, header {{ visibility:hidden; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# 5. HEADER — always visible (guest + logged in)
# ─────────────────────────────────────────────────────────────────────
is_logged_in = st.session_state.logged_in
is_admin     = is_logged_in and _get_role(st.session_state.auth_email) == "admin"

col_logo, col_ctrl = st.columns([5, 2])
with col_logo:
    st.markdown(f"""
    <div class="hero">
        <div class="hero-title">Solar<span>Inspect</span> AI</div>
        <div class="hero-sub">{t('SOLAR PANEL DEFECT DETECTION', 'كشف عيوب الألواح الشمسية')}</div>
    </div>
    """, unsafe_allow_html=True)

with col_ctrl:
    lang_label = "AR" if st.session_state.lang == "en" else "EN"

    if is_logged_in:
        user_initial = st.session_state.auth_email[0].upper()
        short_email  = (st.session_state.auth_email[:18] + "…") if len(st.session_state.auth_email) > 18 else st.session_state.auth_email
        admin_tag    = ' &nbsp;<span style="color:#f5a623;font-weight:800;">ADMIN</span>' if is_admin else ""
        st.markdown(f"""
        <div style="display:flex;flex-direction:column;align-items:flex-end;padding-top:18px;gap:8px;">
            <div class="user-chip">
                <div class="avatar">{user_initial}</div>
                <span class="email">{short_email}{admin_tag}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        bc1, bc2 = st.columns(2)
        with bc1:
            if st.button(f"🌐 {lang_label}", use_container_width=True, key="btn_lang"):
                st.session_state.lang = "ar" if st.session_state.lang == "en" else "en"
                st.rerun()
        with bc2:
            if st.button("🚪 Log Out", use_container_width=True, key="btn_logout"):
                if st.session_state.session_token:
                    _delete_token(st.session_state.session_token)
                st.session_state.logged_in     = False
                st.session_state.auth_email    = ""
                st.session_state.session_token = ""
                st.session_state.history       = []
                st.rerun()
    else:
        # Guest — only language toggle in header
        _, bc1 = st.columns([1, 1])
        with bc1:
            if st.button(f"🌐 {lang_label}", use_container_width=True, key="btn_lang"):
                st.session_state.lang = "ar" if st.session_state.lang == "en" else "en"
                st.rerun()

# ─────────────────────────────────────────────────────────────────────
# 6. LOGIN MODAL (inline, only shown when needed)
# ─────────────────────────────────────────────────────────────────────
if not is_logged_in and st.session_state.get("_show_login", False):
    with st.container():
        st.markdown(f"""
        <div style="background:{BG_CARD};border:1px solid {BORDER};border-radius:12px;
             padding:24px;max-width:420px;margin:0 auto 24px auto;">
            <div style="font-size:1.2rem;font-weight:800;margin-bottom:16px;text-align:center;">
                ☀️ {t('Welcome to SolarInspect AI','مرحباً بك في SolarInspect AI')}
            </div>
        </div>
        """, unsafe_allow_html=True)
        _, mc, _ = st.columns([1, 2, 1])
        with mc:
            tl, ts = st.tabs([t("🔑 Log In","🔑 دخول"), t("✨ Sign Up","✨ تسجيل")])
            with tl:
                with st.form("login_form"):
                    em = st.text_input(t("Email","البريد"), placeholder="you@example.com", key="li_email")
                    pw = st.text_input(t("Password","كلمة المرور"), type="password", key="li_pw")
                    if st.form_submit_button(t("Log In","دخول"), use_container_width=True):
                        ok, msg, tok = auth_login(em, pw)
                        if ok:
                            st.session_state.logged_in     = True
                            st.session_state.auth_email    = em.lower().strip()
                            st.session_state.user_email    = em.lower().strip()
                            st.session_state.session_token = tok
                            st.session_state._show_login   = False
                            st.rerun()
                        else:
                            st.error(msg)
            with ts:
                with st.form("signup_form"):
                    ne  = st.text_input(t("Email","البريد"), placeholder="you@example.com", key="su_email")
                    np1 = st.text_input(t("Password (min 6)","كلمة المرور"), type="password", key="su_pw")
                    np2 = st.text_input(t("Confirm password","تأكيد كلمة المرور"), type="password", key="su_pw2")
                    if st.form_submit_button(t("Create Account","إنشاء حساب"), use_container_width=True):
                        if np1 != np2:
                            st.error(t("Passwords do not match.","كلمتا المرور غير متطابقتين."))
                        else:
                            ok2, msg2 = auth_signup(ne, np1)
                            st.success(msg2) if ok2 else st.error(msg2)

# ─────────────────────────────────────────────────────────────────────
# 7. GUEST BANNER — one clean bar with single login button
# ─────────────────────────────────────────────────────────────────────
if not is_logged_in:
    gb1, gb2 = st.columns([5, 1])
    with gb1:
        st.markdown(f"""
        <div style="background:{BG_CARD};border:1px solid {BORDER};border-radius:10px;
             padding:12px 20px;font-size:0.88rem;color:{TXT_M};">
            👁️ {t('Browsing as guest — scan and forecast work freely. Log in to save history and download reports.',
                   'تتصفح كضيف — الفحص والتوقع يعملان بحرية. سجّل دخولك لحفظ السجل وتحميل التقارير.')}
        </div>
        """, unsafe_allow_html=True)
    with gb2:
        if st.button(f"🔑 {t('Log In','دخول')}", use_container_width=True, key="btn_login_banner"):
            st.session_state._show_login = True
            st.rerun()

# ─────────────────────────────────────────────────────────────────────
# 8. DATA — defect info, model loaders
# ─────────────────────────────────────────────────────────────────────
CLASSES = ["Bird-drop","Clean","Dusty","Electrical-damage","Physical-damage","Snow-covered"]

DEFECT_INFO = {
    "Bird-drop":{"severity":"warning","display_en":"Bird Dropping","display_ar":"إفرازات الطيور","icon":"🐦",
        "desc_en":"Organic contamination causing localised shading and potential hot spots.",
        "desc_ar":"تلوث عضوي يسبب تظليلاً موضعياً ونقاط ساخنة محتملة.",
        "action_en":"Clean with soft cloth and water within 1-2 weeks.",
        "action_ar":"نظف بقطعة قماش ناعمة وماء خلال 1-2 أسبوع.",
        "urgency_en":"Clean within 2 weeks to prevent permanent staining",
        "urgency_ar":"نظف خلال أسبوعين لمنع التلطيخ الدائم",
        "tips_en":["Use distilled water for cleaning","Clean early morning or evening","Inspect for scratches after cleaning"],
        "tips_ar":["استخدم ماء مقطراً","نظف في الصباح الباكر أو المساء","افحص الخدوش بعد التنظيف"],
    },
    "Clean":{"severity":"info","display_en":"Clean Panel","display_ar":"لوح نظيف","icon":"✅",
        "desc_en":"No defects detected. Panel surface appears clean and undamaged.",
        "desc_ar":"لم يتم اكتشاف عيوب. سطح اللوح نظيف وغير تالف.",
        "action_en":"No action required. Continue routine inspection every 3-6 months.",
        "action_ar":"لا يلزم اتخاذ أي إجراء. استمر في الفحص الدوري كل 3-6 أشهر.",
        "urgency_en":"No action needed - next inspection in 3-6 months",
        "urgency_ar":"لا يلزم إجراء - الفحص القادم خلال 3-6 أشهر",
        "tips_en":["Schedule next inspection in 3 months","Monitor energy output monthly","Keep maintenance logs updated"],
        "tips_ar":["جدول الفحص القادم خلال 3 أشهر","راقب إنتاج الطاقة شهرياً","حافظ على تحديث السجلات"],
    },
    "Dusty":{"severity":"info","display_en":"Dust Accumulation","display_ar":"تراكم الغبار","icon":"🌫️",
        "desc_en":"Surface soiling reducing light transmission. Can reduce output by 5-30%.",
        "desc_ar":"تلوث السطح يقلل انتقال الضوء. يمكن أن يقلل الإنتاج بنسبة 5-30٪.",
        "action_en":"Schedule routine cleaning. In desert environments, clean every 2-4 weeks.",
        "action_ar":"جدول التنظيف الدوري. في البيئات الصحراوية، نظف كل 2-4 أسابيع.",
        "urgency_en":"Clean within 1 month for optimal efficiency",
        "urgency_ar":"نظف خلال شهر للحصول على أفضل كفاءة",
        "tips_en":["Use automated cleaning systems if available","Clean before panels heat up","Consider anti-soiling coatings"],
        "tips_ar":["استخدم أنظمة التنظيف التلقائي","نظف في الصباح الباكر أو مساءً","فكر في طلاءات مضادة للأوساخ"],
    },
    "Electrical-damage":{"severity":"critical","display_en":"Electrical Damage","display_ar":"تلف كهربائي","icon":"⚡",
        "desc_en":"Burn marks indicating arc faults or failed bypass diodes. Fire hazard risk.",
        "desc_ar":"علامات حرق تشير إلى أعطال القوس أو صمامات التحويل الفاشلة. خطر حريق.",
        "action_en":"Take panel offline immediately. Contact a certified PV technician.",
        "action_ar":"أوقف تشغيل اللوح فوراً. اتصل بفني PV معتمد.",
        "urgency_en":"URGENT - Take offline immediately, fire risk!",
        "urgency_ar":"عاجل - أوقف التشغيل فوراً، خطر حريق!",
        "tips_en":["Do NOT attempt DIY repair","Document the damage with photos","Check neighboring panels","Review system insurance"],
        "tips_ar":["لا تحاول الإصلاح بنفسك","وثق الضرر بالصور","افحص الألواح المجاورة","راجع تغطية التأمين"],
    },
    "Physical-damage":{"severity":"critical","display_en":"Physical Damage","display_ar":"تلف مادي","icon":"💥",
        "desc_en":"Delamination, frame damage, or cell breakage. Allows moisture ingress.",
        "desc_ar":"تقشر أو تلف الإطار أو كسر الخلايا. يسمح بدخول الرطوبة.",
        "action_en":"Replace panel as soon as possible. Moisture ingress accelerates degradation.",
        "action_ar":"استبدل اللوح في أقرب وقت ممكن. دخول الرطوبة يسرع التدهور.",
        "urgency_en":"Replace within 2 weeks - moisture ingress risk",
        "urgency_ar":"استبدل خلال أسبوعين - خطر دخول الرطوبة",
        "tips_en":["Cover damaged area temporarily","Check warranty for replacement","Inspect mounting structure","Order replacement panel"],
        "tips_ar":["غطِّ المنطقة التالفة مؤقتاً","تحقق من الضمان","افحص هيكل التركيب","اطلب لوحاً بديلاً"],
    },
    "Snow-covered":{"severity":"warning","display_en":"Snow Coverage","display_ar":"تغطية الثلج","icon":"❄️",
        "desc_en":"Snow accumulation blocking sunlight and reducing energy output.",
        "desc_ar":"تراكم الثلج يحجب أشعة الشمس ويقلل إنتاج الطاقة.",
        "action_en":"Remove snow carefully with a soft brush or squeegee.",
        "action_ar":"أزل الثلج بعناية باستخدام فرشاة ناعمة.",
        "urgency_en":"Remove snow within 24-48 hours",
        "urgency_ar":"أزل الثلج خلال 24-48 ساعة",
        "tips_en":["Never use hot water - thermal shock risk","Use a soft roof rake or foam brush","Allow panels to self-clear on sunny days"],
        "tips_ar":["لا تستخدم الماء الساخن","استخدم فرشاة ناعمة","اسمح للألواح بالتنظيف الذاتي"],
    },
}

BADGE = {
    "critical": ('<span class="badge-critical">', "CRITICAL"),
    "warning":  ('<span class="badge-warning">',  "WARNING"),
    "info":     ('<span class="badge-good">',      "GOOD"),
}

@st.cache_resource
def load_effnet():
    path = "best_efficientnet_b0.pth"
    if not os.path.exists(path):
        return None
    m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=6)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m

@st.cache_resource
def load_perf_model():
    files = ["performance_model.pkl","performance_scaler.pkl","model_metadata.json","typical_values.json"]
    if any(not os.path.exists(f) for f in files):
        return None, None, None, None
    try:
        model  = joblib.load("performance_model.pkl")
        scaler = joblib.load("performance_scaler.pkl")
        with open("model_metadata.json") as f: meta    = json.load(f)
        with open("typical_values.json")  as f: typical = json.load(f)
        return model, scaler, meta, typical
    except Exception:
        return None, None, None, None

@st.cache_resource
def load_lstm():
    required = ["lstm_model.keras","lstm_scaler.pkl","lstm_metadata.json","sample_data.csv"]
    if any(not os.path.exists(f) for f in required):
        return None, None, None, None
    try:
        import tensorflow as tf
        model  = tf.keras.models.load_model("lstm_model.keras")
        scaler = joblib.load("lstm_scaler.pkl")
        with open("lstm_metadata.json") as f: meta = json.load(f)
        sample = pd.read_csv("sample_data.csv")
        return model, scaler, meta, sample
    except Exception:
        return None, None, None, None

effnet_model                                     = load_effnet()
perf_model, perf_scaler, perf_meta, typical_vals = load_perf_model()
lstm_model, lstm_scaler, lstm_meta, sample_data  = load_lstm()

def preprocess_image(image):
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return tf(image).unsqueeze(0)

# ─────────────────────────────────────────────────────────────────────
# 9. PDF
# ─────────────────────────────────────────────────────────────────────
AMIRI_PATH = os.path.join("fonts", "Amiri-Regular.ttf")

def _safe_en(text):
    return "".join(c for c in str(text) if _ud.category(c) not in ("So","Cs") and ord(c) < 0x0250).strip()

def _has_amiri():
    return os.path.exists(AMIRI_PATH)

def _ar(text):
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        return get_display(arabic_reshaper.reshape(str(text)))
    except ImportError:
        return str(text)

_AR_SECTIONS = {
    "Detection Result":"نتيجة الكشف","Description":"الوصف",
    "Recommended Action":"الإجراء الموصى به","Maintenance Tips":"نصائح الصيانة",
}

class SolarPDF(FPDF):
    def __init__(self, lang="en"):
        super().__init__()
        self.lang=lang; self.amiri_ok=_has_amiri()
        self.set_margins(18,18,18); self.set_auto_page_break(auto=True,margin=20)
        if self.amiri_ok: self.add_font("Amiri","",AMIRI_PATH,uni=True)
    @property
    def W(self): return self.w-self.l_margin-self.r_margin
    def header(self):
        self.set_fill_color(255,255,255); self.rect(0,0,self.w,18,"F")
        self.set_y(3); self.set_font("Helvetica","B",18); self.set_text_color(245,166,35)
        self.cell(0,10,"SolarInspect AI",align="C"); self.ln(10)
        self.set_draw_color(245,166,35); self.set_line_width(0.6)
        self.line(self.l_margin,self.get_y(),self.w-self.r_margin,self.get_y())
        self.set_line_width(0.2); self.set_draw_color(200,210,220); self.set_text_color(0,0,0); self.ln(6)
    def footer(self):
        self.set_y(-12); self.set_font("Helvetica","I",8); self.set_text_color(180,190,200)
        self.cell(0,8,"SolarInspect AI",align="C"); self.set_text_color(0,0,0)
    def section_header(self,title_en):
        self.ln(4); self.set_fill_color(26,40,60); self.set_text_color(245,166,35)
        if self.lang=="ar" and self.amiri_ok:
            self.set_font("Amiri","",12)
            self.cell(self.W,10,_ar(_AR_SECTIONS.get(title_en,title_en)),ln=True,fill=True,align="R")
        else:
            self.set_font("Helvetica","B",11); self.cell(self.W,9,f"  {title_en}",ln=True,fill=True)
        self.set_text_color(0,0,0); self.ln(3)
    def hr(self):
        self.set_draw_color(200,210,220)
        self.line(self.l_margin,self.get_y(),self.l_margin+self.W,self.get_y()); self.ln(4)
    def body_line(self,label_en,value_en,value_ar="",label_ar=""):
        if self.lang=="ar" and self.amiri_ok:
            self.set_font("Amiri","",10); self.set_text_color(100,120,140)
            self.cell(self.W,7,_ar(label_ar or label_en),ln=True,align="R")
            self.set_font("Amiri","",12); self.set_text_color(20,20,20)
            self.set_x(self.l_margin); self.cell(self.W,8,_ar(value_ar or value_en),ln=True,align="R")
            self.ln(3)
        else:
            self.set_font("Helvetica","B",10); self.set_text_color(60,80,100)
            self.cell(55,7,f"{label_en}:",ln=False)
            self.set_font("Helvetica","",10); self.set_text_color(20,20,20)
            self.cell(self.W-55,7,_safe_en(value_en),ln=True)
        self.set_text_color(0,0,0)
    def body_para(self,text_en,text_ar=""):
        self.set_text_color(30,30,30)
        if self.lang=="ar" and text_ar and self.amiri_ok:
            self.set_font("Amiri","",11); self.multi_cell(self.W,7,_ar(text_ar),align="R")
        else:
            self.set_font("Helvetica","",10); self.multi_cell(self.W,7,_safe_en(text_en))
        self.set_text_color(0,0,0); self.ln(2)
    def info_box(self,text_en,text_ar,severity="info"):
        r,g,b={"critical":(210,60,50),"warning":(200,140,30),"info":(40,160,110)}.get(severity,(80,120,180))
        self.set_fill_color(r,g,b); self.set_text_color(255,255,255)
        if self.lang=="ar" and text_ar and self.amiri_ok:
            self.set_font("Amiri","",11); self.multi_cell(self.W,8,_ar(text_ar),fill=True,align="R")
        else:
            self.set_font("Helvetica","B",10); self.multi_cell(self.W,8,f"  {_safe_en(text_en)}  ",fill=True,align="C")
        self.set_text_color(0,0,0); self.ln(3)
    def tips_table(self,tips_en,tips_ar):
        use_ar=self.lang=="ar" and tips_ar and self.amiri_ok
        tips=tips_ar if use_ar else tips_en
        for i,tip in enumerate(tips):
            self.set_fill_color(*(240,244,250) if i%2==0 else (255,255,255)); self.set_text_color(30,30,30)
            if use_ar:
                self.set_font("Amiri","",11); self.cell(self.W,9,_ar(str(tip)),ln=True,fill=True,align="R")
            else:
                self.set_font("Helvetica","",10); self.cell(self.W,9,f"  {i+1}. {_safe_en(tip)}",ln=True,fill=True)
        self.set_text_color(0,0,0); self.ln(2)

def generate_pdf(pred_class,confidence,info,lang="en",user_email="",underperf=None):
    is_ar=(lang=="ar"); pdf=SolarPDF(lang=lang); pdf.add_page()
    if is_ar and pdf.amiri_ok:
        pdf.set_font("Amiri","",16); pdf.set_text_color(26,40,60)
        pdf.cell(pdf.W,12,_ar("تقرير الفحص"),ln=True,align="C")
        pdf.set_font("Amiri","",10); pdf.set_text_color(100,120,140)
        pdf.cell(pdf.W,6,datetime.now().strftime("%Y-%m-%d  %H:%M"),ln=True,align="C")
        if user_email: pdf.cell(pdf.W,6,_ar(f"الحساب: {user_email}"),ln=True,align="C")
    else:
        pdf.set_font("Helvetica","B",18); pdf.set_text_color(26,40,60)
        pdf.cell(pdf.W,12,"Scan Report",ln=True,align="C")
        pdf.set_font("Helvetica","",10); pdf.set_text_color(100,120,140)
        pdf.cell(pdf.W,6,datetime.now().strftime("%A, %d %B %Y  ·  %H:%M"),ln=True,align="C")
        if user_email: pdf.cell(pdf.W,6,f"Account: {user_email}",ln=True,align="C")
    pdf.set_text_color(0,0,0); pdf.ln(6); pdf.hr()
    pdf.section_header("Detection Result")
    pdf.body_line("Defect Type",info["display_en"],info["display_ar"],"نوع العيب")
    pdf.body_line("Confidence",f"{confidence:.1%}",f"{confidence:.1%}","مستوى الثقة")
    pdf.body_line("Severity",info["severity"].upper(),info["severity"].upper(),"مستوى الخطورة")
    if underperf is not None: pdf.body_line("Underperformance",f"{underperf:.1f}%",f"{underperf:.1f}%","الأداء دون المستوى")
    pdf.ln(4)
    pdf.info_box(f"Urgency: {info['urgency_en']}",f"مهم: {info['urgency_ar']}",severity=info["severity"])
    pdf.section_header("Description"); pdf.body_para(info["desc_en"],info["desc_ar"])
    pdf.section_header("Recommended Action"); pdf.body_para(info["action_en"],info["action_ar"])
    pdf.section_header("Maintenance Tips"); pdf.tips_table(info["tips_en"],info["tips_ar"])
    pdf.ln(6); pdf.hr(); pdf.set_font("Helvetica","I",8); pdf.set_text_color(150,160,170)
    if is_ar and pdf.amiri_ok:
        pdf.set_font("Amiri","",9)
        pdf.multi_cell(pdf.W,5,_ar("تم إنشاء هذا التقرير تلقائياً بواسطة SolarInspect AI. استشر دائماً فني PV معتمد قبل اتخاذ أي إجراء."),align="R")
    else:
        pdf.multi_cell(pdf.W,5,"This report was generated automatically by SolarInspect AI. Always consult a certified PV technician before taking corrective action.",align="C")
    return bytes(pdf.output())

# ─────────────────────────────────────────────────────────────────────
# 10. TABS
# ─────────────────────────────────────────────────────────────────────
if is_admin:
    tab1,tab2,tab3,tab4,tab5 = st.tabs([
        t("🔍 Image Scan","🔍 فحص الصورة"),
        t("📊 Performance","📊 الأداء"),
        t("📈 Power Forecast","📈 توقع الطاقة"),
        t("📋 History","📋 السجل"),
        t("📂 Dataset","📂 البيانات"),
    ])
else:
    tab1,tab2,tab3,tab4 = st.tabs([
        t("🔍 Image Scan","🔍 فحص الصورة"),
        t("📊 Performance","📊 الأداء"),
        t("📈 Power Forecast","📈 توقع الطاقة"),
        t("📋 History","📋 السجل"),
    ])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — IMAGE SCAN
# ═══════════════════════════════════════════════════════════════
with tab1:
    if effnet_model is None:
        st.error(t("Model file not found: best_efficientnet_b0.pth","ملف النموذج غير موجود."))
    else:
        uploaded = st.file_uploader(
            t("Upload solar panel image","رفع صورة اللوح الشمسي"),
            type=["jpg","jpeg","png"], label_visibility="collapsed",
        )
        if uploaded is None:
            st.markdown(f"""
            <div class="upload-zone">
                <div style="font-size:3rem;margin-bottom:16px;">☀️</div>
                <div style="font-size:1.1rem;font-weight:600;color:{TXT_M};">
                    {t('Drop a solar panel image to scan for defects','أضف صورة اللوح الشمسي للكشف عن العيوب')}
                </div>
                <div style="font-family:Space Mono,monospace;font-size:0.78rem;color:{TXT_M};margin-top:8px;">JPG · JPEG · PNG</div>
            </div>""", unsafe_allow_html=True)
        else:
            image = Image.open(uploaded).convert("RGB")
            with st.spinner(t("Scanning panel...","جاري فحص اللوح...")):
                tensor = preprocess_image(image)
                with torch.no_grad():
                    probs = torch.nn.functional.softmax(effnet_model(tensor), dim=1)[0].numpy()

            pred_idx   = int(np.argmax(probs))
            pred_class = CLASSES[pred_idx]
            confidence = float(probs[pred_idx])
            info       = DEFECT_INFO[pred_class]
            display    = info["display_ar"] if IS_AR else info["display_en"]
            sev        = info["severity"]

            import hashlib as _hl
            file_hash    = _hl.md5(uploaded.getvalue()).hexdigest()
            _is_new_scan = is_logged_in and file_hash not in st.session_state.saved_hashes

            col_img, col_res = st.columns([3,2], gap="large")
            with col_img:
                st.markdown(f'<div class="section-title">{t("UPLOADED IMAGE","الصورة المرفوعة")}</div>', unsafe_allow_html=True)
                st.image(image, use_container_width=True)
            with col_res:
                st.markdown(f'<div class="section-title">{t("SEVERITY","مستوى الخطورة")}</div>', unsafe_allow_html=True)
                badge_open, _ = BADGE[sev]
                st.markdown(f'{badge_open}{info["icon"]} {display}</span>', unsafe_allow_html=True)
                st.markdown(f'<div class="section-title" style="margin-top:20px;">{t("CONFIDENCE","مستوى الثقة")}</div>', unsafe_allow_html=True)
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-label">{t("MODEL CONFIDENCE","ثقة النموذج")}</div>
                    <div class="metric-value" style="color:#f5a623;">{confidence:.0%}</div>
                    <div class="conf-bar-bg"><div class="conf-bar-fill" style="width:{int(confidence*100)}%"></div></div>
                </div>""", unsafe_allow_html=True)
                st.markdown(f'<div class="section-title">{t("ANALYSIS","التحليل")}</div>', unsafe_allow_html=True)
                desc    = info["desc_ar"]    if IS_AR else info["desc_en"]
                action  = info["action_ar"]  if IS_AR else info["action_en"]
                urgency = info["urgency_ar"] if IS_AR else info["urgency_en"]
                st.markdown(f"""<div class="defect-card {sev}">
                    <div class="defect-name">{info['icon']} {display}</div>
                    <div class="defect-desc">{desc}</div>
                    <div class="defect-action">→ {action}</div>
                    <div class="urgency-box">{urgency}</div>
                </div>""", unsafe_allow_html=True)

            # ── SENSOR PARAMETERS — always visible, only saved if logged in
            st.markdown(f'<div class="section-title">{t("PANEL SENSOR PARAMETERS — OPTIONAL","معطيات المستشعر — اختياري")}</div>', unsafe_allow_html=True)
            st.markdown(f"""<div class="perf-card" style="margin-bottom:12px;">
                <div style="font-size:0.82rem;color:{TXT_M};">
                    📡 {t('Fill in any sensor readings you have. All fields are optional — leave at 0 to skip.',
                           'أدخل أي قراءات متوفرة. جميع الحقول اختيارية — اترك 0 للتخطي.')}
                    {'' if is_logged_in else f' <b style="color:#f5a623;">{t("Log in to save these readings to your history.","سجّل دخولك لحفظ هذه القراءات في سجلك.")}</b>'}
                </div>
            </div>""", unsafe_allow_html=True)

            sp1, sp2, sp3, sp4 = st.columns(4)
            with sp1: s_irr = st.number_input(t("Irradiation (W/m²/1000)","الإشعاع"), min_value=0.0, max_value=2.0,     value=0.0, step=0.01, key="s_irr", format="%.3f")
            with sp2: s_amb = st.number_input(t("Ambient Temp (°C)","حرارة المحيط"),  min_value=-20.0,max_value=60.0,   value=0.0, step=0.1,  key="s_amb", format="%.1f")
            with sp3: s_mod = st.number_input(t("Module Temp (°C)","حرارة اللوح"),    min_value=-20.0,max_value=90.0,   value=0.0, step=0.1,  key="s_mod", format="%.1f")
            with sp4: s_dc  = st.number_input(t("DC Power (kW)","طاقة DC"),           min_value=0.0, max_value=500000.0,value=0.0, step=0.1,  key="s_dc",  format="%.2f")
            sp5, sp6, sp7, sp8 = st.columns(4)
            with sp5: s_ac  = st.number_input(t("AC Power (kW)","طاقة AC"),           min_value=0.0, max_value=500000.0,value=0.0, step=0.1,  key="s_ac",  format="%.2f")
            with sp6: s_eff = st.number_input(t("Efficiency (%)","الكفاءة %"),        min_value=0.0, max_value=100.0,   value=0.0, step=0.1,  key="s_eff", format="%.1f")
            with sp7: s_cap = st.number_input(t("Panel Capacity (kW)","سعة اللوح"),   min_value=0.0, max_value=1000.0,  value=0.0, step=0.1,  key="s_cap", format="%.2f")
            with sp8: s_age = st.number_input(t("Panel Age (years)","عمر اللوح"),     min_value=0,   max_value=50,      value=0,   step=1,    key="s_age")

            # Save to DB only if logged in
            if _is_new_scan:
                db_save_scan(
                    st.session_state.auth_email, pred_class, info, confidence,
                    irradiation=s_irr if s_irr>0 else None, ambient_temp=s_amb if s_amb>0 else None,
                    module_temp=s_mod if s_mod>0 else None, dc_power=s_dc if s_dc>0 else None,
                    ac_power=s_ac if s_ac>0 else None, efficiency_pct=s_eff if s_eff>0 else None,
                    panel_capacity=s_cap if s_cap>0 else None, panel_age=s_age if s_age>0 else None,
                )
                st.session_state.saved_hashes.add(file_hash)
                st.toast(t("✅ Scan saved to your history","✅ تم حفظ الفحص في سجلك"), icon="✅")

            # ── MAINTENANCE TIPS
            st.markdown(f'<div class="section-title">{t("MAINTENANCE TIPS","نصائح الصيانة")}</div>', unsafe_allow_html=True)
            tips = info["tips_ar"] if IS_AR else info["tips_en"]
            tip_cols = st.columns(len(tips))
            for i, tip in enumerate(tips):
                with tip_cols[i]:
                    st.markdown(f'<div class="tip-card"><div style="font-size:1.2rem;margin-bottom:6px;">💡</div><div class="tip-text">{tip}</div></div>', unsafe_allow_html=True)

            # ── PDF EXPORT — locked for guests
            st.markdown(f'<div class="section-title">{t("EXPORT REPORT","تصدير التقرير")}</div>', unsafe_allow_html=True)
            if not is_logged_in:
                st.markdown(
                    f'<div style="background:{BG_CARD};border:1px solid {BORDER};border-radius:8px;'
                    f'padding:14px 18px;display:flex;align-items:center;justify-content:space-between;gap:16px;">'
                    f'<span style="color:{TXT_M};font-size:0.88rem;">🔒 {t("Log in to generate and download your PDF report.","سجّل دخولك لإنشاء تقرير PDF وتحميله.")}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                if st.button(f"🔑 {t('Log in to download report','سجّل دخولك لتحميل التقرير')}", key="btn_login_pdf"):
                    st.session_state._show_login = True
                    st.rerun()
            else:
                if st.button(t("📄 Generate PDF Report","📄 إنشاء تقرير PDF"), key="gen_pdf"):
                    with st.spinner(t("Generating report...","جاري إنشاء التقرير...")):
                        pdf_bytes = generate_pdf(pred_class, confidence, info,
                            lang=st.session_state.lang, user_email=st.session_state.auth_email)
                    st.download_button(
                        label=t("⬇️ Download Report","⬇️ تحميل التقرير"),
                        data=pdf_bytes,
                        file_name=f"solar_scan_{_now_cairo().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf", key="dl_pdf",
                    )

# ═══════════════════════════════════════════════════════════════
# TAB 2 — PERFORMANCE ANALYZER
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f'<div class="section-title">{t("PANEL PERFORMANCE ANALYZER","محلل أداء اللوح")}</div>', unsafe_allow_html=True)
    if perf_model is None:
        st.warning(t("Performance model files not found.","ملفات نموذج الأداء غير موجودة."))
    else:
        tv = typical_vals
        c1,c2,c3 = st.columns(3)
        with c1: irradiation  = st.number_input(t("Irradiation (W/m2/1000)","الإشعاع"), min_value=0.0, max_value=2.0,   value=float(round(tv["IRRADIATION"],3)),          step=0.01)
        with c2: ambient_temp = st.number_input(t("Ambient Temp (C)","درجة المحيط"),    min_value=-10.0,max_value=60.0, value=float(round(tv["AMBIENT_TEMPERATURE"],1)), step=0.1)
        with c3: module_temp  = st.number_input(t("Module Temp (C)","درجة اللوح"),      min_value=-10.0,max_value=90.0, value=float(round(tv["MODULE_TEMPERATURE"],1)),  step=0.1)
        c4,c5 = st.columns(2)
        with c4: dc_power = st.number_input(t("DC Power (kW)","طاقة DC"), min_value=0.0, max_value=500000.0, value=float(round(tv["DC_POWER"],1)),  step=100.0)
        with c5: ac_power = st.number_input(t("AC Power (kW)","طاقة AC"), min_value=0.0, max_value=500000.0, value=float(round(tv["AC_POWER"],1)), step=100.0)
        if st.button(t("Analyze Performance","تحليل الأداء"), use_container_width=True):
            if irradiation == 0:
                st.info(t("Irradiation is 0 — panel is not generating power.","الإشعاع = 0 — اللوح لا ينتج طاقة."))
            else:
                features    = np.array([[irradiation, ambient_temp, module_temp, datetime.now().hour]])
                features_sc = perf_scaler.transform(features)
                expected_ac = perf_model.predict(features_sc)[0]
                underperf_pct = max(0,(expected_ac-ac_power)/(expected_ac+1e-6)*100)
                dc_ac_eff     = (ac_power/(dc_power+1e-6))*100
                r1,r2,r3 = st.columns(3)
                with r1:
                    col="#2ecc71" if underperf_pct<10 else "#f5a623" if underperf_pct<25 else "#e74c3c"
                    st.markdown(f'<div class="metric-card" style="text-align:center;"><div class="metric-label">{t("UNDERPERFORMANCE","الأداء دون المستوى")}</div><div class="metric-value" style="color:{col};">-{underperf_pct:.1f}%</div></div>', unsafe_allow_html=True)
                with r2:
                    st.markdown(f'<div class="metric-card" style="text-align:center;"><div class="metric-label">{t("EXPECTED AC","طاقة AC المتوقعة")}</div><div class="metric-value" style="color:#f5a623;">{expected_ac:,.0f}</div><div style="font-size:0.8rem;color:{TXT_M};">kW</div></div>', unsafe_allow_html=True)
                with r3:
                    ec="#2ecc71" if dc_ac_eff>90 else "#f5a623" if dc_ac_eff>75 else "#e74c3c"
                    st.markdown(f'<div class="metric-card" style="text-align:center;"><div class="metric-label">{t("DC→AC EFFICIENCY","كفاءة DC→AC")}</div><div class="metric-value" style="color:{ec};">{dc_ac_eff:.1f}%</div></div>', unsafe_allow_html=True)
                if underperf_pct<5:   sc,si,msg_en,msg_ar="#2ecc71","✅",f"Performing normally — {underperf_pct:.1f}% below expected.",f"يعمل بشكل طبيعي — {underperf_pct:.1f}٪."
                elif underperf_pct<15: sc,si,msg_en,msg_ar="#f5a623","⚠️",f"Slightly underperforming by {underperf_pct:.1f}%.",f"أداء أقل بنسبة {underperf_pct:.1f}٪."
                elif underperf_pct<30: sc,si,msg_en,msg_ar="#e67e22","⚠️",f"Underperforming by {underperf_pct:.1f}%.",f"أداء أقل بنسبة {underperf_pct:.1f}٪."
                else:                  sc,si,msg_en,msg_ar="#e74c3c","🚨",f"Severely underperforming by {underperf_pct:.1f}%!",f"أداء أقل بشكل حاد بنسبة {underperf_pct:.1f}٪!"
                st.markdown(f'<div style="background:{BG_CARD};border:1px solid {sc};border-radius:10px;padding:16px 20px;margin-top:16px;"><div style="font-size:1.05rem;font-weight:800;color:{sc};">{si} {msg_ar if IS_AR else msg_en}</div></div>', unsafe_allow_html=True)
                perf_pct=max(0,100-underperf_pct)
                st.markdown(f'<div style="margin-top:16px;"><div style="font-family:Space Mono,monospace;font-size:0.72rem;color:{TXT_M};letter-spacing:3px;margin-bottom:8px;">{t("PERFORMANCE GAUGE","مقياس الأداء")}</div><div style="background:{BAR_BG};border-radius:8px;height:16px;"><div style="width:{perf_pct:.0f}%;height:16px;border-radius:8px;background:linear-gradient(90deg,#e74c3c,#f5a623,#2ecc71);"></div></div><div style="display:flex;justify-content:space-between;font-size:0.75rem;color:{TXT_M};margin-top:4px;"><span>0%</span><span>{t("Performance","الأداء")}: {perf_pct:.0f}%</span><span>100%</span></div></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 3 — POWER FORECAST
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown(f'<div class="section-title">{t("SOLAR POWER FORECAST","توقع الطاقة الشمسية")}</div>', unsafe_allow_html=True)
    f1,f2,f3 = st.columns(3)
    with f1: f_irr = st.slider(t("Irradiation","الإشعاع"),       0.0,1.5,0.7,0.05, key="fc_irr")
    with f2: f_amb = st.slider(t("Ambient Temp (C)","درجة المحيط"),15.0,45.0,28.0,0.5, key="fc_amb")
    with f3: f_mod = st.slider(t("Module Temp (C)","درجة اللوح"), 20.0,70.0,40.0,0.5, key="fc_mod")
    steps = st.slider(t("Forecast hours ahead","ساعات التوقع"),1,12,6, key="fc_steps")
    if st.button(t("Generate Forecast","توليد التوقع"), use_container_width=True):
        with st.spinner(t("Generating forecast...","جاري توليد التوقع...")):
            forecasts, hours = [], []
            PANEL_CAPACITY_KW = 3.5
            INVERTER_EFF      = 0.96
            TEMP_COEFF        = -0.004
            now        = datetime.now()
            # Start from the NEXT 15-min slot so all points are in the future
            mins_past  = now.minute % 15
            start_offset = (15 - mins_past) if mins_past > 0 else 15
            base_dt    = now.replace(second=0, microsecond=0)

            for step in range(steps * 4):
                future_dt     = base_dt + __import__('datetime').timedelta(minutes=start_offset + step * 15)
                hour_of_day   = future_dt.hour
                minute_of_hour = future_dt.minute
                if 6 <= hour_of_day <= 19:
                    angle        = (hour_of_day - 6 + minute_of_hour / 60) / 13 * np.pi
                    solar_factor = max(0.0, float(np.sin(angle)))
                else:
                    solar_factor = 0.0
                temp_factor = 1 + TEMP_COEFF * max(0, f_mod - 25)
                ac_power_fc = max(0.0, f_irr * PANEL_CAPACITY_KW * solar_factor * temp_factor * INVERTER_EFF + float(np.random.normal(0, 0.01)))
                forecasts.append(round(ac_power_fc, 3))
                hours.append(future_dt.strftime("%H:%M"))
        import plotly.graph_objects as go
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=hours,y=forecasts,mode="lines+markers",
            line=dict(color="#f5a623",width=3),marker=dict(size=6,color="#f5a623"),
            fill="tozeroy",fillcolor="rgba(245,166,35,0.1)"))
        fig.update_layout(height=400,paper_bgcolor=BG_CARD,plot_bgcolor=BG_CARD,
            font=dict(color=TXT),xaxis=dict(gridcolor=BORDER,color=TXT_M),
            yaxis=dict(gridcolor=BORDER,color=TXT_M,title=t("AC Power (kW)","طاقة AC")),
            margin=dict(l=20,r=20,t=20,b=40))
        st.plotly_chart(fig, use_container_width=True, key="fc_chart")
        s1,s2,s3=st.columns(3)
        for col,le,la,val,unit,color in [(s1,"AVG POWER","متوسط الطاقة",np.mean(forecasts),"kW","#f5a623"),(s2,"PEAK POWER","ذروة الطاقة",np.max(forecasts),"kW","#e74c3c"),(s3,"EST. ENERGY","الطاقة المتوقعة",sum(forecasts)*0.25/1000,"MWh","#2ecc71")]:
            with col:
                st.markdown(f'<div class="metric-card" style="text-align:center;"><div class="metric-label">{la if IS_AR else le}</div><div class="metric-value" style="color:{color};">{val:,.2f}</div><div style="font-size:0.8rem;color:{TXT_M};margin-top:4px;">{unit}</div></div>', unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 4 — HISTORY
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f'<div class="section-title">{t("SCAN HISTORY","سجل الفحص")}</div>', unsafe_allow_html=True)
    if not is_logged_in:
        st.markdown(f"""
        <div style="background:{BG_CARD};border:1px solid {BORDER};border-radius:12px;
             padding:40px;text-align:center;">
            <div style="font-size:2.5rem;margin-bottom:12px;">📋</div>
            <div style="font-size:1.05rem;font-weight:700;color:{TXT};margin-bottom:8px;">
                {t('Your scan history will appear here','سجل فحوصاتك سيظهر هنا')}
            </div>
            <div style="font-size:0.88rem;color:{TXT_M};margin-bottom:20px;">
                {t('Log in to see all your past scans, defect history, and statistics.','سجّل دخولك لعرض جميع فحوصاتك السابقة.')}
            </div>
        </div>""", unsafe_allow_html=True)
        if st.button(f"🔑 {t('Log In','دخول')}", key="btn_login_history"):
            st.session_state._show_login = True
            st.rerun()
    else:
        try:
            user_history = get_full_history(st.session_state.auth_email, admin=is_admin)
            user_history = user_history[:30]  # last 30 scans only
        except Exception as e:
            st.error(f"Could not load history: {e}")
            user_history = []

        # Build email list for filter (all users if admin)
        if is_admin:
            all_emails = sorted(set(h["email"] for h in user_history if h["email"]))
            filter_opts = [t("All Users","جميع المستخدمين")] + all_emails
            sel_user = st.selectbox(t("Filter by user","تصفية حسب المستخدم"), filter_opts, key="hist_filter")
            if sel_user not in ("All Users", "جميع المستخدمين"):
                user_history = [h for h in user_history if h["email"] == sel_user]

        if not user_history:
            st.markdown(f'<div style="text-align:center;color:{TXT_M};padding:40px;"><div style="font-size:2.5rem;margin-bottom:12px;">📋</div><div>{t("No scans yet.","لا توجد فحوصات بعد.")}</div></div>', unsafe_allow_html=True)
        else:
            total    = len(user_history)
            critical = sum(1 for h in user_history if h["severity"] == "critical")
            warning  = sum(1 for h in user_history if h["severity"] == "warning")
            good     = sum(1 for h in user_history if h["severity"] == "info")

            s1, s2, s3, s4 = st.columns(4)
            for col, le, la, val, color in [
                (s1, "TOTAL",    "الإجمالي",  total,    "#f5a623"),
                (s2, "CRITICAL", "حرج",       critical, "#e74c3c"),
                (s3, "WARNINGS", "تحذيرات",   warning,  "#f5a623"),
                (s4, "GOOD",     "جيد",       good,     "#2ecc71"),
            ]:
                with col:
                    st.markdown(f'<div class="metric-card" style="text-align:center;"><div class="metric-label">{la if IS_AR else le}</div><div class="metric-value" style="color:{color};">{val}</div></div>', unsafe_allow_html=True)

            for h in user_history:
                disp_h    = h.get("display_ar", "") if IS_AR else h.get("display_en", "")
                sev       = h.get("severity", "info")
                sev_color = {"critical": "#e74c3c", "warning": "#f5a623", "info": "#2ecc71"}.get(sev, "#aaa")
                sev_lbl   = {"critical": "حرج", "warning": "تحذير", "info": "جيد"}.get(sev, sev).upper() if IS_AR else sev.upper()
                conf      = h.get("confidence", 0)
                conf_pct  = f"{conf:.0%}" if conf <= 1.0 else f"{conf:.0f}%"
                icon      = h.get("icon", "🔍")
                timestamp = h.get("time", "—")
                email_lbl = f'<span style="color:{TXT_M};font-size:0.78rem;">👤 {h["email"]} · </span>' if is_admin else ""

                st.markdown(
                    f'<div class="history-card">'
                    f'<div style="display:flex;justify-content:space-between;align-items:center;gap:12px;">'
                    f'<div>'
                    f'<span style="font-size:1.1rem;">{icon}</span>'
                    f'<span style="font-weight:700;margin-left:8px;color:{TXT};">{disp_h}</span>'
                    f'<span style="margin-left:10px;color:{TXT_M};font-size:0.8rem;">{conf_pct} {t("confidence","ثقة")}</span>'
                    f'</div>'
                    f'<div style="text-align:right;flex-shrink:0;">'
                    f'{email_lbl}'
                    f'<span style="color:{sev_color};font-size:0.8rem;font-weight:700;">{sev_lbl}</span>'
                    f'<span style="color:{TXT_M};font-size:0.73rem;"> · {timestamp}</span>'
                    f'</div>'
                    f'</div></div>',
                    unsafe_allow_html=True,
                )

            if st.button(t("🗑 Clear My History", "🗑 مسح سجلي"), key="clear_hist"):
                target = st.session_state.get("hist_filter", st.session_state.auth_email)
                if not is_admin or target in ("All Users", "جميع المستخدمين"):
                    target = st.session_state.auth_email
                db_delete_scans(target)
                st.rerun()

# ═══════════════════════════════════════════════════════════════
# TAB 5 — DATASET (ADMIN ONLY)
# ═══════════════════════════════════════════════════════════════
if is_admin:
    with tab5:
        st.markdown(f"""
        <div style="background:{BG_CARD};border:1px solid #f5a623;border-radius:12px;padding:16px 20px;margin-bottom:20px;">
            <div style="font-size:0.8rem;color:#f5a623;font-family:Space Mono,monospace;letter-spacing:2px;margin-bottom:6px;">👑 ADMIN PANEL</div>
            <div style="font-size:0.92rem;color:{TXT_S};">Full access to all users' data, scans, and the dataset browser.</div>
        </div>""", unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">REGISTERED USERS</div>', unsafe_allow_html=True)
        all_users = db_get_all_users()
        for u in all_users:
            role_color="#f5a623" if u["role"]=="admin" else TXT_M
            st.markdown(f'<div class="history-card"><div style="display:flex;justify-content:space-between;"><div style="color:{TXT};">📧 {u["email"]}</div><div><span style="color:{role_color};font-size:0.82rem;font-weight:700;">{u["role"].upper()}</span><span style="color:{TXT_M};font-size:0.78rem;margin-left:12px;">joined {u["created"][:10]}</span></div></div></div>', unsafe_allow_html=True)
        st.markdown(f'<div class="section-title">DATASET BROWSER</div>', unsafe_allow_html=True)
        render_dataset_tab(TXT=TXT,TXT_M=TXT_M,TXT_S=TXT_S,BG_CARD=BG_CARD,BORDER=BORDER,BAR_BG=BAR_BG,IS_AR=IS_AR,DM=DM)

# ─────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="site-footer">
    <div class="footer-brand">☀️ &nbsp;@Solar<span>Inspect</span> AI 2026</div>
    <div class="footer-tagline">{t("SOLAR PANEL DEFECT DETECTION PLATFORM","منصة الكشف عن عيوب الألواح الشمسية")}</div>
</div>""", unsafe_allow_html=True)
