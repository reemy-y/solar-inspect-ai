
# ─────────────────────────────────────────────────────────────────────
# 1. IMPORTS & PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────
import streamlit as st
import torch, timm, numpy as np, os, re, json, joblib, pandas as pd
import unicodedata as _ud, sqlite3, hashlib
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
# 2. AUTH & DATABASE — persistent sessions + role-based access
# ─────────────────────────────────────────────────────────────────────
import secrets as _secrets

DB_PATH    = "users.db"
ADMIN_EMAIL = "reemya185@gmail.com"   # ← only this email gets admin access

# ── Session token cookie name
TOKEN_KEY  = "solar_session_token"

def _db():
    """
    Connect to SQLite and ensure all tables exist.

    Schema:
      users  — stores accounts (email, hashed password, role, created_at)
      scans  — stores every scan result (linked to user by email)
      tokens — stores persistent session tokens (login stays alive)
    """
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")   # safer for concurrent access

    # Users table — role is either "admin" or "user"
    conn.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            email      TEXT    UNIQUE NOT NULL,
            pw_hash    TEXT    NOT NULL,
            role       TEXT    NOT NULL DEFAULT 'user',
            created_at TEXT    NOT NULL
        )
    """)

    # Scans table — every scan result saved here permanently
    conn.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            email       TEXT    NOT NULL,
            scanned_at  TEXT    NOT NULL,
            defect_type TEXT    NOT NULL,
            display_en  TEXT    NOT NULL,
            display_ar  TEXT    NOT NULL,
            confidence  REAL    NOT NULL,
            severity    TEXT    NOT NULL,
            icon        TEXT    NOT NULL,
            pdf_saved   INTEGER DEFAULT 0
        )
    """)

    # Tokens table — persistent login across browser sessions
    conn.execute("""
        CREATE TABLE IF NOT EXISTS tokens (
            token      TEXT PRIMARY KEY,
            email      TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL
        )
    """)

    conn.commit()
    return conn

def _hash(password: str) -> str:
    """SHA-256 password hash."""
    return hashlib.sha256(password.encode()).hexdigest()

# ── Token helpers
def _create_token(email: str) -> str:
    """Create a 30-day session token and save it to DB."""
    from datetime import timedelta
    token      = _secrets.token_hex(32)
    now        = datetime.now()
    expires    = now + timedelta(days=30)
    conn = _db()
    # Remove old tokens for this user first
    conn.execute("DELETE FROM tokens WHERE email = ?", (email,))
    conn.execute(
        "INSERT INTO tokens (token, email, created_at, expires_at) VALUES (?, ?, ?, ?)",
        (token, email, now.isoformat(), expires.isoformat()),
    )
    conn.commit()
    conn.close()
    return token

def _validate_token(token: str):
    """
    Check if a token is valid and not expired.
    Returns email string if valid, None otherwise.
    """
    if not token:
        return None
    conn = _db()
    row = conn.execute(
        "SELECT email, expires_at FROM tokens WHERE token = ?", (token,)
    ).fetchone()
    conn.close()
    if row is None:
        return None
    email, expires_at = row
    if datetime.now() > datetime.fromisoformat(expires_at):
        # Token expired — clean it up
        conn = _db()
        conn.execute("DELETE FROM tokens WHERE token = ?", (token,))
        conn.commit()
        conn.close()
        return None
    return email

def _delete_token(token: str):
    """Delete a token on logout."""
    conn = _db()
    conn.execute("DELETE FROM tokens WHERE token = ?", (token,))
    conn.commit()
    conn.close()

# ── User helpers
def _get_role(email: str) -> str:
    """Return 'admin' or 'user' for a given email."""
    if email.lower() == ADMIN_EMAIL.lower():
        return "admin"
    conn = _db()
    row = conn.execute("SELECT role FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    return row[0] if row else "user"

def auth_signup(email: str, password: str):
    """Register a new user. Returns (success, message)."""
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        return False, "Invalid email format."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    role = "admin" if email.lower() == ADMIN_EMAIL.lower() else "user"
    try:
        conn = _db()
        conn.execute(
            "INSERT INTO users (email, pw_hash, role, created_at) VALUES (?, ?, ?, ?)",
            (email.lower().strip(), _hash(password), role, datetime.now().isoformat()),
        )
        conn.commit()
        conn.close()
        return True, "Account created! You can now log in."
    except sqlite3.IntegrityError:
        return False, "An account with this email already exists."

def auth_login(email: str, password: str):
    """
    Verify credentials.
    Returns (success, message, token_or_None).
    On success, a 30-day session token is created and returned.
    """
    conn = _db()
    row = conn.execute(
        "SELECT pw_hash FROM users WHERE email = ?", (email.lower().strip(),)
    ).fetchone()
    conn.close()
    if row is None:
        return False, "No account found with this email.", None
    if row[0] != _hash(password):
        return False, "Incorrect password.", None
    token = _create_token(email.lower().strip())
    return True, "Logged in successfully.", token

# ── Scan persistence helpers
DATASET_PATH = os.path.join("data", "solar_data.csv")

def _append_scan_to_csv(email: str, pred_class: str, confidence: float, severity: str):
    """
    Append a scan result to the CSV dataset so it appears in the Dataset page.
    Adds a row with the scan's defect type, confidence, and timestamp.
    Creates the CSV with headers if it doesn't exist yet.
    """
    os.makedirs("data", exist_ok=True)
    now = datetime.now()
    new_row = {
        "timestamp":      now.strftime("%Y-%m-%d %H:%M"),
        "date":           now.strftime("%Y-%m-%d"),
        "hour":           now.hour,
        "panel_id":       email,          # user email used as panel identifier
        "irradiation":    "",             # not available from image scan
        "ambient_temp_c": "",
        "module_temp_c":  "",
        "dc_power_kw":    "",
        "ac_power_kw":    "",
        "defect_type":    pred_class,
        "efficiency_pct": "",
        "confidence":     round(confidence * 100, 1),
        "severity":       severity,
        "source":         "scan",         # marks this row as coming from a real scan
    }
    row_df = pd.DataFrame([new_row])
    if os.path.exists(DATASET_PATH):
        row_df.to_csv(DATASET_PATH, mode="a", header=False, index=False)
    else:
        row_df.to_csv(DATASET_PATH, mode="w", header=True, index=False)

def db_save_scan(email: str, pred_class: str, info: dict, confidence: float):
    """
    Save a scan result to:
      1. SQLite scans table (for history tab)
      2. data/solar_data.csv (for dataset page)
    Deduplication prevents the same scan being saved twice within 1 minute.
    """
    conn = _db()
    existing = conn.execute("""
        SELECT id FROM scans
        WHERE email = ? AND defect_type = ?
        AND scanned_at >= datetime('now', '-1 minute')
    """, (email, pred_class)).fetchone()
    if existing is None:
        conn.execute("""
            INSERT INTO scans
              (email, scanned_at, defect_type, display_en, display_ar,
               confidence, severity, icon)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            email,
            datetime.now().strftime("%Y-%m-%d %H:%M"),
            pred_class,
            info["display_en"],
            info["display_ar"],
            round(confidence, 4),
            info["severity"],
            info["icon"],
        ))
        conn.commit()
        # Also write to CSV dataset so it appears in the Dataset page
        _append_scan_to_csv(email, pred_class, confidence, info["severity"])
    conn.close()

def db_get_scans(email: str, admin: bool = False) -> list:
    """
    Load scans from the database.
    - Normal users: only their own scans.
    - Admin: all scans from all users.
    """
    conn = _db()
    if admin:
        rows = conn.execute("""
            SELECT email, scanned_at, defect_type, display_en, display_ar,
                   confidence, severity, icon
            FROM scans ORDER BY scanned_at DESC
        """).fetchall()
    else:
        rows = conn.execute("""
            SELECT email, scanned_at, defect_type, display_en, display_ar,
                   confidence, severity, icon
            FROM scans WHERE email = ? ORDER BY scanned_at DESC
        """, (email,)).fetchall()
    conn.close()
    return [
        {
            "email":      r[0], "time": r[1], "class":      r[2],
            "display_en": r[3], "display_ar": r[4],
            "confidence": r[5], "severity":   r[6], "icon":  r[7],
        }
        for r in rows
    ]

def db_delete_scans(email: str):
    """Delete all scans for a specific user."""
    conn = _db()
    conn.execute("DELETE FROM scans WHERE email = ?", (email,))
    conn.commit()
    conn.close()

def db_get_all_users() -> list:
    """Admin only: get list of all registered users."""
    conn = _db()
    rows = conn.execute(
        "SELECT email, role, created_at FROM users ORDER BY created_at DESC"
    ).fetchall()
    conn.close()
    return [{"email": r[0], "role": r[1], "created": r[2]} for r in rows]

def render_auth_page():
    """
    Full-screen login / signup UI.
    On successful login, a token is stored in st.session_state AND
    in the browser query params so the session survives page refreshes.
    """
    # ── Try to restore session from token stored in session_state
    for k, v in [("logged_in", False), ("auth_email", ""), ("session_token", "")]:
        if k not in st.session_state:
            st.session_state[k] = v

    # ── Auto-login: validate existing token from session_state
    if not st.session_state.logged_in and st.session_state.session_token:
        email = _validate_token(st.session_state.session_token)
        if email:
            st.session_state.logged_in  = True
            st.session_state.auth_email = email
            st.session_state.user_email = email
            return

    if st.session_state.logged_in:
        return  # already authenticated

    # ── Login / signup UI
    _, col, _ = st.columns([1, 1.4, 1])
    with col:
        st.markdown("""
        <div style="text-align:center;margin-bottom:28px;margin-top:40px;">
            <div style="font-size:3rem;">☀️</div>
            <div style="font-size:2rem;font-weight:800;letter-spacing:-1px;">
                Solar<span style="color:#f5a623;">Inspect</span> AI
            </div>
            <div style="color:#8a9ab0;font-size:0.9rem;margin-top:6px;">
                Solar panel defect detection platform
            </div>
        </div>
        """, unsafe_allow_html=True)

        tab_login, tab_signup = st.tabs(["🔑  Log In", "✨  Sign Up"])

        submitted  = False
        submitted2 = False

        with tab_login:
            with st.form("login_form", enter_to_submit=True):
                email    = st.text_input("Email", placeholder="you@example.com")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Log In", use_container_width=True)
            if submitted:
                ok, msg, token = auth_login(email, password)
                if ok:
                    # ── Store token in session — survives page refresh
                    st.session_state.logged_in     = True
                    st.session_state.auth_email    = email.lower().strip()
                    st.session_state.user_email    = email.lower().strip()
                    st.session_state.session_token = token
                    st.rerun()
                else:
                    st.error(msg)

        with tab_signup:
            with st.form("signup_form", enter_to_submit=True):
                new_email = st.text_input("Email", placeholder="you@example.com", key="su_email")
                new_pw    = st.text_input("Password (min 6 chars)", type="password", key="su_pw")
                new_pw2   = st.text_input("Confirm password",        type="password", key="su_pw2")
                submitted2 = st.form_submit_button("Create Account", use_container_width=True)
            if submitted2:
                if new_pw != new_pw2:
                    st.error("Passwords do not match.")
                else:
                    ok2, msg2 = auth_signup(new_email, new_pw)
                    if ok2:
                        st.success(msg2)
                    else:
                        st.error(msg2)

    st.stop()


# ─────────────────────────────────────────────────────────────────────
# 3. THEME — dark mode only (light mode removed)
# ─────────────────────────────────────────────────────────────────────
# Initialise all session-state keys used in the app
for key, default in [
    ("lang",          "en"),
    ("history",       []),
    ("dark_mode",     True),
    ("user_email",    ""),
    ("logged_in",     False),
    ("auth_email",    ""),
    ("session_token", ""),   # persistent login token
]:
    if key not in st.session_state:
        st.session_state[key] = default

# Run auth gate — stops execution here if not logged in
render_auth_page()

# ── Helpers
def t(en, ar):
    return ar if st.session_state.lang == "ar" else en

RTL   = "direction:rtl;text-align:right;" if st.session_state.lang == "ar" else ""
FONT  = "'Cairo', sans-serif"  if st.session_state.lang == "ar" else "'Syne', sans-serif"
IS_AR = st.session_state.lang == "ar"
DM    = True  # always dark

# ── Dark mode color palette (only one theme now)
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

# ── Global CSS
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&family=Cairo:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {{
    background-color:{BG}; color:{TXT}; font-family:{FONT};
    font-size:{"19px" if IS_AR else "17px"};
    font-weight:{"600" if IS_AR else "400"};
}}
.stApp {{ background-color:{BG}; }}

/* Native widget overrides */
.stTextInput>div>div>input,
.stNumberInput>div>div>input {{
    background-color:{INPUT_BG} !important;
    color:{TXT} !important;
    border-color:{BORDER} !important;
    border-radius:8px !important;
}}
/* FIX 6: Make input/slider labels readable — match TXT_S (same as perf-card description) */
.stNumberInput label,
.stTextInput label,
.stSlider label,
.stSelectbox label,
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] {{
    color:{TXT_S} !important;
    font-size:0.92rem !important;
    font-weight:500 !important;
}}
.stButton>button {{
    background-color:{BG_CARD}; color:{TXT};
    border:1px solid {BORDER}; border-radius:8px;
    font-weight:600; transition:all 0.2s;
}}
.stButton>button:hover {{ border-color:#f5a623; color:#f5a623; }}
.stTabs [data-baseweb="tab"] {{ color:{TXT_M}; font-size:0.92rem; }}
.stTabs [aria-selected="true"] {{ color:{TXT} !important; border-bottom:2px solid #f5a623; }}

/* Hero */
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

/* Cards */
.upload-zone {{ border:2px dashed {BORDER}; border-radius:12px; padding:48px; text-align:center; background:{BG_CARD}; {RTL} }}
.metric-card {{ background:{BG_CARD}; border:1px solid {BORDER}; border-radius:12px; padding:20px 24px; margin-bottom:12px; {RTL} }}
.metric-label {{ font-family:'Space Mono',monospace; font-size:0.72rem; color:{TXT_M}; letter-spacing:2px; margin-bottom:6px; }}
.metric-value {{ font-size:2rem; font-weight:800; color:{TXT}; line-height:1; }}
.perf-card {{ background:{BG_CARD}; border:1px solid {BORDER}; border-radius:12px; padding:24px; margin-bottom:16px; {RTL} }}

/* Badges */
.badge-critical {{ background:#c0392b22; border:1px solid #e74c3c; color:#e74c3c; border-radius:8px; padding:8px 20px; font-weight:800; font-size:1.1rem; display:inline-block; }}
.badge-warning  {{ background:#b8860b22; border:1px solid #f5a623; color:#f5a623; border-radius:8px; padding:8px 20px; font-weight:800; font-size:1.1rem; display:inline-block; }}
.badge-good     {{ background:#1a4a2a22; border:1px solid #2ecc71; color:#2ecc71; border-radius:8px; padding:8px 20px; font-weight:800; font-size:1.1rem; display:inline-block; }}

/* Defect card */
.defect-card {{ background:{BG_CARD}; border-radius:0 10px 10px 0; padding:16px 20px; margin-bottom:10px; {RTL} }}
.defect-card.critical {{ border-left:4px solid #e74c3c; }}
.defect-card.warning  {{ border-left:4px solid #f5a623; }}
.defect-card.info     {{ border-left:4px solid #3498db; }}
.defect-name   {{ font-weight:800; font-size:{"1.15rem" if IS_AR else "1.05rem"}; color:{TXT}; margin-bottom:4px; }}
.defect-desc   {{ font-size:{"1rem" if IS_AR else "0.9rem"}; color:{TXT_S}; line-height:{"1.9" if IS_AR else "1.55"}; margin-bottom:8px; }}
.defect-action {{ font-size:{"0.95rem" if IS_AR else "0.88rem"}; color:#f5a623; font-weight:700; }}
.urgency-box   {{ background:{URG_BG}; border:1px solid {URG_C}; border-radius:8px; padding:12px 16px; margin-top:10px; font-size:0.88rem; color:{URG_C}; font-weight:600; {RTL} }}

/* Misc */
.section-title {{ font-family:'Space Mono',monospace; font-size:0.72rem; color:{TXT_M}; letter-spacing:3px; margin-bottom:14px; margin-top:26px; text-transform:uppercase; {RTL} }}
.conf-bar-bg   {{ background:{BAR_BG}; border-radius:4px; height:6px; margin-top:4px; }}
.conf-bar-fill {{ height:6px; border-radius:4px; background:linear-gradient(90deg,#f5a623,#e74c3c); }}
.history-card  {{ background:{BG_CARD}; border:1px solid {BORDER}; border-radius:10px; padding:14px 18px; margin-bottom:8px; transition:border-color 0.2s; {RTL} }}
.history-card:hover {{ border-color:#f5a623; }}
.tip-card {{ background:{TIP_BG}; border:1px solid {TIP_BD}; border-radius:10px; padding:16px; margin-bottom:8px; {RTL} }}
.tip-text {{ font-size:{"0.95rem" if IS_AR else "0.88rem"}; color:{TXT_S}; line-height:1.6; }}
.user-badge {{ background:{BG_CARD}; border:1px solid {BORDER}; border-radius:8px; padding:6px 14px; font-size:0.82rem; color:{TXT_M}; display:inline-block; margin-bottom:8px; }}

/* ── Professional site footer ── */
.site-footer {{
    margin-top: 56px;
    padding: 22px 0 18px 0;
    border-top: 1px solid {BORDER};
    text-align: center;
}}
.site-footer .footer-brand {{
    font-family: 'Syne', sans-serif;
    font-size: 0.88rem;
    font-weight: 600;
    color: {TXT_M};
    letter-spacing: 0.5px;
}}
.site-footer .footer-brand span {{
    color: #f5a623;
}}
.site-footer .footer-tagline {{
    font-family: 'Space Mono', monospace;
    font-size: 0.70rem;
    color: {TXT_M};
    letter-spacing: 2px;
    margin-top: 4px;
    opacity: 0.7;
}}

#MainMenu, footer, header {{ visibility:hidden; }}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# 4. DATA — defect info, model loaders
# ─────────────────────────────────────────────────────────────────────
CLASSES = ["Bird-drop","Clean","Dusty","Electrical-damage","Physical-damage","Snow-covered"]

# Urgency strings have emojis stripped here so the PDF helper doesn't need to
DEFECT_INFO = {
    "Bird-drop":{
        "severity":"warning","display_en":"Bird Dropping","display_ar":"إفرازات الطيور","icon":"🐦",
        "desc_en":"Organic contamination causing localised shading and potential hot spots.",
        "desc_ar":"تلوث عضوي يسبب تظليلاً موضعياً ونقاط ساخنة محتملة.",
        "action_en":"Clean with soft cloth and water within 1-2 weeks.",
        "action_ar":"نظف بقطعة قماش ناعمة وماء خلال 1-2 أسبوع.",
        "urgency_en":"Clean within 2 weeks to prevent permanent staining",
        "urgency_ar":"نظف خلال أسبوعين لمنع التلطيخ الدائم",
        "tips_en":["Use distilled water for cleaning","Clean early morning or evening","Inspect for scratches after cleaning"],
        "tips_ar":["استخدم ماء مقطراً","نظف في الصباح الباكر أو المساء","افحص الخدوش بعد التنظيف"],
    },
    "Clean":{
        "severity":"info","display_en":"Clean Panel","display_ar":"لوح نظيف","icon":"✅",
        "desc_en":"No defects detected. Panel surface appears clean and undamaged.",
        "desc_ar":"لم يتم اكتشاف عيوب. سطح اللوح نظيف وغير تالف.",
        "action_en":"No action required. Continue routine inspection every 3-6 months.",
        "action_ar":"لا يلزم اتخاذ أي إجراء. استمر في الفحص الدوري كل 3-6 أشهر.",
        "urgency_en":"No action needed - next inspection in 3-6 months",
        "urgency_ar":"لا يلزم إجراء - الفحص القادم خلال 3-6 أشهر",
        "tips_en":["Schedule next inspection in 3 months","Monitor energy output monthly","Keep maintenance logs updated"],
        "tips_ar":["جدول الفحص القادم خلال 3 أشهر","راقب إنتاج الطاقة شهرياً","حافظ على تحديث السجلات"],
    },
    "Dusty":{
        "severity":"info","display_en":"Dust Accumulation","display_ar":"تراكم الغبار","icon":"🌫️",
        "desc_en":"Surface soiling reducing light transmission. Can reduce output by 5-30%.",
        "desc_ar":"تلوث السطح يقلل انتقال الضوء. يمكن أن يقلل الإنتاج بنسبة 5-30٪.",
        "action_en":"Schedule routine cleaning. In desert environments, clean every 2-4 weeks.",
        "action_ar":"جدول التنظيف الدوري. في البيئات الصحراوية، نظف كل 2-4 أسابيع.",
        "urgency_en":"Clean within 1 month for optimal efficiency",
        "urgency_ar":"نظف خلال شهر للحصول على أفضل كفاءة",
        "tips_en":["Use automated cleaning systems if available","Clean before panels heat up","Consider anti-soiling coatings"],
        "tips_ar":["استخدم أنظمة التنظيف التلقائي","نظف في الصباح الباكر أو مساءً عندما تكون الألواح غير ساخنة","فكر في طلاءات مضادة للأوساخ"],
    },
    "Electrical-damage":{
        "severity":"critical","display_en":"Electrical Damage","display_ar":"تلف كهربائي","icon":"⚡",
        "desc_en":"Burn marks indicating arc faults or failed bypass diodes. Fire hazard risk.",
        "desc_ar":"علامات حرق تشير إلى أعطال القوس أو صمامات التحويل الفاشلة. خطر حريق.",
        "action_en":"Take panel offline immediately. Contact a certified PV technician.",
        "action_ar":"أوقف تشغيل اللوح فوراً. اتصل بفني PV معتمد.",
        "urgency_en":"URGENT - Take offline immediately, fire risk!",
        "urgency_ar":"عاجل - أوقف التشغيل فوراً، خطر حريق!",
        "tips_en":["Do NOT attempt DIY repair","Document the damage with photos","Check neighboring panels","Review system insurance"],
        "tips_ar":["لا تحاول الإصلاح بنفسك","وثق الضرر بالصور","افحص الألواح المجاورة","راجع تغطية التأمين"],
    },
    "Physical-damage":{
        "severity":"critical","display_en":"Physical Damage","display_ar":"تلف مادي","icon":"💥",
        "desc_en":"Delamination, frame damage, or cell breakage. Allows moisture ingress.",
        "desc_ar":"تقشر أو تلف الإطار أو كسر الخلايا. يسمح بدخول الرطوبة.",
        "action_en":"Replace panel as soon as possible. Moisture ingress accelerates degradation.",
        "action_ar":"استبدل اللوح في أقرب وقت ممكن. دخول الرطوبة يسرع التدهور.",
        "urgency_en":"Replace within 2 weeks - moisture ingress risk",
        "urgency_ar":"استبدل خلال أسبوعين - خطر دخول الرطوبة",
        "tips_en":["Cover damaged area temporarily","Check warranty for replacement","Inspect mounting structure","Order replacement panel"],
        "tips_ar":["غطِّ المنطقة التالفة مؤقتاً","تحقق من الضمان","افحص هيكل التركيب","اطلب لوحاً بديلاً"],
    },
    "Snow-covered":{
        "severity":"warning","display_en":"Snow Coverage","display_ar":"تغطية الثلج","icon":"❄️",
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
        st.error("best_efficientnet_b0.pth not found"); st.stop()
    m = timm.create_model("efficientnet_b0", pretrained=False, num_classes=6)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    m.eval()
    return m

@st.cache_resource
def load_perf_model():
    try:
        model  = joblib.load("performance_model.pkl")
        scaler = joblib.load("performance_scaler.pkl")
        with open("model_metadata.json") as f:  meta    = json.load(f)
        with open("typical_values.json")  as f: typical = json.load(f)
        return model, scaler, meta, typical
    except:
        return None, None, None, None

@st.cache_resource
def load_lstm():
    try:
        import tensorflow as tf
        model  = tf.keras.models.load_model("lstm_model.keras")
        scaler = joblib.load("lstm_scaler.pkl")
        with open("lstm_metadata.json") as f: meta = json.load(f)
        sample = pd.read_csv("sample_data.csv")
        return model, scaler, meta, sample
    except:
        return None, None, None, None

effnet_model                                     = load_effnet()
perf_model, perf_scaler, perf_meta, typical_vals = load_perf_model()
lstm_model, lstm_scaler, lstm_meta, sample_data  = load_lstm()

def preprocess_image(image):
    from torchvision import transforms
    tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tf(image).unsqueeze(0)

# ─────────────────────────────────────────────────────────────────────
# 5. PDF — styled bilingual report with Amiri font for Arabic
# ─────────────────────────────────────────────────────────────────────
AMIRI_PATH = os.path.join("fonts", "Amiri-Regular.ttf")

def _safe_en(text: str) -> str:
    """Strip non-latin / emoji — safe for Helvetica."""
    return "".join(
        c for c in str(text)
        if _ud.category(c) not in ("So", "Cs") and ord(c) < 0x0250
    ).strip()

def _has_amiri() -> bool:
    return os.path.exists(AMIRI_PATH)

def _ar(text: str) -> str:
    """
    FIX 4 — Proper Arabic text rendering for FPDF.
    FPDF renders text left-to-right and doesn't reshape Arabic ligatures.
    We use arabic_reshaper (joins letters correctly) and python-bidi
    (reverses the visual order for RTL display) to pre-process every
    Arabic string before passing it to FPDF.
    Falls back to raw text if libraries are missing.
    """
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        reshaped = arabic_reshaper.reshape(str(text))
        return get_display(reshaped)
    except ImportError:
        return str(text)  # graceful fallback if libs not installed


# Arabic section header translations
_AR_SECTIONS = {
    "Detection Result":    "نتيجة الكشف",
    "Description":         "الوصف",
    "Recommended Action":  "الإجراء الموصى به",
    "Maintenance Tips":    "نصائح الصيانة",
}


class SolarPDF(FPDF):
    """FPDF subclass with branded header/footer and bilingual helpers."""

    def __init__(self, lang="en"):
        super().__init__()
        self.lang     = lang
        self.amiri_ok = _has_amiri()
        self.set_margins(18, 18, 18)
        self.set_auto_page_break(auto=True, margin=20)
        if self.amiri_ok:
            # uni=True enables full Unicode support
            self.add_font("Amiri", "", AMIRI_PATH, uni=True)

    @property
    def W(self):
        """Usable page width (avoids multi_cell width=0 crash)."""
        return self.w - self.l_margin - self.r_margin

    def header(self):
        """
        PDF HEADER — white background, clean typography hierarchy.
        FIX: "SolarInspect AI" is the ONLY header element (18pt bold orange).
             "Scan Report" lives ONCE in the body title block — NOT duplicated here.
        """
        # White header background
        self.set_fill_color(255, 255, 255)
        self.rect(0, 0, self.w, 18, "F")

        # Brand name — large, bold, orange
        self.set_y(3)
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(245, 166, 35)   # brand orange
        self.cell(0, 10, "SolarInspect AI", align="C")
        self.ln(10)

        # Thin brand-orange accent line under the header
        self.set_draw_color(245, 166, 35)
        self.set_line_width(0.6)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.set_line_width(0.2)
        self.set_draw_color(200, 210, 220)  # reset line colour for body

        self.set_text_color(0, 0, 0)
        self.ln(6)

    def footer(self):
        """
        PDF FOOTER — FIX: page numbers removed entirely.
        Only shows the brand name as a subtle watermark-style footer.
        """
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(180, 190, 200)
        self.cell(0, 8, "SolarInspect AI", align="C")
        self.set_text_color(0, 0, 0)

    def section_header(self, title_en: str):
        """
        FIX 4: Bilingual section divider.
        Shows Arabic translation (reshaped+bidi) when in Arabic mode,
        English otherwise.
        """
        self.ln(4)
        self.set_fill_color(26, 40, 60)
        self.set_text_color(245, 166, 35)
        if self.lang == "ar" and self.amiri_ok:
            ar_title = _AR_SECTIONS.get(title_en, title_en)
            self.set_font("Amiri", "", 12)
            self.cell(self.W, 10, _ar(ar_title), ln=True, fill=True, align="R")
        else:
            self.set_font("Helvetica", "B", 11)
            self.cell(self.W, 9, f"  {title_en}", ln=True, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def hr(self):
        """Thin horizontal rule."""
        self.set_draw_color(200, 210, 220)
        self.line(self.l_margin, self.get_y(), self.l_margin + self.W, self.get_y())
        self.ln(4)

    def body_line(self, label_en: str, value_en: str, value_ar: str = "",
                  label_ar: str = ""):
        """
        UPDATED: Label + value row with proper spacing.
        Arabic mode: each item gets its own clearly separated block with
        label on one line and value on the next — no more glued-together text.
        """
        if self.lang == "ar" and self.amiri_ok:
            # ── Arabic: label line (muted colour, smaller)
            self.set_font("Amiri", "", 10)
            self.set_text_color(100, 120, 140)
            lbl = _ar(label_ar) if label_ar else _ar(label_en)
            self.cell(self.W, 7, lbl, ln=True, align="R")

            # ── Arabic: value line (dark, larger, bold-weight via size)
            self.set_font("Amiri", "", 12)
            self.set_text_color(20, 20, 20)
            val = _ar(value_ar) if value_ar else _ar(value_en)
            self.set_x(self.l_margin)
            self.cell(self.W, 8, val, ln=True, align="R")

            # Small gap between items for clear visual separation
            self.ln(3)
        else:
            self.set_font("Helvetica", "B", 10)
            self.set_text_color(60, 80, 100)
            self.cell(55, 7, f"{label_en}:", ln=False)
            self.set_font("Helvetica", "", 10)
            self.set_text_color(20, 20, 20)
            self.cell(self.W - 55, 7, _safe_en(value_en), ln=True)
        self.set_text_color(0, 0, 0)

    def body_para(self, text_en: str, text_ar: str = ""):
        """FIX 4: Paragraph text — Arabic reshaped+bidi when lang=ar."""
        self.set_text_color(30, 30, 30)
        if self.lang == "ar" and text_ar and self.amiri_ok:
            self.set_font("Amiri", "", 11)
            self.multi_cell(self.W, 7, _ar(text_ar), align="R")
        else:
            self.set_font("Helvetica", "", 10)
            self.multi_cell(self.W, 7, _safe_en(text_en))
        self.set_text_color(0, 0, 0)
        self.ln(2)

    def info_box(self, text_en: str, text_ar: str, severity: str = "info"):
        """FIX 4: Coloured urgency box — Arabic processed for RTL."""
        colors = {
            "critical": (210, 60,  50),
            "warning":  (200, 140, 30),
            "info":     (40,  160, 110),
        }
        r, g, b = colors.get(severity, (80, 120, 180))
        self.set_fill_color(r, g, b)
        self.set_text_color(255, 255, 255)
        if self.lang == "ar" and text_ar and self.amiri_ok:
            self.set_font("Amiri", "", 11)
            self.multi_cell(self.W, 8, _ar(text_ar), fill=True, align="R")
        else:
            self.set_font("Helvetica", "B", 10)
            self.multi_cell(self.W, 8, f"  {_safe_en(text_en)}  ", fill=True, align="C")
        self.set_text_color(0, 0, 0)
        self.ln(3)

    def tips_table(self, tips_en: list, tips_ar: list):
        """FIX 4: Zebra-striped tips — Arabic processed for RTL."""
        use_ar = self.lang == "ar" and tips_ar and self.amiri_ok
        tips   = tips_ar if use_ar else tips_en
        for i, tip in enumerate(tips):
            self.set_fill_color(*(240, 244, 250) if i % 2 == 0 else (255, 255, 255))
            self.set_text_color(30, 30, 30)
            if use_ar:
                self.set_font("Amiri", "", 11)
                self.cell(self.W, 9, _ar(str(tip)), ln=True, fill=True, align="R")
            else:
                self.set_font("Helvetica", "", 10)
                self.cell(self.W, 9, f"  {i+1}. {_safe_en(tip)}", ln=True, fill=True)
        self.set_text_color(0, 0, 0)
        self.ln(2)


def generate_pdf(pred_class: str, confidence: float, info: dict,
                 lang: str = "en", user_email: str = "", underperf=None) -> bytes:
    """
    Generate a styled, bilingual PDF scan report.
    FIX 4: Arabic text is pre-processed with arabic_reshaper + python-bidi
            for correct glyph shaping and RTL visual order.
    FIX 7: "SolarInspect AI" header title increased to 14pt (in SolarPDF.header).
    Requires: pip install arabic-reshaper python-bidi
    """
    is_ar = (lang == "ar")
    pdf = SolarPDF(lang=lang)
    pdf.add_page()

    # ── Title block (appears once — header already shows "SolarInspect AI")
    if is_ar and pdf.amiri_ok:
        # Arabic title — single occurrence
        pdf.set_font("Amiri", "", 16)
        pdf.set_text_color(26, 40, 60)
        pdf.cell(pdf.W, 12, _ar("تقرير الفحص"), ln=True, align="C")
        pdf.set_font("Amiri", "", 10)
        pdf.set_text_color(100, 120, 140)
        # FIX: date appears exactly once
        pdf.cell(pdf.W, 6, datetime.now().strftime("%Y-%m-%d  %H:%M"), ln=True, align="C")
        if user_email:
            pdf.cell(pdf.W, 6, _ar(f"الحساب: {user_email}"), ln=True, align="C")
    else:
        # English title — "Scan Report" here only (removed from header to avoid duplication)
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(26, 40, 60)
        pdf.cell(pdf.W, 12, "Scan Report", ln=True, align="C")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(100, 120, 140)
        # FIX: date appears exactly once
        pdf.cell(pdf.W, 6, datetime.now().strftime("%A, %d %B %Y  ·  %H:%M"), ln=True, align="C")
        if user_email:
            pdf.cell(pdf.W, 6, f"Account: {user_email}", ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)
    pdf.hr()

    # ── Detection result
    pdf.section_header("Detection Result")
    pdf.body_line("Defect Type",      info["display_en"],       info["display_ar"],   "نوع العيب")
    pdf.body_line("Confidence",       f"{confidence:.1%}",      f"{confidence:.1%}",  "مستوى الثقة")
    pdf.body_line("Severity",         info["severity"].upper(),  info["severity"].upper(), "مستوى الخطورة")
    if underperf is not None:
        pdf.body_line("Underperformance", f"{underperf:.1f}%",  f"{underperf:.1f}%",  "الأداء دون المستوى")
    pdf.ln(4)

    # ── Urgency box
    pdf.info_box(
        f"Urgency: {info['urgency_en']}",
        f"مهم: {info['urgency_ar']}",
        severity=info["severity"],
    )

    # ── Description
    pdf.section_header("Description")
    pdf.body_para(info["desc_en"], info["desc_ar"])

    # ── Recommended action
    pdf.section_header("Recommended Action")
    pdf.body_para(info["action_en"], info["action_ar"])

    # ── Maintenance tips
    pdf.section_header("Maintenance Tips")
    pdf.tips_table(info["tips_en"], info["tips_ar"])

    # ── Disclaimer
    pdf.ln(6)
    pdf.hr()
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(150, 160, 170)
    if is_ar and pdf.amiri_ok:
        pdf.set_font("Amiri", "", 9)
        pdf.multi_cell(pdf.W, 5, _ar(
            "تم إنشاء هذا التقرير تلقائياً بواسطة SolarInspect AI. "
            "استشر دائماً فني PV معتمد قبل اتخاذ أي إجراء."
        ), align="R")
    else:
        pdf.multi_cell(pdf.W, 5,
            "This report was generated automatically by SolarInspect AI. "
            "Always consult a certified PV technician before taking corrective action.",
            align="C",
        )
    return bytes(pdf.output())


# ─────────────────────────────────────────────────────────────────────
# 6. UI — header, controls, tabs
# ─────────────────────────────────────────────────────────────────────

# ── Top header
col_logo, col_ctrl = st.columns([5, 2])
with col_logo:
    st.markdown(f"""
    <div class="hero">
        <div class="hero-title">Solar<span>Inspect</span> AI</div>
        <div class="hero-sub">
            {t('SOLAR PANEL DEFECT DETECTION ',
               'كشف عيوب الألواح الشمسية ')}
        </div>
    </div>
    """, unsafe_allow_html=True)

with col_ctrl:
    st.markdown("<br>", unsafe_allow_html=True)
    # Show logged-in user
    st.markdown(
        f'<div class="user-badge">👤 {st.session_state.auth_email}</div>',
        unsafe_allow_html=True,
    )
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🌐 AR" if st.session_state.lang == "en" else "🌐 EN"):
            st.session_state.lang = "ar" if st.session_state.lang == "en" else "en"
            st.rerun()
    with c2:
        # Logout — delete token so session doesn't persist after explicit logout
        if st.button("🚪"):
            if st.session_state.session_token:
                _delete_token(st.session_state.session_token)
            st.session_state.logged_in     = False
            st.session_state.auth_email    = ""
            st.session_state.session_token = ""
            st.session_state.history       = []
            st.rerun()

# ── Role check — admin gets extra tab
is_admin = _get_role(st.session_state.auth_email) == "admin"

# ── Tabs — Dataset tab only shown to admin
if is_admin:
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        t("🔍 Image Scan",     "🔍 فحص الصورة"),
        t("📊 Performance",    "📊 الأداء"),
        t("📈 Power Forecast", "📈 توقع الطاقة"),
        t("📋 History",        "📋 السجل"),
        t("📂 Dataset",        "📂 البيانات"),
    ])
else:
    tab1, tab2, tab3, tab4 = st.tabs([
        t("🔍 Image Scan",     "🔍 فحص الصورة"),
        t("📊 Performance",    "📊 الأداء"),
        t("📈 Power Forecast", "📈 توقع الطاقة"),
        t("📋 History",        "📋 السجل"),
    ])

# ═══════════════════════════════════════════════════════════════
# TAB 1 — IMAGE SCAN
# ═══════════════════════════════════════════════════════════════
with tab1:
    uploaded = st.file_uploader(
        t("Upload solar panel image", "رفع صورة اللوح الشمسي"),
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    if uploaded is None:
        st.markdown(f"""
        <div class="upload-zone">
            <div style="font-size:3rem;margin-bottom:16px;">☀️</div>
            <div style="font-size:1.1rem;font-weight:600;color:{TXT_M};">
                {t('Drop a solar panel image to scan for defects',
                   'أضف صورة اللوح الشمسي للكشف عن العيوب')}
            </div>
            <div style="font-family:Space Mono,monospace;font-size:0.78rem;color:{TXT_M};margin-top:8px;">
                JPG · JPEG · PNG
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        image = Image.open(uploaded).convert("RGB")
        with st.spinner(t("Scanning panel...", "جاري فحص اللوح...")):
            tensor = preprocess_image(image)
            with torch.no_grad():
                probs = torch.nn.functional.softmax(effnet_model(tensor), dim=1)[0].numpy()

        pred_idx   = int(np.argmax(probs))
        pred_class = CLASSES[pred_idx]
        confidence = float(probs[pred_idx])
        info       = DEFECT_INFO[pred_class]
        display    = info["display_ar"] if IS_AR else info["display_en"]
        sev        = info["severity"]

        # ── Save scan ONCE per upload using file hash as guard.
        # Streamlit reruns the whole script on every click, so we track
        # which files have already been saved using session_state.
        import hashlib as _hl
        file_hash = _hl.md5(uploaded.getvalue()).hexdigest()
        if "saved_hashes" not in st.session_state:
            st.session_state.saved_hashes = set()
        if file_hash not in st.session_state.saved_hashes:
            db_save_scan(st.session_state.auth_email, pred_class, info, confidence)
            st.session_state.saved_hashes.add(file_hash)

        col_img, col_res = st.columns([3, 2], gap="large")

        with col_img:
            st.markdown(f'<div class="section-title">{t("UPLOADED IMAGE","الصورة المرفوعة")}</div>',
                        unsafe_allow_html=True)
            st.image(image, use_container_width=True)

        with col_res:
            st.markdown(f'<div class="section-title">{t("SEVERITY","مستوى الخطورة")}</div>',
                        unsafe_allow_html=True)
            badge_open, _ = BADGE[sev]
            st.markdown(f'{badge_open}{info["icon"]} {display}</span>', unsafe_allow_html=True)

            st.markdown(f'<div class="section-title" style="margin-top:20px;">{t("CONFIDENCE","مستوى الثقة")}</div>',
                        unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{t("MODEL CONFIDENCE","ثقة النموذج")}</div>
                <div class="metric-value" style="color:#f5a623;">{confidence:.0%}</div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{int(confidence*100)}%"></div>
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown(f'<div class="section-title">{t("ANALYSIS","التحليل")}</div>',
                        unsafe_allow_html=True)
            desc    = info["desc_ar"]    if IS_AR else info["desc_en"]
            action  = info["action_ar"]  if IS_AR else info["action_en"]
            urgency = info["urgency_ar"] if IS_AR else info["urgency_en"]
            st.markdown(f"""
            <div class="defect-card {sev}">
                <div class="defect-name">{info['icon']} {display}</div>
                <div class="defect-desc">{desc}</div>
                <div class="defect-action">→ {action}</div>
                <div class="urgency-box">{urgency}</div>
            </div>""", unsafe_allow_html=True)

        # ── Tips
        st.markdown(f'<div class="section-title">{t("MAINTENANCE TIPS","نصائح الصيانة")}</div>',
                    unsafe_allow_html=True)
        tips     = info["tips_ar"] if IS_AR else info["tips_en"]
        tip_cols = st.columns(len(tips))
        for i, tip in enumerate(tips):
            with tip_cols[i]:
                st.markdown(
                    f'<div class="tip-card"><div style="font-size:1.2rem;margin-bottom:6px;">💡</div>'
                    f'<div class="tip-text">{tip}</div></div>',
                    unsafe_allow_html=True,
                )

        # ── PDF export (generated on demand)
        st.markdown(f'<div class="section-title">{t("EXPORT REPORT","تصدير التقرير")}</div>',
                    unsafe_allow_html=True)

        # Inform user if Arabic PDF is not available
        if not _has_amiri():
            st.info(t(
                "Place fonts/Amiri-Regular.ttf in the app folder to enable Arabic PDF output.",
                "ضع ملف fonts/Amiri-Regular.ttf في مجلد التطبيق لتفعيل PDF العربي.",
            ))

        if st.button(t("📄 Generate PDF Report", "📄 إنشاء تقرير PDF"), key="gen_pdf"):
            with st.spinner(t("Generating report...", "جاري إنشاء التقرير...")):
                pdf_bytes = generate_pdf(
                    pred_class, confidence, info,
                    lang=st.session_state.lang,
                    user_email=st.session_state.auth_email,
                )
            st.download_button(
                label=t("⬇️ Download Report", "⬇️ تحميل التقرير"),
                data=pdf_bytes,
                file_name=f"solar_scan_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                mime="application/pdf",
                key="dl_pdf",
            )

# ═══════════════════════════════════════════════════════════════
# TAB 2 — PERFORMANCE ANALYZER
# ═══════════════════════════════════════════════════════════════
with tab2:
    st.markdown(f'<div class="section-title">{t("PANEL PERFORMANCE ANALYZER","محلل أداء اللوح")}</div>',
                unsafe_allow_html=True)

    if perf_model is None:
        st.warning(t(
            "Performance model files not found. Ensure performance_model.pkl, "
            "performance_scaler.pkl, model_metadata.json and typical_values.json are present.",
            "ملفات نموذج الأداء غير موجودة.",
        ))
    else:
        st.markdown(f"""
        <div class="perf-card">
            <div style="font-size:0.92rem;color:{TXT_S};">
                {t('Enter sensor readings to check if your panel is performing as expected.',
                   'أدخل قراءات المستشعر للتحقق من أداء اللوح.')}
            </div>
        </div>""", unsafe_allow_html=True)

        tv = typical_vals
        c1, c2, c3 = st.columns(3)
        with c1:
            irradiation = st.number_input(
                t("Irradiation (W/m2/1000)", "الإشعاع"),
                min_value=0.0, max_value=2.0, value=float(round(tv["IRRADIATION"], 3)), step=0.01,
            )
        with c2:
            ambient_temp = st.number_input(
                t("Ambient Temp (C)", "درجة حرارة المحيط"),
                min_value=-10.0, max_value=60.0, value=float(round(tv["AMBIENT_TEMPERATURE"], 1)), step=0.1,
            )
        with c3:
            module_temp = st.number_input(
                t("Module Temp (C)", "درجة حرارة اللوح"),
                min_value=-10.0, max_value=90.0, value=float(round(tv["MODULE_TEMPERATURE"], 1)), step=0.1,
            )
        c4, c5 = st.columns(2)
        with c4:
            dc_power = st.number_input(
                t("DC Power (kW)", "طاقة DC"),
                min_value=0.0, max_value=500000.0, value=float(round(tv["DC_POWER"], 1)), step=100.0,
            )
        with c5:
            ac_power = st.number_input(
                t("AC Power (kW)", "طاقة AC"),
                min_value=0.0, max_value=500000.0, value=float(round(tv["AC_POWER"], 1)), step=100.0,
            )

        if st.button(t("Analyze Performance", "تحليل الأداء"), use_container_width=True):
            if irradiation == 0:
                st.info(t(
                    "Irradiation is 0 — panel is not generating power (night or fully shaded).",
                    "الإشعاع = 0 — اللوح لا ينتج طاقة.",
                ))
            else:
                features     = np.array([[irradiation, ambient_temp, module_temp, datetime.now().hour]])
                features_sc  = perf_scaler.transform(features)
                expected_ac  = perf_model.predict(features_sc)[0]
                underperf_pct = max(0, (expected_ac - ac_power) / (expected_ac + 1e-6) * 100)
                dc_ac_eff     = (ac_power / (dc_power + 1e-6)) * 100

                r1, r2, r3 = st.columns(3)
                with r1:
                    col = "#2ecc71" if underperf_pct < 10 else "#f5a623" if underperf_pct < 25 else "#e74c3c"
                    st.markdown(f"""<div class="metric-card" style="text-align:center;">
                        <div class="metric-label">{t("UNDERPERFORMANCE","الأداء دون المستوى")}</div>
                        <div class="metric-value" style="color:{col};">-{underperf_pct:.1f}%</div>
                        <div style="font-size:0.8rem;color:{TXT_M};margin-top:4px;">{t("vs expected","مقارنة بالمتوقع")}</div>
                    </div>""", unsafe_allow_html=True)
                with r2:
                    st.markdown(f"""<div class="metric-card" style="text-align:center;">
                        <div class="metric-label">{t("EXPECTED AC","طاقة AC المتوقعة")}</div>
                        <div class="metric-value" style="color:#f5a623;">{expected_ac:,.0f}</div>
                        <div style="font-size:0.8rem;color:{TXT_M};margin-top:4px;">kW</div>
                    </div>""", unsafe_allow_html=True)
                with r3:
                    ec = "#2ecc71" if dc_ac_eff > 90 else "#f5a623" if dc_ac_eff > 75 else "#e74c3c"
                    st.markdown(f"""<div class="metric-card" style="text-align:center;">
                        <div class="metric-label">{t("DC→AC EFFICIENCY","كفاءة DC→AC")}</div>
                        <div class="metric-value" style="color:{ec};">{dc_ac_eff:.1f}%</div>
                        <div style="font-size:0.8rem;color:{TXT_M};margin-top:4px;">{t("inverter","العاكس")}</div>
                    </div>""", unsafe_allow_html=True)

                if underperf_pct < 5:
                    sc,si = "#2ecc71","✅"
                    msg_en,msg_ar = f"Performing normally — {underperf_pct:.1f}% below expected.", f"يعمل بشكل طبيعي — {underperf_pct:.1f}٪ أقل من المتوقع."
                elif underperf_pct < 15:
                    sc,si = "#f5a623","⚠️"
                    msg_en,msg_ar = f"Slightly underperforming by {underperf_pct:.1f}%. Consider cleaning.", f"أداء أقل بنسبة {underperf_pct:.1f}٪. فكر في التنظيف."
                elif underperf_pct < 30:
                    sc,si = "#e67e22","⚠️"
                    msg_en,msg_ar = f"Underperforming by {underperf_pct:.1f}%. Inspection recommended.", f"أداء أقل بنسبة {underperf_pct:.1f}٪. يُنصح بالفحص."
                else:
                    sc,si = "#e74c3c","🚨"
                    msg_en,msg_ar = f"Severely underperforming by {underperf_pct:.1f}%. Immediate inspection required!", f"أداء أقل بشكل حاد بنسبة {underperf_pct:.1f}٪. يلزم الفحص الفوري!"

                st.markdown(f"""<div style="background:{BG_CARD};border:1px solid {sc};border-radius:10px;
                     padding:16px 20px;margin-top:16px;{RTL}">
                    <div style="font-size:1.05rem;font-weight:800;color:{sc};">
                        {si} {msg_ar if IS_AR else msg_en}
                    </div></div>""", unsafe_allow_html=True)

                perf_pct = max(0, 100 - underperf_pct)
                st.markdown(f"""<div style="margin-top:16px;{RTL}">
                    <div style="font-family:Space Mono,monospace;font-size:0.72rem;color:{TXT_M};
                         letter-spacing:3px;margin-bottom:8px;">{t("PERFORMANCE GAUGE","مقياس الأداء")}</div>
                    <div style="background:{BAR_BG};border-radius:8px;height:16px;">
                        <div style="width:{perf_pct:.0f}%;height:16px;border-radius:8px;
                             background:linear-gradient(90deg,#e74c3c,#f5a623,#2ecc71);"></div>
                    </div>
                    <div style="display:flex;justify-content:space-between;font-size:0.75rem;color:{TXT_M};margin-top:4px;">
                        <span>0%</span><span>{t("Performance","الأداء")}: {perf_pct:.0f}%</span><span>100%</span>
                    </div></div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 3 — POWER FORECAST
# ═══════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════
# TAB 3 — POWER FORECAST
# FIX 4: If LSTM files missing, show a simulation-based forecast
#         so the page is always useful, not just an error message.
# ═══════════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        f'<div class="section-title">{t("SOLAR POWER FORECAST","توقع الطاقة الشمسية")}</div>',
        unsafe_allow_html=True,
    )

    # ── Always show the input controls regardless of model availability
    st.markdown(f"""<div class="perf-card">
        <div style="font-size:0.92rem;color:{TXT_S};">
            {t('Predict solar power output for the next hours based on weather conditions.',
               'توقع إنتاج الطاقة الشمسية للساعات القادمة بناءً على الظروف الجوية.')}
        </div>
        <div style="font-size:0.82rem;color:{TXT_M};margin-top:6px;">
            {t('Adjust the sliders and click Generate Forecast.',
               'اضبط المؤشرات ثم اضغط توليد التوقع.')}
        </div>
    </div>""", unsafe_allow_html=True)

    f1, f2, f3 = st.columns(3)
    with f1: f_irr = st.slider(t("Irradiation", "الإشعاع"),       0.0, 1.5, 0.7, 0.05)
    with f2: f_amb = st.slider(t("Ambient Temp (C)", "درجة المحيط"), 15.0, 45.0, 28.0, 0.5)
    with f3: f_mod = st.slider(t("Module Temp (C)", "درجة اللوح"),  20.0, 70.0, 40.0, 0.5)
    steps = st.slider(t("Forecast hours ahead", "ساعات التوقع"), 1, 12, 6)

    if st.button(t("Generate Forecast", "توليد التوقع"), use_container_width=True):
        with st.spinner(t("Generating forecast...", "جاري توليد التوقع...")):

            if lstm_model is not None:
                # ── Real LSTM forecast
                try:
                    import tensorflow as tf
                    seq_len  = lstm_meta["seq_len"]
                    features = lstm_meta["features"]
                    seq_df   = sample_data[features].tail(seq_len).copy()
                    sequence = lstm_scaler.transform(seq_df.values).copy()
                    forecasts, hours = [], []
                    for step in range(steps * 4):
                        inp  = sequence[-seq_len:].reshape(1, seq_len, len(features))
                        pred = lstm_model.predict(inp, verbose=0)[0][0]
                        hour_val = (datetime.now().hour + step // 4) % 24
                        new_row  = lstm_scaler.transform([[f_irr, f_amb, f_mod, hour_val, 0]])[0]
                        new_row[-1] = pred
                        sequence = np.vstack([sequence, new_row])
                        dummy = np.zeros((1, len(features)))
                        dummy[0, -1] = pred
                        ac_pred = max(0, lstm_scaler.inverse_transform(dummy)[0, -1])
                        forecasts.append(ac_pred)
                        hours.append(f"{(datetime.now().hour + step//4) % 24:02d}:{(step%4)*15:02d}")
                    model_label = t(f"LSTM Model (R²={lstm_meta.get('r2',0):.3f})", "نموذج LSTM")
                except Exception as e:
                    st.error(f"Forecast error: {e}")
                    forecasts, hours, model_label = [], [], ""
            else:
                # ── FIX 4: Simulation fallback when LSTM files not present
                # Uses a physics-based solar power estimate so the page is
                # always functional. Place lstm_model.keras, lstm_scaler.pkl,
                # lstm_metadata.json, sample_data.csv in the app root folder
                # to enable the real ML forecast.
                forecasts, hours = [], []
                base_power = f_irr * 3200 * (1 - (f_mod - 25) * 0.004)
                base_power = max(0, base_power)
                start_hour = datetime.now().hour
                for step in range(steps * 4):
                    hour_of_day = (start_hour + step // 4) % 24
                    # Solar curve — peaks at midday
                    solar_factor = max(0, np.sin((hour_of_day - 6) / 14 * np.pi)) if 6 <= hour_of_day <= 20 else 0
                    noise = np.random.normal(0, base_power * 0.03)
                    ac_pred = max(0, base_power * solar_factor + noise) / 1000
                    forecasts.append(ac_pred)
                    hours.append(f"{hour_of_day:02d}:{(step%4)*15:02d}")
                model_label = t("Physics Simulation (LSTM files not found)", "محاكاة فيزيائية")

            if forecasts:
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=hours, y=forecasts, mode="lines+markers",
                    line=dict(color="#f5a623", width=3),
                    marker=dict(size=6, color="#f5a623"),
                    fill="tozeroy", fillcolor="rgba(245,166,35,0.1)",
                    name=model_label,
                ))
                fig.update_layout(
                    title=dict(
                        text=t("Solar Power Forecast (kW)", "توقع الطاقة الشمسية (كيلوواط)"),
                        font=dict(color=TXT, size=16)
                    ),
                    xaxis=dict(title=t("Time","الوقت"), color=TXT_M, gridcolor=BORDER),
                    yaxis=dict(title=t("AC Power (kW)","طاقة AC"), color=TXT_M, gridcolor=BORDER),
                    paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
                    font=dict(color=TXT), height=400, showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

                if lstm_model is None:
                    st.info(t(
                        "Showing physics-based simulation. To use the real LSTM model, "
                        "place these files in your app root folder: "
                        "lstm_model.keras, lstm_scaler.pkl, lstm_metadata.json, sample_data.csv",
                        "يتم عرض محاكاة فيزيائية. لاستخدام نموذج LSTM الحقيقي، ضع الملفات في مجلد التطبيق."
                    ))

                avg_power    = np.mean(forecasts)
                max_power    = np.max(forecasts)
                total_energy = sum(forecasts) * 0.25 / 1000

                s1, s2, s3 = st.columns(3)
                for col, le, la, val, unit, color in [
                    (s1,"AVG POWER",    "متوسط الطاقة",    avg_power,    "kW",  "#f5a623"),
                    (s2,"PEAK POWER",   "ذروة الطاقة",     max_power,    "kW",  "#e74c3c"),
                    (s3,"EST. ENERGY",  "الطاقة المتوقعة", total_energy, "MWh", "#2ecc71"),
                ]:
                    with col:
                        st.markdown(f"""<div class="metric-card" style="text-align:center;">
                            <div class="metric-label">{la if IS_AR else le}</div>
                            <div class="metric-value" style="color:{color};">{val:,.2f}</div>
                            <div style="font-size:0.8rem;color:{TXT_M};margin-top:4px;">{unit}</div>
                        </div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# TAB 4 — HISTORY
# Normal users: own scans only. Admin: all users' scans.
# Data loaded from SQLite — persists across sessions/redeploys.
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f'<div class="section-title">{t("SCAN HISTORY","سجل الفحص")}</div>',
                unsafe_allow_html=True)

    # ── Load from database (admin sees all, users see own)
    user_history = db_get_scans(st.session_state.auth_email, admin=is_admin)

    # Admin: show user filter + total user count
    if is_admin and user_history:
        all_users_in_scans = sorted(set(h["email"] for h in user_history))
        selected_user = st.selectbox(
            t("Filter by user (admin view)", "تصفية حسب المستخدم"),
            ["All Users"] + all_users_in_scans,
            key="hist_filter"
        )
        if selected_user != "All Users":
            user_history = [h for h in user_history if h["email"] == selected_user]
        st.markdown(
            f'<div style="font-size:0.82rem;color:{TXT_M};margin-bottom:12px;">'
            f'👑 Admin view — showing {len(user_history)} scans'
            f'{"" if selected_user == "All Users" else f" for {selected_user}"}'
            f'</div>',
            unsafe_allow_html=True
        )

    if not user_history:
        st.markdown(f"""
        <div style="text-align:center;color:{TXT_M};padding:40px;">
            <div style="font-size:2rem;margin-bottom:12px;">📋</div>
            <div>{t('No scans yet — upload an image to get started',
                     'لا توجد عمليات فحص بعد — ارفع صورة للبدء')}</div>
        </div>""", unsafe_allow_html=True)
    else:
        total    = len(user_history)
        critical = sum(1 for h in user_history if h["severity"] == "critical")
        warning  = sum(1 for h in user_history if h["severity"] == "warning")
        good     = sum(1 for h in user_history if h["severity"] == "info")

        s1, s2, s3, s4 = st.columns(4)
        for col, le, la, val, color in [
            (s1,"TOTAL SCANS","إجمالي الفحوصات",total,"#f5a623"),
            (s2,"CRITICAL","حرج",critical,"#e74c3c"),
            (s3,"WARNINGS","تحذيرات",warning,"#f5a623"),
            (s4,"GOOD","جيد",good,"#2ecc71"),
        ]:
            with col:
                st.markdown(f"""<div class="metric-card" style="text-align:center;">
                    <div class="metric-label">{la if IS_AR else le}</div>
                    <div class="metric-value" style="color:{color};">{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown(f'<div class="section-title">{t("RECENT SCANS","الفحوصات الأخيرة")}</div>',
                    unsafe_allow_html=True)

        for h in user_history:
            disp_h    = h["display_ar"] if IS_AR else h["display_en"]
            sev_color = {"critical":"#e74c3c","warning":"#f5a623","info":"#2ecc71"}[h["severity"]]
            # Admin: show which user this scan belongs to
            user_tag  = f'<span style="color:{TXT_M};font-size:0.78rem;">👤 {h["email"]}</span><br>' if is_admin else ""
            st.markdown(f"""
            <div class="history-card">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>
                        <span style="font-size:1.2rem;">{h['icon']}</span>
                        <span style="font-weight:700;margin-left:8px;color:{TXT};">{disp_h}</span>
                        <span style="margin-left:12px;color:{TXT_M};font-size:0.82rem;">
                            {h['confidence']:.0%} {t('confidence','ثقة')}
                        </span>
                    </div>
                    <div style="text-align:right;">
                        {user_tag}
                        <span style="color:{sev_color};font-size:0.82rem;font-weight:700;">
                            {h['severity'].upper()}
                        </span><br>
                        <span style="color:{TXT_M};font-size:0.78rem;">{h['time']}</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

        # Clear button — admin clears selected user, normal user clears own
        clear_label = t("🗑 Clear My History", "🗑 مسح سجلي")
        if is_admin and "selected_user" in dir() and selected_user != "All Users":
            clear_label = f"🗑 Clear {selected_user}'s History"
        if st.button(clear_label):
            target = selected_user if (is_admin and selected_user != "All Users") else st.session_state.auth_email
            db_delete_scans(target)
            st.rerun()

if is_admin:
    with tab5:
        st.markdown(f"""
        <div style="background:{BG_CARD};border:1px solid #f5a623;border-radius:12px;
             padding:16px 20px;margin-bottom:20px;">
            <div style="font-size:0.8rem;color:#f5a623;font-family:Space Mono,monospace;
                 letter-spacing:2px;margin-bottom:6px;">👑 ADMIN PANEL</div>
            <div style="font-size:0.92rem;color:{TXT_S};">
                Full access to all users' data, scans, and the dataset browser.
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── All registered users
        st.markdown(f'<div class="section-title">REGISTERED USERS</div>', unsafe_allow_html=True)
        all_users = db_get_all_users()
        if all_users:
            for u in all_users:
                role_color = "#f5a623" if u["role"] == "admin" else TXT_M
                st.markdown(f"""
                <div class="history-card">
                    <div style="display:flex;justify-content:space-between;">
                        <div style="color:{TXT};">📧 {u['email']}</div>
                        <div>
                            <span style="color:{role_color};font-size:0.82rem;font-weight:700;">
                                {u['role'].upper()}
                            </span>
                            <span style="color:{TXT_M};font-size:0.78rem;margin-left:12px;">
                                joined {u['created'][:10]}
                            </span>
                        </div>
                    </div>
                </div>""", unsafe_allow_html=True)

        # ── Dataset browser
        st.markdown(f'<div class="section-title">DATASET BROWSER</div>', unsafe_allow_html=True)
        render_dataset_tab(
            TXT=TXT, TXT_M=TXT_M, TXT_S=TXT_S,
            BG_CARD=BG_CARD, BORDER=BORDER, BAR_BG=BAR_BG,
            IS_AR=IS_AR, DM=DM,
        )
# ─────────────────────────────────────────────────────────────────────
# SITE FOOTER — centered, professional, branded
# ─────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="site-footer">
    <div class="footer-brand">
        ☀️ &nbsp;@Solar<span>Inspect</span> AI 2026
    </div>
    <div class="footer-tagline">
        {t("SOLAR PANEL DEFECT DETECTION PLATFORM",
           "منصة الكشف عن عيوب الألواح الشمسية")}
    </div>
</div>
""", unsafe_allow_html=True)
