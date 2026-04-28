
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

DATABASE_URL = os.environ.get("DATABASE_URL")   # set on Render
ADMIN_EMAIL  = "reemya185@gmail.com"
TOKEN_KEY    = "solar_session_token"

# ── Supabase Storage config (for CSV persistence)
# Storage credentials used only by dataset_tab.py (admin merge).
# app.py does NOT write to the CSV — all mutation is admin-controlled.


def _db():
    conn = psycopg2.connect(
        host=os.environ.get("DB_HOST"),
        port=int(os.environ.get("DB_PORT", "5432")),
        dbname=os.environ.get("DB_NAME"),
        user=os.environ.get("DB_USER"),
        password=os.environ.get("DB_PASSWORD"),
        sslmode="require",
    )
    return conn


def _hash(password: str) -> str:
    """SHA-256 password hash."""
    return hashlib.sha256(password.encode()).hexdigest()


# ── Token helpers
def _create_token(email: str) -> str:
    """Create a 30-day session token and save it to DB."""
    from datetime import timedelta
    token   = _secrets.token_hex(32)
    now     = datetime.now()
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
    """
    Check if a token is valid and not expired.
    Returns email string if valid, None otherwise.
    """
    if not token:
        return None
    conn = _db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT email, expires_at FROM tokens WHERE token = %s", (token,)
            )
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
    """Delete a token on logout."""
    conn = _db()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM tokens WHERE token = %s", (token,))
        conn.commit()
    finally:
        conn.close()


# ── User helpers
def _get_role(email: str) -> str:
    """Return 'admin' or 'user' for a given email."""
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
    """Register a new user. Returns (success, message)."""
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
    """
    Verify credentials.
    Returns (success, message, token_or_None).
    """
    conn = _db()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pw_hash FROM users WHERE email = %s", (email.lower().strip(),)
            )
            row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        return False, "No account found with this email.", None
    if row[0] != _hash(password):
        return False, "Incorrect password.", None
    token = _create_token(email.lower().strip())
    return True, "Logged in successfully.", token



# ── Scan persistence helpers
def db_save_scan(
    email: str, pred_class: str, info: dict, confidence: float,
    irradiation: float = None, ambient_temp: float = None,
    module_temp: float = None, ac_power: float = None,
):
    """
    Save scan to PostgreSQL only.
    The static CSV dataset is NEVER touched here — only the admin can merge via dataset_tab.py.
    merged_into_dataset starts as FALSE so it appears in the admin's pending queue.
    """
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
                       confidence, severity, icon, merged_into_dataset)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, FALSE)
                """, (
                    email,
                    datetime.now(),
                    pred_class, info["display_en"], info["display_ar"],
                    round(confidence, 4), info["severity"], info["icon"],
                ))
            conn.commit()
    finally:
        conn.close()


def db_get_scans(email: str, admin: bool = False) -> list:
    """
    Load scans from the database.
    Normal users: only their own. Admin: all users.
    """
    conn = _db()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            if admin:
                cur.execute("""
                    SELECT email, scanned_at, defect_type, display_en, display_ar,
                           confidence, severity, icon
                    FROM scans ORDER BY scanned_at DESC
                """)
            else:
                cur.execute("""
                    SELECT email, scanned_at, defect_type, display_en, display_ar,
                           confidence, severity, icon
                    FROM scans WHERE email = %s ORDER BY scanned_at DESC
                """, (email,))
            rows = cur.fetchall()
    finally:
        conn.close()
    return [
        {
            "email":      r["email"],
            "time":       r["scanned_at"].strftime("%Y-%m-%d %H:%M") if hasattr(r["scanned_at"], "strftime") else str(r["scanned_at"]),
            "class":      r["defect_type"],
            "display_en": r["display_en"],
            "display_ar": r["display_ar"],
            "confidence": r["confidence"],
            "severity":   r["severity"],
            "icon":       r["icon"],
        }
        for r in rows
    ]


def db_delete_scans(email: str):
    """Delete all scans for a specific user."""
    conn = _db()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM scans WHERE email = %s", (email,))
        conn.commit()
    finally:
        conn.close()


def _ensure_admin():
    """
    Auto-create the admin account on every app startup.
    Password is read from ADMIN_PASSWORD env variable.
    """
    admin_pw = os.environ.get("ADMIN_PASSWORD", "SolarAdmin2026!")
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


# Run on every startup
_ensure_admin()


def db_get_all_users() -> list:
    """Admin only: get list of all registered users."""
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


def render_auth_page():
    """
    Full-screen login / signup UI.
    On successful login, a token is stored in st.session_state.
    """
    for k, v in [("logged_in", False), ("auth_email", ""), ("session_token", "")]:
        if k not in st.session_state:
            st.session_state[k] = v

    if not st.session_state.logged_in and st.session_state.session_token:
        email = _validate_token(st.session_state.session_token)
        if email:
            st.session_state.logged_in  = True
            st.session_state.auth_email = email
            st.session_state.user_email = email
            return

    if st.session_state.logged_in:
        return

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
            with st.form("login_form", enter_to_submit=False):
                email    = st.text_input("Email", placeholder="you@example.com")
                password = st.text_input("Password", type="password")
                submitted = st.form_submit_button("Log In", use_container_width=True)
            if submitted:
                ok, msg, token = auth_login(email, password)
                if ok:
                    st.session_state.logged_in     = True
                    st.session_state.auth_email    = email.lower().strip()
                    st.session_state.user_email    = email.lower().strip()
                    st.session_state.session_token = token
                    st.rerun()
                else:
                    st.error(msg)

        with tab_signup:
            with st.form("signup_form", enter_to_submit=False):
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
# 3. THEME — dark mode only
# ─────────────────────────────────────────────────────────────────────
for key, default in [
    ("lang",          "en"),
    ("history",       []),
    ("dark_mode",     True),
    ("user_email",    ""),
    ("logged_in",     False),
    ("auth_email",    ""),
    ("session_token", ""),
]:
    if key not in st.session_state:
        st.session_state[key] = default

render_auth_page()

def t(en, ar):
    return ar if st.session_state.lang == "ar" else en

RTL   = "direction:rtl;text-align:right;" if st.session_state.lang == "ar" else ""
FONT  = "'Cairo', sans-serif"  if st.session_state.lang == "ar" else "'Syne', sans-serif"
IS_AR = st.session_state.lang == "ar"
DM    = True

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
    background-color:{INPUT_BG} !important;
    color:{TXT} !important;
    border-color:{BORDER} !important;
    border-radius:8px !important;
}}
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
.user-badge {{ background:{BG_CARD}; border:1px solid {BORDER}; border-radius:8px; padding:6px 14px; font-size:0.82rem; color:{TXT_M}; display:inline-block; margin-bottom:8px; }}

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
    files = ["performance_model.pkl", "performance_scaler.pkl", "model_metadata.json", "typical_values.json"]
    if any(not os.path.exists(f) for f in files):
        return None, None, None, None
    try:
        model  = joblib.load("performance_model.pkl")
        scaler = joblib.load("performance_scaler.pkl")
        with open("model_metadata.json") as f:  meta    = json.load(f)
        with open("typical_values.json")  as f: typical = json.load(f)
        return model, scaler, meta, typical
    except Exception:
        return None, None, None, None

@st.cache_resource
def load_lstm():
    required = ["lstm_model.keras", "lstm_scaler.pkl", "lstm_metadata.json", "sample_data.csv"]
    if any(not os.path.exists(f) for f in required):
        return None, None, None, None
    try:
        import tensorflow as tf
        model  = tf.keras.models.load_model("lstm_model.keras")
        scaler = joblib.load("lstm_scaler.pkl")
        with open("lstm_metadata.json") as f: meta = json.load(f)
        sample = pd.read_csv("sample_data.csv")
        assert "seq_len"  in meta
        assert "features" in meta
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
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    return tf(image).unsqueeze(0)

# ─────────────────────────────────────────────────────────────────────
# 5. PDF — styled bilingual report with Amiri font for Arabic
# ─────────────────────────────────────────────────────────────────────
AMIRI_PATH = os.path.join("fonts", "Amiri-Regular.ttf")

def _safe_en(text: str) -> str:
    return "".join(
        c for c in str(text)
        if _ud.category(c) not in ("So", "Cs") and ord(c) < 0x0250
    ).strip()

def _has_amiri() -> bool:
    return os.path.exists(AMIRI_PATH)

def _ar(text: str) -> str:
    try:
        import arabic_reshaper
        from bidi.algorithm import get_display
        reshaped = arabic_reshaper.reshape(str(text))
        return get_display(reshaped)
    except ImportError:
        return str(text)

_AR_SECTIONS = {
    "Detection Result":    "نتيجة الكشف",
    "Description":         "الوصف",
    "Recommended Action":  "الإجراء الموصى به",
    "Maintenance Tips":    "نصائح الصيانة",
}


class SolarPDF(FPDF):
    def __init__(self, lang="en"):
        super().__init__()
        self.lang     = lang
        self.amiri_ok = _has_amiri()
        self.set_margins(18, 18, 18)
        self.set_auto_page_break(auto=True, margin=20)
        if self.amiri_ok:
            self.add_font("Amiri", "", AMIRI_PATH, uni=True)

    @property
    def W(self):
        return self.w - self.l_margin - self.r_margin

    def header(self):
        self.set_fill_color(255, 255, 255)
        self.rect(0, 0, self.w, 18, "F")
        self.set_y(3)
        self.set_font("Helvetica", "B", 18)
        self.set_text_color(245, 166, 35)
        self.cell(0, 10, "SolarInspect AI", align="C")
        self.ln(10)
        self.set_draw_color(245, 166, 35)
        self.set_line_width(0.6)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.set_line_width(0.2)
        self.set_draw_color(200, 210, 220)
        self.set_text_color(0, 0, 0)
        self.ln(6)

    def footer(self):
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(180, 190, 200)
        self.cell(0, 8, "SolarInspect AI", align="C")
        self.set_text_color(0, 0, 0)

    def section_header(self, title_en: str):
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
        self.set_draw_color(200, 210, 220)
        self.line(self.l_margin, self.get_y(), self.l_margin + self.W, self.get_y())
        self.ln(4)

    def body_line(self, label_en: str, value_en: str, value_ar: str = "", label_ar: str = ""):
        if self.lang == "ar" and self.amiri_ok:
            self.set_font("Amiri", "", 10)
            self.set_text_color(100, 120, 140)
            lbl = _ar(label_ar) if label_ar else _ar(label_en)
            self.cell(self.W, 7, lbl, ln=True, align="R")
            self.set_font("Amiri", "", 12)
            self.set_text_color(20, 20, 20)
            val = _ar(value_ar) if value_ar else _ar(value_en)
            self.set_x(self.l_margin)
            self.cell(self.W, 8, val, ln=True, align="R")
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
        colors = {"critical": (210, 60, 50), "warning": (200, 140, 30), "info": (40, 160, 110)}
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
    is_ar = (lang == "ar")
    pdf = SolarPDF(lang=lang)
    pdf.add_page()

    if is_ar and pdf.amiri_ok:
        pdf.set_font("Amiri", "", 16)
        pdf.set_text_color(26, 40, 60)
        pdf.cell(pdf.W, 12, _ar("تقرير الفحص"), ln=True, align="C")
        pdf.set_font("Amiri", "", 10)
        pdf.set_text_color(100, 120, 140)
        pdf.cell(pdf.W, 6, datetime.now().strftime("%Y-%m-%d  %H:%M"), ln=True, align="C")
        if user_email:
            pdf.cell(pdf.W, 6, _ar(f"الحساب: {user_email}"), ln=True, align="C")
    else:
        pdf.set_font("Helvetica", "B", 18)
        pdf.set_text_color(26, 40, 60)
        pdf.cell(pdf.W, 12, "Scan Report", ln=True, align="C")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(100, 120, 140)
        pdf.cell(pdf.W, 6, datetime.now().strftime("%A, %d %B %Y  ·  %H:%M"), ln=True, align="C")
        if user_email:
            pdf.cell(pdf.W, 6, f"Account: {user_email}", ln=True, align="C")
    pdf.set_text_color(0, 0, 0)
    pdf.ln(6)
    pdf.hr()

    pdf.section_header("Detection Result")
    pdf.body_line("Defect Type",      info["display_en"],       info["display_ar"],   "نوع العيب")
    pdf.body_line("Confidence",       f"{confidence:.1%}",      f"{confidence:.1%}",  "مستوى الثقة")
    pdf.body_line("Severity",         info["severity"].upper(),  info["severity"].upper(), "مستوى الخطورة")
    if underperf is not None:
        pdf.body_line("Underperformance", f"{underperf:.1f}%",  f"{underperf:.1f}%",  "الأداء دون المستوى")
    pdf.ln(4)

    pdf.info_box(
        f"Urgency: {info['urgency_en']}",
        f"مهم: {info['urgency_ar']}",
        severity=info["severity"],
    )
    pdf.section_header("Description")
    pdf.body_para(info["desc_en"], info["desc_ar"])
    pdf.section_header("Recommended Action")
    pdf.body_para(info["action_en"], info["action_ar"])
    pdf.section_header("Maintenance Tips")
    pdf.tips_table(info["tips_en"], info["tips_ar"])

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

is_admin = _get_role(st.session_state.auth_email) == "admin"

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
    lang_label   = "AR" if st.session_state.lang == "en" else "EN"
    user_initial = st.session_state.auth_email[0].upper() if st.session_state.auth_email else "?"
    short_email  = st.session_state.auth_email[:20] + "…" if len(st.session_state.auth_email) > 20 else st.session_state.auth_email
    admin_tag    = " · ADMIN" if is_admin else ""

    st.markdown(f"""
    <div style="display:flex;flex-direction:column;align-items:flex-end;padding-top:18px;gap:6px;">
        <div style="background:{BG_CARD};border:1px solid {BORDER};border-radius:8px;
                    padding:5px 10px;display:flex;align-items:center;gap:7px;
                    font-size:0.78rem;color:{TXT_M};white-space:nowrap;">
            <span style="background:#f5a623;color:#000;border-radius:50%;
                         width:20px;height:20px;display:inline-flex;align-items:center;
                         justify-content:center;font-weight:800;font-size:0.68rem;flex-shrink:0;">{user_initial}</span>
            <span>{short_email}<span style="color:#f5a623;font-weight:700;">{admin_tag}</span></span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    b1, b2, b3 = st.columns(3)
    with b1:
        if st.button(f"🌐 {lang_label}", use_container_width=True, key="btn_lang"):
            st.session_state.lang = "ar" if st.session_state.lang == "en" else "en"
            st.rerun()
    with b2:
        st.markdown(f'<div style="height:38px;background:{BG_CARD};border:1px solid {BORDER};border-radius:8px;display:flex;align-items:center;justify-content:center;font-size:0.8rem;color:{TXT_M};">👤</div>', unsafe_allow_html=True)
    with b3:
        if st.button("🚪 Out", use_container_width=True, key="btn_logout"):
            if st.session_state.session_token:
                _delete_token(st.session_state.session_token)
            st.session_state.logged_in     = False
            st.session_state.auth_email    = ""
            st.session_state.session_token = ""
            st.session_state.history       = []
            st.rerun()

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

        st.markdown(f'<div class="section-title">{t("OPTIONAL: ADD SENSOR READINGS TO DATASET","بيانات المستشعر (اختياري)")}</div>', unsafe_allow_html=True)
        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1: s_irr = st.number_input(t("Irradiation","الإشعاع"),     min_value=0.0, max_value=2.0,   value=0.0, step=0.01, key="s_irr", format="%.2f")
        with sc2: s_amb = st.number_input(t("Ambient °C","حرارة المحيط"), min_value=-10.0,max_value=60.0, value=0.0, step=0.1,  key="s_amb", format="%.1f")
        with sc3: s_mod = st.number_input(t("Module °C","حرارة اللوح"),   min_value=-10.0,max_value=90.0, value=0.0, step=0.1,  key="s_mod", format="%.1f")
        with sc4: s_ac  = st.number_input(t("AC Power (kW)","طاقة AC"),   min_value=0.0, max_value=500.0, value=0.0, step=0.1,  key="s_ac",  format="%.2f")

        import hashlib as _hl
        file_hash = _hl.md5(uploaded.getvalue()).hexdigest()
        if "saved_hashes" not in st.session_state:
            st.session_state.saved_hashes = set()
        if file_hash not in st.session_state.saved_hashes:
            db_save_scan(
                st.session_state.auth_email, pred_class, info, confidence,
                irradiation  = s_irr  if s_irr  > 0 else None,
                ambient_temp = s_amb  if s_amb  > 0 else None,
                module_temp  = s_mod  if s_mod  > 0 else None,
                ac_power     = s_ac   if s_ac   > 0 else None,
            )
            st.session_state.saved_hashes.add(file_hash)

        col_img, col_res = st.columns([3, 2], gap="large")

        with col_img:
            st.markdown(f'<div class="section-title">{t("UPLOADED IMAGE","الصورة المرفوعة")}</div>', unsafe_allow_html=True)
            st.image(image, use_container_width=True)

        with col_res:
            st.markdown(f'<div class="section-title">{t("SEVERITY","مستوى الخطورة")}</div>', unsafe_allow_html=True)
            badge_open, _ = BADGE[sev]
            st.markdown(f'{badge_open}{info["icon"]} {display}</span>', unsafe_allow_html=True)

            st.markdown(f'<div class="section-title" style="margin-top:20px;">{t("CONFIDENCE","مستوى الثقة")}</div>', unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">{t("MODEL CONFIDENCE","ثقة النموذج")}</div>
                <div class="metric-value" style="color:#f5a623;">{confidence:.0%}</div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill" style="width:{int(confidence*100)}%"></div>
                </div>
            </div>""", unsafe_allow_html=True)

            st.markdown(f'<div class="section-title">{t("ANALYSIS","التحليل")}</div>', unsafe_allow_html=True)
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

        st.markdown(f'<div class="section-title">{t("MAINTENANCE TIPS","نصائح الصيانة")}</div>', unsafe_allow_html=True)
        tips     = info["tips_ar"] if IS_AR else info["tips_en"]
        tip_cols = st.columns(len(tips))
        for i, tip in enumerate(tips):
            with tip_cols[i]:
                st.markdown(
                    f'<div class="tip-card"><div style="font-size:1.2rem;margin-bottom:6px;">💡</div>'
                    f'<div class="tip-text">{tip}</div></div>',
                    unsafe_allow_html=True,
                )

        st.markdown(f'<div class="section-title">{t("EXPORT REPORT","تصدير التقرير")}</div>', unsafe_allow_html=True)
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
    st.markdown(f'<div class="section-title">{t("PANEL PERFORMANCE ANALYZER","محلل أداء اللوح")}</div>', unsafe_allow_html=True)

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
            irradiation = st.number_input(t("Irradiation (W/m2/1000)", "الإشعاع"), min_value=0.0, max_value=2.0, value=float(round(tv["IRRADIATION"], 3)), step=0.01)
        with c2:
            ambient_temp = st.number_input(t("Ambient Temp (C)", "درجة حرارة المحيط"), min_value=-10.0, max_value=60.0, value=float(round(tv["AMBIENT_TEMPERATURE"], 1)), step=0.1)
        with c3:
            module_temp = st.number_input(t("Module Temp (C)", "درجة حرارة اللوح"), min_value=-10.0, max_value=90.0, value=float(round(tv["MODULE_TEMPERATURE"], 1)), step=0.1)
        c4, c5 = st.columns(2)
        with c4:
            dc_power = st.number_input(t("DC Power (kW)", "طاقة DC"), min_value=0.0, max_value=500000.0, value=float(round(tv["DC_POWER"], 1)), step=100.0)
        with c5:
            ac_power = st.number_input(t("AC Power (kW)", "طاقة AC"), min_value=0.0, max_value=500000.0, value=float(round(tv["AC_POWER"], 1)), step=100.0)

        if st.button(t("Analyze Performance", "تحليل الأداء"), use_container_width=True):
            if irradiation == 0:
                st.info(t("Irradiation is 0 — panel is not generating power (night or fully shaded).", "الإشعاع = 0 — اللوح لا ينتج طاقة."))
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
with tab3:
    st.markdown(
        f'<div class="section-title">{t("SOLAR POWER FORECAST","توقع الطاقة الشمسية")}</div>',
        unsafe_allow_html=True,
    )

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
                forecasts, hours = [], []
                PANEL_CAPACITY_KW = 3.5
                INVERTER_EFF      = 0.96
                TEMP_COEFF        = -0.004
                start_hour = datetime.now().hour
                start_min  = datetime.now().minute

                for step in range(steps * 4):
                    total_minutes  = start_min + step * 15
                    hour_of_day    = (start_hour + total_minutes // 60) % 24
                    minute_of_hour = total_minutes % 60

                    if 6 <= hour_of_day <= 19:
                        solar_angle  = (hour_of_day - 6 + minute_of_hour / 60) / 13 * np.pi
                        solar_factor = max(0, np.sin(solar_angle))
                    else:
                        solar_factor = 0.0

                    temp_factor = 1 + TEMP_COEFF * max(0, f_mod - 25)
                    dc_power    = f_irr * PANEL_CAPACITY_KW * solar_factor * temp_factor
                    ac_power    = max(0, dc_power * INVERTER_EFF)
                    noise       = np.random.normal(0, ac_power * 0.02)
                    ac_power    = max(0, ac_power + noise)
                    forecasts.append(round(ac_power, 3))
                    h_label = (start_hour + total_minutes // 60) % 24
                    hours.append(f"{h_label:02d}:{minute_of_hour:02d}")

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
                    title=dict(text=t("Solar Power Forecast (kW)", "توقع الطاقة الشمسية (كيلوواط)"), font=dict(color=TXT, size=16)),
                    xaxis=dict(title=t("Time","الوقت"), color=TXT_M, gridcolor=BORDER),
                    yaxis=dict(title=t("AC Power (kW)","طاقة AC"), color=TXT_M, gridcolor=BORDER),
                    paper_bgcolor=BG_CARD, plot_bgcolor=BG_CARD,
                    font=dict(color=TXT), height=400, showlegend=False,
                )
                st.plotly_chart(fig, use_container_width=True)

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
# ═══════════════════════════════════════════════════════════════
with tab4:
    st.markdown(f'<div class="section-title">{t("SCAN HISTORY","سجل الفحص")}</div>', unsafe_allow_html=True)

    try:
        user_history = db_get_scans(st.session_state.auth_email, admin=is_admin)
    except Exception as e:
        st.error(t(f"Could not load history: {e}", f"خطأ في تحميل السجل: {e}"))
        user_history = []

    if is_admin and user_history:
        all_emails = sorted(set(h["email"] for h in user_history))
        selected_user = st.selectbox(
            t("Filter by user", "تصفية حسب المستخدم"),
            ["All Users"] + all_emails,
            key="hist_filter"
        )
        if selected_user != "All Users":
            user_history = [h for h in user_history if h["email"] == selected_user]
        st.markdown(
            f'<div style="font-size:0.82rem;color:{TXT_M};margin-bottom:12px;">'
            f'👑 Admin view — {len(user_history)} scans'
            f'</div>', unsafe_allow_html=True
        )

    if not user_history:
        st.markdown(f"""
        <div style="text-align:center;color:{TXT_M};padding:40px;">
            <div style="font-size:2.5rem;margin-bottom:12px;">📋</div>
            <div style="font-size:1rem;">{t(
                'No scans yet — upload an image in the Image Scan tab to get started.',
                'لا توجد فحوصات بعد — ارفع صورة في تبويب الفحص للبدء.')}</div>
        </div>""", unsafe_allow_html=True)
    else:
        total    = len(user_history)
        critical = sum(1 for h in user_history if h["severity"] == "critical")
        warning  = sum(1 for h in user_history if h["severity"] == "warning")
        good     = sum(1 for h in user_history if h["severity"] == "info")

        s1, s2, s3, s4 = st.columns(4)
        for col, le, la, val, color in [
            (s1,"TOTAL","الإجمالي",    total,    "#f5a623"),
            (s2,"CRITICAL","حرج",      critical, "#e74c3c"),
            (s3,"WARNINGS","تحذيرات",  warning,  "#f5a623"),
            (s4,"GOOD","جيد",          good,     "#2ecc71"),
        ]:
            with col:
                st.markdown(f"""<div class="metric-card" style="text-align:center;">
                    <div class="metric-label">{la if IS_AR else le}</div>
                    <div class="metric-value" style="color:{color};">{val}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown(f'<div class="section-title">{t("RECENT SCANS","الفحوصات الأخيرة")}</div>', unsafe_allow_html=True)

        for h in user_history:
            disp_h    = h.get("display_ar","") if IS_AR else h.get("display_en","")
            sev       = h.get("severity","info")
            sev_color = {"critical":"#e74c3c","warning":"#f5a623","info":"#2ecc71"}.get(sev,"#aaa")
            conf      = h.get("confidence", 0)
            icon      = h.get("icon","🔍")
            timestamp = h.get("time","")
            user_line = f"👤 {h['email']} · " if is_admin else ""
            st.markdown(
                f'<div class="history-card">'
                f'<div style="display:flex;justify-content:space-between;align-items:center;gap:12px;">'
                f'<div><span style="font-size:1.1rem;">{icon}</span>'
                f'<span style="font-weight:700;margin-left:8px;color:{TXT};">{disp_h}</span>'
                f'<span style="margin-left:10px;color:{TXT_M};font-size:0.8rem;">{conf:.0%} {t("confidence","ثقة")}</span></div>'
                f'<div style="text-align:right;flex-shrink:0;">'
                f'<span style="color:{TXT_M};font-size:0.73rem;">{user_line}</span>'
                f'<span style="color:{sev_color};font-size:0.8rem;font-weight:700;">{sev.upper()}</span>'
                f'<span style="color:{TXT_M};font-size:0.73rem;"> · {timestamp}</span>'
                f'</div></div></div>',
                unsafe_allow_html=True
            )

        btn_label = t("🗑 Clear My History","🗑 مسح سجلي")
        if is_admin and "selected_user" in st.session_state and st.session_state.get("hist_filter","All Users") != "All Users":
            btn_label = f"🗑 Clear {st.session_state.hist_filter}'s History"
        if st.button(btn_label, key="clear_hist"):
            target = st.session_state.get("hist_filter", st.session_state.auth_email)
            if not is_admin or target == "All Users":
                target = st.session_state.auth_email
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

        st.markdown(f'<div class="section-title">DATASET BROWSER</div>', unsafe_allow_html=True)
        render_dataset_tab(
            TXT=TXT, TXT_M=TXT_M, TXT_S=TXT_S,
            BG_CARD=BG_CARD, BORDER=BORDER, BAR_BG=BAR_BG,
            IS_AR=IS_AR, DM=DM,
        )

# ─────────────────────────────────────────────────────────────────────
# SITE FOOTER
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
