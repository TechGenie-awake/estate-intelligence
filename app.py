"""
Estate Intelligence — Real Estate Property Price Prediction App
Built with Streamlit · Pandas · NumPy · Matplotlib · Joblib
"""

# ──────────────────────────────────────────────────────────────
# IMPORTS
# ──────────────────────────────────────────────────────────────
import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ──────────────────────────────────────────────────────────────
# PAGE CONFIG  (must be first Streamlit call)
# ──────────────────────────────────────────────────────────────
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Agent + report pipeline (Milestone 2)
try:
    from agent.graph import advisory_graph
    from report import AdvisoryReport, generate_pdf
    _AGENT_AVAILABLE = True
    _AGENT_IMPORT_ERROR = None
except Exception as _e:
    advisory_graph = None
    AdvisoryReport = None
    generate_pdf = None
    _AGENT_AVAILABLE = False
    _AGENT_IMPORT_ERROR = str(_e)

st.set_page_config(
    page_title="Estate Intelligence",
    layout="wide",
)

# ──────────────────────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "best_rf_model.joblib")
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "cleaned", "Housing_cleaned.csv")

model = None
scaler = None
feature_importances = None

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(DATA_PATH):
        model = joblib.load(MODEL_PATH)

        _df = pd.read_csv(DATA_PATH)
        _FEATURE_COLS = [c for c in _df.columns if c != "price"]
        _X = _df[_FEATURE_COLS]
        _y = _df["price"]
        _X_train, _, _, _ = train_test_split(_X, _y, test_size=0.2, random_state=42)
        scaler = StandardScaler()
        scaler.fit(_X_train)

        feature_importances = dict(zip(_FEATURE_COLS, model.feature_importances_))
except Exception:
    model = None
    scaler = None
    feature_importances = None

# ──────────────────────────────────────────────────────────────
# REAL FEATURE IMPORTANCES from trained Random Forest
# ──────────────────────────────────────────────────────────────
FEATURE_IMPORTANCES = {
    "area":             0.478,
    "bathrooms":        0.158,
    "airconditioning":  0.068,
    "stories":          0.052,
    "parking":          0.048,
    "furnishingstatus": 0.043,
    "bedrooms":         0.040,
    "basement":         0.030,
    "prefarea":         0.028,
    "hotwaterheating":  0.018,
    "guestroom":        0.016,
    "mainroad":         0.010,
}

# Column order must match exactly what the model was trained on
FEATURE_COLUMNS = [
    "area", "bedrooms", "bathrooms", "stories",
    "mainroad", "guestroom", "basement", "hotwaterheating",
    "airconditioning", "parking", "prefarea", "furnishingstatus"
]

# ──────────────────────────────────────────────────────────────
# HELPER — PREDICTION FUNCTION
# ──────────────────────────────────────────────────────────────
def predict_price(area, bedrooms, bathrooms, stories, mainroad,
                  guestroom, basement, hotwaterheating,
                  airconditioning, parking, prefarea, furnishingstatus):
    """
    Returns predicted price using loaded Random Forest model.
    Falls back to a hand-tuned formula if model is not available.
    Input is passed as a DataFrame to preserve column names for the model.
    """
    input_df = pd.DataFrame([{
        "area":             area,
        "bedrooms":         bedrooms,
        "bathrooms":        bathrooms,
        "stories":          stories,
        "mainroad":         mainroad,
        "guestroom":        guestroom,
        "basement":         basement,
        "hotwaterheating":  hotwaterheating,
        "airconditioning":  airconditioning,
        "parking":          parking,
        "prefarea":         prefarea,
        "furnishingstatus": furnishingstatus,
    }])[FEATURE_COLUMNS]  # enforce column order to match training

    if model is not None and scaler is not None:
        return model.predict(scaler.transform(input_df))[0]
    else:
        return (area * 480 + bedrooms * 45000 + bathrooms * 35000
                + stories * 20000 + airconditioning * 80000
                + prefarea * 60000)


# ──────────────────────────────────────────────────────────────
# HELPER — INDIAN ₹ FORMATTER
# ──────────────────────────────────────────────────────────────
def format_inr(amount):
    """Formats a number into Indian Rupee style with commas.
    Example: 4500000 → ₹45,00,000"""
    amount = int(round(amount))
    s = str(amount)
    if len(s) <= 3:
        return f"₹{s}"
    result = s[-3:]
    s = s[:-3]
    while s:
        result = s[-2:] + "," + result
        s = s[:-2]
    return f"₹{result}"


# ──────────────────────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --canvas:       #f7f8fa;
        --panel:        #ffffff;
        --panel-muted:  #f5f6f8;
        --border:       #e5e7eb;
        --border-soft:  #eef0f3;
        --text:         #0f172a;
        --text-muted:   #64748b;
        --text-subtle:  #94a3b8;
        --accent:       #2563eb;
        --accent-hover: #1d4ed8;
        --accent-soft:  #eff4ff;
        --success:      #059669;
        --success-soft: #ecfdf5;
        --warning:      #d97706;
        --warning-soft: #fffbeb;
        --danger:       #dc2626;
    }

    html, body, [class*="css"], .stApp, .stMarkdown, .stButton, .stTextInput, .stSelectbox, .stRadio {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
        -webkit-font-smoothing: antialiased;
    }

    .stApp { background: var(--canvas); }

    /* Hide Streamlit default chrome (top header bar, main menu, footer) */
    header[data-testid="stHeader"] { display: none !important; }
    #MainMenu { visibility: hidden !important; }
    footer { visibility: hidden !important; }
    div[data-testid="stToolbar"] { display: none !important; }
    div[data-testid="stDecoration"] { display: none !important; }
    div[data-testid="stStatusWidget"] { display: none !important; }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* ── Sidebar ─────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: var(--panel);
        border-right: 1px solid var(--border);
    }
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: var(--text-muted) !important;
        font-size: 0.88rem;
        line-height: 1.55;
    }
    section[data-testid="stSidebar"] .stMarkdown h1,
    section[data-testid="stSidebar"] .stMarkdown h2,
    section[data-testid="stSidebar"] .stMarkdown h3,
    section[data-testid="stSidebar"] .stMarkdown strong {
        color: var(--text) !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: var(--border-soft);
        margin: 1rem 0;
    }

    /* ── Headings ────────────────────────────────── */
    h1, h2, h3, h4, h5 { color: var(--text) !important; letter-spacing: -0.01em; }
    h3 { font-size: 1.15rem !important; font-weight: 600 !important; }
    h4 { font-size: 1rem !important; font-weight: 600 !important; }

    .main-title {
        font-size: 1.7rem;
        font-weight: 700;
        color: var(--text) !important;
        margin: 0 0 0.2rem 0;
        letter-spacing: -0.02em;
    }
    .sub-title {
        color: var(--text-muted) !important;
        font-size: 0.95rem;
        margin: 0 0 1.2rem 0;
    }

    /* ── Tabs ────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.25rem;
        border-bottom: 1px solid var(--border);
        background: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        color: var(--text-muted);
        padding: 0.65rem 1.1rem;
        font-weight: 500;
        font-size: 0.92rem;
        border: none;
        border-bottom: 2px solid transparent;
        border-radius: 0;
        transition: color 0.15s ease, border-color 0.15s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text);
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        color: var(--accent) !important;
        background: transparent !important;
        border-bottom: 2px solid var(--accent) !important;
        font-weight: 600;
    }
    .stTabs [data-baseweb="tab-panel"] { padding-top: 1.5rem; }

    /* ── Buttons ─────────────────────────────────── */
    .stButton > button, .stDownloadButton > button {
        background: var(--accent);
        color: #ffffff;
        border: 1px solid var(--accent);
        border-radius: 8px;
        padding: 0.55rem 1.5rem;
        font-weight: 500;
        font-size: 0.92rem;
        box-shadow: none;
        transition: background 0.15s ease, border-color 0.15s ease;
    }
    .stButton > button:hover, .stDownloadButton > button:hover {
        background: var(--accent-hover);
        border-color: var(--accent-hover);
        box-shadow: none;
        transform: none;
        color: #ffffff;
    }
    .stButton > button:focus, .stDownloadButton > button:focus {
        box-shadow: 0 0 0 3px rgba(37,99,235,0.18);
        outline: none;
    }

    /* ── Inputs ──────────────────────────────────── */
    .stSelectbox label, .stSlider label, .stRadio label,
    .stFileUploader label, .stNumberInput label, .stTextInput label {
        color: var(--text) !important;
        font-weight: 500;
        font-size: 0.88rem;
    }
    .stNumberInput input, .stTextInput input {
        background: var(--panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
        color: var(--text) !important;
        font-size: 0.92rem !important;
    }
    .stNumberInput input:focus, .stTextInput input:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 3px rgba(37,99,235,0.12) !important;
    }
    div[data-baseweb="select"] > div {
        background: var(--panel) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px !important;
    }

    /* ── Price box ───────────────────────────────── */
    .price-box {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 12px;
        padding: 1.5rem 2rem;
        margin: 1.25rem 0;
        display: flex;
        flex-direction: column;
        gap: 0.35rem;
        position: relative;
        overflow: hidden;
    }
    .price-box::before {
        content: "";
        position: absolute;
        left: 0; top: 0; bottom: 0;
        width: 3px;
        background: var(--accent);
    }
    .price-box .label {
        color: var(--text-muted);
        font-size: 0.78rem;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .price-box .price {
        color: var(--text);
        font-size: 2rem;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* ── Status badges ───────────────────────────── */
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.35rem;
        padding: 0.25rem 0.7rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 500;
        border: 1px solid;
    }
    .status-ready {
        background: var(--success-soft);
        color: var(--success);
        border-color: #a7f3d0;
    }
    .status-ready::before {
        content: "";
        width: 6px; height: 6px;
        border-radius: 50%;
        background: var(--success);
    }
    .status-estimate {
        background: var(--warning-soft);
        color: var(--warning);
        border-color: #fde68a;
    }
    .status-estimate::before {
        content: "";
        width: 6px; height: 6px;
        border-radius: 50%;
        background: var(--warning);
    }

    /* ── Insight cards ───────────────────────────── */
    .insight-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 0.85rem 1.1rem;
        margin-bottom: 0.6rem;
        color: var(--text-muted);
        font-size: 0.9rem;
        line-height: 1.55;
    }
    .insight-card strong { color: var(--text); font-weight: 600; }

    /* ── Footer ──────────────────────────────────── */
    .footer {
        text-align: center;
        color: var(--text-subtle);
        font-size: 0.78rem;
        margin-top: 3rem;
        padding: 1rem 0;
        border-top: 1px solid var(--border-soft);
    }

    /* ── Agent trace timeline ────────────────────── */
    .trace-step {
        display: flex;
        align-items: flex-start;
        gap: 0.7rem;
        padding: 0.6rem 0.9rem;
        margin-bottom: 0.4rem;
        background: var(--panel);
        border: 1px solid var(--border);
        border-left: 3px solid var(--success);
        border-radius: 6px;
        color: var(--text);
        font-family: 'JetBrains Mono', ui-monospace, monospace;
        font-size: 0.82rem;
        line-height: 1.5;
    }
    .trace-step .tick {
        color: var(--success);
        font-weight: 600;
        flex-shrink: 0;
    }

    /* ── Advisory section card ───────────────────── */
    .advisory-card {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.7rem;
    }
    .advisory-card h4 {
        color: var(--text) !important;
        margin: 0 0 0.5rem 0;
        font-size: 0.95rem;
        font-weight: 600;
    }
    .advisory-card p, .advisory-card li {
        color: var(--text-muted) !important;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    /* ── Expanders ───────────────────────────────── */
    details[data-testid="stExpander"] {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }
    details[data-testid="stExpander"] summary {
        font-weight: 500;
        color: var(--text);
        padding: 0.65rem 0.9rem;
    }

    /* ── Dataframes ──────────────────────────────── */
    [data-testid="stDataFrame"] {
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
    }

    /* ── Alerts ──────────────────────────────────── */
    .stAlert {
        border-radius: 8px;
        border: 1px solid var(--border);
        font-size: 0.88rem;
    }

    /* ── Caption ─────────────────────────────────── */
    .stCaption, [data-testid="stCaptionContainer"] {
        color: var(--text-muted) !important;
        font-size: 0.85rem;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Estate Intelligence")
    st.markdown("AI-powered real estate price prediction using a trained Random Forest model.")

    st.markdown("---")
    st.markdown("**Team:** Estate Intelligence")
    st.markdown("---")

    # ── Model status indicator ──
    st.markdown("### Model Status")
    if model is not None:
        st.markdown('<span class="status-badge status-ready">Model Ready</span>',
                    unsafe_allow_html=True)
        st.caption("Tuned Random Forest loaded.")
    else:
        st.markdown('<span class="status-badge status-estimate">Estimate Mode</span>',
                    unsafe_allow_html=True)
        st.caption("Model file not found. Using placeholder formula.")

    st.markdown("---")

    # ── Instructions ──
    st.markdown("### How to Use")
    st.markdown("""
    1. **Predict** — Fill in property details in Tab 1 and click *Predict Price*.
    2. **Insights** — View feature importances and correlation heatmap in Tab 2.
    3. **Batch** — Upload a CSV in Tab 3 to predict multiple properties at once.
    4. **AI Advisory** — Run the full LangGraph agent in Tab 4 for a grounded 6-section report with a PDF download.
    """)

    st.markdown("---")
    st.markdown("### Agent Status")
    if _AGENT_AVAILABLE:
        if os.environ.get("GROQ_API_KEY"):
            st.markdown('<span class="status-badge status-ready">Advisory Ready</span>',
                        unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-estimate">API Key Missing</span>',
                        unsafe_allow_html=True)
            st.caption("Set `GROQ_API_KEY` in `.env` to enable Tab 4.")
    else:
        st.markdown('<span class="status-badge status-estimate">Agent Unavailable</span>',
                    unsafe_allow_html=True)
        st.caption(f"Import error: {_AGENT_IMPORT_ERROR}")


# ──────────────────────────────────────────────────────────────
# MAIN HEADER
# ──────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">Estate Intelligence</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub-title">Predict residential property prices with machine learning</p>',
    unsafe_allow_html=True,
)
st.markdown("")


# ──────────────────────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "Predict Price",
    "Data Insights",
    "Batch Prediction",
    "AI Advisory",
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 — SINGLE PROPERTY PREDICTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.markdown("### Enter Property Details")

    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input(
            "Area (sq ft)",
            min_value=500,
            max_value=16200,
            value=3000,
            step=50,
            help="Type the property area in square feet",
        )
        st.caption(
            "Model trained on areas 1,650 - 16,200 sq ft. "
            "Values outside this range may give less accurate predictions."
        )
        bedrooms        = st.selectbox("Bedrooms", options=[1, 2, 3, 4, 5, 6], index=2)
        bathrooms       = st.selectbox("Bathrooms", options=[1, 2, 3, 4], index=0)
        stories         = st.selectbox("Stories", options=[1, 2, 3, 4], index=0)
        parking         = st.selectbox("Parking Spots", options=[0, 1, 2, 3], index=0)
        furnishing      = st.selectbox("Furnishing Status",
                                       options=["Unfurnished", "Semi-Furnished", "Furnished"],
                                       index=0)

    with col2:
        mainroad        = st.radio("Main Road Access",  ["Yes", "No"], index=0, horizontal=True)
        guestroom       = st.radio("Guest Room",         ["Yes", "No"], index=1, horizontal=True)
        basement        = st.radio("Basement",           ["Yes", "No"], index=1, horizontal=True)
        hotwaterheating = st.radio("Hot Water Heating",  ["Yes", "No"], index=1, horizontal=True)
        airconditioning = st.radio("Air Conditioning",   ["Yes", "No"], index=1, horizontal=True)
        prefarea        = st.radio("Preferred Area",      ["Yes", "No"], index=1, horizontal=True)

    # Encode Yes/No inputs to 1/0
    yes_no         = lambda x: 1 if x == "Yes" else 0
    furnishing_val = {"Unfurnished": 0, "Semi-Furnished": 1, "Furnished": 2}[furnishing]

    st.markdown("")

    # ── Predict button ──
    if st.button("Predict Price", use_container_width=True, key="single_predict"):
        # Store prediction in session state so it persists across rerenders
        st.session_state["predicted"] = predict_price(
            area, bedrooms, bathrooms, stories,
            yes_no(mainroad), yes_no(guestroom), yes_no(basement),
            yes_no(hotwaterheating), yes_no(airconditioning),
            parking, yes_no(prefarea), furnishing_val,
        )

    # Show result if prediction exists in session state
    if "predicted" in st.session_state:
        predicted = st.session_state["predicted"]

        # Styled price output box
        st.markdown(
            f"""
            <div class="price-box">
                <div class="label">Estimated Property Price</div>
                <div class="price">{format_inr(predicted)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Feature importance chart using REAL model values
        st.markdown("#### Feature Importance (Trained Random Forest)")
        st.caption("How much each feature contributed to the model's predictions.")

        _label_map = {
            "area": "Area", "bedrooms": "Bedrooms", "bathrooms": "Bathrooms",
            "stories": "Stories", "mainroad": "Main Road", "guestroom": "Guest Room",
            "basement": "Basement", "hotwaterheating": "Hot Water", "airconditioning": "AC",
            "parking": "Parking", "prefarea": "Preferred Area", "furnishingstatus": "Furnishing",
        }
        if feature_importances:
            raw = {_label_map.get(k, k): v for k, v in feature_importances.items()}
        else:
            raw = FEATURE_IMPORTANCES
        fi_sorted = dict(sorted(raw.items(), key=lambda x: x[1]))

        fig, ax = plt.subplots(figsize=(8, 4.5))
        fig.patch.set_facecolor("#ffffff")
        ax.set_facecolor("#ffffff")

        bars = ax.barh(
            list(fi_sorted.keys()),
            list(fi_sorted.values()),
            color="#2563eb",
            edgecolor="none",
            height=0.55,
            zorder=3,
        )

        ax.set_xlim(0, 0.56)
        ax.set_xlabel("Importance Score", color="#64748b", fontsize=9.5)
        ax.tick_params(colors="#64748b", labelsize=9)
        for spine in ax.spines.values():
            spine.set_color("#e5e7eb")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="x", color="#eef0f3", linewidth=0.8, zorder=0)

        for bar in bars:
            w = bar.get_width()
            ax.text(w + 0.008, bar.get_y() + bar.get_height() / 2,
                    f"{w:.1%}", va="center", color="#0f172a",
                    fontsize=8.5, fontweight=500)

        plt.tight_layout()
        st.pyplot(fig)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 — DATA INSIGHTS
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.markdown("### Model & Data Insights")

    fi_img_path   = os.path.join(os.path.dirname(__file__), "notebooks", "feature_importance.png")
    corr_img_path = os.path.join(os.path.dirname(__file__), "notebooks", "correlation_heatmap.png")

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Feature Importance")
        if os.path.exists(fi_img_path):
            st.image(fi_img_path, caption="Feature Importance — Tuned Random Forest",
                     use_container_width=True)
        else:
            st.warning("feature_importance.png not found in notebooks/")

    with col_b:
        st.markdown("#### Correlation Heatmap")
        if os.path.exists(corr_img_path):
            st.image(corr_img_path, caption="Correlation Heatmap — Feature Relationships",
                     use_container_width=True)
        else:
            st.warning("correlation_heatmap.png not found in notebooks/")

    st.markdown("#### Key Findings")
    st.markdown("""
    <div class="insight-card">
        <strong>Area dominates (47.8%)</strong> — Property size is by far the strongest predictor of price.
        Larger homes consistently command higher prices in this dataset.
    </div>
    <div class="insight-card">
        <strong>Bathrooms are second (15.8%)</strong> — Number of bathrooms has strong influence,
        likely because it correlates with overall property quality and size.
    </div>
    <div class="insight-card">
        <strong>Air Conditioning matters (6.8%)</strong> — Presence of AC significantly
        adds to property value, reflecting buyer preferences in the Indian market.
    </div>
    <div class="insight-card">
        <strong>Stories & Parking (5.2%, 4.8%)</strong> — Multi-storey homes and parking
        availability contribute meaningfully to price.
    </div>
    <div class="insight-card">
        <strong>Main Road access has lowest impact (1.0%)</strong> — Despite being a common
        amenity listed, it has the least influence on price among all features.
    </div>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 — BATCH CSV PREDICTION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown("### Upload CSV for Batch Prediction")
    st.caption(
        "Your CSV must contain: "
        "`area`, `bedrooms`, `bathrooms`, `stories`, `mainroad`, `guestroom`, "
        "`basement`, `hotwaterheating`, `airconditioning`, `parking`, "
        "`prefarea`, `furnishingstatus`  "
        "(binary columns as 0/1, furnishingstatus as 0=Unfurnished / 1=Semi / 2=Furnished)"
    )

    # Sample template download
    sample_data = pd.DataFrame([{
        "area": 1500, "bedrooms": 3, "bathrooms": 2, "stories": 2,
        "mainroad": 1, "guestroom": 0, "basement": 0,
        "hotwaterheating": 0, "airconditioning": 1,
        "parking": 1, "prefarea": 0, "furnishingstatus": 1
    }])
    st.download_button(
        label="Download Sample CSV Template",
        data=sample_data.to_csv(index=False),
        file_name="sample_input.csv",
        mime="text/csv",
    )

    st.markdown("")
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.markdown("#### Uploaded Data")
        st.dataframe(df, use_container_width=True)

        missing_cols = [c for c in FEATURE_COLUMNS if c not in df.columns]

        if missing_cols:
            st.error(f"Missing columns: **{', '.join(missing_cols)}**. Please fix your CSV.")
        else:
            if st.button("Predict All", use_container_width=True, key="batch_predict"):
                with st.spinner("Running predictions..."):
                    df["predicted_price"] = df.apply(
                        lambda row: predict_price(
                            row["area"], row["bedrooms"], row["bathrooms"],
                            row["stories"], row["mainroad"], row["guestroom"],
                            row["basement"], row["hotwaterheating"],
                            row["airconditioning"], row["parking"],
                            row["prefarea"], row["furnishingstatus"],
                        ),
                        axis=1,
                    )
                    df["predicted_price_INR"] = df["predicted_price"].apply(format_inr)

                st.markdown("#### Predictions Complete")
                st.dataframe(df, use_container_width=True)

                export_df = df.drop(columns=["predicted_price"])
                st.download_button(
                    label="Download Results as CSV",
                    data=export_df.to_csv(index=False),
                    file_name="estate_intelligence_predictions.csv",
                    mime="text/csv",
                    use_container_width=True,
                )


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 — AI ADVISORY (LangGraph agent + RAG + PDF export)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.markdown("### AI-Powered Property Advisory")
    st.caption(
        "Runs a LangGraph agent: validate → predict → retrieve (FAISS RAG) → "
        "generate (Groq LLM). Produces a 6-section advisory report with a PDF download."
    )

    if not _AGENT_AVAILABLE:
        st.error(f"Agent modules failed to import: `{_AGENT_IMPORT_ERROR}`. "
                 "Run `pip install -r requirements.txt` and restart.")
        st.stop()

    if not os.environ.get("GROQ_API_KEY"):
        st.warning(
            "`GROQ_API_KEY` is not set. Get a free key at "
            "[console.groq.com](https://console.groq.com) and add it to your `.env` file."
        )

    # ── Inputs ──
    ac1, ac2 = st.columns(2)
    with ac1:
        adv_area            = st.number_input("Area (sq ft)", min_value=500,
                                              max_value=20000, value=1800, step=50,
                                              key="adv_area")
        adv_bedrooms        = st.selectbox("Bedrooms", [1, 2, 3, 4, 5, 6],
                                           index=2, key="adv_bedrooms")
        adv_bathrooms       = st.selectbox("Bathrooms", [1, 2, 3, 4],
                                           index=1, key="adv_bathrooms")
        adv_stories         = st.selectbox("Stories", [1, 2, 3, 4],
                                           index=1, key="adv_stories")
        adv_parking         = st.selectbox("Parking Spots", [0, 1, 2, 3],
                                           index=2, key="adv_parking")
        adv_furnishing      = st.selectbox(
            "Furnishing", ["Unfurnished", "Semi-Furnished", "Furnished"],
            index=1, key="adv_furnishing",
        )
    with ac2:
        adv_mainroad        = st.radio("Main Road Access", ["Yes", "No"],
                                       index=0, horizontal=True, key="adv_mainroad")
        adv_guestroom       = st.radio("Guest Room", ["Yes", "No"],
                                       index=1, horizontal=True, key="adv_guestroom")
        adv_basement        = st.radio("Basement", ["Yes", "No"],
                                       index=1, horizontal=True, key="adv_basement")
        adv_hotwater        = st.radio("Hot Water Heating", ["Yes", "No"],
                                       index=1, horizontal=True, key="adv_hotwater")
        adv_ac              = st.radio("Air Conditioning", ["Yes", "No"],
                                       index=0, horizontal=True, key="adv_ac")
        adv_prefarea        = st.radio("Preferred Area", ["Yes", "No"],
                                       index=0, horizontal=True, key="adv_prefarea")

    st.markdown("")

    if st.button("Generate Advisory Report", use_container_width=True, key="run_agent"):
        yn = lambda v: 1 if v == "Yes" else 0
        fmap = {"Unfurnished": 0, "Semi-Furnished": 1, "Furnished": 2}
        property_input = {
            "area":             adv_area,
            "bedrooms":         adv_bedrooms,
            "bathrooms":        adv_bathrooms,
            "stories":          adv_stories,
            "mainroad":         yn(adv_mainroad),
            "guestroom":        yn(adv_guestroom),
            "basement":         yn(adv_basement),
            "hotwaterheating":  yn(adv_hotwater),
            "airconditioning":  yn(adv_ac),
            "parking":          adv_parking,
            "prefarea":         yn(adv_prefarea),
            "furnishingstatus": fmap[adv_furnishing],
        }

        initial_state = {
            "property_input":     property_input,
            "validated_input":    None,
            "validation_errors":  [],
            "predicted_price":    None,
            "market_trends":      None,
            "report":             None,
            "report_structured":  None,
            "agent_steps":        [],
        }

        with st.spinner("Running agent pipeline — validating, predicting, retrieving, generating…"):
            try:
                st.session_state["advisory_state"] = advisory_graph.invoke(initial_state)
                st.session_state["advisory_error"] = None
            except Exception as e:
                st.session_state["advisory_state"] = None
                st.session_state["advisory_error"] = f"{type(e).__name__}: {e}"

    # ── Render result ──
    if st.session_state.get("advisory_error"):
        st.error(f"Agent run failed: {st.session_state['advisory_error']}")

    adv_state = st.session_state.get("advisory_state")
    if adv_state:
        # Agent trace
        st.markdown("#### Agent Trace")
        st.caption("Each LangGraph node logs its step here — full pipeline transparency.")
        for step in adv_state.get("agent_steps") or []:
            st.markdown(
                f'<div class="trace-step"><span class="tick">✓</span><span>{step}</span></div>',
                unsafe_allow_html=True,
            )

        # Validation warnings
        val_errors = adv_state.get("validation_errors") or []
        if val_errors:
            with st.expander(f"Data Quality Notes ({len(val_errors)})", expanded=False):
                for err in val_errors:
                    st.markdown(f"- {err}")

        # Predicted price
        price = adv_state.get("predicted_price")
        if price is not None:
            st.markdown(
                f"""
                <div class="price-box">
                    <div class="label">ML-Predicted Price</div>
                    <div class="price">{format_inr(price)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        # 6-section advisory
        structured = adv_state.get("report_structured")
        if structured:
            report_model = AdvisoryReport(**structured)
            st.markdown("#### Advisory Report")
            for title, body in report_model.sections():
                with st.expander(title, expanded=True):
                    st.markdown(body or "_No content generated for this section._")

            # PDF download
            try:
                pdf_bytes = generate_pdf(
                    report_model,
                    predicted_price=price or 0,
                    property_input=adv_state.get("validated_input") or {},
                )
                st.download_button(
                    label="Download Report as PDF",
                    data=pdf_bytes,
                    file_name="estate_intelligence_advisory.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as e:
                st.error(f"PDF generation failed: {type(e).__name__}: {e}")
        elif adv_state.get("report"):
            st.markdown("#### Advisory Report (raw)")
            st.markdown(adv_state["report"])


# ──────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">'
    "Predictions are estimates only. Not financial advice. "
    "| Estate Intelligence · Built with Streamlit"
    "</div>",
    unsafe_allow_html=True,
)