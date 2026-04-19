import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from agent.state import AgentState

# ── Paths ────────────────────────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.dirname(__file__))
_MODEL_PATH = os.path.join(_BASE, "models", "best_rf_model.joblib")
_DATA_PATH = os.path.join(_BASE, "data", "cleaned", "Housing_cleaned.csv")

# ── Feature config ────────────────────────────────────────────────────────────
FEATURE_COLS = [
    "area", "bedrooms", "bathrooms", "stories",
    "mainroad", "guestroom", "basement", "hotwaterheating",
    "airconditioning", "parking", "prefarea", "furnishingstatus",
]

FIELD_DEFAULTS = {
    "area": 5000,
    "bedrooms": 3,
    "bathrooms": 1,
    "stories": 1,
    "mainroad": 1,
    "guestroom": 0,
    "basement": 0,
    "hotwaterheating": 0,
    "airconditioning": 0,
    "parking": 1,
    "prefarea": 0,
    "furnishingstatus": 1,
}

FIELD_RANGES = {
    "area": (500, 20000),
    "bedrooms": (1, 10),
    "bathrooms": (1, 6),
    "stories": (1, 4),
    "parking": (0, 3),
    "furnishingstatus": (0, 2),
}

BINARY_FIELDS = {"mainroad", "guestroom", "basement", "hotwaterheating", "airconditioning", "prefarea"}


# ── Load model + reconstruct scaler once at import time ──────────────────────
def _init_model_and_scaler():
    model = joblib.load(_MODEL_PATH)

    df = pd.read_csv(_DATA_PATH)
    X = df[FEATURE_COLS]
    y = df["price"]
    X_train, _, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    scaler.fit(X_train)

    return model, scaler


_MODEL, _SCALER = _init_model_and_scaler()


# ═════════════════════════════════════════════════════════════════════════════
# NODE 1 — Validate & clean property input
# ═════════════════════════════════════════════════════════════════════════════
def validate_input_node(state: AgentState) -> AgentState:
    """
    Checks every required field. Missing or invalid values get a default so the
    pipeline never hard-fails. All corrections are recorded in validation_errors
    so the LLM can acknowledge them in the report.
    """
    raw = state["property_input"]
    errors: list[str] = []
    validated: dict = {}

    for field in FEATURE_COLS:
        default = FIELD_DEFAULTS[field]
        val = raw.get(field)

        # ── Missing ──
        if val is None or val == "":
            errors.append(f"'{field}' not provided — using default ({default})")
            validated[field] = default
            continue

        # ── Type coercion ──
        try:
            val = float(val) if field == "area" else int(float(val))
        except (ValueError, TypeError):
            errors.append(f"'{field}' has invalid value {val!r} — using default ({default})")
            validated[field] = default
            continue

        # ── Range check ──
        if field in FIELD_RANGES:
            lo, hi = FIELD_RANGES[field]
            if not (lo <= val <= hi):
                errors.append(f"'{field}' = {val} is outside range [{lo}, {hi}] — clamped")
                val = max(lo, min(hi, val))

        # ── Binary check ──
        if field in BINARY_FIELDS and val not in (0, 1):
            errors.append(f"'{field}' must be 0 or 1, got {val} — using default ({default})")
            val = default

        validated[field] = val

    return {**state, "validated_input": validated, "validation_errors": errors}


# ═════════════════════════════════════════════════════════════════════════════
# NODE 2 — Predict price using the trained Random Forest model
# ═════════════════════════════════════════════════════════════════════════════
def predict_price_node(state: AgentState) -> AgentState:
    inp = state["validated_input"]
    features = pd.DataFrame([[inp[col] for col in FEATURE_COLS]], columns=FEATURE_COLS)
    features_scaled = _SCALER.transform(features)
    price = float(_MODEL.predict(features_scaled)[0])
    return {**state, "predicted_price": price}


# ═════════════════════════════════════════════════════════════════════════════
# NODE 3 — Retrieve market trends via FAISS RAG over the knowledge base
# ═════════════════════════════════════════════════════════════════════════════
_FALLBACK_TRENDS = (
    "Indian real estate market (2024-25): Tier-1 cities report 8–12% YoY price appreciation. "
    "RBI repo rate held at 6.5%, keeping home loan rates between 8.5–9.5%. "
    "Government PMAY scheme supports affordable housing (under ₹45L). "
    "Properties with metro/highway connectivity command a 15–20% premium. "
    "Air-conditioned and furnished properties in preferred localities see faster transactions."
)


def _build_retrieval_query(inp: dict, price: float | None) -> str:
    """
    Craft a retrieval query from the validated property features so the FAISS
    search surfaces the most relevant market/regulatory chunks.
    """
    furnish_map = {0: "unfurnished", 1: "semi-furnished", 2: "furnished"}
    parts = [
        f"{inp.get('bedrooms', '?')} BHK {inp.get('area', '?')} sq ft residential property",
        furnish_map.get(inp.get("furnishingstatus"), ""),
    ]
    if inp.get("airconditioning"):
        parts.append("with air conditioning")
    if inp.get("prefarea"):
        parts.append("in preferred area")
    if inp.get("parking"):
        parts.append(f"{inp['parking']} parking")
    if price is not None:
        if price < 45_00_000:
            parts.append("affordable housing PMAY subsidy")
        elif price > 1_00_00_000:
            parts.append("premium market gated community")
    parts.append("market trend price per sq ft home loan RBI regulation")
    return " ".join(p for p in parts if p)


def retrieve_trends_node(state: AgentState) -> AgentState:
    """
    Queries the FAISS index built from rag/knowledge_base/ and returns a
    grounded market/regulatory context. On any failure, falls back to a
    static summary so the pipeline never hard-fails — the data-quality
    note will surface the degradation.
    """
    inp = state.get("validated_input") or {}
    price = state.get("predicted_price")
    errors = list(state.get("validation_errors") or [])

    query = _build_retrieval_query(inp, price)

    try:
        from rag.retriever import retrieve, format_context
        results = retrieve(query, k=4)
        trends = format_context(results) or _FALLBACK_TRENDS
    except Exception as e:
        errors.append(f"RAG retrieval unavailable ({type(e).__name__}) — using fallback market context")
        trends = _FALLBACK_TRENDS

    return {**state, "market_trends": trends, "validation_errors": errors}


# ═════════════════════════════════════════════════════════════════════════════
# NODE 4 — Generate structured advisory report via Groq LLM
# ═════════════════════════════════════════════════════════════════════════════
def generate_report_node(state: AgentState) -> AgentState:
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.3,
        api_key=os.environ.get("GROQ_API_KEY"),
    )

    inp = state["validated_input"]
    price = state["predicted_price"]
    trends = state["market_trends"]
    errors = state["validation_errors"]

    furnish_map = {0: "Unfurnished", 1: "Semi-furnished", 2: "Furnished"}

    property_desc = f"""
- Area: {inp['area']} sq ft
- Bedrooms: {inp['bedrooms']} | Bathrooms: {inp['bathrooms']} | Stories: {inp['stories']}
- Main road access: {'Yes' if inp['mainroad'] else 'No'}
- Air conditioning: {'Yes' if inp['airconditioning'] else 'No'}
- Preferred area: {'Yes' if inp['prefarea'] else 'No'}
- Parking spots: {inp['parking']}
- Furnishing: {furnish_map.get(inp['furnishingstatus'], 'Unknown')}
- Guest room: {'Yes' if inp['guestroom'] else 'No'}
- Basement: {'Yes' if inp['basement'] else 'No'}
- Hot water heating: {'Yes' if inp['hotwaterheating'] else 'No'}
""".strip()

    data_quality_note = ""
    if errors:
        data_quality_note = (
            "\n\nData Quality Note: The following fields had missing or invalid values "
            "and were filled with defaults:\n" + "\n".join(f"  • {e}" for e in errors)
        )

    system_prompt = (
        "You are a senior real estate advisor in India. "
        "Generate a professional, structured property advisory report. "
        "Be factual and concise. Never make specific investment promises. "
        "Always include disclaimers. Use ₹ for prices."
    )

    user_prompt = f"""Generate a real estate advisory report for the following property.

PROPERTY DETAILS:
{property_desc}

ML-PREDICTED PRICE: ₹{price:,.0f}
{data_quality_note}

MARKET CONTEXT:
{trends}

Write the report with exactly these 6 sections (use bold markdown headers):

**1. Property Summary**
Describe the property based on its features in 2–3 sentences.

**2. Price Prediction Interpretation**
Explain whether the predicted price seems fair, high, or low relative to the property features and market. Mention the model's MAE is ~₹10,25,000 so the actual price could vary by that margin.

**3. Market Trend Insights**
Relate this property to the current market conditions provided above.

**4. Recommended Actions**
Give 2–3 specific, actionable recommendations for a buyer or investor.

**5. Supporting Sources & References**
List the data sources and model used to generate this report.

**6. Legal & Financial Disclaimers**
Include standard real estate and financial advice disclaimers (2–3 sentences).
"""

    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])

    return {**state, "report": response.content}
