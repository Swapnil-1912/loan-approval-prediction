import streamlit as st
import numpy as np
import joblib
import os

# ─────────────────────────────────────────────────────────────
# Page Config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Approval Prediction",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Custom CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Dark gradient background */
.stApp {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    min-height: 100vh;
}

/* Header */
.hero-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.15));
    border: 1px solid rgba(139,92,246,0.3);
    border-radius: 20px;
    margin-bottom: 2rem;
    backdrop-filter: blur(10px);
}

.hero-header h1 {
    font-size: 2.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #818cf8, #c084fc, #f472b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.4rem;
}

.hero-header p {
    color: #94a3b8;
    font-size: 1.05rem;
    font-weight: 400;
    margin: 0;
}

/* Section card */
.section-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.09);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 1.5rem;
    backdrop-filter: blur(12px);
}

.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: #818cf8;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

/* Override Streamlit widget styling */
.stSelectbox label, .stNumberInput label, .stSlider label {
    color: #cbd5e1 !important;
    font-size: 0.875rem !important;
    font-weight: 500 !important;
}

div[data-baseweb="select"] {
    border-radius: 10px !important;
}

div[data-baseweb="input"] {
    border-radius: 10px !important;
}

/* Predict Button */
div.stButton > button {
    width: 100%;
    padding: 0.85rem 2rem;
    font-size: 1.1rem;
    font-weight: 700;
    border: none;
    border-radius: 12px;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4);
    letter-spacing: 0.03em;
}

div.stButton > button:hover {
    background: linear-gradient(135deg, #4f46e5, #7c3aed);
    box-shadow: 0 6px 30px rgba(99,102,241,0.6);
    transform: translateY(-2px);
}

/* Result banners */
.result-approved {
    background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(5,150,105,0.15));
    border: 2px solid rgba(16,185,129,0.5);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    text-align: center;
    margin-top: 1.5rem;
    animation: fadeIn 0.5s ease;
}

.result-rejected {
    background: linear-gradient(135deg, rgba(239,68,68,0.15), rgba(185,28,28,0.15));
    border: 2px solid rgba(239,68,68,0.5);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    text-align: center;
    margin-top: 1.5rem;
    animation: fadeIn 0.5s ease;
}

.result-emoji {
    font-size: 3.5rem;
    margin-bottom: 0.8rem;
}

.result-title-approved {
    font-size: 2rem;
    font-weight: 800;
    color: #10b981;
    margin-bottom: 0.4rem;
}

.result-title-rejected {
    font-size: 2rem;
    font-weight: 800;
    color: #ef4444;
    margin-bottom: 0.4rem;
}

.result-subtitle {
    color: #94a3b8;
    font-size: 1rem;
    font-weight: 400;
}

/* Stat pills */
.stats-row {
    display: flex;
    gap: 1rem;
    justify-content: center;
    margin-top: 1.5rem;
    flex-wrap: wrap;
}

.stat-pill {
    background: rgba(255,255,255,0.07);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 30px;
    padding: 0.5rem 1.2rem;
    color: #e2e8f0;
    font-size: 0.85rem;
    font-weight: 500;
}

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: rgba(15, 12, 41, 0.8) !important;
    border-right: 1px solid rgba(139,92,246,0.2);
}

section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3,
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] li {
    color: #e2e8f0 !important;
}

/* Accuracy badge */
.accuracy-badge {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    font-weight: 700;
    font-size: 1.4rem;
    padding: 0.5rem 1.5rem;
    border-radius: 50px;
    margin: 0.5rem 0;
}

/* Divider */
hr {
    border-color: rgba(255,255,255,0.08) !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Load Model & Scaler
# ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(BASE_DIR, "loan_model.joblib")
    scaler_path = os.path.join(BASE_DIR, "..", "Model", "scaler.pkl")

    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model file not found at: {model_path}")
        st.stop()
    if not os.path.exists(scaler_path):
        st.error(f"❌ Scaler file not found at: {scaler_path}")
        st.stop()

    model  = joblib.load(MODEL_PATH)
    scaler = joblib.load(scaler_path)
    return model, scaler

model, scaler = load_artifacts()

# ─────────────────────────────────────────────────────────────
# Label encoding maps (matching LabelEncoder fit order from notebook)
# LabelEncoder sorts unique values alphabetically by default
# ─────────────────────────────────────────────────────────────
GENDER_MAP        = {"Female": 0, "Male": 1}
MARRIED_MAP       = {"No": 0, "Yes": 1}
DEPENDENTS_MAP    = {"0": 0, "1": 1, "2": 2, "3+": 3}
EDUCATION_MAP     = {"Graduate": 0, "Not Graduate": 1}
SELF_EMPLOYED_MAP = {"No": 0, "Yes": 1}
PROPERTY_MAP      = {"Rural": 0, "Semiurban": 1, "Urban": 2}

# ─────────────────────────────────────────────────────────────
# Sidebar – Info Panel
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 About This App")
    st.markdown("""
    This tool uses a **Logistic Regression** model trained on the Loan Prediction Dataset
    from Kaggle to predict whether a loan application will be **approved or rejected**.
    """)
    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    st.markdown('<div class="accuracy-badge">86.18%</div>', unsafe_allow_html=True)
    st.markdown("**Accuracy** on validation data")
    st.markdown("---")
    st.markdown("### 🔑 Key Features")
    st.markdown("""
    - **Credit History** – Strongest predictor
    - **Income** – Applicant + Co-applicant
    - **Loan Amount & Term**
    - **Property Area**
    - **Personal Profile** – Gender, Education, Employment
    """)
    st.markdown("---")
    st.markdown("### ℹ️ How to Use")
    st.markdown("""
    1. Fill in your details in the form
    2. Click **Predict Loan Status**
    3. View your result instantly
    """)
    st.markdown("---")
    st.markdown(
        '<p style="font-size:0.78rem; color:#64748b;">Built with ❤️ using Streamlit · Summer Internship 2026</p>',
        unsafe_allow_html=True
    )

# ─────────────────────────────────────────────────────────────
# Hero Header
# ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-header">
    <h1>🏦 Loan Approval Prediction</h1>
    <p>Instantly predict your loan eligibility using Machine Learning · Powered by Logistic Regression</p>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Input Form
# ─────────────────────────────────────────────────────────────
with st.form("loan_form"):

    # ── Personal Information ──────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">👤 Personal Information</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox(
            "Gender",
            options=list(GENDER_MAP.keys()),
            help="Applicant's gender"
        )
    with col2:
        married = st.selectbox(
            "Marital Status",
            options=list(MARRIED_MAP.keys()),
            help="Is the applicant married?"
        )
    with col3:
        dependents = st.selectbox(
            "Number of Dependents",
            options=list(DEPENDENTS_MAP.keys()),
            help="Number of persons dependent on the applicant"
        )

    col4, col5 = st.columns(2)
    with col4:
        education = st.selectbox(
            "Education Level",
            options=list(EDUCATION_MAP.keys()),
            help="Highest level of education attained"
        )
    with col5:
        self_employed = st.selectbox(
            "Self Employed",
            options=list(SELF_EMPLOYED_MAP.keys()),
            help="Is the applicant self-employed?"
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Financial Details ─────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">💰 Financial Details</div>', unsafe_allow_html=True)

    col6, col7 = st.columns(2)
    with col6:
        applicant_income = st.number_input(
            "Applicant Income (₹/month)",
            min_value=0,
            max_value=1_000_000,
            value=5000,
            step=500,
            help="Monthly income of the primary applicant in ₹"
        )
    with col7:
        coapplicant_income = st.number_input(
            "Co-applicant Income (₹/month)",
            min_value=0,
            max_value=500_000,
            value=0,
            step=500,
            help="Monthly income of the co-applicant (enter 0 if not applicable)"
        )

    col8, col9 = st.columns(2)
    with col8:
        loan_amount = st.number_input(
            "Loan Amount (₹ thousands)",
            min_value=0,
            max_value=10_000,
            value=120,
            step=10,
            help="Requested loan amount in thousands of ₹ (e.g., 120 = ₹1,20,000)"
        )
    with col9:
        loan_term = st.selectbox(
            "Loan Amount Term (months)",
            options=[12, 36, 60, 84, 120, 180, 240, 300, 360, 480],
            index=8,
            help="Duration of loan repayment in months"
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Loan Details ──────────────────────────────────────────
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">📋 Credit & Property Details</div>', unsafe_allow_html=True)

    col10, col11 = st.columns(2)
    with col10:
        credit_history = st.selectbox(
            "Credit History",
            options={"Has credit history (meets guidelines)": 1.0, "No credit history": 0.0},
            help="1 = Has credit history that meets lending guidelines; 0 = No credit history"
        )
        credit_history_value = 1.0 if credit_history == "Has credit history (meets guidelines)" else 0.0
    with col11:
        property_area = st.selectbox(
            "Property Area",
            options=list(PROPERTY_MAP.keys()),
            help="Location type of the property to be purchased"
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ── Submit Button ─────────────────────────────────────────
    submitted = st.form_submit_button("🔍 Predict Loan Status")

# ─────────────────────────────────────────────────────────────
# Prediction Logic
# ─────────────────────────────────────────────────────────────
if submitted:
    # Encode categorical features
    gender_enc        = GENDER_MAP[gender]
    married_enc       = MARRIED_MAP[married]
    dependents_enc    = DEPENDENTS_MAP[dependents]
    education_enc     = EDUCATION_MAP[education]
    self_employed_enc = SELF_EMPLOYED_MAP[self_employed]
    property_enc      = PROPERTY_MAP[property_area]

    # Build feature vector (same order as training: 11 features, no Loan_ID, no Loan_Status)
    features = np.array([[
        gender_enc,
        married_enc,
        dependents_enc,
        education_enc,
        self_employed_enc,
        applicant_income,
        coapplicant_income,
        loan_amount,
        float(loan_term),
        credit_history_value,
        property_enc,
    ]])

    # Scale
    features_scaled = scaler.transform(features)

    # Predict
    prediction      = model.predict(features_scaled)[0]
    probabilities   = model.predict_proba(features_scaled)[0]
    confidence      = round(max(probabilities) * 100, 2)

    # Total income & ratio for display
    total_income   = applicant_income + coapplicant_income
    emi_estimate   = round((loan_amount * 1000) / loan_term, 0) if loan_term > 0 else 0
    income_ratio   = round((emi_estimate / total_income) * 100, 1) if total_income > 0 else "N/A"

    # ── Result Banner ─────────────────────────────────────────
    if prediction == 1:
        st.markdown(f"""
        <div class="result-approved">
            <div class="result-emoji">🎉</div>
            <div class="result-title-approved">Loan Approved!</div>
            <p class="result-subtitle">
                Congratulations! Based on the details provided, your loan application is
                likely to be <strong>approved</strong>.
            </p>
            <div class="stats-row">
                <span class="stat-pill">🎯 Confidence: {confidence}%</span>
                <span class="stat-pill">💵 Total Income: ₹{total_income:,}/mo</span>
                <span class="stat-pill">📅 Term: {loan_term} months</span>
                <span class="stat-pill">💳 EMI ≈ ₹{int(emi_estimate):,}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="result-rejected">
            <div class="result-emoji">❌</div>
            <div class="result-title-rejected">Loan Rejected</div>
            <p class="result-subtitle">
                Unfortunately, based on the details provided, your loan application is
                likely to be <strong>rejected</strong>. Consider improving your credit history
                or reducing the requested loan amount.
            </p>
            <div class="stats-row">
                <span class="stat-pill">🎯 Confidence: {confidence}%</span>
                <span class="stat-pill">💵 Total Income: ₹{total_income:,}/mo</span>
                <span class="stat-pill">📅 Term: {loan_term} months</span>
                <span class="stat-pill">💳 EMI ≈ ₹{int(emi_estimate):,}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Tips Section ──────────────────────────────────────────
    st.markdown("---")
    if prediction == 0:
        st.markdown("### 💡 Tips to Improve Your Chances")
        tips_col1, tips_col2 = st.columns(2)
        with tips_col1:
            st.info("📈 **Increase Income** – Add a co-applicant with a steady income to boost your combined monthly income.")
            st.info("💳 **Build Credit History** – Maintain a good repayment track record on existing loans or credit cards.")
        with tips_col2:
            st.info("📉 **Reduce Loan Amount** – Request a smaller loan amount to lower the EMI-to-income ratio.")
            st.info("⏱️ **Extend Loan Term** – A longer repayment term lowers monthly EMI and improves eligibility.")
    else:
        st.markdown("### 🌟 What Helped Your Approval?")
        tips_col1, tips_col2 = st.columns(2)
        with tips_col1:
            if credit_history_value == 1.0:
                st.success("✅ **Good Credit History** – A positive credit history is the strongest factor in approval.")
            if total_income >= 5000:
                st.success("✅ **Healthy Income** – Your combined income comfortably supports loan repayment.")
        with tips_col2:
            if income_ratio != "N/A" and float(income_ratio) < 50:
                st.success(f"✅ **Low EMI Ratio** – Your EMI is only {income_ratio}% of your income — well within safe limits.")
            if education == "Graduate":
                st.success("✅ **Education Level** – Being a graduate positively influences loan decisions.")
