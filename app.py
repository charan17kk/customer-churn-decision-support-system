import streamlit as st
import pandas as pd
import joblib
import base64
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import seaborn as sns

print("APP STARTED")
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

# ==========================================
# ENTERPRISE UI THEME (visual layer only)
# No ML logic lives in this block.
# ==========================================
st.markdown(
    """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap" rel="stylesheet">
    <style>

        html, body, [class*="css"] {
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
        }

        :root {
            --enterprise-blue: #1657C6;
            --enterprise-blue-dark: #0F3F94;
            --enterprise-blue-soft: #EAF1FD;
            --enterprise-green: #157A4B;
            --enterprise-green-soft: #EAF7F0;
            --enterprise-red: #B3261E;
            --enterprise-red-soft: #FDEDEC;
            --enterprise-amber: #9A6400;
            --enterprise-amber-soft: #FFF6E5;
            --enterprise-ink: #1A2333;
            --enterprise-muted: #5B6474;
            --enterprise-border: #E3E7EE;
            --enterprise-bg: #F5F3ED;
        }

        .stApp {
            background-color: var(--enterprise-bg);
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
            max-width: 1180px;
        }

        h1, h2, h3, h4 {
            color: var(--enterprise-ink);
            letter-spacing: -0.01em;
        }

        h3 {
            font-size: 1.55rem !important;
            font-weight: 800 !important;
        }

        p, li, span, label {
            color: var(--enterprise-ink);
        }

        .briefing-card p, .briefing-card li {
            font-size: 1.02rem;
        }

        /* Hero banner */
        .hero-banner {
            background: linear-gradient(135deg, var(--enterprise-blue-dark) 0%, var(--enterprise-blue) 100%);
            border-radius: 16px;
            padding: 2.75rem 3rem;
            margin-bottom: 2rem;
            box-shadow: 0 8px 24px rgba(15, 63, 148, 0.18);
        }
        .hero-icon-badge {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 64px;
            height: 64px;
            background: rgba(255, 255, 255, 0.14);
            border-radius: 14px;
            font-size: 30px;
            margin-bottom: 1.1rem;
            overflow: hidden;
        }
        .hero-icon-badge img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            border-radius: 14px;
        }
        .hero-title {
            color: #FFFFFF;
            font-size: 2.5rem;
            font-weight: 800;
            margin: 0 0 0.6rem 0;
            letter-spacing: -0.02em;
        }
        .hero-subtitle {
            color: rgba(255, 255, 255, 0.9);
            font-size: 1.12rem;
            font-weight: 400;
            max-width: 780px;
            line-height: 1.6;
            margin: 0;
        }

        /* Section label */
        .section-eyebrow {
            display: flex;
            align-items: center;
            gap: 0.55rem;
            font-size: 1.3rem;
            font-weight: 800;
            color: var(--enterprise-ink);
            margin: 0.25rem 0 1.1rem 0;
            padding-bottom: 0.65rem;
            border-bottom: 1px solid var(--enterprise-border);
            letter-spacing: -0.01em;
        }
        .section-eyebrow-icon {
            font-size: 1.35rem;
        }

        /* Card wrapper used around groups of native Streamlit inputs */
        div[data-testid="stVerticalBlockBorderWrapper"] {
            background: #FFFFFF;
            border-radius: 14px;
            border: 1px solid var(--enterprise-border);
            box-shadow: 0 1px 3px rgba(16, 24, 40, 0.04);
            transition: box-shadow 0.2s ease;
        }
        div[data-testid="stVerticalBlockBorderWrapper"]:hover {
            box-shadow: 0 4px 14px rgba(16, 24, 40, 0.07);
        }

        /* Inputs */
        div[data-baseweb="select"] > div, .stNumberInput input {
            border-radius: 8px !important;
            border-color: var(--enterprise-border) !important;
        }
        .stSlider {
            padding-top: 0.4rem;
        }

        /* Primary button */
        .stButton > button {
            background: var(--enterprise-blue);
            color: #FFFFFF;
            border: none;
            border-radius: 10px;
            padding: 0.7rem 1rem;
            font-weight: 600;
            font-size: 1rem;
            letter-spacing: 0.01em;
            transition: background 0.15s ease, transform 0.05s ease;
            box-shadow: 0 2px 6px rgba(22, 87, 198, 0.25);
        }
        .stButton > button:hover {
            background: var(--enterprise-blue-dark);
            color: #FFFFFF;
        }
        .stButton > button:active {
            transform: translateY(1px);
        }

        /* KPI metric cards */
        .kpi-card {
            border-radius: 14px;
            padding: 1.4rem 1.5rem;
            border: 1px solid var(--enterprise-border);
            background: #FFFFFF;
            box-shadow: 0 1px 3px rgba(16, 24, 40, 0.04);
            height: 100%;
        }
        .kpi-label {
            font-size: 0.88rem;
            font-weight: 700;
            color: var(--enterprise-muted);
            text-transform: uppercase;
            letter-spacing: 0.04em;
            margin-bottom: 0.55rem;
        }
        .kpi-value {
            font-size: 2.2rem;
            font-weight: 800;
            line-height: 1.15;
        }
        .kpi-blue .kpi-value { color: var(--enterprise-blue-dark); }
        .kpi-green .kpi-value { color: var(--enterprise-green); }
        .kpi-red .kpi-value { color: var(--enterprise-red); }
        .kpi-amber .kpi-value { color: var(--enterprise-amber); }
        .kpi-green { background: var(--enterprise-green-soft); }
        .kpi-red { background: var(--enterprise-red-soft); }
        .kpi-amber { background: var(--enterprise-amber-soft); }

        /* Status banners */
        .status-banner {
            border-radius: 12px;
            padding: 1.1rem 1.4rem;
            font-size: 1rem;
            font-weight: 500;
            display: flex;
            align-items: flex-start;
            gap: 0.7rem;
            margin: 1.2rem 0;
        }
        .status-banner-error {
            background: var(--enterprise-red-soft);
            color: var(--enterprise-red);
            border: 1px solid rgba(179, 38, 30, 0.18);
        }
        .status-banner-success {
            background: var(--enterprise-green-soft);
            color: var(--enterprise-green);
            border: 1px solid rgba(21, 122, 75, 0.18);
        }

        /* Action / recommendation cards */
        .action-card {
            display: flex;
            gap: 0.9rem;
            align-items: flex-start;
            background: #FFFFFF;
            border: 1px solid var(--enterprise-border);
            border-left: 4px solid var(--enterprise-blue);
            border-radius: 12px;
            padding: 1rem 1.2rem;
            margin-bottom: 0.8rem;
            box-shadow: 0 1px 2px rgba(16, 24, 40, 0.03);
            transition: box-shadow 0.15s ease;
        }
        .action-card:hover {
            box-shadow: 0 3px 10px rgba(16, 24, 40, 0.07);
        }
        .action-icon {
            font-size: 1.3rem;
            line-height: 1;
            margin-top: 0.1rem;
        }
        .action-text {
            font-size: 0.98rem;
            color: var(--enterprise-ink);
            line-height: 1.5;
        }

        /* Executive briefing container */
        .briefing-card {
            background: #FFFFFF;
            border: 1px solid var(--enterprise-border);
            border-radius: 16px;
            padding: 2rem 2.2rem;
            box-shadow: 0 1px 4px rgba(16, 24, 40, 0.05);
        }
        .briefing-card h3 {
            font-size: 1.3rem !important;
            margin-top: 1.3rem;
            margin-bottom: 0.7rem;
        }
        .briefing-card ul {
            margin: 0 0 1rem 0;
            padding-left: 1.2rem;
        }
        .briefing-card li {
            margin-bottom: 0.35rem;
            line-height: 1.5;
        }
        .briefing-insight {
            background: var(--enterprise-blue-soft);
            border-radius: 10px;
            padding: 0.9rem 1.1rem;
            margin-top: 0.8rem;
            font-size: 0.97rem;
            line-height: 1.55;
        }

        /* Chart explanation card */
        .chart-note-card {
            background: #FFFFFF;
            border: 1px solid var(--enterprise-border);
            border-radius: 12px;
            padding: 1.1rem 1.3rem;
            margin-bottom: 1.2rem;
            font-size: 0.95rem;
            line-height: 1.55;
        }

        hr {
            border-color: var(--enterprise-border) !important;
        }

    </style>
    """,
    unsafe_allow_html=True
)


@st.cache_resource
def load_pipeline():

    artifacts = joblib.load(
        "model/customer_churn_pipeline.pkl"
    )

    return (
        artifacts["model"],
        artifacts["preprocessor"],
        artifacts["label_encoder"],
        artifacts["threshold"],
        artifacts["explainer"]
    )

model, preprocessor, label_encoder, threshold, explainer = load_pipeline()
print("PIPELINE LOADED")


def get_logo_html():
    """
    Presentation-only helper: looks for the project logo at assets/churn.jpg
    and returns an <img> tag (base64-embedded) for the hero banner.
    Falls back to an emoji badge if the file isn't found, so the app
    never breaks if the asset path differs.
    """
    logo_path = os.path.join("assets", "churn.jpg")
    try:
        with open(logo_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode()
        return f'<img src="data:image/jpeg;base64,{encoded}" alt="Customer Churn logo">'
    except FileNotFoundError:
        return "📊"


# ==========================================
# HERO SECTION (presentation only)
# ==========================================
st.markdown(
    f"""
    <div class="hero-banner">
        <div class="hero-icon-badge">{get_logo_html()}</div>
        <p class="hero-title">Customer Churn Decision Support System</p>
        <p class="hero-subtitle">
            An AI-powered platform that predicts customer churn, explains the key drivers behind each prediction, and recommends actionable business strategies to support proactive customer retention.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-eyebrow"><span class="section-eyebrow-icon">🧾</span>Customer Information</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-eyebrow"><span class="section-eyebrow-icon">👤</span>Customer Profile</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

    col1, col2 = st.columns(2)

    with col1:

        gender = st.selectbox(
            "Gender",
            ["Female", "Male"]
        )

        senior_citizen = st.selectbox(
            "Senior Citizen",
            ["No", "Yes"]
        )

    with col2:

        partner = st.selectbox(
            "Partner",
            ["Yes", "No"]
        )

        dependents = st.selectbox(
            "Dependents",
            ["Yes", "No"]
        )

st.markdown(
    '<div class="section-eyebrow" style="margin-top:1.5rem;"><span class="section-eyebrow-icon">📡</span>Services</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

    col1, col2, col3 = st.columns(3)

    with col1:

        phone_service = st.selectbox(
            "Phone Service",
            ["Yes", "No"]
        )

        online_security = st.selectbox(
            "Online Security",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streaming_tv = st.selectbox(
            "Streaming TV",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with col2:

        multiple_lines = st.selectbox(
            "Multiple Lines",
            [
                "Yes",
                "No",
                "No phone service"
            ]
        )

        online_backup = st.selectbox(
            "Online Backup",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        streaming_movies = st.selectbox(
            "Streaming Movies",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

    with col3:

        internet_service = st.selectbox(
            "Internet Service",
            [
                "DSL",
                "Fiber optic",
                "No"
            ]
        )

        device_protection = st.selectbox(
            "Device Protection",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

        tech_support = st.selectbox(
            "Tech Support",
            [
                "Yes",
                "No",
                "No internet service"
            ]
        )

st.markdown(
    '<div class="section-eyebrow" style="margin-top:1.5rem;"><span class="section-eyebrow-icon">💳</span>Account Information</div>',
    unsafe_allow_html=True
)

with st.container(border=True):

    col1, col2, col3 = st.columns(3)

    with col1:

        tenure = st.slider(
            "Tenure (Months)",
            0,
            72,
            12
        )

        paperless_billing = st.selectbox(
            "Paperless Billing",
            ["Yes", "No"]
        )

    with col2:

        contract = st.selectbox(
            "Contract",
            [
                "Month-to-month",
                "One year",
                "Two year"
            ]
        )

        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    with col3:

        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0
        )

        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=850.0
        )

st.markdown("<div style='margin-top:1.8rem;'></div>", unsafe_allow_html=True)

predict = st.button(
    "Predict Customer Churn",
    use_container_width=True
)

if predict:

    with st.spinner("Running the prediction model..."):

        customer_df = pd.DataFrame({

            "gender": [gender],

            "seniorcitizen": [
                1 if senior_citizen == "Yes" else 0
            ],

            "partner": [partner],

            "dependents": [dependents],

            "tenure": [tenure],

            "phoneservice": [phone_service],

            "multiplelines": [multiple_lines],

            "internetservice": [internet_service],

            "onlinesecurity": [online_security],

            "onlinebackup": [online_backup],

            "deviceprotection": [device_protection],

            "techsupport": [tech_support],

            "streamingtv": [streaming_tv],

            "streamingmovies": [streaming_movies],

            "contract": [contract],

            "paperlessbilling": [paperless_billing],

            "paymentmethod": [payment_method],

            "monthlycharges": [monthly_charges],

            "totalcharges": [total_charges]

        })

        X = preprocessor.transform(customer_df)
        probability = model.predict_proba(X)[0][1]
        prediction_encoded = 1 if probability >= threshold else 0
        prediction = label_encoder.inverse_transform(
            [prediction_encoded]
            )[0]

        if probability < 0.30:
            risk = "🟢 LOW"
        elif probability < threshold:
            risk = "🟡 MEDIUM"
        else:
            risk = "🔴 HIGH"

    st.markdown("<div style='margin-top:0.5rem;'></div>", unsafe_allow_html=True)
    st.markdown(
        '<div class="section-eyebrow"><span class="section-eyebrow-icon">🎯</span>Customer Risk Assessment</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    if prediction == "Yes":
        outcome = "Will Churn"
    else:
        outcome = "Will Stay"

    # Presentation-only color mapping derived from the unchanged prediction/risk values
    outcome_card_class = "kpi-red" if prediction == "Yes" else "kpi-green"

    if risk.endswith("LOW"):
        risk_card_class = "kpi-green"
    elif risk.endswith("MEDIUM"):
        risk_card_class = "kpi-amber"
    else:
        risk_card_class = "kpi-red"

    with col1:
        st.markdown(
            f"""
            <div class="kpi-card {outcome_card_class}">
                <div class="kpi-label">Customer Outcome</div>
                <div class="kpi-value">{outcome}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="kpi-card kpi-blue">
                <div class="kpi-label">Estimated Churn Risk</div>
                <div class="kpi-value">{probability*100:.2f}%</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="kpi-card {risk_card_class}">
                <div class="kpi-label">Risk Level</div>
                <div class="kpi-value">{risk}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    if prediction == "Yes":

        st.markdown(
            """
            <div class="status-banner status-banner-error">
                <span>⚠️</span>
                <span>This customer is likely to discontinue the service. Immediate retention efforts are recommended.</span>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="status-banner status-banner-success">
                <span>✅</span>
                <span>This customer is expected to remain with the company. Continue delivering a positive customer experience.</span>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ==========================================
    # BUSINESS RECOMMENDATIONS
    # =========================================

    st.markdown(
        '<div class="section-eyebrow" style="margin-top:1.8rem;"><span class="section-eyebrow-icon">📋</span>Recommended Actions</div>',
        unsafe_allow_html=True
    )

    recommendations = []

    # Contract
    if contract == "Month-to-month":
        recommendations.append(
            "📌 Offer an annual or two-year contract discount to improve retention."
        )

    # Tech Support
    if tech_support == "No":
        recommendations.append(
            "🛠 Offer a free Tech Support trial."
        )

    # Online Security
    if online_security == "No":
        recommendations.append(
            "🔒 Promote Online Security add-on services."
        )

    # Online Backup
    if online_backup == "No":
        recommendations.append(
            "💾 Recommend Online Backup services."
        )

    # Device Protection
    if device_protection == "No":
        recommendations.append(
            "🖥 Offer Device Protection as a bundled add-on."
        )

    # High Monthly Charges
    if monthly_charges > 80:
        recommendations.append(
            "💰 Consider providing a loyalty discount due to high monthly charges."
        )

    # New Customer
    if tenure < 12:
        recommendations.append(
            "🤝 Assign a customer success representative for proactive engagement."
        )

    # Fiber Customers
    if internet_service == "Fiber optic":
        recommendations.append(
            "📡 Check customer satisfaction with Fiber Optic service quality."
        )

    # High Risk
    if probability >= threshold:
        recommendations.append(
            "🚨 Immediate retention campaign recommended."
        )

    # Default
    if len(recommendations) == 0:
        recommendations.append(
            "✅ No immediate retention action required."
        )



    for recommendation in recommendations:
        # Presentation-only: split the leading emoji (icon) from the message (title/description).
        # The underlying recommendation string/logic above is untouched.
        parts = recommendation.split(" ", 1)
        if len(parts) == 2:
            icon, text = parts[0], parts[1]
        else:
            icon, text = "•", recommendation

        st.markdown(
            f"""
            <div class="action-card">
                <div class="action-icon">{icon}</div>
                <div class="action-text">{text}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<div style='margin-top:0.8rem;'></div>", unsafe_allow_html=True)

    # ==========================================
    # MODEL EXPLANATION
    # ==========================================

    shap_values = explainer.shap_values(X)
    feature_names = preprocessor.get_feature_names_out()

    shap_df = pd.DataFrame({
    "Feature": feature_names,
    "SHAP Value": shap_values[0]
    })
    shap_df["Importance"] = shap_df["SHAP Value"].abs()

    shap_df = shap_df.sort_values(
    by="Importance",
    ascending=False
    )


    shap_df["Feature"] = (
    shap_df["Feature"]
    .str.replace("num__", "", regex=False)
    .str.replace("cat__", "", regex=False)
    .str.replace("_", " ", regex=False)
    )

    feature_mapping = {

    "tenure": "Customer Tenure",

    "monthlycharges": "Monthly Charges",

    "totalcharges": "Total Charges",

    "gender Male": "Gender (Male)",

    "gender Female": "Gender (Female)",

    "partner Yes": "Has a Partner",

    "dependents Yes": "Has Dependents",

    "phoneservice Yes": "Phone Service",

    "multiplelines Yes": "Multiple Phone Lines",

    "internetservice DSL": "Internet Service (DSL)",

    "internetservice Fiber optic": "Internet Service (Fiber Optic)",

    "internetservice No": "No Internet Service",

    "onlinesecurity Yes": "Online Security",

    "onlinebackup Yes": "Online Backup",

    "deviceprotection Yes": "Device Protection",

    "techsupport Yes": "Technical Support",

    "streamingtv Yes": "Streaming TV",

    "streamingmovies Yes": "Streaming Movies",

    "contract Month-to-month": "Month-to-Month Contract",

    "contract One year": "One-Year Contract",

    "contract Two year": "Two-Year Contract",

    "paperlessbilling Yes": "Paperless Billing"

    }

    shap_df["Feature"] = shap_df["Feature"].replace(feature_mapping)

    top10 = shap_df.head(10)

    st.markdown(
        '<div class="section-eyebrow" style="margin-top:1.8rem;"><span class="section-eyebrow-icon">📝</span>Executive Summary</div>',
        unsafe_allow_html=True
    )

    top_positive = top10[top10["SHAP Value"] > 0]["Feature"].head(3).tolist()
    top_negative = top10[top10["SHAP Value"] < 0]["Feature"].head(3).tolist()

    # NOTE: using a real st.container(border=True) here (instead of a raw
    # markdown <div> spanning multiple st calls) is what fixes the empty
    # white box that used to render above this text — a bare opening
    # <div> in one st.markdown call is its own isolated DOM node in
    # Streamlit and never actually wraps the st.write() call after it.
    with st.container(border=True):

        if prediction == "Yes":

            st.markdown(
                f"""
                Based on the customer's profile, the model predicts that this customer is **likely to churn**.

                ### Primary factors increasing churn risk
                • {top_positive[0] if len(top_positive) > 0 else "N/A"}

                • {top_positive[1] if len(top_positive) > 1 else "N/A"}

                • {top_positive[2] if len(top_positive) > 2 else "N/A"}

                ### Factors helping retain this customer
                • {top_negative[0] if len(top_negative) > 0 else "N/A"}

                • {top_negative[1] if len(top_negative) > 1 else "N/A"}

                • {top_negative[2] if len(top_negative) > 2 else "N/A"}

                **Business Insight**

                Although several customer characteristics support retention, the overall churn risk remains elevated. Targeted retention initiatives should focus on the factors contributing most to the customer's likelihood of leaving.
                """
                )

        else:

            st.markdown(
            f"""
            Based on the customer's profile, the model predicts that this customer is **likely to remain with the company**.

            ### Primary factors supporting retention
            • {top_negative[0] if len(top_negative) > 0 else "N/A"}

            • {top_negative[1] if len(top_negative) > 1 else "N/A"}

            • {top_negative[2] if len(top_negative) > 2 else "N/A"}

            ### Factors that still increase churn risk
            • {top_positive[0] if len(top_positive) > 0 else "N/A"}

            • {top_positive[1] if len(top_positive) > 1 else "N/A"}

            • {top_positive[2] if len(top_positive) > 2 else "N/A"}

            **Business Insight**

            The customer currently shows a strong likelihood of remaining with the company. Maintaining service quality and customer satisfaction should help preserve this positive outlook.
            """
            )

    st.markdown(
        '<div class="section-eyebrow" style="margin-top:1.8rem;"><span class="section-eyebrow-icon">🔍</span>Key Factors Behind This Assessment</div>',
        unsafe_allow_html=True
    )

    if prediction == "Yes":

        st.markdown(
            """
            <div class="chart-note-card">
            The visualization below summarizes the customer characteristics that most influenced the prediction.
            <ul style="margin-top:0.6rem;">
                <li><strong>Right side:</strong> Factors that increased the customer's likelihood of churning.</li>
                <li><strong>Left side:</strong> Factors that helped reduce the customer's likelihood of churning.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    else:

        st.markdown(
            """
            <div class="chart-note-card">
            The visualization below summarizes the customer characteristics that most influenced the prediction.
            <ul style="margin-top:0.6rem;">
                <li><strong>Left side:</strong> Factors that helped the customer remain with the company.</li>
                <li><strong>Right side:</strong> Factors that still contributed to churn risk.</li>
            </ul>
            </div>
            """,
            unsafe_allow_html=True
        )

    # ------------------------------------------
    # Advanced presentation for the SAME SHAP data.
    # The underlying values (top10["SHAP Value"], top10["Feature"]) and the
    # red = increases risk / green = decreases risk convention are unchanged.
    # Everything below only concerns rendering: gradient shading, bar
    # shadows/texture, value labels, and seaborn-based styling.
    # ------------------------------------------
    sns.set_style("whitegrid", {
        "axes.edgecolor": "#E3E7EE",
        "grid.color": "#E9ECF2",
    })

    plt.rcParams["font.family"] = "DejaVu Sans"

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FCFCFB")

    values = top10["SHAP Value"].to_numpy()
    features = top10["Feature"].tolist()
    max_abs = max(abs(values.max()), abs(values.min()), 1e-9)

    # Gradient intensity scaled by each bar's own magnitude (unchanged sign
    # logic: positive -> red family, negative -> green family)
    red_cmap = plt.get_cmap("Reds")
    green_cmap = plt.get_cmap("Greens")

    bar_colors = []
    for v in values:
        intensity = 0.35 + 0.55 * (abs(v) / max_abs)
        if v > 0:
            bar_colors.append(red_cmap(intensity))
        else:
            bar_colors.append(green_cmap(intensity))

    y_pos = np.arange(len(features))

    # Soft drop-shadow layer for a bit of texture/depth
    ax.barh(
        y_pos,
        values,
        height=0.62,
        color="#1A2333",
        alpha=0.06,
        zorder=1,
        left=np.where(values > 0, 0.0, values) - (max_abs * 0.012),
    )

    bars = ax.barh(
        y_pos,
        values,
        height=0.62,
        color=bar_colors,
        edgecolor="#FFFFFF",
        linewidth=0.6,
        zorder=3,
    )

    # Value labels at the end of each bar
    for bar, v in zip(bars, values):
        label_x = v + (max_abs * 0.025 if v >= 0 else -max_abs * 0.025)
        ha = "left" if v >= 0 else "right"
        ax.text(
            label_x,
            bar.get_y() + bar.get_height() / 2,
            f"{v:+.2f}",
            va="center",
            ha=ha,
            fontsize=9.5,
            fontweight="bold",
            color="#1A2333",
            path_effects=[path_effects.withStroke(linewidth=2.2, foreground="#FCFCFB")],
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()

    ax.set_xlabel("Impact on Customer Outcome", fontsize=12, color="#1A2333", labelpad=12, fontweight="medium")
    ax.set_ylabel("Customer Attributes", fontsize=12, color="#1A2333", labelpad=12, fontweight="medium")
    ax.set_title("Top Factors Influencing the Prediction", fontsize=16, fontweight="bold", color="#1A2333", pad=18)
    ax.axvline(0, color="#1A2333", linewidth=1.2, zorder=4)

    pad = max_abs * 0.18
    ax.set_xlim(values.min() - pad, values.max() + pad)

    ax.tick_params(axis="both", labelsize=10.5, colors="#1A2333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#E3E7EE")
    ax.grid(axis="x", color="#E9ECF2", linewidth=0.9, zorder=0)
    ax.grid(axis="y", visible=False)
    ax.set_axisbelow(True)

    # Legend clarifying the unchanged color convention
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=red_cmap(0.75), edgecolor="none", label="Increases churn risk"),
        Patch(facecolor=green_cmap(0.75), edgecolor="none", label="Decreases churn risk"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="lower right",
        frameon=False,
        fontsize=9.5,
        labelcolor="#1A2333",
    )

    fig.tight_layout()

    st.pyplot(fig)