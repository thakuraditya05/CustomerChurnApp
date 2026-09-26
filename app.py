from pathlib import Path
import pickle

import pandas as pd
import streamlit as st
import tensorflow as tf

BASE_DIR = Path(__file__).resolve().parent

# --- Page Configuration ---
st.set_page_config(
    page_title="Customer Churn Intelligence Engine",
    page_icon="📉",
    layout="centered",
    initial_sidebar_state="collapsed",
)


@st.cache_resource
def load_artifacts():
    """Load the model and preprocessing objects once per Streamlit process."""
    model = tf.keras.models.load_model(BASE_DIR / "model.h5", compile=False)
    with (BASE_DIR / "label_encoder_gender.pkl").open("rb") as file:
        label_encoder_gender = pickle.load(file)
    with (BASE_DIR / "onehot_encoder_geo.pkl").open("rb") as file:
        onehot_encoder_geo = pickle.load(file)
    with (BASE_DIR / "scaler.pkl").open("rb") as file:
        scaler = pickle.load(file)
    return model, label_encoder_gender, onehot_encoder_geo, scaler

model, label_encoder_gender, onehot_encoder_geo, scaler = load_artifacts()

# --- Dark Theme & Custom CSS ---
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root { 
        --bg-main: #0b0f12;
        --bg-panel: #141b20;
        --bg-card: #1b242b;
        --text-bright: #f0f4f8;
        --text-muted: #9ba8b3;
        --accent-green: #00d285;
        --border-color: #26333d;
    }

    /* Global styling */
    .stApp { background: var(--bg-main); color: var(--text-bright); }
    [data-testid="stHeader"] { background: rgba(11, 15, 18, 0.9); }
    [data-testid="stMainBlockContainer"] { max-width: 1280px; padding-top: 1.5rem; padding-bottom: 3rem; }
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    h1, h2, h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -.035em; }

    /* Fix input labels visibility */
    label, p, div[data-testid="stMarkdownContainer"] p {
        color: #d1dbe3 !important;
        font-weight: 500;
    }

    .eyebrow { color: var(--accent-green); text-transform: uppercase; font-size: .72rem; font-weight: 700; letter-spacing: .14em; margin-bottom: .4rem; }
    
    /* Hero Banner */
    .hero { 
        background: linear-gradient(135deg, #0d2820 0%, #134235 60%, #1a5847 100%); 
        border: 1px solid #1f5444;
        border-radius: 20px; 
        padding: 1.8rem 2.2rem; 
        color: #fff; 
        margin-bottom: 1.5rem; 
        margin-top: 1.5rem; 
    }
    .hero h1 { color: #fff; font-size: 2rem; margin: 0 0 .4rem; }
    .hero p { color: #b0d4c8; margin: 0; font-size: .95rem; }

    /* Panels */
    .panel { 
        background: var(--bg-panel); 
        border: 1px solid var(--border-color); 
        border-radius: 18px; 
        padding: 1.4rem 1.6rem; 
        margin-bottom: 1rem;
    }
    .section-title { font-family: 'Manrope', sans-serif; font-size: 1.2rem; font-weight: 800; margin: .05rem 0 .25rem; color: var(--text-bright); }
    .section-copy { color: var(--text-muted); font-size: .87rem; margin: 0 0 1rem; }

    /* Custom Result Card (First Image Style) */
    .result-card { 
        background: var(--bg-card); 
        border: 1px solid var(--border-color); 
        border-radius: 16px; 
        padding: 1.6rem; 
        margin-top: 0.5rem; 
        margin-bottom: 1rem;
    }
    .result-label { color: var(--text-muted); font-size: .75rem; text-transform: uppercase; letter-spacing: .1em; font-weight: 700; }
    .result-value { font-family: 'Manrope', sans-serif; font-size: 3.2rem; font-weight: 800; margin: .2rem 0; }
    .result-description { color: #d1dbe3; font-size: .92rem; margin: .6rem 0 .8rem; line-height: 1.4; }
    
    .insight-chip { display: inline-block; border-radius: 999px; padding: .38rem .85rem; font-size: .78rem; font-weight: 700; }
    .chip-low { color: var(--accent-green); background: rgba(0, 210, 133, 0.12); border: 1px solid rgba(0, 210, 133, 0.3); }
    .chip-medium { color: #ffc107; background: rgba(255, 193, 7, 0.12); border: 1px solid rgba(255, 193, 7, 0.3); }
    .chip-high { color: #ff6b6b; background: rgba(255, 107, 107, 0.12); border: 1px solid rgba(255, 107, 107, 0.3); }

    /* Form Input Controls Styling */
    div[data-baseweb="select"] > div, .stNumberInput input { 
        background-color: #1a232a !important; 
        color: #ffffff !important; 
        border-color: #2e3e4a !important;
        border-radius: 8px;
    }
    
    /* Submit Button */
    .stButton > button, [data-testid="stFormSubmitButton"] button { 
        background: var(--accent-green); 
        color: #051a12; 
        border: 0; 
        border-radius: 10px; 
        min-height: 3rem; 
        font-weight: 700; 
        width: 100%; 
        transition: all .2s ease; 
    }
    .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover { 
        background: #00ef96; 
        color: #051a12; 
        box-shadow: 0 4px 15px rgba(0, 210, 133, 0.3);
    }
    
    .score-caption { color: var(--text-muted); font-size: .8rem; margin-top: .4rem; }
    .footer { color: var(--text-muted); font-size: .78rem; text-align: center; margin-top: 2rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Hero Header ---
st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">Customer Intelligence · Neural Network Assessor</div>
      <h1>📊 Customer Attrition & Churn Predictor</h1>
      <p>Analyze customer profile metrics to estimate retention probability using Deep Learning Neural Networks.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# --- Educational Explanations (Collapsible) ---
with st.expander("ℹ️ What does this app do and how is Churn calculated? (Click to expand)"):
    st.markdown("""
    - **Churn Risk Probability:** The likelihood (0% to 100%) that a banking customer will cancel their account or leave the service.
    - **How to use it?** Configure the customer demographics, account parameters, and engagement details below, then click **Assess Churn Risk**.
    - **Model Architecture:** Powered by an **Artificial Neural Network (ANN)** trained with normalized features and encoded categorical inputs.
    """)

with st.expander("📊 What do the input parameters mean?"):
    st.markdown("""
    - **Geography & Demographics:** Customer country location, age, and gender profile.
    - **Credit Score:** Standard evaluation score ranging from 300 to 850.
    - **Tenure:** Total active years the customer has maintained an account with the bank.
    - **Products Owned:** Total active banking products (e.g., loans, savings, investments).
    - **Account Balance & Estimated Salary:** Customer financial liquidity and expected annual income.
    - **Activity Level & Credit Card:** Indicators showing regular account transactions and credit card status.
    """)

st.divider()

# --- Main Columns ---
left, right = st.columns([1.18, 0.82], gap="large")

with left:
    # st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="eyebrow panel ">Data Entry</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Build Customer Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-copy">Provide customer metrics below. Hover over tooltips (❓) for guidance.</div>', unsafe_allow_html=True)

    with st.form("customer_details"):
        st.markdown("#### 👤 1. Personal & Demographic Profile")
        personal_a, personal_b, personal_c = st.columns(3)
        with personal_a:
            geography = st.selectbox(
                "Geography", 
                onehot_encoder_geo.categories_[0],
                help="Primary country where the account is registered."
            )
        with personal_b:
            gender = st.selectbox(
                "Gender", 
                label_encoder_gender.classes_,
                help="Customer recorded gender."
            )
        with personal_c:
            age = st.number_input(
                "Age (Years)", 
                min_value=18, 
                max_value=92, 
                value=40, 
                step=1,
                help="Customer age in years."
            )

        st.markdown("---")
        st.markdown("#### 💳 2. Financial & Account Metrics")
        account_a, account_b, account_c = st.columns(3)
        with account_a:
            credit_score = st.number_input(
                "Credit Score", 
                min_value=300, 
                max_value=850, 
                value=650, 
                step=1,
                help="Standard credit score (300 to 850)."
            )
        with account_b:
            tenure = st.number_input(
                "Tenure (Years)", 
                min_value=0, 
                max_value=10, 
                value=5, 
                step=1,
                help="Years maintained with bank."
            )
        with account_c:
            num_of_products = st.selectbox(
                "Products Owned", 
                [1, 2, 3, 4], 
                index=0,
                help="Active banking products."
            )

        account_d, account_e = st.columns(2)
        with account_d:
            balance = st.number_input(
                "Account Balance ($)", 
                min_value=0.0, 
                value=50000.0, 
                step=1000.0, 
                format="%.2f",
                help="Available total account balance."
            )
        with account_e:
            estimated_salary = st.number_input(
                "Estimated Salary ($)", 
                min_value=0.0, 
                value=100000.0, 
                step=1000.0, 
                format="%.2f",
                help="Estimated annual income."
            )

        st.markdown("---")
        st.markdown("#### 📈 3. Engagement & Activity")
        engagement_a, engagement_b = st.columns(2)
        with engagement_a:
            has_cr_card = st.selectbox(
                "Credit Card Status", 
                [1, 0], 
                format_func=lambda x: "Holds Credit Card (1.0)" if x else "No Credit Card (0.0)",
                help="Customer holds an active credit card."
            )
        with engagement_b:
            is_active_member = st.selectbox(
                "Member Activity Level", 
                [1, 0], 
                format_func=lambda x: "Active Member (1.0)" if x else "Inactive Member (0.0)",
                help="Customer transacts regularly."
            )

        st.write("")
        submitted = st.form_submit_button("🚀 Assess Churn Risk")
    st.markdown('</div>', unsafe_allow_html=True)

with right:
    # st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown('<div class="panel eyebrow">Output Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Risk Analysis</div>', unsafe_allow_html=True)

    if submitted:
        # Preprocessing Data
        input_data = pd.DataFrame(
            {
                "CreditScore": [credit_score],
                "Gender": [label_encoder_gender.transform([gender])[0]],
                "Age": [age],
                "Tenure": [tenure],
                "Balance": [balance],
                "NumOfProducts": [num_of_products],
                "HasCrCard": [has_cr_card],
                "IsActiveMember": [is_active_member],
                "EstimatedSalary": [estimated_salary],
            }
        )
        geo_input = pd.DataFrame({"Geography": [geography]})
        geo_encoded = onehot_encoder_geo.transform(geo_input).toarray()
        geo_encoded_df = pd.DataFrame(
            geo_encoded,
            columns=onehot_encoder_geo.get_feature_names_out(["Geography"]),
        )
        input_data = pd.concat([input_data, geo_encoded_df], axis=1)
        input_data = input_data.reindex(columns=scaler.feature_names_in_)
        input_data_scaled = scaler.transform(input_data)

        probability = float(model.predict(input_data_scaled, verbose=0)[0][0])

        # Dynamic Status Configuration
        if probability < 0.30:
            val_color = "#00d285"
            chip_class = "chip-low"
            chip_text = "Low Attrition Risk"
            summary_desc = "Customer engagement looks stable. No immediate intervention required."
        elif probability <= 0.60:
            val_color = "#ffc107"
            chip_class = "chip-medium"
            chip_text = "Moderate Attrition Risk"
            summary_desc = "Customer exhibits mixed engagement signals. Routine outreach recommended."
        else:
            val_color = "#ff6b6b"
            chip_class = "chip-high"
            chip_text = "High Attrition Risk"
            summary_desc = "Customer shows severe churn indicators. Immediate retention action required."

        # First Image Card HTML
        st.markdown(
            f'''
            <div class="result-card">
                <div class="result-label">ESTIMATED CHURN LIKELIHOOD</div>
                <div class="result-value" style="color: {val_color};">{probability:.1%}</div>
                <div class="result-description">{summary_desc}</div>
                <span class="insight-chip {chip_class}">{chip_text}</span>
            </div>
            ''',
            unsafe_allow_html=True,
        )

        # Progress bar & score range caption (First Image Style)
        st.progress(probability)
        st.markdown(
            '<div class="score-caption">Score range: 0% (Loyal) to 100% (High risk of leaving)</div>',
            unsafe_allow_html=True,
        )

        st.write("")

        # Banner Alert (Second Image Style)
        if probability < 0.30:
            st.info(
                f"🟢 **Low Churn Risk ({probability:.1%}):** The customer exhibits strong engagement metrics. "
                "There is low probability of account cancellation."
            )
        elif probability <= 0.60:
            st.warning(
                f"🟡 **Moderate Churn Risk ({probability:.1%}):** The customer exhibits mixed engagement signals. "
                "Consider offering loyalty rewards or initiating a routine service follow-up."
            )
        else:
            st.error(
                f"🔴 **High Churn Risk ({probability:.1%}):** This customer has a critical risk of leaving! "
                "Immediate retention efforts, promotional incentives, or personal calls are recommended."
            )
    else:
        st.markdown(
            '''
            <div class="result-card">
                <div class="result-label">ESTIMATED CHURN LIKELIHOOD</div>
                <div class="result-value" style="color:#4a5a67;">—</div>
                <div class="result-description">Fill in the profile details on the left and click <b>Assess Churn Risk</b> to run the evaluation.</div>
                <span class="insight-chip" style="color:#8092a1; background:rgba(255,255,255,0.05); border:1px solid #2e3e4a;">Ready to evaluate</span>
            </div>
            ''',
            unsafe_allow_html=True,
        )
        st.progress(0.0)
        st.markdown(
            '<div class="score-caption">Score range: 0% (Loyal) to 100% (High risk of leaving)</div>',
            unsafe_allow_html=True,
        )

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div class="footer">Customer Churn Intelligence Engine &nbsp;·&nbsp; Deep Learning Assessment</div>', unsafe_allow_html=True)



# from pathlib import Path
# import pickle

# import pandas as pd
# import streamlit as st
# import tensorflow as tf

# BASE_DIR = Path(__file__).resolve().parent

# # --- Page Configuration ---
# st.set_page_config(
#     page_title="Customer Churn | Insight",
#     page_icon="◈",
#     layout="wide",
#     initial_sidebar_state="collapsed",
# )

# @st.cache_resource
# def load_artifacts():
#     """Load the model and preprocessing objects once per Streamlit process."""
#     model = tf.keras.models.load_model(BASE_DIR / "model.h5", compile=False)
#     with (BASE_DIR / "label_encoder_gender.pkl").open("rb") as file:
#         label_encoder_gender = pickle.load(file)
#     with (BASE_DIR / "onehot_encoder_geo.pkl").open("rb") as file:
#         onehot_encoder_geo = pickle.load(file)
#     with (BASE_DIR / "scaler.pkl").open("rb") as file:
#         scaler = pickle.load(file)
#     return model, label_encoder_gender, onehot_encoder_geo, scaler

# model, label_encoder_gender, onehot_encoder_geo, scaler = load_artifacts()

# # --- Dark Theme & Custom CSS ---
# st.markdown(
#     """
#     <style>
#     @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

#     :root { 
#         --bg-main: #0b0f12;
#         --bg-panel: #141b20;
#         --bg-card: #1b242b;
#         --text-bright: #f0f4f8;
#         --text-muted: #9ba8b3;
#         --accent-green: #00d285;
#         --accent-mint: #103b2f;
#         --border-color: #26333d;
#     }

#     /* Global styling */
#     .stApp { background: var(--bg-main); color: var(--text-bright); }
#     [data-testid="stHeader"] { background: rgba(11, 15, 18, 0.9); }
#     [data-testid="stMainBlockContainer"] { max-width: 1280px; padding-top: 1.5rem; padding-bottom: 3rem; }
#     html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
#     h1, h2, h3 { font-family: 'Manrope', sans-serif !important; letter-spacing: -.035em; }

#     /* Fix input labels visibility */
#     label, p, div[data-testid="stMarkdownContainer"] p {
#         color: #d1dbe3 !important;
#         font-weight: 500;
#     }

#     .eyebrow { color: var(--accent-green); text-transform: uppercase; font-size: .72rem; font-weight: 700; letter-spacing: .14em; margin-bottom: .4rem; }
    
#     /* Hero Banner */
#     .hero { 
#         background: linear-gradient(135deg, #0d2820 0%, #134235 60%, #1a5847 100%); 
#         border: 1px solid #1f5444;
#         border-radius: 20px; 
#         padding: 1.8rem 2.2rem; 
#         color: #fff; 
#         margin-bottom: 1.5rem; 
#         margin-top: 1.5rem; 
#     }
#     .hero h1 { color: #fff; font-size: 2rem; margin: 0 0 .4rem; }
#     .hero p { color: #b0d4c8; margin: 0; font-size: .95rem; }

#     /* Panels */
#     .panel { 
#         background: var(--bg-panel); 
#         border: 1px solid var(--border-color); 
#         border-radius: 18px; 
#         padding: 1.4rem 1.6rem; 
#         margin-bottom: 1rem;
#     }
#     .section-title { font-family: 'Manrope', sans-serif; font-size: 1.2rem; font-weight: 800; margin: .05rem 0 .25rem; color: var(--text-bright); }
#     .section-copy { color: var(--text-muted); font-size: .87rem; margin-0 0 1rem; }

#     /* Cards & Badges */
#     .result-card { 
#         background: var(--bg-card); 
#         border: 1px solid var(--border-color); 
#         border-radius: 16px; 
#         padding: 1.4rem; 
#         margin-top: 1rem; 
#     }
#     .result-label { color: var(--text-muted); font-size: .75rem; text-transform: uppercase; letter-spacing: .1em; font-weight: 700; }
#     .result-value { font-family: 'Manrope', sans-serif; font-size: 2.8rem; font-weight: 800; color: var(--accent-green); margin: .3rem 0; }
#     .insight-chip { display: inline-block; border-radius: 999px; padding: .38rem .75rem; font-size: .78rem; font-weight: 700; margin-top: .8rem; }
#     .chip-high { color: #ff6b6b; background: rgba(255, 107, 107, 0.15); border: 1px solid rgba(255, 107, 107, 0.3); }
#     .chip-low { color: var(--accent-green); background: rgba(0, 210, 133, 0.15); border: 1px solid rgba(0, 210, 133, 0.3); }

#     /* Form Input Controls Styling */
#     div[data-baseweb="select"] > div, .stNumberInput input { 
#         background-color: #1a232a !important; 
#         color: #ffffff !important; 
#         border-color: #2e3e4a !important;
#         border-radius: 8px;
#     }
    
#     /* Submit Button */
#     .stButton > button, [data-testid="stFormSubmitButton"] button { 
#         background: var(--accent-green); 
#         color: #051a12; 
#         border: 0; 
#         border-radius: 10px; 
#         min-height: 3rem; 
#         font-weight: 700; 
#         width: 100%; 
#         transition: all .2s ease; 
#     }
#     .stButton > button:hover, [data-testid="stFormSubmitButton"] button:hover { 
#         background: #00ef96; 
#         color: #051a12; 
#         box-shadow: 0 4px 15px rgba(0, 210, 133, 0.3);
#     }
    
#     .footer { color: var(--text-muted); font-size: .78rem; text-align: center; margin-top: 2rem; }
#     </style>
#     """,
#     unsafe_allow_html=True,
# )

# # --- Hero Section ---
# st.markdown(
#     """
#     <div class="hero">
#       <div class="eyebrow">Customer Intelligence · Neural Network Assessor</div>
#       <h1>Predict & Prevent Customer Attrition</h1>
#       <p>Fill out the customer profile details below to get an instant risk prediction powered by Artificial Neural Networks.</p>
#     </div>
#     """,
#     unsafe_allow_html=True,
# )

# left, right = st.columns([1.18, 0.82], gap="large")

# with left:
#     st.markdown('<div class="panel eyebrow">Data Entry</div>', unsafe_allow_html=True)
#     # st.markdown('<div class="eyebrow">', unsafe_allow_html=True)
#     st.markdown('<div class="section-title">Build Customer Snapshot</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-copy">Provide the customer details below. Tooltips (❓) explain each parameter.</div>', unsafe_allow_html=True)

#     with st.form("customer_details"):
#         st.markdown("#### 1. Demographic Profile")
#         personal_a, personal_b, personal_c = st.columns(3)
#         with personal_a:
#             geography = st.selectbox(
#                 "Geography", 
#                 onehot_encoder_geo.categories_[0],
#                 help="The primary country where the customer's account is registered."
#             )
#         with personal_b:
#             gender = st.selectbox(
#                 "Gender", 
#                 label_encoder_gender.classes_,
#                 help="Customer's recorded gender."
#             )
#         with personal_c:
#             age = st.number_input(
#                 "Age (Years)", 
#                 min_value=18, 
#                 max_value=92, 
#                 value=40, 
#                 step=1,
#                 help="Customer age in years. Older demographics historically show varied retention patterns."
#             )

#         st.markdown("---")
#         st.markdown("#### 2. Financial & Account Metrics")
#         account_a, account_b, account_c = st.columns(3)
#         with account_a:
#             credit_score = st.number_input(
#                 "Credit Score", 
#                 min_value=300, 
#                 max_value=850, 
#                 value=650, 
#                 step=1,
#                 help="Standard credit evaluation score (300 to 850)."
#             )
#         with account_b:
#             tenure = st.number_input(
#                 "Tenure (Years)", 
#                 min_value=0, 
#                 max_value=10, 
#                 value=5, 
#                 step=1,
#                 help="How many years the customer has maintained an account with the bank."
#             )
#         with account_c:
#             num_of_products = st.selectbox(
#                 "Products Owned", 
#                 [1, 2, 3, 4], 
#                 index=0,
#                 help="Number of bank products (loans, savings, investment accounts) active."
#             )

#         account_d, account_e = st.columns(2)
#         with account_d:
#             balance = st.number_input(
#                 "Account Balance ($)", 
#                 min_value=0.0, 
#                 value=50000.0, 
#                 step=1000.0, 
#                 format="%.2f",
#                 help="Current total money available in customer's account."
#             )
#         with account_e:
#             estimated_salary = st.number_input(
#                 "Estimated Salary ($)", 
#                 min_value=0.0, 
#                 value=100000.0, 
#                 step=1000.0, 
#                 format="%.2f",
#                 help="Estimated annual income of the customer."
#             )

#         st.markdown("---")
#         st.markdown("#### 3. Activity & Engagement")
#         engagement_a, engagement_b = st.columns(2)
#         with engagement_a:
#             has_cr_card = st.selectbox(
#                 "Credit Card Status", 
#                 [1, 0], 
#                 format_func=lambda x: "Holds an active Credit Card" if x else "No Credit Card active",
#                 help="Indicates if the customer holds a credit card issued by the institution."
#             )
#         with engagement_b:
#             is_active_member = st.selectbox(
#                 "Member Activity Level", 
#                 [1, 0], 
#                 format_func=lambda x: "Active Member (Regular transactions)" if x else "Inactive Member (Low activity)",
#                 help="Whether the customer actively transacts or uses digital banking."
#             )

#         st.write("")
#         submitted = st.form_submit_button("✦  Run Churn Prediction Assessment")
#     st.markdown('</div>', unsafe_allow_html=True)

# with right:
#     st.markdown('<div class="panel eyebrow">Output Analytics</div>', unsafe_allow_html=True)
#     # st.markdown('<div class="eyebrow">Output Analytics</div>', unsafe_allow_html=True)
#     st.markdown('<div class="section-title">Risk Analysis</div>', unsafe_allow_html=True)

#     if submitted:
#         # Data Preparation
#         input_data = pd.DataFrame(
#             {
#                 "CreditScore": [credit_score],
#                 "Gender": [label_encoder_gender.transform([gender])[0]],
#                 "Age": [age],
#                 "Tenure": [tenure],
#                 "Balance": [balance],
#                 "NumOfProducts": [num_of_products],
#                 "HasCrCard": [has_cr_card],
#                 "IsActiveMember": [is_active_member],
#                 "EstimatedSalary": [estimated_salary],
#             }
#         )
#         geo_input = pd.DataFrame({"Geography": [geography]})
#         geo_encoded = onehot_encoder_geo.transform(geo_input).toarray()
#         geo_encoded_df = pd.DataFrame(
#             geo_encoded,
#             columns=onehot_encoder_geo.get_feature_names_out(["Geography"]),
#         )
#         input_data = pd.concat([input_data, geo_encoded_df], axis=1)
#         input_data = input_data.reindex(columns=scaler.feature_names_in_)
#         input_data_scaled = scaler.transform(input_data)

#         probability = float(model.predict(input_data_scaled, verbose=0)[0][0])
#         likely_to_churn = probability > 0.5
#         status = "High Attrition Risk" if likely_to_churn else "Low Attrition Risk"
#         recommendation = (
#             "Customer exhibits key indicators of churning. Offer personalized loyalty incentives or schedule a retention call."
#             if likely_to_churn
#             else "Customer engagement looks stable. No immediate intervention required."
#         )
#         chip_class = "chip-high" if likely_to_churn else "chip-low"

#         st.markdown(
#             f'<div class="result-card"><div class="result-label">Estimated Churn Likelihood</div>'
#             f'<div class="result-value" style="color:{"#ff6b6b" if likely_to_churn else "#00d285"}">{probability:.1%}</div>'
#             f'<p style="color:#b0c0ce; font-size:.9rem; margin:.5rem 0;">{recommendation}</p>'
#             f'<span class="insight-chip {chip_class}">{status}</span></div>',
#             unsafe_allow_html=True,
#         )
#         st.write("")
#         st.progress(probability)
#         st.caption("Score range: 0% (Loyal) to 100% (High risk of leaving)")
#     else:
#         st.markdown(
#             '<div class="result-card"><div class="result-label">Awaiting Input</div>'
#             '<div class="result-value" style="font-size:2rem; color:#4a5a67;">—</div>'
#             '<p style="color:#8092a1; font-size:.88rem;">Fill in the profile details on the left and click <b>Run Churn Prediction</b> to generate risk analysis.</p>'
#             '<span class="insight-chip chip-low" style="color:#8092a1; background:rgba(255,255,255,0.05); border:1px solid #2e3e4a;">● &nbsp; Ready to evaluate</span></div>',
#             unsafe_allow_html=True,
#         )
#         st.markdown("#### Input Variables")
#         st.caption("The neural network analyzes Demographics (Age, Country, Gender), Financial standing (Balance, Salary, Credit Score), and Product Engagement.")
#     st.markdown('</div>', unsafe_allow_html=True)

# st.markdown('<div class="footer">Customer Churn Intelligence Engine &nbsp;·&nbsp; Deep Learning Assessment</div>', unsafe_allow_html=True)