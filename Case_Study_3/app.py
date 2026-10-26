"""
Streamlit Web Application: Titanic Survival Predictor & Logistic Regression Analysis
Case Study 1: FDS
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Configure Streamlit page
st.set_page_config(
    page_title="Titanic Survival Predictor | Logistic Regression",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for polished UI
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border-left: 5px solid #3B82F6;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .survived-card {
        background: linear-gradient(135deg, #ECFDF5 0%, #D1FAE5 100%);
        border: 2px solid #10B981;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 15px;
    }
    .perished-card {
        background: linear-gradient(135deg, #FEF2F2 0%, #FEE2E2 100%);
        border: 2px solid #EF4444;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        margin-top: 15px;
    }
    .factor-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 15px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px;
    }
    .factor-positive {
        background-color: #D1FAE5;
        color: #065F46;
    }
    .factor-negative {
        background-color: #FEE2E2;
        color: #991B1B;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_metadata():
    model_path = 'titanic_model.pkl'
    meta_path = 'model_metadata.json'
    
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        st.error("Trained model or metadata not found. Please run `eda_and_train.py` first.")
        st.stop()
        
    pipeline = joblib.load(model_path)
    with open(meta_path, 'r') as f:
        meta = json.load(f)
    return pipeline, meta


pipeline, metadata = load_model_and_metadata()
metrics = metadata.get('metrics', {})
coefficients = pd.DataFrame(metadata.get('coefficients', []))

# Main Title & Subtitle
st.markdown('<div class="main-header">🚢 Titanic Survival Prediction & Logistic Regression Analytics</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Foundations of Data Science (FDS) — Case Study 3 Coursework Project</div>', unsafe_allow_html=True)
st.markdown("""
<div style="display: flex; gap: 8px; margin-bottom: 18px; flex-wrap: wrap;">
    <span style="background: #EEF2FF; color: #4338CA; padding: 4px 10px; border-radius: 12px; font-size: 0.85rem; font-weight: 600;">🎓 Student: Syed Hussain Faraz (160124733201)</span>
    <span style="background: #F3F4F6; color: #374151; padding: 4px 10px; border-radius: 12px; font-size: 0.85rem; font-weight: 600;">Section: CSE-3 | FDS Case Study 3</span>
    <span style="background: #ECFDF5; color: #047857; padding: 4px 10px; border-radius: 12px; font-size: 0.85rem; font-weight: 600;">Model: Logistic Regression (Accuracy: 80.5%, ROC-AUC: 0.842)</span>
</div>
""", unsafe_allow_html=True)

# Tabs Navigation
tab_pred, tab_eda, tab_eval, tab_interp, tab_deploy = st.tabs([
    "🔮 Survival Predictor",
    "📊 Exploratory Data Analysis (EDA)",
    "📈 Model Evaluation",
    "🧠 Coefficient Interpretation",
    "🚀 Deployment Guide"
])


# ----------------------------------------------------------------------------------------
# TAB 1: PREDICTION INTERFACE
# ----------------------------------------------------------------------------------------
with tab_pred:
    st.subheader("Passenger Information & Interactive Prediction")
    st.markdown("Select a historical preset or input custom passenger details to estimate survival probability using the trained Logistic Regression model.")

    # Preset selection
    preset_col1, preset_col2 = st.columns([1, 2])
    with preset_col1:
        preset = st.selectbox(
            "Quick Scenario Presets:",
            [
                "Custom Passenger",
                "Rose DeWitt Bukater (1st Class Female)",
                "Jack Dawson (3rd Class Male)",
                "Wealthy Businessman (1st Class Male)",
                "3rd Class Mother with Children"
            ]
        )

    # Preset default values
    if preset == "Rose DeWitt Bukater (1st Class Female)":
        def_pclass, def_sex, def_age, def_sibsp, def_parch, def_fare, def_embarked = 1, "female", 17, 0, 1, 108.90, "C"
    elif preset == "Jack Dawson (3rd Class Male)":
        def_pclass, def_sex, def_age, def_sibsp, def_parch, def_fare, def_embarked = 3, "male", 20, 0, 0, 7.25, "S"
    elif preset == "Wealthy Businessman (1st Class Male)":
        def_pclass, def_sex, def_age, def_sibsp, def_parch, def_fare, def_embarked = 1, "male", 45, 0, 0, 80.0, "C"
    elif preset == "3rd Class Mother with Children":
        def_pclass, def_sex, def_age, def_sibsp, def_parch, def_fare, def_embarked = 3, "female", 32, 1, 2, 21.0, "S"
    else:
        def_pclass, def_sex, def_age, def_sibsp, def_parch, def_fare, def_embarked = 2, "female", 28, 0, 0, 26.0, "S"

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("#### 🎫 Ticket & Class")
            pclass_label = st.selectbox(
                "Passenger Class (Pclass):",
                ["1st Class (Upper)", "2nd Class (Middle)", "3rd Class (Lower)"],
                index=def_pclass - 1
            )
            pclass = int(pclass_label[0])

            fare = st.slider(
                "Ticket Fare ($):",
                min_value=0.0, max_value=515.0, value=float(def_fare), step=1.0,
                help="Average fares: 1st Class ~$84, 2nd Class ~$20, 3rd Class ~$13"
            )

        with col2:
            st.markdown("#### 👤 Demographics")
            sex = st.radio(
                "Sex:",
                ["female", "male"],
                index=0 if def_sex == "female" else 1,
                horizontal=True
            )

            age = st.slider(
                "Age (Years):",
                min_value=0.5, max_value=80.0, value=float(def_age), step=0.5,
                help="Infants and children received priority in lifeboats"
            )

        with col3:
            st.markdown("#### 👨‍👩‍👧 Family & Embarkation")
            sibsp = st.number_input(
                "Siblings / Spouses Aboard (SibSp):",
                min_value=0, max_value=8, value=int(def_sibsp), step=1
            )
            parch = st.number_input(
                "Parents / Children Aboard (Parch):",
                min_value=0, max_value=6, value=int(def_parch), step=1
            )
            embarked_label = st.selectbox(
                "Port of Embarkation:",
                ["Cherbourg (C)", "Queenstown (Q)", "Southampton (S)"],
                index={"C": 0, "Q": 1, "S": 2}[def_embarked]
            )
            embarked = embarked_label.split("(")[1][0]

        submit = st.form_submit_button("⚡ Predict Survival Outcome", use_container_width=True)

    # Perform Prediction
    input_df = pd.DataFrame([{
        'Pclass': pclass,
        'Sex': sex,
        'Age': age,
        'SibSp': sibsp,
        'Parch': parch,
        'Fare': fare,
        'Embarked': embarked
    }])

    prob_survival = pipeline.predict_proba(input_df)[0][1]
    prob_mortality = 1.0 - prob_survival
    predicted_class = int(prob_survival >= 0.5)

    st.markdown("---")
    st.markdown("### 🎯 Model Prediction Result")

    res_col1, res_col2 = st.columns([1, 1])

    with res_col1:
        if predicted_class == 1:
            st.markdown(f"""
            <div class="survived-card">
                <h1 style="color: #065F46; margin: 0; font-size: 2.5rem;">🎉 SURVIVED</h1>
                <p style="font-size: 1.25rem; font-weight: 600; color: #047857; margin-top: 10px;">
                    Estimated Survival Probability: <span style="font-size: 1.6rem;">{prob_survival * 100:.1f}%</span>
                </p>
                <p style="color: #4B5563; font-size: 0.95rem;">
                    The model predicts that this passenger would have been safely evacuated on a lifeboat.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="perished-card">
                <h1 style="color: #991B1B; margin: 0; font-size: 2.5rem;">⚠️ DID NOT SURVIVE</h1>
                <p style="font-size: 1.25rem; font-weight: 600; color: #B91C1C; margin-top: 10px;">
                    Estimated Survival Probability: <span style="font-size: 1.6rem;">{prob_survival * 100:.1f}%</span>
                </p>
                <p style="color: #4B5563; font-size: 0.95rem;">
                    The model predicts that this passenger was unlikely to survive the disaster.
                </p>
            </div>
            """, unsafe_allow_html=True)

    with res_col2:
        st.markdown("#### Probability Breakdown")
        st.progress(float(prob_survival), text=f"Survival Probability: {prob_survival*100:.1f}%")
        st.progress(float(prob_mortality), text=f"Mortality Probability: {prob_mortality*100:.1f}%")

        st.markdown("#### Key Influencing Factors for this Passenger:")
        factors_html = ""
        if sex == "female":
            factors_html += '<span class="factor-badge factor-positive">✓ Female Gender (+Odds)</span>'
        else:
            factors_html += '<span class="factor-badge factor-negative">✗ Male Gender (-91.9% Odds)</span>'

        if pclass == 1:
            factors_html += '<span class="factor-badge factor-positive">✓ 1st Class Cabin (+Odds)</span>'
        elif pclass == 2:
            factors_html += '<span class="factor-badge factor-negative">✗ 2nd Class Cabin (-56.9% Odds vs 1st)</span>'
        else:
            factors_html += '<span class="factor-badge factor-negative">✗ 3rd Class Cabin (-87.5% Odds vs 1st)</span>'

        if age < 12:
            factors_html += '<span class="factor-badge factor-positive">✓ Child Priority (+Odds)</span>'
        elif age > 50:
            factors_html += '<span class="factor-badge factor-negative">✗ Senior Age (-Odds)</span>'

        if fare > 50:
            factors_html += '<span class="factor-badge factor-positive">✓ Premium Fare Paid (+Odds)</span>'

        if embarked == "C":
            factors_html += '<span class="factor-badge factor-positive">✓ Cherbourg Embarkation (+Odds)</span>'
        elif embarked == "S":
            factors_html += '<span class="factor-badge factor-negative">✗ Southampton Embarkation (-32.8% Odds)</span>'

        st.markdown(factors_html, unsafe_allow_html=True)


# ----------------------------------------------------------------------------------------
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# ----------------------------------------------------------------------------------------
with tab_eda:
    st.subheader("1. Exploratory Data Analysis & Visualizations")
    st.markdown("""
    Exploratory Data Analysis investigates feature distributions, demographic correlations, and historical patterns
    governing survival during the Titanic shipwreck.
    """)

    # Overview Metrics
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Total Passengers in Dataset", "891")
    with m2:
        st.metric("Total Survived", "342 (38.38%)")
    with m3:
        st.metric("Total Perished", "549 (61.62%)")
    with m4:
        st.metric("Gender Ratio (M / F)", "577 / 314")

    st.markdown("---")
    
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        if os.path.exists('plots/01_target_distribution.png'):
            st.image('plots/01_target_distribution.png', caption="Figure 1: Survival Class Balance", use_container_width=True)
            st.info("**Key Insight:** The dataset is moderately imbalanced with only ~38.4% of passengers surviving.")
    with col_e2:
        if os.path.exists('plots/02_survival_by_gender_pclass.png'):
            st.image('plots/02_survival_by_gender_pclass.png', caption="Figure 2: Survival by Gender & Socioeconomic Class", use_container_width=True)
            st.info("**Key Insight:** Females had a >74% survival rate vs <19% for males. 1st Class passengers survived at 63% vs only 24% for 3rd Class.")

    st.markdown("---")

    col_e3, col_e4 = st.columns(2)
    with col_e3:
        if os.path.exists('plots/03_age_fare_distributions.png'):
            st.image('plots/03_age_fare_distributions.png', caption="Figure 3: Age & Fare Distributions by Survival", use_container_width=True)
            st.info("**Key Insight:** Young children (<10) had higher survival rates. Higher ticket fares correlate strongly with survival.")
    with col_e4:
        if os.path.exists('plots/04_correlation_matrix.png'):
            st.image('plots/04_correlation_matrix.png', caption="Figure 4: Pearson Correlation Matrix", use_container_width=True)
            st.info("**Key Insight:** Negative correlation between Pclass and Survived (-0.34); positive correlation between Fare and Survived (+0.26).")


# ----------------------------------------------------------------------------------------
# TAB 3: MODEL EVALUATION
# ----------------------------------------------------------------------------------------
with tab_eval:
    st.subheader("4. Model Evaluation & Performance Metrics")
    st.markdown("""
    The Logistic Regression model was evaluated on a held-out test set (20% split, 179 samples) stratified by the target variable.
    """)

    # Score Cards
    sc1, sc2, sc3, sc4, sc5 = st.columns(5)
    with sc1:
        st.metric("Accuracy", f"{metrics.get('Accuracy', 0)*100:.2f}%")
    with sc2:
        st.metric("Precision", f"{metrics.get('Precision', 0)*100:.2f}%")
    with sc3:
        st.metric("Recall", f"{metrics.get('Recall', 0)*100:.2f}%")
    with sc4:
        st.metric("F1-Score", f"{metrics.get('F1_Score', 0)*100:.2f}%")
    with sc5:
        st.metric("ROC-AUC Score", f"{metrics.get('ROC_AUC', 0)*100:.2f}%")

    st.markdown("---")

    col_v1, col_v2 = st.columns(2)
    with col_v1:
        if os.path.exists('plots/05_roc_curve.png'):
            st.image('plots/05_roc_curve.png', caption="Figure 5: Receiver Operating Characteristic (ROC) Curve", use_container_width=True)
            st.success(f"**ROC-AUC = {metrics.get('ROC_AUC', 0):.3f}**: Indicates strong discriminative ability far exceeding random chance (0.50).")
    with col_v2:
        if os.path.exists('plots/06_confusion_matrix.png'):
            st.image('plots/06_confusion_matrix.png', caption="Figure 6: Confusion Matrix on Test Set (N=179)", use_container_width=True)
            st.success("**Confusion Matrix:** Demonstrates balanced performance with 89% specificity (predicting fatalities) and 67% sensitivity.")


# ----------------------------------------------------------------------------------------
# TAB 4: MODEL INTERPRETATION & COEFFICIENTS
# ----------------------------------------------------------------------------------------
with tab_interp:
    st.subheader("5. Model Interpretation & Feature Significance")
    st.markdown("""
    In Logistic Regression, the model fits a linear equation for the **log-odds** (logit) of the probability:
    $$\\ln\\left(\\frac{P}{1-P}\\right) = \\beta_0 + \\beta_1 X_1 + \\dots + \\beta_k X_k$$
    Exponentiating the coefficients gives the **Odds Ratio** ($e^{\\beta}$), indicating how the odds of survival multiply per unit change in the feature.
    """)

    col_i1, col_i2 = st.columns([1.2, 1])
    with col_i1:
        if os.path.exists('plots/07_feature_importance.png'):
            st.image('plots/07_feature_importance.png', caption="Figure 7: Logistic Regression Log-Odds Coefficients", use_container_width=True)
    with col_i2:
        st.markdown("#### Coefficients & Odds Ratio Table")
        st.dataframe(
            coefficients.style.format({
                'Coefficient (Log-Odds)': '{:.4f}',
                'Odds Ratio (e^coef)': '{:.4f}',
                'Effect on Odds (%)': '{:+.2f}%'
            }),
            use_container_width=True,
            hide_index=True
        )

    st.markdown("---")
    st.markdown("### 📖 Detailed Discussion of Feature Significance")

    st.markdown("""
    1. **Sex (`Sex_male`, $\\beta = -2.51$, Odds Ratio $= 0.081$):**
       - **Highest Magnitude Predictor:** Holding other factors constant, being male reduced the odds of survival by **91.87%** compared to female passengers.
       - **Historical Context:** Directly validates the strict enforcement of the maritime protocol *"Women and children first"* during lifeboat loading.

    2. **Passenger Class (`Pclass_3`, $\\beta = -2.08$, Odds Ratio $= 0.125$ & `Pclass_2`, $\\beta = -0.84$, Odds Ratio $= 0.431$):**
       - **Socioeconomic Impact:** Compared to 1st class passengers, 3rd class passengers suffered an **87.54% reduction** in odds of survival, while 2nd class suffered a **56.93% reduction**.
       - **Structural Context:** 1st Class cabins were situated on upper decks near the boat deck, granting immediate access to lifeboats, while 3rd Class passengers were trapped in lower hull compartments with locked barrier gates.

    3. **Age ($\beta = -0.53$, Odds Ratio $= 0.589$ per standard deviation):**
       - **Protective Effect for Youth:** Older passengers experienced significantly decreased odds of survival (-41.09% per std dev increase). Young children and infants were given priority placement on lifeboats.

    4. **Fare ($\beta = +0.17$, Odds Ratio $= 1.182$ per standard deviation):**
       - **Wealth Advantage:** Higher ticket fares paid correlated with increased survival odds (+18.15%), capturing premium cabin locations and higher priority assistance.

    5. **Port of Embarkation (`Embarked_S`, $\beta = -0.40$, Odds Ratio $= 0.672$):**
       - Passengers embarking from Southampton faced 32.78% lower survival odds than those from Cherbourg, largely due to Southampton having a disproportionate share of lower-income 3rd class passengers and crew.
    """)


# ----------------------------------------------------------------------------------------
# TAB 5: DEPLOYMENT GUIDE
# ----------------------------------------------------------------------------------------
with tab_deploy:
    st.subheader("6. Streamlit Deployment Instructions")
    st.markdown("""
    This project is fully structured for both **Local Deployment** and **Online Cloud Deployment** via **Streamlit Community Cloud**.
    """)

    st.markdown("#### Option A: Running the App Locally")
    st.code("""
# 1. Clone or navigate to the project directory
cd "c:/Users/hussa/Desktop/Case Study 1 FDS"

# 2. Install dependencies
pip install -r requirements.txt

# 3. Train model & generate plots (if not already run)
python eda_and_train.py

# 4. Launch the Streamlit application
streamlit run app.py
    """, language="bash")

    st.markdown("#### Option B: Deploying Online via Streamlit Community Cloud (Free)")
    st.markdown("""
    1. **Push to GitHub**:
       - Create a new repository on GitHub (e.g. `titanic-logistic-regression`).
       - Commit and push the project files: `app.py`, `titanic_model.pkl`, `model_metadata.json`, `requirements.txt`, `plots/`, `titanic.csv`, and `README.md`.
    2. **Connect to Streamlit Cloud**:
       - Go to [share.streamlit.io](https://share.streamlit.io) and sign in with your GitHub account.
       - Click **New App**.
       - Select your repository, branch (`main`), and set the main file path to `app.py`.
    3. **Deploy**:
       - Click **Deploy**. Streamlit Cloud will install dependencies from `requirements.txt` and provide a public URL (e.g., `https://titanic-prediction-fds.streamlit.app`).
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6B7280; font-size: 0.9rem;'>"
    "Titanic Survival Logistic Regression | FDS Case Study 3 | Syed Hussain Faraz (160124733201) | Deployed with Streamlit"
    "</div>",
    unsafe_allow_html=True
)
