# Titanic Survival Prediction using Logistic Regression
### Foundations of Data Science (FDS) — Case Study 3

**Coursework Submission Details:**
- **Course:** Foundations of Data Science (FDS)
- **Case Study:** Case Study 3 — Logistic Regression & Streamlit Deployment
- **Student Name:** Syed Hussain Faraz
- **Roll Number:** 160124733201
- **Section / Batch:** CSE-3
- **Academic Year:** 2026 – 2027
- **GitHub Repository:** [https://github.com/Syed-Hussain-Faraz/FDS-CaseStudy.git](https://github.com/Syed-Hussain-Faraz/FDS-CaseStudy.git)

An end-to-end coursework project implementing **Logistic Regression** on the Titanic dataset, covering data cleaning, statistical exploratory data analysis, train-test evaluation, mathematical odds ratio interpretation, and an interactive **Streamlit** web application.

---

## Table of Contents
- [Overview & Objectives](#overview--objectives)
- [Project Structure](#project-structure)
- [Requirements & Installation](#requirements--installation)
- [Task 1: Exploratory Data Analysis (EDA)](#task-1-exploratory-data-analysis-eda)
- [Task 2: Data Preprocessing](#task-2-data-preprocessing)
- [Task 3 & 4: Model Building & Evaluation](#task-3--4-model-building--evaluation)
- [Task 5: Interpretation & Feature Significance](#task-5-interpretation--feature-significance)
- [Task 6: Streamlit Web Deployment](#task-6-streamlit-web-deployment)

---

## Overview & Objectives

The goal of this case study is to build and interpret a predictive model that estimates the probability of survival for passengers aboard the RMS Titanic based on demographic, socioeconomic, and travel characteristics:
- **Target Variable:** `Survived` (0 = Perished, 1 = Survived)
- **Predictor Variables:** `Pclass` (Passenger Class), `Sex`, `Age`, `SibSp` (Siblings/Spouses), `Parch` (Parents/Children), `Fare`, and `Embarked` (Port of Embarkation).

---

## Project Structure

```text
Case_Study_3/
├── titanic.csv                 # Dataset (891 passenger records, 12 features)
├── eda_and_train.py            # Complete EDA, preprocessing, model training & evaluation script
├── app.py                      # Interactive Streamlit Web Application
├── titanic_analysis.ipynb      # Executed Jupyter Notebook with complete analysis and outputs
├── Case_Study_1_Titanic_Survival_Report.pdf  # Comprehensive academic submission report
├── titanic_model.pkl           # Serialized Scikit-Learn Pipeline (Preprocessor + LogisticRegression)
├── model_metadata.json         # Evaluation metrics, coefficients, odds ratios, and metadata
├── requirements.txt            # Python dependencies for local and cloud deployment
├── README.md                   # Comprehensive project documentation
└── plots/                      # High-resolution generated charts
    ├── 01_target_distribution.png
    ├── 02_survival_by_gender_pclass.png
    ├── 03_age_fare_distributions.png
    ├── 04_correlation_matrix.png
    ├── 05_roc_curve.png
    ├── 06_confusion_matrix.png
    └── 07_feature_importance.png
```

---

## Requirements & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Syed-Hussain-Faraz/FDS-CaseStudy.git
   ```

2. **Navigate to the Case Study folder:**
   ```bash
   cd FDS-CaseStudy/Case_Study_3
   ```

3. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Launch the Streamlit web application:**
   ```bash
   streamlit run app.py
   ```

---

## Task 1: Exploratory Data Analysis (EDA)

Run `python eda_and_train.py` to regenerate all summary statistics and plots into the `plots/` directory:

- **Dataset Size:** 891 rows, 12 columns.
- **Target Distribution:** 
  - `0` (Did Not Survive): 549 passengers (**61.62%**)
  - `1` (Survived): 342 passengers (**38.38%**)
- **Key Demographic Observations:**
  - **Gender Effect:** 74.2% of female passengers survived, compared to only 18.9% of male passengers.
  - **Socioeconomic Class Effect:** 1st Class survival was 63.0%, 2nd Class was 47.3%, and 3rd Class was only 24.2%.
  - **Age Dynamics:** Young children under age 10 had higher survival likelihoods due to lifeboat prioritization.
  - **Fare Distribution:** Higher fares strongly correlate with survival; the median fare for survivors ($26.00) was double that of non-survivors ($10.50).

---

## Task 2: Data Preprocessing

1. **Handling Missing Values:**
   - `Age` (177 missing / 19.9%): Imputed conditionally using median age per `Pclass` and `Sex` group.
   - `Embarked` (2 missing / 0.2%): Imputed with the statistical mode (`'S'` for Southampton).
   - `Cabin` (687 missing / 77.1%): Excluded due to excessive sparsity.
   - `Fare`: Median imputation safeguard.
2. **Feature Selection:** Excluded identifiers and text columns (`PassengerId`, `Name`, `Ticket`).
3. **Categorical Encoding & Scaling:**
   - One-Hot Encoded `Pclass`, `Sex`, and `Embarked` with `drop='first'` to prevent collinearity.
   - Scaled numerical features (`Age`, `Fare`, `SibSp`, `Parch`) with `StandardScaler`.
4. **Data Splitting:** Stratified 80/20 train/test split (712 training, 179 testing).

---

## Task 3 & 4: Model Building & Evaluation

A `scikit-learn` `Pipeline` combining `ColumnTransformer` and `LogisticRegression(solver='lbfgs', max_iter=1000)` was trained.

### Test Set Performance (N=179):
| Metric | Score | Percentage |
| :--- | :--- | :--- |
| **Accuracy** | `0.8045` | **80.45%** |
| **Precision** | `0.7931` | **79.31%** |
| **Recall (Sensitivity)** | `0.6667` | **66.67%** |
| **F1-Score** | `0.7244` | **72.44%** |
| **ROC-AUC Score** | `0.8419` | **84.19%** |

- **ROC-AUC (0.842):** Confirms strong discriminative power across various decision thresholds.
- **Specificity (0.891):** Correctly identifies 98 out of 110 non-survivors in the test set.

---

## Task 5: Interpretation & Feature Significance

The logistic regression equation models the log-odds of survival:
$$\ln\left(\frac{P}{1-P}\right) = \beta_0 + \beta_1 X_1 + \dots + \beta_k X_k$$
Exponentiating the coefficients yields the **Odds Ratio (OR = $e^\beta$)**:

| Feature | Coefficient ($\beta$) | Odds Ratio ($e^\beta$) | Impact on Odds (%) | Significance & Historical Context |
| :--- | :---: | :---: | :---: | :--- |
| `Sex_male` | **-2.5095** | **0.0813** | **-91.87%** | Strongest predictor. Males had 91.9% lower odds of survival than females (*"Women and children first"*). |
| `Pclass_3` | **-2.0828** | **0.1246** | **-87.54%** | Third class passengers had 87.5% lower odds of survival compared to 1st class (trapped on lower decks). |
| `Pclass_2` | **-0.8424** | **0.4307** | **-56.93%** | Second class passengers had 56.9% lower odds of survival compared to 1st class. |
| `Age` | **-0.5291** | **0.5891** | **-41.09%** | Each 1 standard deviation increase in age reduced survival odds by 41.1% (priority given to children). |
| `Embarked_S` | **-0.3973** | **0.6722** | **-32.78%** | Southampton passengers had 32.8% lower odds compared to Cherbourg passengers. |
| `SibSp` | **-0.2802** | **0.7557** | **-24.43%** | Traveling with many siblings/spouses slightly reduced survival odds due to difficulty evacuating together. |
| `Parch` | **-0.0677** | **0.9345** | **-6.55%** | Parents/children effect in presence of SibSp and Age. |
| `Fare` | **+0.1668** | **1.1815** | **+18.15%** | Each 1 standard deviation increase in fare increased survival odds by 18.2% (premium access). |
| `Embarked_Q` | **+0.1754** | **1.1917** | **+19.17%** | Embarking at Queenstown had a small positive adjustment vs Cherbourg reference. |

---

## Task 6: Streamlit Web Deployment

### Local Execution:
Launch the interactive web application using:
```bash
streamlit run app.py
```
This automatically opens the dashboard in your default browser at `http://localhost:8501`.

### App Features:
1. **Interactive Predictor:** Input passenger features or select presets (e.g. *Rose DeWitt Bukater*, *Jack Dawson*) to see live survival probabilities.
2. **Visual Gauge & Breakdown:** Dynamic progress bars displaying survival vs mortality probabilities with factor explanations.
3. **EDA Gallery:** High-resolution charts and statistical distributions.
4. **Model Performance Hub:** Confusion matrix, ROC curve, and classification metrics.
5. **Coefficient & Odds Ratio Table:** Interactive explanation of all learned logistic weights.
