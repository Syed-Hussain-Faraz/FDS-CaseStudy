# Foundations of Data Science (FDS) — Case Studies

A collection of coursework projects completed as part of the **Foundations of Data Science (FDS)** course. This repository contains data analysis, machine learning, model evaluation, statistical interpretation, and application development projects.

## Student Information

- **Student Name:** Syed Hussain Faraz
- **Roll Number:** 160124733201
- **Section:** CSE-3
- **Academic Year:** 2026–2027
- **Course:** Foundations of Data Science (FDS)
- **Repository:** https://github.com/Syed-Hussain-Faraz/FDS-CaseStudy

---

## Case Studies

### Case Study 3 — Titanic Survival Prediction

**Topic:** Logistic Regression and Streamlit Deployment

[View Case Study 3](./Case_Study_3/)

An end-to-end machine learning project that predicts whether a Titanic passenger survived based on demographic, socioeconomic, and travel-related features.

**Key components:**
- Exploratory Data Analysis (EDA) and statistical visualizations.
- Missing-value handling and data preprocessing.
- Categorical encoding and numerical feature scaling.
- Logistic Regression model training using Scikit-learn.
- Model evaluation using accuracy, precision, recall, F1-score, and ROC-AUC.
- Interpretation of model coefficients and odds ratios.
- Interactive prediction application developed using Streamlit.

**Technologies:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, Jupyter Notebook, and Streamlit.

**Main files:**
- `titanic.csv` — Dataset.
- `eda_and_train.py` — EDA, preprocessing, model training, and evaluation.
- `app.py` — Interactive Streamlit application.
- `titanic_analysis.ipynb` — Executed notebook containing analysis and outputs.
- `titanic_model.pkl` — Serialized trained model pipeline.
- `model_metadata.json` — Model metrics and interpretation metadata.
- `requirements.txt` — Project dependencies.
- `Case_Study_1_Titanic_Survival_Report.pdf` — Academic report.
- `plots/` — Generated analysis and evaluation visualizations.

**Run locally:**

```bash
git clone https://github.com/Syed-Hussain-Faraz/FDS-CaseStudy.git
cd FDS-CaseStudy/Case_Study_3
pip install -r requirements.txt
streamlit run app.py
```

The application is available locally at `http://localhost:8501`.

**Reported test-set performance:**

| Metric | Score |
|---|---:|
| Accuracy | 80.45% |
| Precision | 79.31% |
| Recall | 66.67% |
| F1-score | 72.44% |
| ROC-AUC | 84.19% |

For the complete methodology, results, and interpretation, see the [Case Study 3 README](./Case_Study_3/README.md).

---

### Case Study 6 — Bank Marketing Classification

**Topic:** Comparative Classification Using Machine Learning

[View Case Study 6](./Case_Study_6/)

A classification project that predicts whether a bank customer will subscribe to a term deposit based on customer characteristics and campaign-related information.

The project uses the Bank Marketing dataset and compares multiple classification algorithms to examine their predictive performance.

**Key components:**
- Data loading, cleaning, and exploratory analysis.
- Analysis of target-class distribution and class imbalance.
- Categorical feature encoding and numerical feature scaling.
- Stratified training and testing split.
- Training and comparison of multiple classification models.
- Evaluation using accuracy, precision, recall, F1-score, and ROC-AUC.
- Confusion matrix analysis and model comparison.

**Models implemented:**
- Logistic Regression
- Decision Tree Classifier
- K-Nearest Neighbors (KNN)
- Random Forest Classifier

**Technologies:** Python, Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn, and Jupyter Notebook.

**Main files:**
- `bank-full.csv` — Bank Marketing dataset.
- `Bank_Marketing_Classification.ipynb` — Notebook containing the analysis, model training, and evaluation.
- `bank_marketing_analysis.py` — Python analysis script.
- `Bank_Marketing_Classification_Case_Study.pdf` — Academic report.
- `README.md` — Case study documentation.

**Run locally:**

```bash
cd FDS-CaseStudy/Case_Study_6
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
python bank_marketing_analysis.py
```

To open the notebook, run:

```bash
jupyter notebook Bank_Marketing_Classification.ipynb
```

**Reported test-set performance:**

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 89.33% | 66.32% | 17.86% | 28.15% | 77.17% |
| Decision Tree | 83.09% | 29.82% | 32.89% | 31.28% | 61.32% |
| KNN | 88.74% | 55.29% | 19.75% | 29.11% | 70.27% |
| Random Forest | 89.54% | 63.73% | 24.57% | 35.47% | 78.88% |

Random Forest achieved the highest accuracy, F1-score, and ROC-AUC among the four evaluated models. The results also illustrate why accuracy alone may be insufficient when the target classes are imbalanced.

For the full methodology, results, and analysis, see the [Case Study 6 README](./Case_Study_6/README.md).

---

## Repository Structure

```text
FDS-CaseStudy/
├── README.md
│
├── Case_Study_3/
│   ├── README.md
│   ├── titanic.csv
│   ├── eda_and_train.py
│   ├── app.py
│   ├── titanic_analysis.ipynb
│   ├── titanic_model.pkl
│   ├── model_metadata.json
│   ├── requirements.txt
│   ├── Case_Study_1_Titanic_Survival_Report.pdf
│   └── plots/
│
└── Case_Study_6/
    ├── README.md
    ├── bank-full.csv
    ├── Bank_Marketing_Classification.ipynb
    ├── bank_marketing_analysis.py
    └── Bank_Marketing_Classification_Case_Study.pdf
```

*The tree shows the expected project organization; individual folders may contain additional files.*

---

## Learning Outcomes

These case studies demonstrate practical applications of foundational data science and machine learning concepts:

- Data cleaning, preprocessing, and exploratory data analysis.
- Statistical summaries and visualization.
- Handling missing values and categorical variables.
- Feature scaling and encoding.
- Classification model development and evaluation.
- Understanding precision, recall, F1-score, ROC-AUC, and confusion matrices.
- Interpreting logistic regression coefficients and odds ratios.
- Comparing algorithms and examining the effects of class imbalance.
- Deploying a machine learning model through an interactive web application.

---

## How to Use This Repository

1. Open the relevant case study folder.
2. Read its `README.md` for the methodology and instructions.
3. Run the Python script or open the Jupyter Notebook to inspect the analysis.
4. Refer to the PDF report for the documented academic submission, where available.

Each case study has its own files, documentation, dependencies, and execution instructions.

---

**Coursework repository:** Foundations of Data Science (FDS)  
**Academic Year:** 2026–2027
