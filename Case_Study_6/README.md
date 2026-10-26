# Bank Marketing Prediction using Machine Learning Classification Algorithms
### Foundations of Data Science (FDS) — Case Study 6

**Coursework Submission Details:**
- **Course:** Foundations of Data Science (FDS)
- **Case Study:** Case Study 6 — Comparative Analysis of Classification Algorithms
- **Student Name:** Syed Hussain Faraz
- **Roll Number:** 160124733201
- **Section / Batch:** CSE-3
- **Academic Year:** 2026–2027
- **GitHub Repository:** https://github.com/Syed-Hussain-Faraz/FDS-CaseStudy.git

An end-to-end machine learning case study that compares Logistic Regression, Decision Tree, K-Nearest Neighbors (KNN), and Random Forest for predicting whether a bank customer will subscribe to a term deposit. The project covers exploratory data analysis, data preprocessing, model training, evaluation metrics, and interpretation of results.

---

## Table of Contents

- [Overview & Objectives](#overview--objectives)
- [Project Structure](#project-structure)
- [Requirements & Installation](#requirements--installation)
- [Task 1: Exploratory Data Analysis](#task-1-exploratory-data-analysis)
- [Task 2: Data Preprocessing](#task-2-data-preprocessing)
- [Task 3: Model Building](#task-3-model-building)
- [Task 4: Model Evaluation & Comparison](#task-4-model-evaluation--comparison)
- [Task 5: Findings & Interpretation](#task-5-findings--interpretation)
- [Conclusion](#conclusion)

---

## Overview & Objectives

The objective is to analyze bank marketing campaign data and develop classification models to predict whether a customer will subscribe to a term deposit.

- **Dataset:** Bank Marketing (`bank-full.csv`)
- **Dataset Size:** 45,211 records and 17 columns
- **Target Variable:** `y` — `yes` or `no`
- **Problem Type:** Binary classification
- **Algorithms:** Logistic Regression, Decision Tree, KNN, and Random Forest

### Objectives

1. Analyze customer and campaign data using exploratory data analysis.
2. Examine the target distribution and relevant statistical patterns.
3. Preprocess numerical and categorical features.
4. Train four classification algorithms using the same train-test split.
5. Compare models using accuracy, precision, recall, F1-score, and ROC-AUC.
6. Interpret the results and identify limitations.

---

## Project Structure

```text
FDS-CaseStudy/
└── Case_Study_6/
    ├── bank-full.csv
    ├── Bank_Marketing_Classification.ipynb
    ├── bank_marketing_analysis.py
    ├── Bank_Marketing_Classification_Case_Study.pdf
    └── README.md
```

The structure above lists the intended project files. Keep the names consistent with the actual files in your repository. If your analysis generates an `outputs/` folder, you can include it as well.

---

## Requirements & Installation

### Libraries Used

- `pandas` — data loading and manipulation
- `numpy` — numerical operations
- `matplotlib` — visualization
- `seaborn` — statistical visualization
- `scikit-learn` — preprocessing, classification, and evaluation

### Run Locally

**1. Clone the repository:**

```bash
git clone https://github.com/Syed-Hussain-Faraz/FDS-CaseStudy.git
```

**2. Navigate to Case Study 6:**

```bash
cd FDS-CaseStudy/Case_Study_6
```

**3. Install the dependencies:**

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

**4. Run the Python script:**

Ensure that `bank-full.csv` is in the same folder as the script.

```bash
python bank_marketing_analysis.py
```

**5. Open the Jupyter Notebook:**

```bash
jupyter notebook Bank_Marketing_Classification.ipynb
```

Alternatively, open the notebook in Google Colab and upload or connect the dataset as required.

**Note:** The dataset is semicolon-separated, so it must be loaded using `pd.read_csv('bank-full.csv', sep=';')`.

---

## Task 1: Exploratory Data Analysis (EDA)

The dataset was examined to understand its structure, data quality, and customer subscription patterns.

### Dataset Summary

| Property | Value |
|---|---:|
| Total records | 45,211 |
| Total columns | 17 |
| Missing values | 0 |
| Duplicate records | 0 |
| Customers who did not subscribe | 39,922 (88.30%) |
| Customers who subscribed | 5,289 (11.70%) |

### Key Findings

- Customers whose previous campaign outcome was successful had a subscription rate of approximately **64.73%**.
- Students had a subscription rate of approximately **28.68%**.
- Retired customers had a subscription rate of approximately **22.79%**.
- Unemployed customers had a subscription rate of approximately **15.50%**.
- Management employees had a subscription rate of approximately **13.76%**.
- Blue-collar employees had a subscription rate of approximately **7.27%**.

These patterns show that subscription rates vary across customer groups and previous campaign outcomes. They indicate associations rather than proof of causation.

---

## Task 2: Data Preprocessing

The following preprocessing steps were used:

1. **Data inspection:** Loaded the semicolon-separated CSV and examined data types, summary statistics, missing values, and duplicate records.
2. **Target encoding:** Converted `y` into a binary target, with `no = 0` and `yes = 1`.
3. **Feature selection:** Excluded `duration` because it is only known after a call, whereas the objective is to predict subscription before the call.
4. **Numerical scaling:** Used `StandardScaler` to standardize numerical features.
5. **Categorical encoding:** Used `OneHotEncoder(handle_unknown='ignore')` to represent categorical features numerically.
6. **Train-test split:** Used an 80:20 stratified split with `random_state=42` to preserve the target-class proportions.
7. **Preprocessing workflow:** Fitted transformations on training data to reduce the risk of data leakage.

---

## Task 3: Model Building

Four classification algorithms were trained and evaluated.

### 1. Logistic Regression

A binary classification algorithm that estimates the probability of a customer subscribing to a term deposit. It serves as a baseline model.

### 2. Decision Tree

A tree-based model that learns decision rules by splitting observations according to feature values. It can capture nonlinear patterns.

### 3. K-Nearest Neighbors (KNN)

A distance-based algorithm that predicts a class from the labels of nearby training observations. The model used `n_neighbors=5`.

### 4. Random Forest

An ensemble model that combines predictions from multiple decision trees. The model used `n_estimators=100` and `random_state=42`.

---

## Task 4: Model Evaluation & Comparison

The models were compared using five metrics:

- **Accuracy:** Percentage of all predictions that are correct.
- **Precision:** Percentage of predicted subscribers who actually subscribed.
- **Recall:** Percentage of actual subscribers correctly identified.
- **F1-score:** Harmonic mean of precision and recall.
- **ROC-AUC:** Measures the ability to distinguish between subscribers and non-subscribers across classification thresholds.

### Test Set Results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 89.33% | 66.32% | 17.86% | 28.15% | 0.7717 |
| Decision Tree | 83.09% | 29.82% | 32.89% | 31.28% | 0.6132 |
| KNN | 88.74% | 55.29% | 19.75% | 29.11% | 0.7027 |
| **Random Forest** | **89.54%** | 63.73% | 24.57% | **35.47%** | **0.7888** |

### Interpretation

- **Best overall performance:** Random Forest achieved the highest accuracy (89.54%), F1-score (35.47%), and ROC-AUC (0.7888).
- **Highest precision:** Logistic Regression achieved 66.32% precision.
- **Highest recall:** Decision Tree achieved 32.89% recall.
- **Important limitation:** All four models had relatively low recall for subscribers, meaning many actual subscribers were classified as non-subscribers.

### Random Forest Confusion Matrix

| Actual / Predicted | Predicted No | Predicted Yes |
|---|---:|---:|
| Actual No | 7,837 | 148 |
| Actual Yes | 798 | 260 |

The model correctly classified 7,837 non-subscribers and 260 subscribers. However, it missed 798 actual subscribers, highlighting the difficulty of predicting the minority class.

---

## Task 5: Findings & Interpretation

1. **Accuracy alone is insufficient:** Approximately 88.30% of the records belong to the non-subscriber class, so accuracy can conceal poor subscriber detection.
2. **Random Forest performed best overall:** It achieved the highest accuracy, F1-score, and ROC-AUC among the evaluated models.
3. **Recall needs improvement:** Random Forest identified approximately 24.57% of actual subscribers at the default classification threshold.
4. **Decision Tree achieved the highest recall:** It identified a greater proportion of actual subscribers, although its overall accuracy and ROC-AUC were lower.
5. **Previous campaign outcome is informative:** The exploratory analysis found a substantially higher subscription rate among customers whose previous campaign outcome was successful.
6. **Potential improvements:** Future experiments could investigate class weighting, resampling applied only to training data, hyperparameter tuning, and classification-threshold adjustment.

The reported results apply to the selected dataset, preprocessing workflow, model configurations, and train-test split.

---

## Conclusion

This case study compared four machine learning algorithms for predicting bank term-deposit subscriptions. The workflow included exploratory data analysis, feature preprocessing, model training, and evaluation using multiple classification metrics.

**Random Forest achieved the strongest overall results** among the four evaluated models, with 89.54% accuracy, a 35.47% F1-score, and a ROC-AUC of 0.7888. However, its recall of 24.57% indicates that identifying actual subscribers remains a significant challenge.

The project demonstrates the importance of evaluating classification models using several metrics, particularly when the target variable is imbalanced. Further work could focus on improving recall and evaluating model performance on additional data.
