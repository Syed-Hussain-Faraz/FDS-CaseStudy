"""
Titanic Logistic Regression - Exploratory Data Analysis, Modeling & Interpretation
Case Study 1: FDS
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
    classification_report
)

warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

os.makedirs('plots', exist_ok=True)


def run_eda(df: pd.DataFrame):
    print("=" * 70)
    print("1. EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    print("\n--- Dataset Shape ---")
    print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")

    print("\n--- Feature Data Types & Non-Null Counts ---")
    print(df.info())

    print("\n--- Missing Values Summary ---")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({'Missing_Count': missing, 'Missing_Pct': missing_pct})
    print(missing_df[missing_df['Missing_Count'] > 0])

    print("\n--- Summary Statistics (Numerical Features) ---")
    print(df.describe().round(2))

    print("\n--- Summary Statistics (Categorical Features) ---")
    print(df.describe(include=['object']))

    print("\n--- Target Variable Distribution (Survived) ---")
    survived_counts = df['Survived'].value_counts()
    survived_pct = df['Survived'].value_counts(normalize=True) * 100
    print(pd.DataFrame({'Count': survived_counts, 'Percentage (%)': survived_pct.round(2)}))

    # 1. Target Distribution Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(['Did Not Survive (0)', 'Survived (1)'], survived_counts.values, 
                  color=['#d9534f', '#5cb85c'], width=0.55, edgecolor='black', linewidth=0.8)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2.0, yval + 10, 
                f"{yval} ({yval/len(df)*100:.1f}%)", ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_title("Survival Distribution on the Titanic", fontsize=14, pad=15, fontweight='bold')
    ax.set_ylabel("Passenger Count", fontsize=12)
    ax.set_ylim(0, 650)
    plt.tight_layout()
    plt.savefig('plots/01_target_distribution.png', dpi=300)
    plt.close()

    # 2. Survival by Gender & Class
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Sex vs Survived
    sex_survival = df.groupby('Sex')['Survived'].value_counts(normalize=True).unstack() * 100
    sex_survival.plot(kind='bar', stacked=True, ax=axes[0], color=['#d9534f', '#5cb85c'], edgecolor='black', width=0.5)
    axes[0].set_title("Survival Rate by Gender", fontsize=13, fontweight='bold')
    axes[0].set_ylabel("Percentage (%)", fontsize=11)
    axes[0].set_xlabel("Sex", fontsize=11)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    axes[0].legend(['Did Not Survive', 'Survived'], loc='upper right')
    for p in axes[0].patches:
        width, height = p.get_width(), p.get_height()
        if height > 5:
            x, y = p.get_xy()
            axes[0].text(x + width/2, y + height/2, f"{height:.1f}%", ha='center', va='center', color='white', fontweight='bold')

    # Pclass vs Survived
    pclass_survival = df.groupby('Pclass')['Survived'].value_counts(normalize=True).unstack() * 100
    pclass_survival.plot(kind='bar', stacked=True, ax=axes[1], color=['#d9534f', '#5cb85c'], edgecolor='black', width=0.5)
    axes[1].set_title("Survival Rate by Passenger Class (Pclass)", fontsize=13, fontweight='bold')
    axes[1].set_ylabel("Percentage (%)", fontsize=11)
    axes[1].set_xlabel("Passenger Class", fontsize=11)
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_xticklabels(['1st Class', '2nd Class', '3rd Class'], rotation=0)
    axes[1].legend(['Did Not Survive', 'Survived'], loc='upper right')
    for p in axes[1].patches:
        width, height = p.get_width(), p.get_height()
        if height > 5:
            x, y = p.get_xy()
            axes[1].text(x + width/2, y + height/2, f"{height:.1f}%", ha='center', va='center', color='white', fontweight='bold')

    plt.tight_layout()
    plt.savefig('plots/02_survival_by_gender_pclass.png', dpi=300)
    plt.close()

    # 3. Distributions of Age & Fare by Survival Status
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Age Histogram / KDE
    sns.histplot(data=df, x='Age', hue='Survived', kde=True, ax=axes[0, 0], 
                 palette=['#d9534f', '#5cb85c'], element='step', common_norm=False, bins=30)
    axes[0, 0].set_title("Age Distribution by Survival Status", fontsize=13, fontweight='bold')
    axes[0, 0].set_xlabel("Age (years)", fontsize=11)
    axes[0, 0].legend(['Survived', 'Did Not Survive'])

    # Age Boxplot
    sns.boxplot(data=df, x='Survived', y='Age', hue='Survived', ax=axes[0, 1], palette=['#d9534f', '#5cb85c'], width=0.4, legend=False)
    axes[0, 1].set_title("Age Boxplot by Survival Status", fontsize=13, fontweight='bold')
    axes[0, 1].set_xticks([0, 1])
    axes[0, 1].set_xticklabels(['Did Not Survive (0)', 'Survived (1)'])

    # Fare Histogram
    sns.histplot(data=df[df['Fare'] < 200], x='Fare', hue='Survived', kde=True, ax=axes[1, 0], 
                 palette=['#d9534f', '#5cb85c'], element='step', common_norm=False, bins=30)
    axes[1, 0].set_title("Fare Distribution (< $200) by Survival Status", fontsize=13, fontweight='bold')
    axes[1, 0].set_xlabel("Fare ($)", fontsize=11)
    axes[1, 0].legend(['Survived', 'Did Not Survive'])

    # Fare Boxplot
    sns.boxplot(data=df, x='Survived', y='Fare', hue='Survived', ax=axes[1, 1], palette=['#d9534f', '#5cb85c'], width=0.4, showfliers=False, legend=False)
    axes[1, 1].set_title("Fare Boxplot (Excl. extreme outliers)", fontsize=13, fontweight='bold')
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_xticklabels(['Did Not Survive (0)', 'Survived (1)'])

    plt.tight_layout()
    plt.savefig('plots/03_age_fare_distributions.png', dpi=300)
    plt.close()

    # 4. Correlation Matrix
    numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
    corr = df[numeric_cols].corr()

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", linewidths=1, ax=ax, vmin=-1, vmax=1)
    ax.set_title("Pearson Correlation Matrix", fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('plots/04_correlation_matrix.png', dpi=300)
    plt.close()

    print("\nEDA plots saved successfully in 'plots/' directory.")


def preprocess_and_train():
    print("\n" + "=" * 70)
    print("2. DATA PREPROCESSING & FEATURE ENGINEERING")
    print("=" * 70)

    df = pd.read_csv('titanic.csv')

    # Selected predictors for Logistic Regression
    # We drop: PassengerId (identifier), Name (high-cardinality text), Ticket (high-cardinality), Cabin (>77% missing)
    # Features kept: Pclass, Sex, Age, SibSp, Parch, Fare, Embarked
    
    print(f"Missing values before imputation:\n{df[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']].isnull().sum()}")

    # Handle missing values
    # Median age by Pclass and Sex for robust imputation
    age_medians = df.groupby(['Pclass', 'Sex'])['Age'].median()
    df['Age'] = df.apply(
        lambda row: age_medians[row['Pclass'], row['Sex']] if pd.isnull(row['Age']) else row['Age'],
        axis=1
    )

    # Impute Embarked with Mode ('S')
    mode_embarked = df['Embarked'].mode()[0]
    df['Embarked'] = df['Embarked'].fillna(mode_embarked)

    # Impute Fare with median if missing (for generalizability)
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    X = df[features].copy()
    y = df['Survived'].copy()

    # Split: 80% train, 20% test (stratified by survival status)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    print(f"\nTrain set size: {X_train.shape[0]} samples")
    print(f"Test set size:  {X_test.shape[0]} samples")

    # Define Preprocessing Pipelines
    numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']
    categorical_features = ['Pclass', 'Sex', 'Embarked']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ]
    )

    # Logistic Regression Pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42, max_iter=1000, C=1.0, solver='lbfgs'))
    ])

    print("\n" + "=" * 70)
    print("3. MODEL TRAINING")
    print("=" * 70)
    model_pipeline.fit(X_train, y_train)
    print("Logistic Regression model trained successfully.")

    print("\n" + "=" * 70)
    print("4. MODEL EVALUATION (TEST SET)")
    print("=" * 70)

    y_pred = model_pipeline.predict(X_test)
    y_pred_proba = model_pipeline.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    metrics = {
        'Accuracy': float(accuracy),
        'Precision': float(precision),
        'Recall': float(recall),
        'F1_Score': float(f1),
        'ROC_AUC': float(roc_auc),
        'Train_Samples': int(len(X_train)),
        'Test_Samples': int(len(X_test))
    }

    print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
    print(f"F1-Score:  {f1:.4f} ({f1*100:.2f}%)")
    print(f"ROC-AUC:   {roc_auc:.4f} ({roc_auc*100:.2f}%)")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Did Not Survive (0)', 'Survived (1)']))

    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax,
                xticklabels=['Pred: Died (0)', 'Pred: Survived (1)'],
                yticklabels=['Actual: Died (0)', 'Actual: Survived (1)'])
    ax.set_title("Confusion Matrix (Test Set, N=179)", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plt.savefig('plots/06_confusion_matrix.png', dpi=300)
    plt.close()

    # ROC Curve Plot
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot(fpr, tpr, color='#0275d8', lw=2.5, label=f'Logistic Regression (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='#888888', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.500)')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax.set_ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=11)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=13, fontweight='bold', pad=12)
    ax.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    plt.savefig('plots/05_roc_curve.png', dpi=300)
    plt.close()

    print("\n" + "=" * 70)
    print("5. MODEL INTERPRETATION & COEFFICIENT ANALYSIS")
    print("=" * 70)

    # Extract feature names after One-Hot Encoding
    ohe_features = model_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = numeric_features + list(ohe_features)
    coefficients = model_pipeline.named_steps['classifier'].coef_[0]
    intercept = model_pipeline.named_steps['classifier'].intercept_[0]

    odds_ratios = np.exp(coefficients)
    pct_change = (odds_ratios - 1) * 100

    coef_df = pd.DataFrame({
        'Feature': all_feature_names,
        'Coefficient (Log-Odds)': coefficients,
        'Odds Ratio (e^coef)': odds_ratios,
        'Effect on Odds (%)': pct_change
    }).sort_values(by='Coefficient (Log-Odds)', ascending=False).reset_index(drop=True)

    print(f"Model Intercept (log-odds): {intercept:.4f} (Base Odds: {np.exp(intercept):.4f})")
    print("\nFeature Coefficients and Odds Ratios:")
    print(coef_df.to_string(index=False, justify='left'))

    # Plot Feature Coefficients
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['#5cb85c' if c > 0 else '#d9534f' for c in coef_df['Coefficient (Log-Odds)']]
    bars = ax.barh(coef_df['Feature'], coef_df['Coefficient (Log-Odds)'], color=colors, edgecolor='black', height=0.6)
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.set_title('Logistic Regression Coefficients (Impact on Log-Odds of Survival)', fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Coefficient Value (Log-Odds)', fontsize=11)
    for bar in bars:
        width = bar.get_width()
        ha = 'left' if width > 0 else 'right'
        offset = 0.04 if width > 0 else -0.04
        ax.text(width + offset, bar.get_y() + bar.get_height()/2, f"{width:.2f}", va='center', ha=ha, fontsize=10, fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/07_feature_importance.png', dpi=300)
    plt.close()

    # Save artifacts for Streamlit deployment
    joblib.dump(model_pipeline, 'titanic_model.pkl')
    
    # Save metadata dictionary
    meta = {
        'metrics': metrics,
        'coefficients': coef_df.to_dict(orient='records'),
        'intercept': float(intercept),
        'features': features,
        'numeric_features': numeric_features,
        'categorical_features': categorical_features
    }
    with open('model_metadata.json', 'w') as f:
        json.dump(meta, f, indent=4)

    print("\nPipeline model saved to 'titanic_model.pkl'")
    print("Model metadata saved to 'model_metadata.json'")
    print("Evaluation and Interpretation plots saved to 'plots/'")
    print("\n" + "=" * 70)


if __name__ == '__main__':
    df = pd.read_csv('titanic.csv')
    run_eda(df)
    preprocess_and_train()
