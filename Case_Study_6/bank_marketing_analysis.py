"""
Written by: Syed Hussain Faraz (160124733201)
Bank Marketing Classification Analysis
This script must be run in the same folder as bank-full.csv.

Requirements:
    pandas, numpy, matplotlib, scikit-learn

The script loads the semicolon-separated dataset, performs a stratified 80:20 split,
preprocesses numerical/categorical features inside pipelines, trains four classifiers,
prints evaluation metrics, and saves visualizations in a folder named outputs.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    classification_report, ConfusionMatrixDisplay, roc_curve, auc
)

DATA_PATH = Path("bank-full.csv")
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# 1. Load data: the original dataset uses semicolons as delimiters.
df = pd.read_csv(DATA_PATH, sep=";")
print("Dataset shape:", df.shape)
print(df.head(10))
print("\nMissing values:\n", df.isnull().sum())
print("Duplicate rows:", df.duplicated().sum())
print("\nTarget distribution:\n", df["y"].value_counts())
print("\nTarget percentages:\n", (df["y"].value_counts(normalize=True) * 100).round(2))

# 2. Visualize target distribution.
target_counts = df["y"].value_counts().reindex(["no", "yes"])
plt.figure(figsize=(7, 4.5))
bars = plt.bar(target_counts.index, target_counts.values)
plt.title("Distribution of Term-Deposit Subscription")
plt.xlabel("Subscription Outcome (y)")
plt.ylabel("Number of Customers")
for bar, value in zip(bars, target_counts.values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f"{value:,}",
             ha="center", va="bottom")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "target_distribution.png", dpi=160)
plt.close()

# 3. Exclude duration to model prediction before a call takes place.
X = df.drop(columns=["y", "duration"])
y = df["y"].map({"no": 0, "yes": 1})

numeric_features = X.select_dtypes(include=np.number).columns.tolist()
categorical_features = X.select_dtypes(include=["object", "category"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

preprocessor = ColumnTransformer([
    ("num", StandardScaler(), numeric_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
])

# 4. Train all models with the same split and preprocessing approach.
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, random_state=42, n_jobs=-1
    )
}

results = []
trained = {}
predictions = {}

for name, estimator in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", estimator)
    ])
    pipeline.fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    prob = pipeline.predict_proba(X_test)[:, 1]
    trained[name] = pipeline
    predictions[name] = (pred, prob)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "Precision": precision_score(y_test, pred, zero_division=0),
        "Recall": recall_score(y_test, pred, zero_division=0),
        "F1-Score": f1_score(y_test, pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_test, prob)
    })
    print(f"\n--- {name} ---")
    print(classification_report(
        y_test, pred, labels=[0, 1],
        target_names=["No subscription", "Subscribed"],
        zero_division=0
    ))

results_df = pd.DataFrame(results).set_index("Model")
print("\nModel comparison:\n", results_df.round(4))
results_df.to_csv(OUTPUT_DIR / "model_comparison.csv")

# 5. Metric comparison chart.
ax = results_df.plot(kind="bar", figsize=(9, 5), ylim=(0, 1), width=0.8)
ax.set_title("Comparison of Classification Models")
ax.set_xlabel("Classification Model")
ax.set_ylabel("Metric Score")
plt.xticks(rotation=20, ha="right")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "model_comparison.png", dpi=160)
plt.close()

# 6. Confusion matrix for each model.
for name, pipeline in trained.items():
    pred = predictions[name][0]
    disp = ConfusionMatrixDisplay.from_predictions(
        y_test, pred, labels=[0, 1],
        display_labels=["No", "Yes"], colorbar=False
    )
    disp.ax_.set_title(f"Confusion Matrix - {name}")
    disp.ax_.set_xlabel("Predicted Label")
    disp.ax_.set_ylabel("Actual Label")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(OUTPUT_DIR / f"confusion_matrix_{safe_name}.png", dpi=160)
    plt.close()

# 7. ROC curves and AUC.
plt.figure(figsize=(7, 5.5))
for name, (pred, prob) in predictions.items():
    fpr, tpr, _ = roc_curve(y_test, prob)
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc(fpr, tpr):.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
plt.title("ROC Curves for Classification Models")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right", fontsize=8)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curves.png", dpi=160)
plt.close()

print("\nBest model by F1-score:", results_df["F1-Score"].idxmax())
print("All plots and model metrics saved in:", OUTPUT_DIR.resolve())
