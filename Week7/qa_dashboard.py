#!/usr/bin/env python3
"""
QA Dashboard & Defect Prediction
- Load build/test metrics
- Clean data and run EDA
- Train linear regression to predict defects_reported
- Generate charts and save predictions
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split

# -----------------------------
# Config
# -----------------------------
DATA_PATH = "builds.csv"
OUTPUT_DIR = "outputs"
SEED = 42

CHART_STYLE = {
    "palette": "deep",
    "figsize": (14, 9),
    "dpi": 120,
}

FEATURE_COLS = [
    "lines_changed",
    "files_changed",
    "cyclomatic_complexity",
    "coverage",
    "pass_rate",
    "failures",
]

TARGET_COL = "defects_reported"

# -----------------------------
# Utilities
# -----------------------------
def ensure_output_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

def load_data(path: str) -> pd.DataFrame:
    if not os.path.isfile(path):
        print(f"ERROR: Data file not found: {path}")
        sys.exit(1)
    df = pd.read_csv(path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    # Basic validations
    required_cols = ["build_id", "date", "environment"] + FEATURE_COLS + [TARGET_COL]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Convert types
    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # Handle missing values
    # Numeric columns: fill with median; categorical: fill with mode
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mode().iloc[0])

    # Clip rates to [0, 1] for sanity
    df["coverage"] = df["coverage"].clip(0, 1)
    df["pass_rate"] = df["pass_rate"].clip(0, 1)

    # Ensure non-negative numeric values
    for c in ["lines_changed", "files_changed", "cyclomatic_complexity", "test_count", "failures", "avg_runtime_sec", TARGET_COL]:
        if c in df.columns:
            df[c] = df[c].clip(lower=0)

    return df

def print_eda(df: pd.DataFrame):
    print("\n=== Data Snapshot ===")
    print(df.head(5))
    print("\n=== Info ===")
    print(df.info())
    print("\n=== Describe (numeric) ===")
    print(df.describe())

    print("\n=== Null counts ===")
    print(df.isnull().sum())

    # Simple environment breakdown
    env_summary = df.groupby("environment")["failures"].agg(["count", "mean", "sum"]).reset_index()
    print("\n=== Failures by environment ===")
    print(env_summary)

def train_regression(df: pd.DataFrame):
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=max(0.3, min(0.5, 0.3)), random_state=SEED
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    coef_map = dict(zip(FEATURE_COLS, model.coef_))
    print("\n=== Linear Regression Results ===")
    print(f"R^2: {r2:.3f}")
    print(f"MAE: {mae:.3f}")
    print("Coefficients:")
    for k, v in coef_map.items():
        print(f"  {k:22s}: {v:+.4f}")
    print(f"Intercept: {model.intercept_:+.4f}")

    return model, r2, mae

def save_predictions(df: pd.DataFrame, model, out_path: str):
    preds = model.predict(df[FEATURE_COLS].values)
    out = df[["build_id", "date", "environment"]].copy()
    out["predicted_defects"] = np.round(preds, 2)
    out["actual_defects"] = df[TARGET_COL]
    out["error"] = np.round(out["predicted_defects"] - out["actual_defects"], 2)
    out.to_csv(out_path, index=False)
    print(f"\nSaved predictions -> {out_path}")

def make_dashboard(df: pd.DataFrame, model):
    sns.set_theme(style="whitegrid", palette=CHART_STYLE["palette"])
    fig = plt.figure(figsize=CHART_STYLE["figsize"], dpi=CHART_STYLE["dpi"])
    fig.suptitle("QA Test Quality & Defect Prediction Dashboard", fontsize=16, y=0.98)

    # 1) Line plot: pass_rate over time by environment
    ax1 = plt.subplot(2, 3, 1)
    for env, dsub in df.sort_values("date").groupby("environment"):
        ax1.plot(dsub["date"], dsub["pass_rate"], marker="o", label=env)
    ax1.set_title("Pass rate over builds")
    ax1.set_xlabel("Date")
    ax1.set_ylabel("Pass rate")
    ax1.legend(fontsize=8, loc="lower right")

    # 2) Bar chart: total failures by environment
    ax2 = plt.subplot(2, 3, 2)
    env_fail = df.groupby("environment")["failures"].sum().sort_values(ascending=False)
    env_fail.plot(kind="bar", ax=ax2, color=sns.color_palette()[:len(env_fail)])
    ax2.set_title("Failures by environment")
    ax2.set_xlabel("Environment")
    ax2.set_ylabel("Total failures")

    # 3) Histogram: avg runtime distribution
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(df["avg_runtime_sec"], bins=8, color="#4C72B0", edgecolor="white")
    ax3.set_title("Avg runtime distribution")
    ax3.set_xlabel("Runtime (sec)")
    ax3.set_ylabel("Frequency")

    # 4) Scatter + regression: coverage vs defects
    ax4 = plt.subplot(2, 3, 4)
    sns.scatterplot(data=df, x="coverage", y="defects_reported", hue="environment", ax=ax4, s=60)
    # Regression line using model (fix other features to median)
    med = df[FEATURE_COLS].median()
    cov_range = np.linspace(df["coverage"].min(), df["coverage"].max(), 50)
    X_line = []
    for c in cov_range:
        row = [
            med["lines_changed"],
            med["files_changed"],
            med["cyclomatic_complexity"],
            c,
            med["pass_rate"],
            med["failures"],
        ]
        X_line.append(row)
    y_line = model.predict(np.array(X_line))
    ax4.plot(cov_range, y_line, color="black", linestyle="--", label="Regression (coverage axis)")
    ax4.set_title("Coverage vs Defects (with regression trace)")
    ax4.set_xlabel("Coverage")
    ax4.set_ylabel("Defects reported")
    ax4.legend(fontsize=8, loc="best")

    # 5) Boxplot: failures distribution by environment
    ax5 = plt.subplot(2, 3, 5)
    sns.boxplot(data=df, x="environment", y="failures", ax=ax5)
    ax5.set_title("Failures distribution by environment")
    ax5.set_xlabel("Environment")
    ax5.set_ylabel("Failures")

    # 6) Feature importance (coefficients)
    ax6 = plt.subplot(2, 3, 6)
    model_feats = dict(zip(FEATURE_COLS, model.coef_))
    coef_ser = pd.Series(model_feats).sort_values()
    coef_ser.plot(kind="barh", ax=ax6, color="#55A868")
    ax6.set_title("Linear regression coefficients")
    ax6.set_xlabel("Weight")
    ax6.set_ylabel("Feature")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig

def main():
    ensure_output_dir(OUTPUT_DIR)
    df = load_data(DATA_PATH)
    df = clean_data(df)

    print_eda(df)

    model, r2, mae = train_regression(df)
    predictions_path = os.path.join(OUTPUT_DIR, "predictions.csv")
    save_predictions(df, model, predictions_path)

    fig = make_dashboard(df, model)
    dashboard_path = os.path.join(OUTPUT_DIR, "qa_dashboard.png")
    fig.savefig(dashboard_path)
    print(f"Saved dashboard -> {dashboard_path}")

    print("\nDone.")

if __name__ == "__main__":
    main()
