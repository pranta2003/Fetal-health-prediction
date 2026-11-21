"""
Stream 3 - Clinical Risk Model
--------------------------------
Trains an XGBoost model on maternal clinical risk factors from the combined
CDC Fetal Death + Live Births dataset (built via the Drive pipeline scripts)
to predict pregnancy outcome (0 = live birth, 1 = stillbirth).

HOW TO USE:
1. Update DATA_PATH below to point at clinical_training_data.csv
   - In Colab:   /content/drive/MyDrive/fetal-health-project/data/processed/clinical_training_data.csv
   - Locally:    data/processed/clinical_training_data.csv
2. Run: python src/stream3_clinical/train_clinical_model.py

The real column names (MAGER, COMBGEST, RF_GDIAB, etc.) are already the
actual CDC names — no renaming/mapping step is needed anymore since the
dataset-build scripts already saved the file with real CDC column names.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, roc_auc_score
import xgboost as xgb
import os

RANDOM_SEED = 42
DATA_PATH = "data/processed/clinical_training_data.csv"
MODEL_OUT_PATH = "models/stream3_clinical_xgb.json"

TARGET_COLUMN = "OUTCOME"

# Numeric clinical fields — imputed with median
NUMERIC_FEATURES = [
    "MAGER",       # maternal age
    "BMI",         # maternal BMI
    "COMBGEST",    # combined gestational age (weeks)
    "PRECARE",     # month prenatal care began
    "PRIORDEAD",   # prior fetal deaths
    "PRIORLIVE",   # prior live births
    "MEDUC",       # mother's education level
]

# Categorical / coded flag fields — imputed with most-frequent value, then one-hot encoded
CATEGORICAL_FEATURES = [
    "RF_GDIAB",    # gestational diabetes
    "RF_GHYPE",    # gestational hypertension
    "RF_EHYPE",    # chronic hypertension
    "RF_CESAR",    # previous cesarean
    "RF_ARTEC",    # assisted reproductive technology
    "RF_INFTR",    # infertility treatment
    "RF_FEDRG",    # fertility-enhancing drugs
    "CIG_REC",     # smoking recode
    "WIC",         # WIC program participation
    "DPLURAL",     # plurality (single/twin/triplet+)
    "SEX",         # fetal sex
    "DMETH_REC",   # delivery method
    "MRACE6",      # maternal race (6-category recode)
    "MBSTATE_REC", # mother's residence status
]

# NOTE: DBWT (birth weight) is intentionally excluded — for fetal death
# records it behaves very differently from live births and risks leaking
# outcome information directly into the model rather than genuine risk signal.


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Could not find {path}. Update DATA_PATH to point at your "
            f"clinical_training_data.csv (built via the Drive pipeline scripts)."
        )
    df = pd.read_csv(path)
    print(f"Loaded combined clinical dataset: {df.shape[0]} rows, {df.shape[1]} columns")

    missing_needed = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET_COLUMN]
                       if c not in df.columns]
    if missing_needed:
        raise KeyError(
            f"Expected column(s) not found in the dataset: {missing_needed}. "
            f"Check clinical_training_data.csv actually contains these fields."
        )
    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(transformers=[
        ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ("categorical", Pipeline(steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_FEATURES),
    ])

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=RANDOM_SEED,
    )

    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def train(df: pd.DataFrame):
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    preds = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n--- Stream 3 Clinical Risk Model: Evaluation ---")
    print(classification_report(y_test, preds, target_names=["Healthy/Live Birth", "Stillbirth"]))
    print(f"AUC-ROC: {roc_auc_score(y_test, proba):.4f}")

    return pipeline


if __name__ == "__main__":
    df = load_data()
    pipeline = train(df)

    os.makedirs(os.path.dirname(MODEL_OUT_PATH), exist_ok=True)
    pipeline.named_steps["model"].save_model(MODEL_OUT_PATH)
    print(f"\nModel saved to {MODEL_OUT_PATH}")
