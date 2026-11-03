"""Pipeline creation and model fitting for Stream 3 clinical XGBoost risk model."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import xgboost as xgb


def make_pipe(num, cat, n_estimators=300, random_state=42):
    """Build preprocessing and XGBClassifier pipeline verbatim from notebook cell 5."""
    parts = [("n", SimpleImputer(strategy="median"), num)]
    if cat:
        parts.append(
            (
                "c",
                Pipeline(
                    [
                        ("i", SimpleImputer(strategy="most_frequent")),
                        ("o", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat,
            )
        )
    model = xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=random_state,
        tree_method="hist",
        n_jobs=-1,
    )
    return Pipeline([("pre", ColumnTransformer(parts)), ("m", model)])


def fit_eval(d, num, cat, mode="random", n_estimators=300, random_state=42):
    """Fit pipeline and evaluate risk predictions.

    mode='random': stratified 80/20 train/test split.
    mode='temporal': train on YEAR <= 2021, test on YEAR >= 2022.
    """
    if mode == "temporal":
        tr, te = d[d.YEAR <= 2021], d[d.YEAR >= 2022]
    else:
        tr, te = train_test_split(
            d, test_size=0.2, random_state=random_state, stratify=d.OUTCOME
        )
    pipe = make_pipe(
        num, cat, n_estimators=n_estimators, random_state=random_state
    ).fit(tr[num + cat], tr.OUTCOME)
    p = pipe.predict_proba(te[num + cat])[:, 1]
    return pipe, tr, te, p
