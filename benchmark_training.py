"""Chronological cloudburst benchmark with leakage-labelled feature sets."""

from __future__ import annotations

import argparse
import importlib.util
import logging
import sys
import warnings
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "cleaned_cloudburst (2).csv"
OUTPUT = ROOT / "outputs"
RANDOM_STATE = 42
TARGET = "cloudburst"
ATMOSPHERIC = ["t2m", "u10", "v10", "sp", "tcwv", "wind_speed"]
RAIN = [
    "rain_mm", "rain_3h", "rain_6h", "rain_peak_3h", "rain_lag1", "rain_lag2", "tp",
]
FEATURE_SETS = {
    "A_full": ["t2m", "u10", "v10", "sp", "tcwv", "wind_speed"],
    "A_lean": ["t2m", "u10", "sp", "tcwv"],
    "A_lean2": ["t2m", "u10", "sp", "tcwv", "wind_speed"],
}
VERBATIM_LIMITATION = (
    "D4 shows that 38 of 47 equal-width rain_3h bins carry a constant cloudburst label, "
    "with a transition band of rain_3h ∈ [0.40, 2.73]. D3 confirms rain_peak_3h alone "
    "reproduces cloudburst at F1=0.931, precision=0.928, recall=0.935. This proves the "
    "label is approximately a threshold on rain_3h. Any model with access to rain_mm, "
    "rain_3h, rain_6h, rain_peak_3h, rain_lag1, rain_lag2, or tp is solving a lookup, "
    "not a prediction. Those configurations are reported for completeness but must not "
    "be used operationally. The honest deployable system is Config A (atmospheric only), "
    "whose realistic ceiling is PR-AUC ≈ 0.15–0.25 and F1(pos) ≈ 0.22–0.38 — modest but "
    "real, driven primarily by tcwv (moisture) and u10 (zonal wind)."
)


def setup_logger() -> logging.Logger:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("cloudburst_benchmark")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def load_data() -> pd.DataFrame:
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")
    frame = pd.read_csv(DATA_PATH)
    needed = {"time", TARGET, *ATMOSPHERIC, *RAIN, "tcwv_3h", "tcwv_6h", "sp_drop_3h", "t2m_grad"}
    missing = needed.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    frame["time"] = pd.to_datetime(frame["time"], errors="raise")
    frame = frame.sort_values("time", kind="stable").reset_index(drop=True)
    if frame[TARGET].isna().any() or not set(frame[TARGET].unique()).issubset({0, 1}):
        raise ValueError("cloudburst must contain only non-missing binary values 0 and 1")
    return frame


def make_model_specs(y_train: pd.Series) -> dict[str, Pipeline]:
    negative, positive = np.bincount(y_train.astype(int), minlength=2)
    scale_pos_weight = negative / max(positive, 1)
    estimators: dict[str, tuple[object, bool]] = {
        "Dummy": (DummyClassifier(strategy="most_frequent"), False),
        "LogReg": (
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
            True,
        ),
        "KNN": (KNeighborsClassifier(n_neighbors=7), True),
        "DecisionTree": (
            DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE),
            False,
        ),
        "RandomForest": (
            RandomForestClassifier(
                n_estimators=300, class_weight="balanced", n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            False,
        ),
        "ExtraTrees": (
            ExtraTreesClassifier(
                n_estimators=300, class_weight="balanced", n_jobs=-1,
                random_state=RANDOM_STATE,
            ),
            False,
        ),
        "GradientBoosting": (GradientBoostingClassifier(random_state=RANDOM_STATE), False),
        "AdaBoost": (AdaBoostClassifier(random_state=RANDOM_STATE), False),
        "GaussianNB": (GaussianNB(), True),
    }
    if importlib.util.find_spec("xgboost"):
        from xgboost import XGBClassifier

        estimators["XGBoost"] = (
            XGBClassifier(
                n_estimators=300, scale_pos_weight=scale_pos_weight,
                eval_metric="aucpr", tree_method="hist", n_jobs=1,
                random_state=RANDOM_STATE,
            ),
            False,
        )
    if importlib.util.find_spec("lightgbm"):
        from lightgbm import LGBMClassifier

        estimators["LightGBM"] = (
            LGBMClassifier(
                n_estimators=300, class_weight="balanced",
                random_state=RANDOM_STATE, verbosity=-1, n_jobs=1,
            ),
            False,
        )
    if importlib.util.find_spec("catboost"):
        from catboost import CatBoostClassifier

        estimators["CatBoost"] = (
            CatBoostClassifier(
                iterations=300, auto_class_weights="Balanced", verbose=0,
                random_seed=RANDOM_STATE, thread_count=1,
            ),
            False,
        )
    expected = {
        "Dummy", "LogReg", "KNN", "DecisionTree", "RandomForest", "ExtraTrees",
        "GradientBoosting", "AdaBoost", "XGBoost", "LightGBM", "CatBoost", "GaussianNB",
    }
    if estimators.keys() != expected:
        raise ImportError(f"Missing requested model dependencies: {sorted(expected - estimators.keys())}")
    specs = {}
    for name, (estimator, scale) in estimators.items():
        steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale:
            steps.append(("scaler", StandardScaler()))
        steps.append(("classifier", estimator))
        specs[name] = Pipeline(steps)
    return specs


def fit_pipeline(
    pipeline: Pipeline, name: str, X: pd.DataFrame, y: pd.Series,
) -> Pipeline:
    fit_params = {}
    if name in {"GradientBoosting", "AdaBoost"}:
        counts = y.value_counts()
        weights = y.map({cls: len(y) / (2 * count) for cls, count in counts.items()})
        fit_params["classifier__sample_weight"] = weights.to_numpy()
    pipeline.fit(X, y, **fit_params)
    return pipeline


def positive_scores(model, X: pd.DataFrame) -> np.ndarray:
    classes = list(model.classes_)
    return model.predict_proba(X)[:, classes.index(1)]


def score_metrics(model, X: pd.DataFrame, y: pd.Series) -> tuple[dict, np.ndarray]:
    scores = positive_scores(model, X)
    predictions = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "precision_pos": float(precision_score(y, predictions, zero_division=0)),
        "recall_pos": float(recall_score(y, predictions, zero_division=0)),
        "f1_pos": float(f1_score(y, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, scores)) if y.nunique() == 2 else np.nan,
        "pr_auc": float(average_precision_score(y, scores)),
        "mcc": float(matthews_corrcoef(y, predictions)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }, scores


def evaluate_cv(
    pipeline: Pipeline, name: str, X: pd.DataFrame, y: pd.Series,
    splitter: TimeSeriesSplit,
) -> dict[str, float]:
    f1_values = []
    ap_values = []
    for train_index, valid_index in splitter.split(X):
        fold_model = clone(pipeline)
        fit_pipeline(fold_model, name, X.iloc[train_index], y.iloc[train_index])
        fold_scores = positive_scores(fold_model, X.iloc[valid_index])
        fold_predictions = (fold_scores >= 0.5).astype(int)
        f1_values.append(f1_score(y.iloc[valid_index], fold_predictions, zero_division=0))
        ap_values.append(average_precision_score(y.iloc[valid_index], fold_scores))
    return {
        "cv_f1_mean": float(np.mean(f1_values)),
        "cv_f1_std": float(np.std(f1_values, ddof=1)),
        "cv_pr_auc_mean": float(np.mean(ap_values)),
        "cv_pr_auc_std": float(np.std(ap_values, ddof=1)),
    }


def run_ablation(X_train: pd.DataFrame, y_train: pd.Series,
                 X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    if not importlib.util.find_spec("xgboost"):
        raise ImportError("XGBoost is required for the specified Config A ablation.")
    from xgboost import XGBClassifier

    variants = [("all_6_features", None)]
    variants.extend((f"drop_{feature}", feature) for feature in ATMOSPHERIC)
    rows = []
    negative, positive = np.bincount(y_train.astype(int), minlength=2)
    for variant, dropped in variants:
        columns = [feature for feature in ATMOSPHERIC if feature != dropped]
        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", XGBClassifier(
                n_estimators=300, scale_pos_weight=negative / max(positive, 1),
                eval_metric="aucpr", tree_method="hist", n_jobs=1,
                random_state=RANDOM_STATE,
            )),
        ])
        model.fit(X_train[columns], y_train)
        rows.append({
            "ablation": variant,
            "dropped_feature": dropped or "",
            "features": ",".join(columns),
            "test_pr_auc": average_precision_score(y_test, positive_scores(model, X_test[columns])),
        })
    result = pd.DataFrame(rows).sort_values("test_pr_auc", ascending=False)
    result.to_csv(OUTPUT / "ablation_A_drop_one_feature.csv", index=False)
    print("\nCONFIG A DROP-ONE-FEATURE ABLATION (XGBoost; chronological holdout PR-AUC)")
    print(result.to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    return result


def run_smote_cv(
    top_names: list[str], specs: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series,
    logger: logging.Logger,
) -> pd.DataFrame:
    if not importlib.util.find_spec("imblearn"):
        raise ImportError("imbalanced-learn is required for the requested SMOTE comparisons.")
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    rows = []
    splitter = TimeSeriesSplit(n_splits=5)
    for name in top_names:
        base = specs[name]
        for train_index, valid_index in splitter.split(X):
            steps = [(key, clone(value)) for key, value in base.steps]
            insert_at = next(i for i, (key, _) in enumerate(steps) if key == "classifier")
            steps.insert(insert_at, ("smote", SMOTE(k_neighbors=5, random_state=RANDOM_STATE)))
            model = ImbPipeline(steps)
            model.fit(X.iloc[train_index], y.iloc[train_index])
            scores = positive_scores(model, X.iloc[valid_index])
            rows.append({
                "model": name,
                "f1_pos": f1_score(y.iloc[valid_index], scores >= 0.5, zero_division=0),
                "pr_auc": average_precision_score(y.iloc[valid_index], scores),
            })
        logger.info("Completed 5-fold TimeSeriesSplit SMOTE CV for %s", name)
    result = pd.DataFrame(rows).groupby("model").agg(
        smote_cv_f1_mean=("f1_pos", "mean"),
        smote_cv_f1_std=("f1_pos", "std"),
        smote_cv_pr_auc_mean=("pr_auc", "mean"),
        smote_cv_pr_auc_std=("pr_auc", "std"),
    ).reset_index()
    result.to_csv(OUTPUT / "smote_cv_A_lean_top2.csv", index=False)
    return result


def parameter_distributions(name: str) -> dict[str, object]:
    from scipy.stats import loguniform, randint

    spaces = {
        "LogReg": {"classifier__C": loguniform(0.001, 1000)},
        "KNN": {"classifier__n_neighbors": randint(3, 22),
                "classifier__weights": ["uniform", "distance"]},
        "DecisionTree": {"classifier__max_depth": [None, 8, 16, 24],
                         "classifier__min_samples_leaf": [1, 2, 5, 10],
                         "classifier__max_features": ["sqrt", 0.7, None]},
        "RandomForest": {"classifier__n_estimators": [100, 200, 300],
                         "classifier__max_depth": [None, 12, 24],
                         "classifier__min_samples_leaf": [1, 2, 5],
                         "classifier__max_features": ["sqrt", 0.7, 1.0, 0.5]},
        "ExtraTrees": {"classifier__n_estimators": [100, 200, 300],
                       "classifier__max_depth": [None, 12, 24],
                       "classifier__min_samples_leaf": [1, 2, 5],
                       "classifier__max_features": ["sqrt", 0.7, 1.0, 0.5]},
        "GradientBoosting": {"classifier__n_estimators": [50, 100, 150],
                             "classifier__learning_rate": [0.03, 0.05, 0.1],
                             "classifier__max_depth": [2, 3],
                             "classifier__min_samples_leaf": [1, 2, 5]},
        "AdaBoost": {"classifier__n_estimators": [50, 100, 200],
                     "classifier__learning_rate": [0.03, 0.1, 0.5, 1.0]},
        "XGBoost": {"classifier__max_depth": [3, 5, 7],
                    "classifier__learning_rate": [0.03, 0.1, 0.2],
                    "classifier__n_estimators": [100, 200, 300],
                    "classifier__min_child_weight": [1, 3, 5]},
        "LightGBM": {"classifier__max_depth": [-1, 5, 10],
                     "classifier__learning_rate": [0.03, 0.1, 0.2],
                     "classifier__n_estimators": [100, 200, 300],
                     "classifier__num_leaves": [15, 31, 63]},
        "CatBoost": {"classifier__depth": [4, 6, 8],
                     "classifier__learning_rate": [0.03, 0.1, 0.2],
                     "classifier__iterations": [100, 200, 300],
                     "classifier__l2_leaf_reg": [1, 3, 5]},
        "GaussianNB": {"classifier__var_smoothing": loguniform(1e-12, 1e-6)},
    }
    return spaces[name]


def tune_top_two(
    names: list[str], specs: dict[str, Pipeline], X: pd.DataFrame, y: pd.Series,
) -> tuple[dict[str, Pipeline], dict[str, dict]]:
    fitted = {}
    summaries = {}
    for name in names:
        search = RandomizedSearchCV(
            estimator=specs[name],
            param_distributions=parameter_distributions(name),
            n_iter=10,
            scoring="average_precision",
            cv=TimeSeriesSplit(n_splits=5),
            random_state=RANDOM_STATE,
            n_jobs=-1,
            pre_dispatch="n_jobs",
            refit=True,
            error_score="raise",
        )
        fit_params = {}
        if name in {"GradientBoosting", "AdaBoost"}:
            counts = y.value_counts()
            weights = y.map({cls: len(y) / (2 * count) for cls, count in counts.items()})
            fit_params["classifier__sample_weight"] = weights.to_numpy()
        search.fit(X, y, **fit_params)
        fitted[name] = search.best_estimator_
        summaries[name] = {
            "best_params": search.best_params_,
            "best_cv_pr_auc_mean": float(search.best_score_),
            "best_cv_pr_auc_std": float(
                search.cv_results_["std_test_score"][search.best_index_]
            ),
        }
    (OUTPUT / "best_params_A_lean.json").write_text(
        pd.Series(summaries).to_json(indent=2), encoding="utf-8"
    )
    return fitted, summaries


def threshold_metrics(
    model, X_test: pd.DataFrame, y_test: pd.Series,
) -> tuple[dict, pd.DataFrame]:
    scores = positive_scores(model, X_test)
    thresholds = np.arange(0.02, 0.981, 0.01)
    rows = []
    for threshold in thresholds:
        pred = scores >= threshold
        rows.append({
            "threshold": float(threshold),
            "f1_pos": float(f1_score(y_test, pred, zero_division=0)),
            "precision_pos": float(precision_score(y_test, pred, zero_division=0)),
            "recall_pos": float(recall_score(y_test, pred, zero_division=0)),
        })
    table = pd.DataFrame(rows)
    best_f1 = table.loc[table["f1_pos"].idxmax()].to_dict()
    eligible = table[table["precision_pos"] >= 0.7]
    best_precision = None if eligible.empty else eligible.sort_values(
        ["recall_pos", "threshold"], ascending=[False, True]
    ).iloc[0].to_dict()
    return {"best_f1": best_f1, "precision_at_least_0_7": best_precision}, table


def validation_thresholds(model, X_val: pd.DataFrame, y_val: pd.Series) -> dict[str, dict]:
    scores = positive_scores(model, X_val)
    rows = []
    for threshold in np.arange(0.02, 0.981, 0.01):
        prediction = scores >= threshold
        rows.append({
            "threshold": float(threshold),
            "precision": float(precision_score(y_val, prediction, zero_division=0)),
            "recall": float(recall_score(y_val, prediction, zero_division=0)),
            "f1": float(f1_score(y_val, prediction, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_val, prediction)),
        })
    table = pd.DataFrame(rows)
    f1_row = table.loc[table["f1"].idxmax()].to_dict()
    balanced_row = table.loc[table["balanced_accuracy"].idxmax()].to_dict()
    eligible = table[table["precision"] >= 0.70]
    recall_row = None if eligible.empty else eligible.sort_values(
        ["recall", "threshold"], ascending=[False, True]
    ).iloc[0].to_dict()
    return {
        "default": {"threshold": 0.50},
        "t_F1": f1_row,
        "t_recall_at_P70": recall_row,
        "t_balanced": balanced_row,
    }


def evaluate_test_thresholds(
    model, X_test: pd.DataFrame, y_test: pd.Series, thresholds: dict[str, dict],
) -> tuple[pd.DataFrame, float]:
    scores = positive_scores(model, X_test)
    rows = []
    for source, selection in thresholds.items():
        if selection is None:
            rows.append({
                "threshold_source": source, "threshold": np.nan,
                "precision": np.nan, "recall": np.nan, "f1": np.nan,
                "balanced_accuracy": np.nan, "tn": np.nan, "fp": np.nan,
                "fn": np.nan, "tp": np.nan,
            })
            continue
        threshold = float(selection["threshold"])
        prediction = scores >= threshold
        tn, fp, fn, tp = confusion_matrix(y_test, prediction, labels=[0, 1]).ravel()
        rows.append({
            "threshold_source": source,
            "threshold": threshold,
            "precision": float(precision_score(y_test, prediction, zero_division=0)),
            "recall": float(recall_score(y_test, prediction, zero_division=0)),
            "f1": float(f1_score(y_test, prediction, zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        })
    return pd.DataFrame(rows), float(average_precision_score(y_test, scores))


def shap_summary(model, X_train: pd.DataFrame, X_test: pd.DataFrame) -> list[tuple[str, float]]:
    import shap

    classifier = model.named_steps["classifier"]
    sample = X_test.sample(n=min(300, len(X_test)), random_state=RANDOM_STATE)
    try:
        if hasattr(classifier, "feature_importances_"):
            explainer = shap.TreeExplainer(classifier)
            values = explainer.shap_values(sample)
        elif isinstance(classifier, LogisticRegression):
            transformed = model[:-1].transform(sample)
            background = model[:-1].transform(X_train.sample(
                n=min(100, len(X_train)), random_state=RANDOM_STATE
            ))
            values = shap.LinearExplainer(classifier, background).shap_values(transformed)
        else:
            background = X_train.sample(n=min(50, len(X_train)), random_state=RANDOM_STATE)
            explainer = shap.Explainer(
                lambda rows: positive_scores(model, pd.DataFrame(rows, columns=X_train.columns)),
                background,
                algorithm="permutation",
            )
            values = explainer(sample, max_evals=2 * X_train.shape[1] + 1).values
    except Exception as exc:
        raise RuntimeError(f"SHAP computation failed for the Config A winner: {exc}") from exc

    values = np.asarray(values)
    if values.ndim == 3:
        values = values[..., 1]
    if values.ndim == 1:
        values = values.reshape(1, -1)
    importances = np.abs(values).mean(axis=0)
    ranked = sorted(zip(X_train.columns, importances), key=lambda item: item[1], reverse=True)
    plt.figure(figsize=(8, 5))
    shap.summary_plot(values, sample, show=False, max_display=min(10, len(ranked)))
    plt.tight_layout()
    plt.savefig(OUTPUT / "shap_A.png", dpi=160, bbox_inches="tight")
    plt.close()
    return [(str(feature), float(value)) for feature, value in ranked]


def lean2_shap_summary(
    model, X_fit: pd.DataFrame, X_test: pd.DataFrame,
) -> list[tuple[str, float]]:
    import shap

    sample = X_test.sample(n=min(300, len(X_test)), random_state=RANDOM_STATE)
    transform = model[:-1]
    background = transform.transform(X_fit.sample(
        n=min(100, len(X_fit)), random_state=RANDOM_STATE
    ))
    transformed_sample = transform.transform(sample)
    classifier = model.named_steps["classifier"]
    try:
        values = shap.LinearExplainer(classifier, background).shap_values(transformed_sample)
    except Exception as exc:
        raise RuntimeError(f"SHAP computation failed for the A_lean2 model: {exc}") from exc
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[..., 1]
    importances = np.abs(values).mean(axis=0)
    ranked = sorted(zip(FEATURE_SETS["A_lean2"], importances),
                    key=lambda item: item[1], reverse=True)
    plt.figure(figsize=(8, 5))
    shap.summary_plot(values, sample, show=False, max_display=len(FEATURE_SETS["A_lean2"]))
    plt.tight_layout()
    plt.savefig(OUTPUT / "shap_A_lean2.png", dpi=160, bbox_inches="tight")
    plt.close()
    return [(str(feature), float(value)) for feature, value in ranked]


def write_lean2_model_card(
    model, validation_cutoffs: dict[str, dict], test_metrics: pd.DataFrame,
    test_pr_auc: float, shap_values: list[tuple[str, float]], frame: pd.DataFrame,
    fit_end: int, train_end: int,
) -> None:
    comparison = {
        name: float(pd.read_csv(OUTPUT / f"results_{name}.csv")
                    .loc[lambda data: data["model"].eq("LogReg"), "pr_auc"].iloc[0])
        for name in ("A_full", "A_lean", "A_lean2")
    }
    test_by_source = test_metrics.set_index("threshold_source")
    cv = pd.read_csv(OUTPUT / "results_A_lean2.csv")
    cv_logreg = cv.loc[cv["model"].eq("LogReg")].iloc[0]
    test_table = [
        "| Threshold source | Threshold | Precision | Recall | F1 | BalAcc | TN/FP/FN/TP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for source in ("default", "t_F1", "t_recall_at_P70", "t_balanced"):
        row = test_by_source.loc[source]
        if pd.isna(row["threshold"]):
            test_table.append(f"| {source} | not found | — | — | — | — | — |")
        else:
            test_table.append(
                f"| {source} | {row['threshold']:.2f} | {row['precision']:.4f} | "
                f"{row['recall']:.4f} | {row['f1']:.4f} | "
                f"{row['balanced_accuracy']:.4f} | "
                f"{int(row['tn'])}/{int(row['fp'])}/{int(row['fn'])}/{int(row['tp'])} |"
            )
    threshold_section = "\n".join(test_table)
    val_lines = []
    for source, values in validation_cutoffs.items():
        if values is None:
            val_lines.append(f"- **{source}:** no validation threshold achieved precision ≥ 0.70.")
        elif source == "default":
            val_lines.append("- **default:** 0.50 (pre-specified; not selected on validation).")
        else:
            val_lines.append(
                f"- **{source}:** {values['threshold']:.2f} "
                f"(validation precision={values.get('precision', np.nan):.4f}, "
                f"recall={values.get('recall', np.nan):.4f}, "
                f"F1={values.get('f1', np.nan):.4f}, "
                f"balanced accuracy={values.get('balanced_accuracy', np.nan):.4f})"
            )
    text = [
        "# Cloudburst model card — Config A_lean2",
        "",
        "**Config A_lean2 is the deployable model** (five features; comparison PR-AUC "
        f"{comparison['A_lean2']:.4f}, highest of the three atmospheric-only feature sets).",
        "",
        "**Config A_lean (four features) is NOT the winner despite earlier guidance**; "
        "LogReg benefits from `wind_speed`, unlike XGBoost, which could reconstruct wind "
        "speed from `u10` and `v10` components. **Config A_full also "
        f"underperforms A_lean2 by a small margin** ({comparison['A_full']:.4f} vs. "
        f"{comparison['A_lean2']:.4f}).",
        "",
        "The earlier F1=0.327 at threshold 0.82 was measured after tuning that threshold "
        "on the test set and is optimistic. It is not used as the deployable estimate here; "
        f"the prior five-fold chronological training CV F1 at threshold 0.50 is "
        f"{cv_logreg['cv_f1_mean']:.4f} ± {cv_logreg['cv_f1_std']:.4f}.",
        "",
        "## Model",
        "",
        "- **Estimator:** `LogisticRegression(class_weight='balanced', C=0.0022310108)`",
        f"- **Features:** {', '.join(FEATURE_SETS['A_lean2'])}",
        "- **Split:** chronological 80% train / 20% test; training split further divided chronologically into 85% fit / 15% validation.",
        f"- **Fit rows:** {fit_end:,}; **validation rows:** {train_end - fit_end:,}; **test rows:** {len(frame) - train_end:,}.",
        f"- **Fit period:** {frame['time'].iloc[0]} to {frame['time'].iloc[fit_end - 1]}",
        f"- **Validation period:** {frame['time'].iloc[fit_end]} to {frame['time'].iloc[train_end - 1]}",
        f"- **Test period:** {frame['time'].iloc[train_end]} to {frame['time'].iloc[-1]}",
        "",
        "## Validation-selected thresholds",
        "",
        *val_lines,
        "",
        "## Test metrics (each threshold applied once)",
        "",
        f"Test PR-AUC from this fitted model: {test_pr_auc:.4f}.",
        "",
        threshold_section,
        "",
        "The model used to generate these metrics was fitted on the 85% fit portion only. "
        "The validation partition was used only for threshold selection; no test metric "
        "participated in choosing thresholds.",
        "",
        "## Top SHAP features",
        "",
        *[f"{i}. {feature}: mean |SHAP| = {importance:.8f}"
          for i, (feature, importance) in enumerate(shap_values, 1)],
        "",
        "## Feature-set comparison",
        "",
        "| Feature set | LogReg comparison PR-AUC |",
        "|---|---:|",
        *[f"| {name} | {value:.4f} |" for name, value in comparison.items()],
        "",
        "## Rain-label leakage diagnostic",
        "",
        "D4 found 38 of 47 equal-width `rain_3h` bins with a constant label and a "
        "transition band around 0.40–2.73. D3 found that `rain_peak_3h` alone achieved "
        "F1=0.931, precision=0.928, and recall=0.935. Rain-derived features are therefore "
        "excluded from this atmospheric-only deployable model.",
        "",
        "The rain-derived reference configurations were not retrained for this corrected model.",
        "",
        "## Operating recommendation",
        "",
        "For nowcasting where missed events cost more than false alarms, use the validation "
        "F1-maximizing threshold. Consult the corresponding test row above for the observed "
        "recall, precision, and false-alarm burden. A validation-tuned operating point is "
        "still subject to temporal-regime drift and should be prospectively monitored.",
        (
            f"- At the validation-selected F1 threshold, test precision={test_by_source.loc['t_F1', 'precision']:.4f}, "
            f"or approximately {1 / test_by_source.loc['t_F1', 'precision'] - 1:.2f} false-alarm hours "
            "per true hit."
            if test_by_source.loc["t_F1", "precision"] > 0 else
            "- At the validation-selected F1 threshold, precision is zero, so false alarms per true hit are undefined."
        ),
        "",
    ]
    (OUTPUT / "model_card_A_lean2.md").write_text("\n".join(text), encoding="utf-8")


def finalize_a_lean2() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    logger = setup_logger()
    frame = load_data()
    features = FEATURE_SETS["A_lean2"]
    y = frame[TARGET].astype(int)
    train_end = int(len(frame) * 0.8)
    fit_end = int(train_end * 0.85)
    X_fit = frame.loc[:fit_end - 1, features]
    y_fit = y.iloc[:fit_end]
    X_val = frame.loc[fit_end:train_end - 1, features]
    y_val = y.iloc[fit_end:train_end]
    X_test = frame.loc[train_end:, features]
    y_test = y.iloc[train_end:]
    logger.info(
        "A_lean2 chronological split: fit=%d, validation=%d, test=%d",
        len(X_fit), len(X_val), len(X_test),
    )
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            class_weight="balanced", C=0.0022310108,
            max_iter=2000, random_state=RANDOM_STATE,
        )),
    ])
    model.fit(X_fit, y_fit)
    thresholds = validation_thresholds(model, X_val, y_val)
    test_metrics, test_pr_auc = evaluate_test_thresholds(model, X_test, y_test, thresholds)
    test_metrics.to_csv(OUTPUT / "threshold_metrics_A_lean2.csv", index=False)
    validation_report = pd.DataFrame([
        {"threshold_source": source, **(selection or {})}
        for source, selection in thresholds.items()
    ])
    validation_report.to_csv(OUTPUT / "threshold_selection_A_lean2.csv", index=False)
    joblib.dump(model, OUTPUT / "best_model_A_lean2.joblib")
    shap_values = lean2_shap_summary(model, X_fit, X_test)
    write_lean2_model_card(
        model, thresholds, test_metrics, test_pr_auc, shap_values,
        frame, fit_end, train_end,
    )
    (OUTPUT / "model_card_A.md").unlink(missing_ok=True)

    print("=== VALIDATION-SELECTED THRESHOLDS ===")
    for source, selection in thresholds.items():
        print(f"{source}: {selection}")
    print("\n=== TEST METRICS (thresholds applied once) ===")
    print(test_metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nTest PR-AUC: {test_pr_auc:.4f}")
    print("\n=== TOP SHAP FEATURES (A_lean2) ===")
    for rank, (feature, importance) in enumerate(shap_values, 1):
        print(f"{rank}. {feature}: mean |SHAP|={importance:.8f}")
    logger.info("Saved corrected A_lean2 model, test threshold metrics, SHAP plot, and model card.")


def metric_table(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.sort_values("pr_auc", ascending=False).reset_index(drop=True)
    ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
    return ranked


def display_table(frame: pd.DataFrame) -> str:
    rows = []
    for _, row in frame.iterrows():
        rows.append(
            f"| {int(row['rank'])} | {row['model']} | {row['pr_auc']:.4f} | "
            f"{row['f1_pos']:.4f} | {row['recall_pos']:.4f} | {row['roc_auc']:.4f} | "
            f"{row['balanced_accuracy']:.4f} | {row['accuracy']:.4f} | "
            f"{int(row['tn'])}/{int(row['fp'])}/{int(row['fn'])}/{int(row['tp'])} |"
        )
    return "\n".join([
        "| Rank | Model | PR-AUC | F1(pos) | Recall(pos) | ROC-AUC | BalAcc | Acc | TN/FP/FN/TP |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        *rows,
    ])


def save_pr_auc_plot(results: pd.DataFrame) -> None:
    ordered_models = results.sort_values("pr_auc", ascending=False)["model"].tolist()
    pivot = results.pivot(index="model", columns="feature_set", values="pr_auc")
    pivot = pivot.reindex(ordered_models)
    ax = pivot.plot(kind="bar", figsize=(14, 7), color=["#4472C4", "#70AD47", "#ED7D31"])
    ax.set(title="Chronological holdout PR-AUC by model and atmospheric feature set",
           xlabel="Model", ylabel="PR-AUC")
    ax.tick_params(axis="x", labelrotation=35)
    ax.legend(title="Feature set")
    plt.tight_layout()
    plt.savefig(OUTPUT / "pr_auc_models_A.png", dpi=160, bbox_inches="tight")
    plt.close()


def write_model_card(
    winner: pd.Series, model, threshold_info: dict, shap_values: list[tuple[str, float]],
    ablation: pd.DataFrame, tuned: dict[str, dict], frame: pd.DataFrame,
    feature_set_pr_auc: dict[str, float],
) -> None:
    f1_threshold = threshold_info["best_f1"]
    precision_threshold = threshold_info["precision_at_least_0_7"]
    text = [
        "# Config A_lean model card — FINAL DEPLOYABLE MODEL",
        "",
        "## Model and intended use",
        "",
        f"- **Model:** {winner['model']}",
        f"- **Feature set:** {winner['feature_set']}",
        f"- **Features:** {', '.join(FEATURE_SETS[str(winner['feature_set'])])}",
        "- **Use:** atmospheric-only cloudburst risk ranking. Rain-derived features are excluded.",
        "- **Split:** sorted chronological 80/20 holdout; all preprocessing and fitting use the train portion only.",
        f"- **Train interval:** {frame['time'].iloc[0]} to {frame['time'].iloc[int(len(frame) * 0.8) - 1]}",
        f"- **Test interval:** {frame['time'].iloc[int(len(frame) * 0.8)]} to {frame['time'].iloc[-1]}",
        "",
        "## Atmospheric feature-set comparison",
        "",
        "| Feature set | Holdout PR-AUC |",
        "|---|---:|",
        *[f"| {name} | {value:.6f} |" for name, value in feature_set_pr_auc.items()],
        "- A_lean is promoted as requested; A_lean2 had the highest observed holdout PR-AUC. This single split does not establish that it will generalize better.",
        "",
        "## Holdout performance",
        "",
        f"- **PR-AUC:** {winner['pr_auc']:.6f}; **ROC-AUC:** {winner['roc_auc']:.6f}",
        f"- **F1(pos):** {winner['f1_pos']:.6f}; **precision:** {winner['precision_pos']:.6f}; **recall:** {winner['recall_pos']:.6f}",
        f"- **Accuracy:** {winner['accuracy']:.6f}; **balanced accuracy:** {winner['balanced_accuracy']:.6f}; **MCC:** {winner['mcc']:.6f}",
        f"- **Confusion matrix:** TN={int(winner['tn'])}, FP={int(winner['fp'])}, FN={int(winner['fn'])}, TP={int(winner['tp'])}",
        f"- **5-fold TimeSeriesSplit train CV:** PR-AUC={winner['cv_pr_auc_mean']:.6f} ± {winner['cv_pr_auc_std']:.6f}; F1={winner['cv_f1_mean']:.6f} ± {winner['cv_f1_std']:.6f}",
        "- Holdout metrics and confusion matrix above use threshold 0.50; the threshold-selected scores below are a separate evaluation on the same holdout.",
        "",
        "## Thresholds",
        "",
        f"- **F1-maximizing threshold:** {f1_threshold['threshold']:.2f} (F1={f1_threshold['f1_pos']:.4f}, precision={f1_threshold['precision_pos']:.4f}, recall={f1_threshold['recall_pos']:.4f})",
        (
            "- **Precision >= 0.7 threshold:** not found on the tested threshold grid."
            if precision_threshold is None else
            f"- **Precision >= 0.7 threshold (highest recall):** {precision_threshold['threshold']:.2f} (precision={precision_threshold['precision_pos']:.4f}, recall={precision_threshold['recall_pos']:.4f})"
        ),
        "",
        "## Top SHAP features",
        "",
        *[f"{i}. {feature}: mean |SHAP| = {importance:.8f}"
          for i, (feature, importance) in enumerate(shap_values[:10], 1)],
        "",
        "## Config A drop-one-feature ablation",
        "",
        "| Ablation | Dropped feature | Holdout PR-AUC |",
        "|---|---|---:|",
        *[
            f"| {row.ablation} | {row.dropped_feature or 'none'} | {row.test_pr_auc:.6f} |"
            for row in ablation.itertuples(index=False)
        ],
        "",
        "## Tuning",
        "",
        f"- **Selected model parameters:** `{model.get_params()['classifier']}`",
        f"- **Top-2 A_lean randomized search summary (10 iterations each):** `{tuned}`",
        "",
        "## Leakage finding and operational restriction",
        "",
        VERBATIM_LIMITATION,
        "",
        "## Limitations",
        "",
        "The event base rate is low (4.08% overall), and a chronological split can have substantial regime shift. Threshold metrics are evaluated on the held-out period and are therefore calibration diagnostics, not an independent deployment guarantee. This dataset alone does not establish causal atmospheric precursors or prove forecast lead time; deployment requires timestamp-aware availability checks and prospective validation.",
        "",
    ]
    (OUTPUT / "model_card_A.md").write_text("\n".join(text), encoding="utf-8")


def write_readme() -> None:
    text = """# Cloudburst benchmark — Config A

Only scientifically valid atmospheric feature sets are trained:

- `A_full`: `t2m`, `u10`, `v10`, `sp`, `tcwv`, `wind_speed`
- `A_lean`: `t2m`, `u10`, `sp`, `tcwv`
- `A_lean2`: `t2m`, `u10`, `sp`, `tcwv`, `wind_speed`

Leaky reference configurations B–D are deliberately skipped. SVC and MLP are
also omitted for runtime and tabular-model considerations.

Run the three Config A feature sets with:

```powershell
python run_pipeline.py --config A
```

The comparison run uses a chronological 80/20 split, five-fold TimeSeriesSplit
CV, and top-two-only SMOTE/tuning on A_lean. Outputs include
`results_A_full.csv`, `results_A_lean.csv`, and `results_A_lean2.csv`.

Refit the corrected deployable model and select thresholds without test leakage:

```powershell
python run_pipeline.py --finalize-a-lean2
```

This uses an 85/15 chronological fit/validation split within the training period,
then evaluates each selected threshold once on the final test period.
It saves `best_model_A_lean2.joblib`, `threshold_metrics_A_lean2.csv`,
`shap_A_lean2.png`, and `model_card_A_lean2.md`.

"""
    (OUTPUT / "README_A.md").write_text(text, encoding="utf-8")


def run() -> None:
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    logger = setup_logger()
    frame = load_data()
    y = frame[TARGET].astype(int)
    cut = int(len(frame) * 0.8)
    y_train, y_test = y.iloc[:cut], y.iloc[cut:]
    split_time = frame["time"].iloc[cut]
    logger.info(
        "Rows=%d, positives=%d (%.4f%%); chronological train=%d / test=%d; test starts %s",
        len(frame), int(y.sum()), y.mean() * 100, cut, len(frame) - cut, split_time,
    )
    logger.info(
        "Requested models: Dummy, LogReg, KNN, DecisionTree, RandomForest, ExtraTrees, "
        "GradientBoosting, AdaBoost, XGBoost, LightGBM, CatBoost, GaussianNB"
    )

    ablation_path = OUTPUT / "ablation_A_drop_one_feature.csv"
    if ablation_path.is_file():
        ablation = pd.read_csv(ablation_path)
        if len(ablation) != len(ATMOSPHERIC) + 1:
            ablation = None
    else:
        ablation = None
    if ablation is None:
        ablation = run_ablation(
            frame.loc[:cut - 1, ATMOSPHERIC], y_train,
            frame.loc[cut:, ATMOSPHERIC], y_test,
        )
    else:
        print("\nCONFIG A DROP-ONE-FEATURE ABLATION (reused completed XGBoost holdout run)")
        print(ablation.to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    all_results = []
    lean_fitted: dict[str, Pipeline] = {}
    tuning_summary: dict[str, dict] = {}
    lean_smote: pd.DataFrame | None = None
    lean_train = frame.loc[:cut - 1, FEATURE_SETS["A_lean"]]
    lean_test = frame.loc[cut:, FEATURE_SETS["A_lean"]]

    for feature_set, features in FEATURE_SETS.items():
        X_train = frame.loc[:cut - 1, features]
        X_test = frame.loc[cut:, features]
        logger.info("Feature set %s — SCIENTIFICALLY VALID", feature_set)
        specs = make_model_specs(y_train)
        records = []
        fitted = {}
        splitter = TimeSeriesSplit(n_splits=5)
        for name, spec in specs.items():
            logger.info("%s: CV and fit %s", feature_set, name)
            cv = evaluate_cv(spec, name, X_train, y_train, splitter)
            model = fit_pipeline(clone(spec), name, X_train, y_train)
            metrics, _ = score_metrics(model, X_test, y_test)
            records.append({
                "model": name, "feature_set": feature_set,
                **metrics, **cv,
            })
            fitted[name] = model
        results = pd.DataFrame(records)

        if feature_set == "A_lean":
            top_two = results.sort_values("cv_pr_auc_mean", ascending=False).head(2)["model"].tolist()
            logger.info("A_lean top-2 by train CV PR-AUC for tuning/SMOTE: %s", top_two)
            lean_smote = run_smote_cv(top_two, specs, X_train, y_train, logger)
            lean_smote.to_csv(OUTPUT / "smote_cv_A_lean_top2.csv", index=False)
            tuned_estimators, tuning_summary = tune_top_two(top_two, specs, X_train, y_train)
            for name in top_two:
                tuned_model = tuned_estimators[name]
                tuned_cv = evaluate_cv(tuned_model, name, X_train, y_train, splitter)
                tuned_metrics, _ = score_metrics(tuned_model, X_test, y_test)
                results.loc[results["model"].eq(name), list(tuned_metrics)] = list(tuned_metrics.values())
                results.loc[results["model"].eq(name), list(tuned_cv)] = list(tuned_cv.values())
                fitted[name] = tuned_model
                results.loc[results["model"].eq(name), "tuned"] = True
            results["tuned"] = results["tuned"].fillna(False).astype(bool)
            results = results.merge(lean_smote, on="model", how="left")
            lean_fitted = fitted
        else:
            results["tuned"] = False

        results = metric_table(results)
        results.to_csv(OUTPUT / f"results_{feature_set}.csv", index=False)
        all_results.append(results)
        print(f"\nCONFIG {feature_set} — SCIENTIFICALLY VALID")
        print(display_table(results))

    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results.to_csv(OUTPUT / "results_A.csv", index=False)
    save_pr_auc_plot(combined_results)
    lean_results = next(
        result for result in all_results if result["feature_set"].eq("A_lean").all()
    )
    winner = lean_results.iloc[0]
    winner_name = str(winner["model"])
    winner_model = lean_fitted[winner_name]
    threshold_info, threshold_table = threshold_metrics(winner_model, lean_test, y_test)
    threshold_table.to_csv(OUTPUT / "thresholds_A_lean.csv", index=False)
    joblib.dump(winner_model, OUTPUT / "best_model_A.joblib")
    joblib.dump(winner_model, OUTPUT / "best_model_A_lean.joblib")
    shap_values = shap_summary(winner_model, lean_train, lean_test)
    write_model_card(
        winner, winner_model, threshold_info, shap_values, ablation,
        tuning_summary, frame,
        {
            str(result.iloc[0]["feature_set"]): float(result.iloc[0]["pr_auc"])
            for result in all_results
        },
    )
    write_readme()
    print("\n=== FINAL DEPLOYABLE MODEL ===")
    precision_threshold = threshold_info["precision_at_least_0_7"]
    precision_threshold_text = (
        "none found" if precision_threshold is None
        else f"{precision_threshold['threshold']:.2f}"
    )
    print(
        f"FINAL DEPLOYABLE MODEL\nConfig: A_lean\nModel: {winner_name}\n"
        f"Threshold: {threshold_info['best_f1']['threshold']:.2f} "
        f"(F1-maximizing; precision>=0.7 threshold={precision_threshold_text})\n"
        f"Metrics: PR-AUC={winner['pr_auc']:.4f}, F1={winner['f1_pos']:.4f}, "
        f"Recall={winner['recall_pos']:.4f}, Precision={winner['precision_pos']:.4f}, "
        f"BalAcc={winner['balanced_accuracy']:.4f}\n"
        f"Confusion matrix: TN={int(winner['tn'])}, FP={int(winner['fp'])}, "
        f"FN={int(winner['fn'])}, TP={int(winner['tp'])}\n"
        f"CV (TimeSeriesSplit 5-fold): PR-AUC = {winner['cv_pr_auc_mean']:.4f} ± "
        f"{winner['cv_pr_auc_std']:.4f}, F1 = {winner['cv_f1_mean']:.4f} ± "
        f"{winner['cv_f1_std']:.4f}"
    )
    print("\n=== TOP-10 SHAP FEATURES (A_lean winner) ===")
    for rank, (feature, value) in enumerate(shap_values[:10], 1):
        print(f"{rank}. {feature}: mean |SHAP|={value:.8f}")
    print("\n=== HONEST INTERPRETATION ===")
    print(
        f"A_lean holdout F1 is {winner['f1_pos']:.3f}; "
        f"{'it is' if 0.20 <= winner['f1_pos'] <= 0.40 else 'it is not'} in the "
        "requested 0.20–0.40 range. Compare PR-AUC against the 4.08% event base rate "
        "and CV stability before treating this as real predictive signal. "
        "Configs B–D were skipped as reference tautologies. "
        "The requested interpretation paragraph appears verbatim in model_card_A.md."
    )
    print("\n=== REPRODUCE ===\npython run_pipeline.py --config A")
    logger.info("All requested Config A feature-set files are saved under %s", OUTPUT)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the chronological cloudburst benchmark.")
    parser.add_argument("--config", choices=["A"], default="A",
                        help="Run the three scientifically valid Config A feature sets.")
    parser.add_argument(
        "--finalize-a-lean2", action="store_true",
        help="Refit A_lean2 and select thresholds on train validation before test evaluation.",
    )
    args = parser.parse_args()
    if args.finalize_a_lean2:
        finalize_a_lean2()
        return
    run()


if __name__ == "__main__":
    main()
