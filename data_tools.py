"""
AI Data Scientist Agent - data_tools.py

Core utilities for dataset inspection, preprocessing, model training,
evaluation, and prediction.

Important design choices:
- Training uses the original CSV, not the globally imputed preview file.
- Imputers, encoders, and scalers are fitted inside sklearn Pipelines
  using training folds only.
- Exact duplicate rows are removed before the train/test split by default
  to reduce the risk of identical records appearing in both sets.
- Identifier columns are excluded from model features.
- Known student-grade leakage columns are excluded for the relevant targets.
- The final model is selected using cross-validation on the training set
  when possible; the test set is reserved for final evaluation.
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
)
from sklearn.model_selection import (
    KFold,
    StratifiedKFold,
    cross_val_score,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


MODEL_FILE = "model_bundle.joblib"
METRICS_FILE = "model_metrics.json"
PREDICTIONS_FILE = "test_predictions.csv"
PROCESSED_FILE = "processed_data.csv"
NEW_PREDICTIONS_FILE = "new_data_predictions.csv"

RANDOM_STATE = 42

ID_COLUMNS = {
    "id",
    "student_id",
    "record_id",
    "row_id",
    "index",
}

SUBJECT_SCORE_COLUMNS = {
    "math_score",
    "science_score",
    "english_score",
}


# ============================================================
# GENERAL DATASET UTILITIES
# ============================================================

def read_csv(file_path):
    """Read a CSV and raise a clear error if it cannot be loaded."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        raise ValueError(f"Could not read CSV file: {exc}") from exc

    if len(df.columns) == 0:
        raise ValueError("The CSV contains no columns.")

    # Duplicate column names make feature selection ambiguous.
    if df.columns.duplicated().any():
        duplicates = df.columns[df.columns.duplicated()].tolist()
        raise ValueError(
            f"The CSV contains duplicate column names: {duplicates}"
        )

    return df


def inspect_dataset(file_path):
    """Return basic dataset information and data-quality warnings."""
    df = read_csv(file_path)

    repeated_id_columns = {}
    for column in df.columns:
        if (
            str(column).strip().lower() in ID_COLUMNS
            or str(column).strip().lower().endswith("_id")
        ):
            duplicated_count = int(df[column].duplicated(keep=False).sum())
            if duplicated_count:
                repeated_id_columns[str(column)] = duplicated_count

    return {
        "file": str(file_path),
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": df.columns.tolist(),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_cells": int(df.isna().sum().sum()),
        "missing_values_by_column": {
            str(column): int(count)
            for column, count in df.isna().sum().items()
        },
        "repeated_identifier_rows": repeated_id_columns,
        "memory_usage_kb": round(
            float(df.memory_usage(deep=True).sum() / 1024), 2
        ),
    }


def check_missing_values(file_path):
    """Return missing-value counts by column."""
    df = read_csv(file_path)
    return {
        str(column): int(count)
        for column, count in df.isna().sum().items()
    }


def detect_data_types(file_path):
    """Return pandas data types by column."""
    df = read_csv(file_path)
    return {
        str(column): str(dtype)
        for column, dtype in df.dtypes.items()
    }


def summary_statistics(file_path):
    """Return summary statistics in a JSON-friendly format."""
    df = read_csv(file_path)
    summary = df.describe(include="all")
    summary = summary.astype(object).where(pd.notna(summary), None)
    return summary.to_dict()


def preprocess_data(file_path, output_file=PROCESSED_FILE):
    df = read_csv(file_path)

    if df.empty:
        raise ValueError("The CSV contains no rows.")

    for column in df.columns:
        if pd.api.types.is_numeric_dtype(df[column]):
            median = df[column].median()
            fill_value = 0 if pd.isna(median) else median
            df[column] = df[column].fillna(fill_value)
        else:
            mode = df[column].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            df[column] = df[column].fillna(fill_value)

    df.to_csv(output_file, index=False)

    return {
        "message": "Dataset preprocessing completed.",
        "output_file": output_file,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "missing_values_after": {
            str(column): int(count)
            for column, count in df.isna().sum().items()
        },
        "note": (
            "This file is for inspection. Model training should use the "
            "original file so imputation is fitted on training data only."
        ),
    }


# ============================================================
# PROBLEM TYPE, FEATURES, AND PREPROCESSING
# ============================================================

def detect_problem_type(file_path, target_column):
    df = read_csv(file_path)

    if target_column not in df.columns:
        raise ValueError(
            f"Target '{target_column}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    target = df[target_column].dropna()

    if target.empty:
        raise ValueError("The target column contains no usable values.")

    if (
        pd.api.types.is_object_dtype(target)
        or isinstance(target.dtype, pd.CategoricalDtype)
        or pd.api.types.is_bool_dtype(target)
    ):
        problem_type = "classification"
    elif target.nunique() <= 20:
        problem_type = "classification"
    else:
        problem_type = "regression"

    return {"problem_type": problem_type}


def get_feature_columns(df, target_column):
    excluded = {target_column}

    for column in df.columns:
        normalized = str(column).strip().lower()
        if normalized in ID_COLUMNS or normalized.endswith("_id"):
            excluded.add(column)

    target_name = str(target_column).strip().lower()

    if target_name == "final_grade":
        for column in df.columns:
            if str(column).strip().lower() == "overall_score":
                excluded.add(column)
            if str(column).strip().lower() in SUBJECT_SCORE_COLUMNS:
                excluded.add(column)

    elif target_name == "overall_score":
        for column in df.columns:
            normalized = str(column).strip().lower()
            if normalized == "final_grade" or normalized in SUBJECT_SCORE_COLUMNS:
                excluded.add(column)

    return [column for column in df.columns if column not in excluded]


def build_preprocessor(X):
    """Build a preprocessing transformer based on feature data types."""
    numeric_columns = X.select_dtypes(include=np.number).columns.tolist()
    categorical_columns = [
        column for column in X.columns
        if column not in numeric_columns
    ]

    transformers = []

    if numeric_columns:
        numeric_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )
        transformers.append(
            ("numeric", numeric_pipeline, numeric_columns)
        )

    if categorical_columns:
        categorical_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="most_frequent",
                        keep_empty_features=True,
                    ),
                ),
                (
                    "onehot",
                    OneHotEncoder(handle_unknown="ignore"),
                ),
            ]
        )
        transformers.append(
            ("categorical", categorical_pipeline, categorical_columns)
        )

    if not transformers:
        raise ValueError(
            "No usable feature columns remain after excluding the target "
            "and identifier/leakage columns."
        )

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )


def get_models(problem_type):
    """Return candidate estimators for the selected problem type."""
    if problem_type == "regression":
        return {
            "LinearRegression": LinearRegression(),
            "DecisionTreeRegressor": DecisionTreeRegressor(
                random_state=RANDOM_STATE
            ),
            "RandomForestRegressor": RandomForestRegressor(
                n_estimators=200,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            ),
        }

    if problem_type == "classification":

        return {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),

        "LogisticRegressionBalanced": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),

        "DecisionTreeClassifier": DecisionTreeClassifier(
            random_state=RANDOM_STATE
        ),

        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    raise ValueError(
        "problem_type must be either 'classification' or 'regression'."
    )


def make_pipeline(preprocessor, estimator):
    """Create a fresh pipeline for one model."""
    return Pipeline(
        steps=[
            ("preprocessor", clone(preprocessor)),
            ("model", clone(estimator)),
        ]
    )


def get_cv_splitter(problem_type, y_train):
    """Return a suitable CV splitter, or None if data is too limited."""
    if problem_type == "classification":
        counts = pd.Series(y_train).value_counts()
        if counts.empty or counts.min() < 2:
            return None

        folds = min(5, int(counts.min()))
        return StratifiedKFold(
            n_splits=folds,
            shuffle=True,
            random_state=RANDOM_STATE,
        )

    if len(y_train) < 2:
        return None

    folds = min(5, len(y_train))
    return KFold(
        n_splits=folds,
        shuffle=True,
        random_state=RANDOM_STATE,
    )


# ============================================================
# METRICS
# ============================================================

def calculate_regression_metrics(y_true, predictions):
    """Calculate common regression metrics."""
    r2 = r2_score(y_true, predictions) if len(y_true) >= 2 else np.nan

    return {
        "r2_score": _safe_round(r2),
        "mae": _safe_round(mean_absolute_error(y_true, predictions)),
        "rmse": _safe_round(
            np.sqrt(mean_squared_error(y_true, predictions))
        ),
    }


def calculate_classification_metrics(y_true, predictions):
    """Calculate weighted classification metrics."""
    return {
        "accuracy": _safe_round(accuracy_score(y_true, predictions)),
        "precision_weighted": _safe_round(
            precision_score(
                y_true,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "recall_weighted": _safe_round(
            recall_score(
                y_true,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
        "f1_weighted": _safe_round(
            f1_score(
                y_true,
                predictions,
                average="weighted",
                zero_division=0,
            )
        ),
    }


def _safe_round(value):
    """Return a JSON-friendly rounded float or None for non-finite values."""
    try:
        value = float(value)
        if not np.isfinite(value):
            return None
        return round(value, 4)
    except (TypeError, ValueError):
        return None


def _json_default(value):
    """Convert common NumPy and pandas values to JSON-compatible values."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if pd.isna(value):
        return None
    return str(value)


def _find_identifier_columns(df):
    """Find identifier columns that should not be used as model features."""
    found = []
    for column in df.columns:
        normalized = str(column).strip().lower()
        if normalized in ID_COLUMNS or normalized.endswith("_id"):
            found.append(column)
    return found


def _target_leakage_warnings(target_column, feature_columns, df):
    """Return explainable warnings about common leakage risks."""
    warnings_list = []
    target_name = str(target_column).strip().lower()

    if target_name in {"final_grade", "overall_score"}:
        related = [
            col for col in df.columns
            if str(col).strip().lower() in (
                SUBJECT_SCORE_COLUMNS | {"overall_score", "final_grade"}
            )
            and col != target_column
        ]
        if related:
            warnings_list.append(
                "Related grade/score columns were excluded or should be "
                f"reviewed for target leakage: {related}."
            )

    if not feature_columns:
        warnings_list.append("No usable feature columns remain.")

    return warnings_list


# ============================================================
# MODEL TRAINING
# ============================================================



def train_models(
    file_path,
    target_column,
    problem_type=None,
    drop_exact_duplicates=True,
):
    """
    Train, compare, and save candidate models.
    """
    df = read_csv(file_path)

    if df.empty:
        raise ValueError("The CSV contains no rows.")

    # Detect columns that contain no usable values.
    empty_columns = [
        column
        for column in df.columns
        if df[column].isna().all()
    ]

    if empty_columns:
        raise ValueError(
            "These columns contain only missing values: "
            f"{empty_columns}. Remove them or provide valid values "
            "before training."
        )

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' not found. "
            f"Available columns: {df.columns.tolist()}"
        )

    original_rows = len(df)
    original_duplicate_rows = int(df.duplicated().sum())
    repeated_identifiers = {}

    for column in _find_identifier_columns(df):
        count = int(df[column].duplicated(keep=False).sum())
        if count:
            repeated_identifiers[str(column)] = count

    # Remove exact duplicate records before splitting.
    if drop_exact_duplicates:
        df = df.drop_duplicates().copy()

    rows_removed_as_duplicates = original_rows - len(df)

    # Remove rows whose target is missing.
    rows_without_target = int(df[target_column].isna().sum())

    if rows_without_target > 0:
        print(
            f"Removed {rows_without_target} rows "
            "with missing target values."
        )

    df = df.dropna(subset=[target_column]).copy()

    if len(df) < 10:
        raise ValueError(
            "Fewer than 10 usable rows remain after removing duplicate "
            "records and rows with missing target values."
        )

    # Detect the problem type using the cleaned target.
    if problem_type is None:
        target = df[target_column]

        if (
            pd.api.types.is_object_dtype(target)
            or isinstance(target.dtype, pd.CategoricalDtype)
            or pd.api.types.is_bool_dtype(target)
        ):
            problem_type = "classification"
        elif target.nunique() <= 20:
            problem_type = "classification"
        else:
            problem_type = "regression"

    if problem_type not in {"classification", "regression"}:
        raise ValueError(
            "problem_type must be 'classification' or 'regression'."
        )

    if (
        problem_type == "classification"
        and df[target_column].nunique() < 2
    ):
        raise ValueError(
            "Classification requires at least two target classes."
        )

    # Keep only rows with finite numeric target values for regression.
    if problem_type == "regression":
        numeric_target = pd.to_numeric(
            df[target_column], errors="coerce"
        )
        valid_target = (
            numeric_target.notna()
            & np.isfinite(numeric_target)
        )
        rows_invalid_target = int((~valid_target).sum())
        df = df.loc[valid_target].copy()
        df[target_column] = numeric_target.loc[valid_target]
    else:
        rows_invalid_target = 0

    if len(df) < 10:
        raise ValueError(
            "Fewer than 10 usable rows remain after target validation."
        )

    feature_columns = get_feature_columns(df, target_column)

    if not feature_columns:
        raise ValueError(
            "No feature columns remain after excluding the target, "
            "identifiers, and known leakage columns."
        )

    X = df[feature_columns].copy()
    y = df[target_column].copy()

    # Stratify only when every class has at least two examples.
    stratify_values = None

    if problem_type == "classification":
        class_counts = y.value_counts()

        if len(class_counts) >= 2 and class_counts.min() >= 2:
            stratify_values = y

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=RANDOM_STATE,
            stratify=stratify_values,
        )
    except ValueError as exc:
        raise ValueError(
            "Could not create a train/test split. Check the number of "
            f"rows per target class. Details: {exc}"
        ) from exc

    if (
        problem_type == "classification"
        and y_train.nunique() < 2
    ):
        raise ValueError(
            "The training split contains only one target class. "
            "Add more examples for each class and retry."
        )

    preprocessor = build_preprocessor(X_train)
    models = get_models(problem_type)
    cv_splitter = get_cv_splitter(problem_type, y_train)

    scoring = (
        "r2" if problem_type == "regression" else "f1_weighted"
    )

    fitted_models = {}
    results = {}

    for model_name, estimator in models.items():
        result = {}
        pipeline = make_pipeline(preprocessor, estimator)

        # Cross-validation uses training data only.
        if cv_splitter is not None:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    scores = cross_val_score(
                        make_pipeline(preprocessor, estimator),
                        X_train,
                        y_train,
                        cv=cv_splitter,
                        scoring=scoring,
                        n_jobs=None,
                        error_score="raise",
                    )

                result["cv_mean"] = _safe_round(np.mean(scores))
                result["cv_std"] = _safe_round(np.std(scores))

            except Exception as exc:
                result["cv_mean"] = None
                result["cv_std"] = None
                result["cv_warning"] = str(exc)
        else:
            result["cv_mean"] = None
            result["cv_std"] = None
            result["cv_warning"] = (
                "Cross-validation was skipped because the available "
                "examples per class or total row count were insufficient."
            )

        # Fit the candidate model.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(X_train, y_train)

        fitted_models[model_name] = pipeline
        train_predictions = pipeline.predict(X_train)

        if problem_type == "regression":
            train_score = (
                r2_score(y_train, train_predictions)
                if len(y_train) >= 2
                else np.nan
            )
            result["training_selection_score"] = _safe_round(
                train_score
            )

            test_predictions = pipeline.predict(X_test)
            result.update(
                calculate_regression_metrics(
                    y_test, test_predictions
                )
            )

        else:
            train_score = f1_score(
                y_train,
                train_predictions,
                average="weighted",
                zero_division=0,
            )
            result["training_selection_score"] = _safe_round(
                train_score
            )

            test_predictions = pipeline.predict(X_test)
            result.update(
                calculate_classification_metrics(
                    y_test, test_predictions
                )
            )

            result["classification_report"] = classification_report(
                y_test,
                test_predictions,
                output_dict=True,
                zero_division=0,
            )

            class_labels = pipeline.named_steps["model"].classes_

            result["confusion_matrix"] = confusion_matrix(
                y_test,
                test_predictions,
                labels=class_labels,
            ).tolist()

            result["class_labels"] = [
                str(label) for label in class_labels
            ]

        results[model_name] = result

    # Select using CV if every candidate has a valid CV score.
    cv_available_for_all = all(
        results[name].get("cv_mean") is not None
        for name in results
    )

    if cv_available_for_all:
        best_model_name = max(
            results,
            key=lambda name: results[name]["cv_mean"],
        )

        selection_metric = (
            "cross-validation R²"
            if problem_type == "regression"
            else "cross-validation weighted F1"
        )
    else:
        # Never use test scores to select the model.
        best_model_name = max(
            results,
            key=lambda name: (
                results[name].get("training_selection_score")
                if results[name].get("training_selection_score")
                is not None
                else float("-inf")
            ),
        )

        selection_metric = (
            "training R² (CV unavailable)"
            if problem_type == "regression"
            else "training weighted F1 (CV unavailable)"
        )

    best_pipeline = fitted_models[best_model_name]
    best_result = results[best_model_name]
    best_test_predictions = best_pipeline.predict(X_test)

    excluded_columns = [
        column
        for column in df.columns
        if column not in feature_columns
        and column != target_column
    ]

    leakage_warnings = _target_leakage_warnings(
        target_column, feature_columns, df
    )

    # Save all fitted models, the selected model, and test data.
    model_bundle = {
        "model": best_pipeline,
        "models": fitted_models,
        "target_column": target_column,
        "problem_type": problem_type,
        "feature_columns": feature_columns,
        "selected_model": best_model_name,
        "source_file": str(file_path),
        "excluded_columns": excluded_columns,
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "X_test": X_test,
        "y_test": y_test,
    }

    joblib.dump(model_bundle, MODEL_FILE)

    prediction_output = pd.DataFrame({
        "actual": y_test.reset_index(drop=True),
        "predicted": pd.Series(
            best_test_predictions
        ).reset_index(drop=True),
    })
    prediction_output.to_csv(PREDICTIONS_FILE, index=False)

    report = {
        "source_file": str(file_path),
        "original_rows": int(original_rows),
        "original_duplicate_rows": int(original_duplicate_rows),
        "rows_removed_as_duplicates": int(
            rows_removed_as_duplicates
        ),
        "rows_without_target_removed": int(rows_without_target),
        "invalid_regression_targets_removed": int(
            rows_invalid_target
        ),
        "rows_used": int(len(df)),
        "training_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "target_column": target_column,
        "problem_type": problem_type,
        "identifier_columns_with_repeated_values": repeated_identifiers,
        "excluded_columns": [
            str(column) for column in excluded_columns
        ],
        "feature_columns": [
            str(column) for column in feature_columns
        ],
        "selected_model": best_model_name,
        "selection_metric": selection_metric,
        "selected_cv_mean": best_result.get("cv_mean"),
        "selected_cv_std": best_result.get("cv_std"),
        "models": results,
        "leakage_warnings": leakage_warnings,
        "important_note": (
            "Exact duplicate rows were removed before splitting when "
            "drop_exact_duplicates=True. Review repeated identifiers and "
            "the feature list for dataset-specific leakage. Test metrics "
            "are estimates, not a guarantee of real-world performance."
        ),
        "model_file": MODEL_FILE,
        "predictions_file": PREDICTIONS_FILE,
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2, default=_json_default)

    return report
# ============================================================
# COMPATIBILITY WRAPPERS FOR THE CLI AGENT
# ============================================================

def train_regression_models(file_path, target_column):
    """Train regression models and return per-model results."""
    return train_models(
        file_path,
        target_column,
        problem_type="regression",
    )["models"]


def train_classification_models(file_path, target_column):
    """Train classification models and return per-model results."""
    return train_models(
        file_path,
        target_column,
        problem_type="classification",
    )["models"]


# Singular aliases retained for older agent.py versions.
def train_model(file_path, target_column):
    """Compatibility alias for regression training."""
    return train_regression_models(file_path, target_column)


def train_classification_model(file_path, target_column):
    """Compatibility alias for classification training."""
    return train_classification_models(file_path, target_column)



def compare_regression_models(results):
    """Compare regression results using CV R² or training R²."""
    if not results:
        raise ValueError("No regression results to compare.")

    all_cv_available = all(
        values.get("cv_mean") is not None
        for values in results.values()
    )

    if all_cv_available:
        metric = "cv_mean"
        selection_metric = "cross-validation R²"
    else:
        metric = "training_selection_score"
        selection_metric = "training R² (CV unavailable)"

    best_name = max(
        results,
        key=lambda name: (
            results[name].get(metric)
            if results[name].get(metric) is not None
            else float("-inf")
        ),
    )

    return {
        "best_model": best_name,
        "selection_metric": selection_metric,
        "all_models": results,
    }


def compare_classification_models(results):
    """Compare classification results using CV weighted F1 or training F1."""
    if not results:
        raise ValueError("No classification results to compare.")

    all_cv_available = all(
        values.get("cv_mean") is not None
        for values in results.values()
    )

    if all_cv_available:
        metric = "cv_mean"
        selection_metric = "cross-validation weighted F1"
    else:
        metric = "training_selection_score"
        selection_metric = "training weighted F1 (CV unavailable)"

    best_name = max(
        results,
        key=lambda name: (
            results[name].get(metric)
            if results[name].get(metric) is not None
            else float("-inf")
        ),
    )

    return {
        "best_model": best_name,
        "selection_metric": selection_metric,
        "all_models": results,
    }

# ============================================================
# PREDICTION ON NEW DATA
# ============================================================


def predict_new_data(model_file, new_file):
    """Predict new rows after validating the uploaded feature data."""

    if not os.path.isfile(model_file):
        raise FileNotFoundError(
            f"Saved model not found: {model_file}. Train a model first."
        )

    bundle = joblib.load(model_file)
    new_df = read_csv(new_file)

    if new_df.empty:
        raise ValueError("The prediction CSV contains no rows.")

    feature_columns = bundle.get("feature_columns")

    if not feature_columns:
        raise ValueError(
            "The saved model has no expected feature schema. "
            "Please train the model again."
        )

    model = bundle["model"]

    # Find features supplied by the user.
    provided_features = [
        column for column in feature_columns
        if column in new_df.columns
    ]

    # Do not predict if none of the expected features were supplied.
    if not provided_features:
        raise ValueError(
            "Your CSV does not contain any expected feature columns. "
            f"Expected at least one of: {feature_columns}"
        )

    # Reject rows where all supplied feature values are missing.
    blank_rows = new_df[provided_features].isna().all(axis=1)

    if blank_rows.any():
        row_numbers = (new_df.index[blank_rows] + 2).tolist()
        raise ValueError(
            "Some rows have no usable feature values. "
            f"Check CSV row(s): {row_numbers}."
        )

    missing_features = [
        column for column in feature_columns
        if column not in new_df.columns
    ]

    extra_columns = [
        column for column in new_df.columns
        if column not in feature_columns
    ]

    # Add missing columns so the saved pipeline can impute their values.
    X_new = new_df.copy()

    for column in missing_features:
        X_new[column] = np.nan

    # Keep exactly the features and their order used during training.
    X_new = X_new[feature_columns]

    predictions = model.predict(X_new)

    output = new_df.copy()
    target_column = bundle["target_column"]
    output[f"predicted_{target_column}"] = predictions

    output.to_csv(NEW_PREDICTIONS_FILE, index=False)

    return {
        "message": "Predictions completed.",
        "model": bundle.get("selected_model"),
        "problem_type": bundle.get("problem_type"),
        "rows_predicted": int(len(output)),
        "missing_features_filled_by_pipeline": missing_features,
        "extra_input_columns_ignored_for_model": extra_columns,
        "output_file": NEW_PREDICTIONS_FILE,
    }

# ============================================================
# EVALUATION UTILITIES
# ============================================================

def evaluate_classification_model(file_path):
    """Evaluate a CSV containing columns named actual and predicted."""
    df = read_csv(file_path)
    required = {"actual", "predicted"}

    if not required.issubset(df.columns):
        raise ValueError(
            "Evaluation CSV must contain 'actual' and 'predicted' columns."
        )

    return {
        **calculate_classification_metrics(
            df["actual"], df["predicted"]
        ),
        "classification_report": classification_report(
            df["actual"],
            df["predicted"],
            output_dict=True,
            zero_division=0,
        ),
        "confusion_matrix": confusion_matrix(
            df["actual"], df["predicted"]
        ).tolist(),
    }


def evaluate_regression_model(file_path):
    """Evaluate a CSV containing columns named actual and predicted."""
    df = read_csv(file_path)
    required = {"actual", "predicted"}

    if not required.issubset(df.columns):
        raise ValueError(
            "Evaluation CSV must contain 'actual' and 'predicted' columns."
        )

    return calculate_regression_metrics(
        pd.to_numeric(df["actual"], errors="raise"),
        pd.to_numeric(df["predicted"], errors="raise"),
    )
