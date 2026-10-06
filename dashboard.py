import hashlib
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

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
from sklearn.model_selection import train_test_split

from data_tools import train_models, predict_new_data


# =========================================================
# 1. PAGE CONFIGURATION AND FILES
# =========================================================
st.set_page_config(
    page_title="AI Data Scientist Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

METRICS_FILE = "model_metrics.json"
MODEL_FILE = "model_bundle.joblib"
TEST_PREDICTIONS_FILE = "test_predictions.csv"
NEW_PREDICTIONS_FILE = "new_data_predictions.csv"
TRAINING_DATA_FILE = "dashboard_uploaded_dataset.csv"
PREDICTION_INPUT_FILE = "dashboard_prediction_input.csv"
REPORT_FILE = "analysis_report.md"


# =========================================================
# 2. STYLING
# =========================================================
st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 3rem;}
      .hero {
        padding: 1.4rem 1.6rem; border-radius: 18px;
        background: linear-gradient(120deg, rgba(48,104,190,.24), rgba(26,160,150,.13));
        border: 1px solid rgba(130,160,200,.25); margin-bottom: 1rem;
      }
      .hero h1 {margin: 0 0 .35rem 0;}
      .muted {opacity: .75;}
      div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 12px; padding: 12px 14px;
      }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================
def read_json(path):
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError):
        return {}


def first_value(mapping, keys, default=None):
    if not isinstance(mapping, dict):
        return default
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return default


def safe_float(value):
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def format_number(value):
    number = safe_float(value)
    if number is None:
        return "N/A"
    return f"{number:,.4f}"


def infer_problem_type(target):
    if pd.api.types.is_numeric_dtype(target) and target.nunique(dropna=True) > 20:
        return "regression"
    return "classification"


def load_bundle():
    if not os.path.exists(MODEL_FILE):
        return None
    try:
        return joblib.load(MODEL_FILE)
    except Exception as exc:
        st.warning(f"Could not load the saved model bundle: {exc}")
        return None


def find_estimator(bundle):
    """Support common bundle formats from data_tools.py."""
    if bundle is None:
        return None
    if hasattr(bundle, "predict"):
        return bundle
    if isinstance(bundle, dict):
        for key in ("model", "best_model", "estimator", "pipeline", "selected_estimator"):
            candidate = bundle.get(key)
            if hasattr(candidate, "predict"):
                return candidate
    return None


def find_model_name(bundle, report):
    if isinstance(bundle, dict):
        name = first_value(bundle, ("selected_model_name", "model_name", "best_model_name", "selected_model"))
        if isinstance(name, str):
            return name
        candidate = bundle.get("model")
        if candidate is not None and not isinstance(candidate, str):
            return candidate.__class__.__name__
    return first_value(report, ("selected_model", "best_model", "model_name"), "Saved model")


def normalize_metrics_table(report):
    """Extract a model-comparison table from several common report layouts."""
    if not isinstance(report, dict):
        return pd.DataFrame()

    for key in ("model_comparison", "comparison", "results", "models", "all_models"):
        value = report.get(key)
        if isinstance(value, list):
            df = pd.DataFrame(value)
            if not df.empty:
                return df
        if isinstance(value, dict):
            # A dictionary of model_name -> metrics
            rows = []
            for name, metrics in value.items():
                if isinstance(metrics, dict):
                    rows.append({"Model": name, **metrics})
            if rows:
                return pd.DataFrame(rows)

    # Some implementations save the rows directly in the JSON root.
    for key in ("model_results", "comparison_results"):
        value = report.get(key)
        if isinstance(value, list):
            return pd.DataFrame(value)

    return pd.DataFrame()


def metric_column(df, candidates):
    lookup = {str(column).lower(): column for column in df.columns}
    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]
    return None


def get_target_from_report(report, fallback=None):
    return first_value(report, ("target_column", "target", "target_col"), fallback)


def get_problem_from_report(report, fallback=None):
    value = first_value(report, ("problem_type", "task_type"), fallback)
    if isinstance(value, str):
        value = value.lower()
        if "class" in value:
            return "classification"
        if "regress" in value:
            return "regression"
    return fallback


def feature_importance_frame(estimator, bundle):
    """Extract coefficients or tree importance, including a preprocessing pipeline."""
    if estimator is None:
        return pd.DataFrame()

    final_estimator = estimator
    preprocessor = None
    if hasattr(estimator, "steps") and estimator.steps:
        final_estimator = estimator.steps[-1][1]
        if len(estimator.steps) > 1:
            preprocessor = estimator[:-1]

    values = None
    kind = None
    if hasattr(final_estimator, "feature_importances_"):
        values = np.asarray(final_estimator.feature_importances_).ravel()
        kind = "Feature importance"
    elif hasattr(final_estimator, "coef_"):
        coef = np.asarray(final_estimator.coef_)
        if coef.ndim > 1:
            values = np.mean(np.abs(coef), axis=0)
        else:
            values = np.abs(coef).ravel()
        kind = "Absolute coefficient"

    if values is None:
        return pd.DataFrame()

    names = None
    try:
        if preprocessor is not None and hasattr(preprocessor, "get_feature_names_out"):
            names = list(preprocessor.get_feature_names_out())
    except Exception:
        names = None

    if not names and isinstance(bundle, dict):
        names = bundle.get("feature_names") or bundle.get("transformed_feature_names")

    if not names:
        names = [f"Feature {i + 1}" for i in range(len(values))]

    if len(names) != len(values):
        names = [f"Feature {i + 1}" for i in range(len(values))]

    result = pd.DataFrame({"Feature": names, "Importance": values, "Measure": kind})
    return result.sort_values("Importance", ascending=False).head(25)


def extract_model_metrics(report, selected_model=None):
    """Read metrics from current and older model_metrics.json layouts."""
    if not isinstance(report, dict):
        return {}

    metrics = {}
    model_map = report.get("models", {})
    if isinstance(model_map, dict) and model_map:
        chosen = None
        if selected_model:
            chosen = model_map.get(selected_model)
            if chosen is None:
                wanted = str(selected_model).strip().lower()
                for name, values in model_map.items():
                    if str(name).strip().lower() == wanted:
                        chosen = values
                        break
        if chosen is None:
            chosen_name = first_value(report, ("selected_model", "best_model", "model_name", "best_model_name"))
            if chosen_name:
                for name, values in model_map.items():
                    if str(name).strip().lower() == str(chosen_name).strip().lower():
                        chosen = values
                        break
        if isinstance(chosen, dict):
            metrics.update(chosen)

    # Merge nested metric containers without discarding the model-specific values.
    for container_key in ("test_metrics", "metrics", "evaluation"):
        nested = report.get(container_key)
        if isinstance(nested, dict):
            metrics.update(nested)

    # Some versions store each model as a row in model_comparison.
    rows = report.get("model_comparison", [])
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, dict):
                continue
            row_name = first_value(row, ("Model", "model", "name", "model_name"))
            if selected_model and row_name and str(row_name).strip().lower() == str(selected_model).strip().lower():
                metrics.update({k: v for k, v in row.items() if k.lower() not in ("model", "name", "model_name")})
                break

    # Root-level metrics are a backwards-compatible fallback.
    metric_keys = (
        "mae", "test_mae", "rmse", "test_rmse", "r2_score", "r2", "test_r2",
        "accuracy", "test_accuracy", "precision", "weighted_precision", "recall",
        "weighted_recall", "f1", "f1_score", "weighted_f1", "cv_mean", "cv_std",
    )
    for key in metric_keys:
        if key in report and key not in metrics:
            metrics[key] = report[key]
    return metrics


def write_analysis_report(dataset, target, problem_type, metrics, selected_model, feature_df, duplicate_count):
    problem = str(problem_type or "").strip().lower()
    lines = [
        "# AI Data Scientist Agent — Analysis Report",
        "",
        f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Dataset rows: {len(dataset):,}",
        f"- Dataset columns: {len(dataset.columns):,}",
        f"- Duplicate rows: {duplicate_count:,}",
        f"- Missing cells: {int(dataset.isna().sum().sum()):,}",
        f"- Target: `{target}`",
        f"- Problem type: {problem_type}",
        f"- Selected model: {selected_model}",
        "",
        "## Dataset quality",
        "",
        f"- Columns containing missing values: {int((dataset.isna().sum() > 0).sum())}",
        f"- Exact duplicate rows: {duplicate_count}",
        "",
        "## Model evaluation",
        "",
    ]
    if metrics:
        for key, value in metrics.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                lines.append(f"- {key}: {float(value):.4f}")
            elif value is not None and not isinstance(value, (dict, list, tuple)):
                lines.append(f"- {key}: {value}")
    else:
        lines.append("Model metrics were not available in the saved report. Train the model again to refresh the metrics file.")

    lines.extend(["", "## Most influential features", ""])
    if feature_df is not None and not feature_df.empty:
        for _, row in feature_df.head(10).iterrows():
            lines.append(f"- {row['Feature']}: {row['Importance']:.6g} ({row['Measure']})")
    else:
        lines.append("Feature importance or coefficients were not available for this model.")

    lines.extend(["", "## Interpretation and limitations", ""])
    if problem == "classification":
        lines.extend([
            "- Accuracy is the fraction of evaluated examples classified correctly.",
            "- Precision measures how often positive predictions are correct; recall measures how many actual positive cases are found.",
            "- F1 combines precision and recall. For imbalanced datasets, inspect per-class metrics and the confusion matrix instead of relying on accuracy alone.",
        ])
    elif problem == "regression":
        lines.extend([
            "- R² describes the proportion of target variance explained on the evaluated data; it is not prediction accuracy.",
            "- MAE is the average absolute prediction error, in the target's units.",
            "- RMSE penalizes larger errors more strongly than MAE, in the target's units.",
        ])
    else:
        lines.append("- Choose the correct problem type before training so the appropriate metrics are calculated.")
    lines.extend([
        "- Cross-validation estimates can vary across folds. Review the standard deviation as well as the mean.",
        "- Results depend on dataset quality, feature availability, split strategy, and whether the target leaks into the features.",
        "- This report is a technical summary, not a guarantee that predictions will be accurate for future or unseen populations.",
    ])
    return "\n".join(lines)


def dataset_signature(df):
    """Stable hash for the uploaded data and column order."""
    payload = df.to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def clean_target_column(df, target):
    return df[target].notna()


# =========================================================
# 4. SIDEBAR: REQUIRE UPLOAD BEFORE SHOWING THE DASHBOARD
# =========================================================
st.sidebar.title("🤖 AI Data Scientist Agent")
st.sidebar.caption("Upload a CSV to explore, train, evaluate, and predict.")

uploaded_dataset = st.sidebar.file_uploader(
    "Upload your dataset",
    type=["csv"],
    key="dataset_upload",
    help="The main dashboard stays empty until you upload a CSV.",
)

if uploaded_dataset is None:
    st.markdown(
        """
        <div class="hero">
          <h1>🤖 AI Data Scientist Agent</h1>
          <p>Upload a CSV dataset to begin automated data exploration and machine-learning analysis.</p>
          <p class="muted">Your dashboard will appear after a dataset is uploaded.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

try:
    dataset = pd.read_csv(uploaded_dataset)
    dataset_name = uploaded_dataset.name
except Exception as exc:
    st.error(f"Could not read this CSV file: {exc}")
    st.stop()

if dataset.empty or len(dataset.columns) == 0:
    st.error("The uploaded CSV is empty. Please upload a CSV containing rows and columns.")
    st.stop()
if dataset.columns.duplicated().any():
    st.error("This CSV has duplicate column names. Rename the columns and upload it again.")
    st.stop()

# Convert blank strings to missing values for quality checks and modelling.
dataset = dataset.replace(r"^\s*$", np.nan, regex=True)
duplicate_count = int(dataset.duplicated().sum())
current_dataset_signature = dataset_signature(dataset)

report = read_json(METRICS_FILE)
# Do not display results from a different upload as if they belong to this CSV.
if not report or report.get("dataset_signature") != current_dataset_signature:
    report = None
    bundle = None
    estimator = None
else:
    bundle = load_bundle()
    estimator = find_estimator(bundle)


# =========================================================
# 5. HERO AND NAVIGATION
# =========================================================
st.markdown(
    f"""
    <div class="hero">
      <h1>🤖 AI Data Scientist Agent</h1>
      <p>Dataset: <b>{dataset_name}</b></p>
      <p class="muted">Explore your data, train candidate models, understand performance, and generate predictions.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

tabs = st.tabs([
    "📊 Explore data",
    "🧠 Train & compare",
    "🔍 Evaluate model",
    "✨ Predict new data",
    "📄 Report",
])


# =========================================================
# TAB 1: EXPLORE DATA
# =========================================================
with tabs[0]:
    st.header("Dataset overview")
    missing_cells = int(dataset.isna().sum().sum())
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(dataset):,}")
    c2.metric("Columns", f"{len(dataset.columns):,}")
    c3.metric("Missing cells", f"{missing_cells:,}")
    c4.metric("Duplicate rows", f"{duplicate_count:,}")

    with st.expander("Preview dataset", expanded=True):
        st.dataframe(dataset.head(20), use_container_width=True)

    st.subheader("Data quality")
    quality = pd.DataFrame({
        "Column": dataset.columns,
        "Data type": [str(dtype) for dtype in dataset.dtypes],
        "Missing values": [int(dataset[column].isna().sum()) for column in dataset.columns],
        "Missing %": [round(dataset[column].isna().mean() * 100, 2) for column in dataset.columns],
        "Unique values": [int(dataset[column].nunique(dropna=True)) for column in dataset.columns],
    })
    st.dataframe(quality, use_container_width=True, hide_index=True)

    if missing_cells == 0:
        st.success("No missing values were found.")
    else:
        st.warning(f"Found {missing_cells:,} missing cells. The training pipeline should impute missing values using training data only.")

    if duplicate_count:
        st.info(f"There are {duplicate_count:,} exact duplicate rows. Review whether these are genuine repeated records before training.")

    st.subheader("Explore a column")
    selected_column = st.selectbox("Choose a column", list(dataset.columns), key="explore_column")
    series = dataset[selected_column]
    st.caption(f"Unique non-missing values: {series.nunique(dropna=True):,}")

    if pd.api.types.is_numeric_dtype(series):
        stats = series.describe().rename_axis("Statistic").reset_index(name="Value")
        st.dataframe(stats, use_container_width=True, hide_index=True)
        plot_data = series.dropna()
        if not plot_data.empty:
            left, right = st.columns(2)
            with left:
                st.plotly_chart(
                    px.histogram(plot_data, x=selected_column, title=f"Distribution of {selected_column}", marginal="box"),
                    use_container_width=True,
                )
            with right:
                st.plotly_chart(
                    px.box(dataset, y=selected_column, title=f"Outlier overview: {selected_column}"),
                    use_container_width=True,
                )
    else:
        counts = series.fillna("(missing)").astype(str).value_counts().head(30).rename_axis("Value").reset_index(name="Count")
        st.dataframe(counts, use_container_width=True, hide_index=True)
        st.plotly_chart(
            px.bar(counts, x="Value", y="Count", title=f"Top categories in {selected_column}"),
            use_container_width=True,
        )

    numeric_df = dataset.select_dtypes(include=np.number)
    if numeric_df.shape[1] >= 2:
        st.subheader("Numeric correlation")
        corr = numeric_df.corr(numeric_only=True)
        st.plotly_chart(
            px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlation matrix"),
            use_container_width=True,
        )



# =========================================================
# TAB 2: TRAIN AND COMPARE
# =========================================================
with tabs[1]:
    st.header("Train machine-learning models")
    st.write(
        "Choose the column you want the agent to predict. "
        "The target is the answer the model learns from."
    )

    target_default = get_target_from_report(report)
    target_options = list(dataset.columns)

    default_index = (
        target_options.index(target_default)
        if target_default in target_options
        else len(target_options) - 1
    )

    target_column = st.selectbox(
        "Target column",
        target_options,
        index=default_index,
        key="training_target",
    )

    inferred_type = infer_problem_type(
        dataset[target_column].dropna()
    )

    problem_type = st.radio(
        "Problem type",
        ["regression", "classification"],
        index=0 if inferred_type == "regression" else 1,
        horizontal=True,
        help=(
            "Regression predicts a numeric quantity. Classification "
            "predicts a category. You can override the automatic guess."
        ),
        key="problem_type",
    )

    st.info(
        f"Selected target: **{target_column}** · Task: **{problem_type}**. "
        "Check that the target or a direct copy of it is not present "
        "among the input features."
    )

    with st.expander("Before training: leakage and data checks"):
        st.markdown(
            """
            - Remove identifiers that do not carry meaningful predictive information.
            - Do not include columns that directly reveal the target.
            - For a house-price task, `price` should be the target, not an input feature.
            - If rows represent the same property or source, consider whether a random split gives an overly optimistic estimate.
            - A test set should remain separate from the data used to fit preprocessing and models.
            """
        )

    if st.button(
        "🚀 Train and compare models",
        type="primary",
        use_container_width=True,
    ):
        if dataset[target_column].isna().all():
            st.error("The selected target contains no usable values.")

        elif (
            problem_type == "regression"
            and not pd.api.types.is_numeric_dtype(
                dataset[target_column]
            )
        ):
            st.error(
                "Regression requires a numeric target. "
                "Select classification or choose a numeric target."
            )

        else:
            try:
                dataset.to_csv(TRAINING_DATA_FILE, index=False)

                with st.spinner(
                    "Training candidate models and evaluating them..."
                ):
                    try:
                        result = train_models(
                            file_path=TRAINING_DATA_FILE,
                            target_column=target_column,
                            problem_type=problem_type,
                        )
                    except TypeError:
                        # Compatibility with earlier versions.
                        result = train_models(
                            file_path=TRAINING_DATA_FILE,
                            target_column=target_column,
                        )

                st.session_state["last_training_target"] = target_column
                st.session_state["last_training_problem"] = problem_type
                st.session_state["last_training_dataset"] = dataset_name
                st.session_state["last_training_result"] = (
                    str(result)
                    if result is not None
                    else "Training completed."
                )

                # Associate the report with the current uploaded dataset.
                latest_report = read_json(METRICS_FILE) or {}
                latest_report["dataset_signature"] = (
                    current_dataset_signature
                )
                latest_report["uploaded_dataset_name"] = dataset_name

                with open(
                    METRICS_FILE, "w", encoding="utf-8"
                ) as report_file:
                    json.dump(
                        latest_report,
                        report_file,
                        indent=2,
                        default=str,
                    )

                st.success(f"Training completed for {dataset_name}.")
                st.rerun()

            except Exception as exc:
                st.error(f"Training failed: {exc}")
                st.exception(exc)

    # -----------------------------------------------------
    # DISPLAY SAVED MODEL COMPARISON
    # -----------------------------------------------------
    report = read_json(METRICS_FILE)
    saved_target = get_target_from_report(report)

    saved_source = (
        first_value(
            report,
            ("dataset", "dataset_name", "file_path", "training_file"),
            "",
        )
        if isinstance(report, dict)
        else ""
    )

    if report and isinstance(report, dict):
        st.subheader("Latest saved model comparison")

        saved_problem = get_problem_from_report(
            report,
            st.session_state.get(
                "last_training_problem", "regression"
            ),
        )

        # Warn if the saved target differs from the current selection.
        if saved_target and saved_target != target_column:
            st.warning(
                f"The saved report uses target `{saved_target}`, "
                f"while your current selection is `{target_column}`. "
                "Train again to refresh the comparison."
            )

        if saved_source and os.path.basename(str(saved_source)) not in (
            os.path.basename(dataset_name),
            os.path.basename(TRAINING_DATA_FILE),
        ):
            st.caption(f"Saved report source: {saved_source}")

        # -------------------------------------------------
        # DATA QUALITY AND LEAKAGE CHECKS
        # -------------------------------------------------
        leakage_warnings = report.get("leakage_warnings", [])
        excluded_columns = report.get("excluded_columns", [])

        if excluded_columns:
            st.info(
                "Excluded from model features: "
                + ", ".join(map(str, excluded_columns))
            )

        if leakage_warnings:
            with st.expander(
                "⚠️ Data leakage review", expanded=True
            ):
                for warning in leakage_warnings:
                    st.warning(str(warning))

        elif report.get("feature_columns"):
            st.success(
                "Automated checks found no configured target-specific "
                "leakage warning. Still review feature meanings manually; "
                "automatic checks cannot detect every form of leakage."
            )

        # -------------------------------------------------
        # WHY THE WINNING MODEL WAS SELECTED
        # -------------------------------------------------
        selected_name = report.get("selected_model")
        selection_metric = report.get("selection_metric")
        selected_cv = report.get("selected_cv_mean")
        selected_std = report.get("selected_cv_std")

        if selected_name:
            st.markdown("### Why this model was selected")

            st.write(
                f"The agent selected **{selected_name}** using "
                f"**{selection_metric or 'the available evaluation metric'}**."
            )

            cv_value = safe_float(selected_cv)
            if cv_value is not None:
                st.write(
                    f"Cross-validation mean: **{cv_value:.4f}**"
                )
            else:
                st.write(
                    "Cross-validation mean was not available."
                )

            std_value = safe_float(selected_std)
            if std_value is not None:
                st.write(
                    f"Cross-validation standard deviation: "
                    f"**{std_value:.4f}**. A larger value indicates "
                    "greater variation across folds."
                )

            st.caption(
                "The test set is intended for final evaluation, not model "
                "tuning. Cross-validation results are estimates and do not "
                "guarantee future performance."
            )

        # -------------------------------------------------
        # MODEL COMPARISON TABLE AND CHART
        # -------------------------------------------------
        comparison = normalize_metrics_table(report)

        if not comparison.empty:
            model_col = metric_column(
                comparison, ("model", "model_name", "name")
            )

            if model_col:
                comparison = comparison.rename(
                    columns={model_col: "Model"}
                )

            st.dataframe(
                comparison,
                use_container_width=True,
                hide_index=True,
            )

            metric_candidates = [
                (
                    "Cross-validation mean",
                    ("cv_mean", "mean_test_score", "cv_score"),
                ),
                (
                    "Test R² / accuracy",
                    ("r2_score", "test_r2", "accuracy", "test_accuracy"),
                ),
                ("MAE", ("mae", "test_mae")),
                ("RMSE", ("rmse", "test_rmse")),
                (
                    "F1 score",
                    ("f1", "f1_score", "weighted_f1"),
                ),
            ]

            available = [
                (label, metric_column(comparison, keys))
                for label, keys in metric_candidates
            ]
            available = [
                (label, column)
                for label, column in available
                if column is not None
            ]

            if available:
                label_to_col = dict(available)

                selected_metric_label = st.selectbox(
                    "Compare models using",
                    list(label_to_col.keys()),
                    key="comparison_metric",
                )

                selected_metric = label_to_col[
                    selected_metric_label
                ]

                chart_df = comparison.copy()

                if "Model" not in chart_df.columns:
                    chart_df["Model"] = chart_df.index.astype(str)

                chart_df[selected_metric] = pd.to_numeric(
                    chart_df[selected_metric],
                    errors="coerce",
                )

                chart_df = chart_df.dropna(
                    subset=[selected_metric]
                )

                if not chart_df.empty:
                    st.plotly_chart(
                        px.bar(
                            chart_df,
                            x="Model",
                            y=selected_metric,
                            text=selected_metric,
                            title=(
                                "Model comparison: "
                                f"{selected_metric_label}"
                            ),
                        ).update_traces(
                            texttemplate="%{text:.4f}",
                            textposition="outside",
                        ),
                        use_container_width=True,
                    )

        else:
            st.caption(
                "The metrics report exists, but its comparison-table "
                "format was not recognized. Training again may refresh it."
            )

        # -------------------------------------------------
        # AUTOMATIC MODEL COMPARISON SUMMARY
        # -------------------------------------------------
        st.subheader("🤖 Automatic model comparison summary")

        model_results = report.get("models", [])

        # Support reports where model results are stored as a dictionary.
        if isinstance(model_results, dict):
            normalized_results = []

            for name, values in model_results.items():
                if isinstance(values, dict):
                    item = dict(values)
                    item.setdefault("model", name)
                    normalized_results.append(item)

            model_results = normalized_results

        if model_results:
            results_df = pd.DataFrame(model_results)

            model_name_col = metric_column(
                results_df,
                ("model", "model_name", "name"),
            )

            if model_name_col:
                results_df = results_df.rename(
                    columns={model_name_col: "Model"}
                )

            # Find the model with the best cross-validation score.
            cv_col = metric_column(
                results_df,
                ("cv_mean", "mean_test_score", "cv_score"),
            )

            if cv_col:
                results_df[cv_col] = pd.to_numeric(
                    results_df[cv_col],
                    errors="coerce",
                )

                valid_cv = results_df.dropna(subset=[cv_col])

                if not valid_cv.empty:
                    best_cv_row = valid_cv.loc[
                        valid_cv[cv_col].idxmax()
                    ]

                    st.write(
                        f"**Best cross-validation score:** "
                        f"{best_cv_row.get('Model', 'Unknown model')} "
                        f"({best_cv_row[cv_col]:.4f})"
                    )

            if saved_problem == "classification":
                st.markdown("#### Understanding classification results")

                st.markdown(
                    """
                    - **Cross-validation weighted F1:** compares overall
                      classification performance while accounting for
                      the support of each class.
                    - **Positive-class recall:** measures how many actual
                      positive cases the model identifies.
                    - **Positive-class precision:** measures how many
                      predicted positive cases are actually positive.
                    - **Trade-off:** improving recall may increase false
                      positives.
                    """
                )

                # Per-class recall may not be stored in the comparison
                # report, so do not invent it from aggregate metrics.
                yes_recall_col = metric_column(
                    results_df,
                    (
                        "recall_yes",
                        "yes_recall",
                        "positive_recall",
                        "recall_positive",
                    ),
                )

                if yes_recall_col:
                    results_df[yes_recall_col] = pd.to_numeric(
                        results_df[yes_recall_col],
                        errors="coerce",
                    )

                    valid_recall = results_df.dropna(
                        subset=[yes_recall_col]
                    )

                    if not valid_recall.empty:
                        best_recall_row = valid_recall.loc[
                            valid_recall[yes_recall_col].idxmax()
                        ]

                        st.write(
                            f"**Highest positive-class recall:** "
                            f"{best_recall_row.get('Model', 'Unknown model')} "
                            f"({best_recall_row[yes_recall_col]:.4f})"
                        )
                else:
                    st.info(
                        "Per-class recall is not stored in the model "
                        "comparison table. Open the Evaluate Model tab "
                        "and select each model to compare its positive-class "
                        "precision, recall, F1 score, and confusion matrix."
                    )

                # Explain the selected model using the recorded CV score.
                if selected_name and cv_value is not None:
                    st.write(
                        f"The agent selected **{selected_name}** based on "
                        f"the recorded model-selection criterion. Its "
                        f"cross-validation mean was **{cv_value:.4f}**."
                    )

                st.warning(
                    "For imbalanced datasets, don't choose a model based "
                    "on accuracy alone. Review the positive-class results "
                    "and the costs of false positives and false negatives."
                )

            elif saved_problem == "regression":
                st.markdown("#### Understanding regression results")
                st.write(
                    "Compare cross-validation scores alongside MAE, RMSE, "
                    "and R² where available. The appropriate model depends "
                    "on the size and pattern of prediction errors, not just "
                    "one score."
                )

        else:
            st.info(
                "No per-model results were found in the saved report. "
                "Train the models again to generate the comparison."
            )

    else:
        st.caption(
            "Train models to generate a comparison report."
        )

# =========================================================
# TAB 3: EVALUATION
# =========================================================
with tabs[2]:
    st.header("Model evaluation")

    report = read_json(METRICS_FILE)
    bundle = load_bundle()

    if not isinstance(report, dict) or not isinstance(bundle, dict):
        st.info("Train a model first to see evaluation metrics.")

    else:
        saved_target = get_target_from_report(
            report,
            st.session_state.get("last_training_target"),
        )
        saved_problem = get_problem_from_report(
            report,
            st.session_state.get("last_training_problem", "regression"),
        )

        # Get all saved fitted models.
        saved_models = bundle.get("models", {})
        if not saved_models:
            # Backward compatibility with older saved bundles.
            saved_models = {}
            old_estimator = find_estimator(bundle)
            old_name = find_model_name(bundle, report)
            if old_estimator is not None:
                saved_models[old_name or "Saved model"] = old_estimator

        if not saved_models:
            st.warning("No saved models found. Please train the models again.")
        else:
            model_names = list(saved_models.keys())
            default_name = find_model_name(bundle, report)

            if default_name not in model_names:
                default_name = model_names[0]

            selected_name = st.selectbox(
                "Choose a model to evaluate",
                model_names,
                index=model_names.index(default_name),
                key="evaluation_model_choice",
            )

            estimator = saved_models[selected_name]
            metrics_dict = extract_model_metrics(report, selected_name)

            st.write(f"**Problem type:** {saved_problem or 'Not recorded'}")
            st.write(f"**Selected model:** {selected_name}")

            if saved_target:
                st.write(f"**Target:** `{saved_target}`")

            # Generate predictions using the selected model and
            # the same held-out test set for every model.
            X_test = bundle.get("X_test")
            y_test = bundle.get("y_test")

            if X_test is not None and y_test is not None:
                predictions = estimator.predict(X_test)
                actual = pd.Series(y_test).reset_index(drop=True)
                predicted = pd.Series(predictions).reset_index(drop=True)

                if saved_problem == "classification":
                    st.subheader("Classification metrics")

                    accuracy = accuracy_score(actual, predicted)
                    precision = precision_score(
                        actual, predicted,
                        average="weighted", zero_division=0
                    )
                    recall = recall_score(
                        actual, predicted,
                        average="weighted", zero_division=0
                    )
                    f1 = f1_score(
                        actual, predicted,
                        average="weighted", zero_division=0
                    )

                    cols = st.columns(4)
                    for col, label, value in zip(
                        cols,
                        ("Accuracy", "Weighted precision",
                         "Weighted recall", "Weighted F1"),
                        (accuracy, precision, recall, f1),
                    ):
                        col.metric(label, f"{value:.4f}")

                    st.caption(
                        "For imbalanced datasets, inspect the positive "
                        "class's precision, recall, F1, and the confusion "
                        "matrix instead of relying on accuracy alone."
                    )

                    st.subheader("Per-class performance")
                    report_dict = classification_report(
                        actual.astype(str),
                        predicted.astype(str),
                        output_dict=True,
                        zero_division=0,
                    )
                    st.dataframe(
                        pd.DataFrame(report_dict).T,
                        use_container_width=True,
                    )

                    labels = sorted(
                        set(actual.astype(str)) |
                        set(predicted.astype(str))
                    )
                    matrix = confusion_matrix(
                        actual.astype(str),
                        predicted.astype(str),
                        labels=labels,
                    )

                    st.plotly_chart(
                        px.imshow(
                            matrix,
                            x=labels,
                            y=labels,
                            text_auto=True,
                            labels={
                                "x": "Predicted",
                                "y": "Actual",
                                "color": "Count",
                            },
                            title=f"Confusion matrix — {selected_name}",
                        ),
                        use_container_width=True,
                    )

                    st.text("Per-class classification report")
                    st.code(
                        classification_report(
                            actual.astype(str),
                            predicted.astype(str),
                            zero_division=0,
                        )
                    )

                elif saved_problem == "regression":
                    mae = mean_absolute_error(actual, predictions)
                    rmse = np.sqrt(
                        mean_squared_error(actual, predictions)
                    )
                    r2 = r2_score(actual, predictions)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("MAE", f"{mae:.4f}")
                    c2.metric("RMSE", f"{rmse:.4f}")
                    c3.metric("R²", f"{r2:.4f}")

                    plot_df = pd.DataFrame({
                        "Actual": actual,
                        "Predicted": predictions,
                    })
                    st.plotly_chart(
                        px.scatter(
                            plot_df,
                            x="Actual",
                            y="Predicted",
                            title=f"Actual vs predicted — {selected_name}",
                            opacity=0.65,
                        ),
                        use_container_width=True,
                    )

            else:
                st.warning(
                    "This saved model bundle does not contain the shared "
                    "test data. Please retrain the models using the "
                    "updated data_tools.py."
                )

            st.subheader("Feature influence")
            feature_df = feature_importance_frame(estimator, bundle)

            if not feature_df.empty:
                st.caption(
                    "For tree models, values are feature importances. "
                    "For linear models, absolute coefficient magnitudes "
                    "are shown. These are not necessarily causal effects."
                )

                st.plotly_chart(
                    px.bar(
                        feature_df.sort_values("Importance"),
                        x="Importance",
                        y="Feature",
                        orientation="h",
                        title=f"Feature influence — {selected_name}",
                    ),
                    use_container_width=True,
                )
                st.dataframe(
                    feature_df,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info(
                    "Feature influence is unavailable for this model."
                )

# =========================================================
# TAB 4: PREDICT NEW DATA
# =========================================================
with tabs[3]:
    st.header("Predict new data")
    st.write(
        "Upload a separate CSV containing the model's input features. "
        "For a normal prediction workflow, do not include the target column."
    )

    if estimator is None:
        st.warning("Train a model first. A saved model bundle was not found or did not contain a recognizable estimator.")

    prediction_file = st.file_uploader(
        "Upload a CSV for prediction",
        type=["csv"],
        key="prediction_upload",
    )

    if prediction_file is not None:
        prediction_df = pd.read_csv(prediction_file).replace(r"^\s*$", np.nan, regex=True)
        st.subheader("Prediction input preview")
        st.dataframe(prediction_df.head(20), use_container_width=True)

        expected_features = None
        if isinstance(bundle, dict):
            expected_features = bundle.get("feature_columns") or bundle.get("features")
        if expected_features:
            missing_features = [column for column in expected_features if column not in prediction_df.columns]
            extra_features = [column for column in prediction_df.columns if column not in expected_features]
            if missing_features:
                st.warning(
                    "Some expected features are missing: "
                    + ", ".join(map(str, missing_features))
                    + ". The prediction utility may reject or impute these depending on your saved pipeline."
                )
            if extra_features:
                st.info("Extra columns will be ignored by the saved prediction pipeline if supported: " + ", ".join(map(str, extra_features)))

        if st.button("✨ Generate predictions", type="primary", use_container_width=True):
            try:
                prediction_df.to_csv(PREDICTION_INPUT_FILE, index=False)
                with st.spinner("Generating predictions..."):
                    try:
                        result = predict_new_data(MODEL_FILE, PREDICTION_INPUT_FILE)
                    except TypeError:
                        result = predict_new_data(model_file=MODEL_FILE, new_file=PREDICTION_INPUT_FILE)

                if isinstance(result, pd.DataFrame):
                    output_df = result
                elif isinstance(result, str) and os.path.exists(result):
                    output_df = pd.read_csv(result)
                elif os.path.exists(NEW_PREDICTIONS_FILE):
                    output_df = pd.read_csv(NEW_PREDICTIONS_FILE)
                else:
                    output_df = None

                if output_df is not None:
                    st.session_state["last_prediction_output"] = output_df
                    st.success("Predictions generated successfully.")
                else:
                    st.success("Prediction function completed. Check its output file for results.")
                st.rerun()
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
                st.exception(exc)

    output_df = st.session_state.get("last_prediction_output")
    if output_df is None and os.path.exists(NEW_PREDICTIONS_FILE):
        try:
            output_df = pd.read_csv(NEW_PREDICTIONS_FILE)
        except Exception:
            output_df = None

    if isinstance(output_df, pd.DataFrame):
        st.subheader("Prediction results")
        st.dataframe(output_df, use_container_width=True)
        st.download_button(
            "⬇️ Download predictions CSV",
            data=output_df.to_csv(index=False).encode("utf-8"),
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )


# =========================================================
# TAB 5: DOWNLOADABLE REPORT
# =========================================================
with tabs[4]:
    st.header("Analysis report")
    current_report = read_json(METRICS_FILE)
    if not isinstance(current_report, dict):
        current_report = {}

    if (
        not current_report
        or current_report.get("dataset_signature") != current_dataset_signature
    ):
        current_report = {}
        current_bundle
saved_signature = current_report.get("dataset_signature")
same_dataset_signature = saved_signature == current_dataset_signature
same_dataset_name = (
    current_report.get("uploaded_dataset_name") == dataset_name
)

# Prefer an exact signature match. If an older saved report has a
# different signature, allow it only when the uploaded filename matches.
if not current_report or not (
    same_dataset_signature or same_dataset_name
):
    current_report = {}
    current_bundle = None
else:
    current_bundle = load_bundle()
    if not same_dataset_signature:
        st.warning(
            "The saved report matches this dataset's filename, "
            "but its content signature differs. Verify that the uploaded "
            "file is the same version used for training."
        ) == None
    else:
        current_bundle = load_bundle()
    current_estimator = find_estimator(current_bundle)
    current_feature_df = feature_importance_frame(current_estimator, current_bundle)
    current_target = get_target_from_report(current_report, st.session_state.get("last_training_target"))
    current_problem = get_problem_from_report(
        current_report,
        st.session_state.get("last_training_problem", "regression"),
    )
    current_model_name = find_model_name(current_bundle, current_report)

    metric_values = extract_model_metrics(current_report, current_model_name)

    st.write("This report combines dataset quality checks with the latest saved model results.")
    report_text = write_analysis_report(
        dataset=dataset,
        target=current_target or "(not recorded)",
        problem_type=current_problem or "(not recorded)",
        metrics=metric_values,
        selected_model=current_model_name,
        feature_df=current_feature_df,
        duplicate_count=duplicate_count,
    )
    if isinstance(current_report, dict) and current_report:
        extra_lines = ["", "## Automated training checks", ""]
        removed = current_report.get("rows_removed_as_duplicates")
        if removed is not None:
            extra_lines.append(f"- Duplicate rows removed before splitting: {removed}")
        if current_report.get("excluded_columns"):
            extra_lines.append("- Excluded columns: " + ", ".join(map(str, current_report["excluded_columns"])))
        if current_report.get("leakage_warnings"):
            extra_lines.append("- Leakage review warnings:")
            extra_lines.extend(f"  - {warning}" for warning in current_report["leakage_warnings"])
        else:
            extra_lines.append("- No configured target-specific leakage warning was recorded; manual review is still necessary.")
        if current_report.get("selection_metric"):
            extra_lines.append(f"- Model selection criterion: {current_report['selection_metric']}")
        if current_report.get("important_note"):
            extra_lines.extend(["", str(current_report["important_note"])])
        report_text += "\n" + "\n".join(extra_lines)

    st.markdown(report_text)
    st.download_button(
        "⬇️ Download analysis report (Markdown)",
        data=report_text.encode("utf-8"),
        file_name="ai_data_scientist_report.md",
        mime="text/markdown",
        use_container_width=True,
    )

st.divider()
st.caption("AI Data Scientist Agent · Review the dataset and evaluation setup before relying on predictions.")
