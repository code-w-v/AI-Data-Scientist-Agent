"""Command-line interface for the AI Data Scientist Agent (no API key required)."""

import json
import shlex

from data_tools import (
    check_missing_values,
    detect_data_types,
    detect_problem_type,
    inspect_dataset,
    predict_new_data,
    preprocess_data,
    summary_statistics,
    train_models,
)


def print_result(result):
    print(json.dumps(result, indent=2, default=str, ensure_ascii=False))


def help_text():
    print(r"""
AI Data Scientist Agent — commands

  analyze <csv>                         Inspect dimensions, missing cells and duplicates
  missing <csv>                         Show missing-value counts
  types <csv>                           Show column data types
  summary <csv>                         Show descriptive statistics
  preprocess <csv>                      Save an imputed exploration copy as processed_data.csv
  detect <csv> <target>                 Guess classification or regression
  train <csv> <target> [task]           Train and compare models
  run <csv> <target> <task>             Run complete data science workflow
  predict <new.csv>                     Predict using model_bundle.joblib
  help                                  Show this help
  exit                                  Quit

Task:
  regression
  classification

Examples:
  analyze Housing.csv
  train Housing.csv price regression
  train std.csv final_grade classification
  run salary_test.csv salary regression
  run heart.csv HeartDisease classification
  predict new_data.csv

Tip: use quotes around file names or target names that contain spaces.
""")


def run_workflow(file_path, target_column, task):
    """
    Run the main data science workflow automatically.

    Steps:
    1. Inspect dataset
    2. Check missing values
    3. Detect data types
    4. Preprocess an exploration copy
    5. Train and compare models
    6. Select the best model
    """

    print("\n========== AI DATA SCIENTIST WORKFLOW ==========\n")

    # --------------------------------------------------
    # Step 1: Inspect dataset
    # --------------------------------------------------

    print("STEP 1: Inspecting dataset...")

    inspection = inspect_dataset(file_path)

    print_result({
        "rows": inspection.get("rows"),
        "columns": inspection.get("columns"),
        "duplicate_rows": inspection.get("duplicate_rows"),
        "missing_cells": inspection.get("missing_cells"),
    })

    # --------------------------------------------------
    # Step 2: Check missing values
    # --------------------------------------------------

    print("\nSTEP 2: Checking missing values...")

    missing = check_missing_values(file_path)

    missing_columns = {
        column: count
        for column, count in missing.items()
        if count > 0
    }

    if missing_columns:
        print_result(missing_columns)
    else:
        print("No missing values found.")

    # --------------------------------------------------
    # Step 3: Detect data types
    # --------------------------------------------------

    print("\nSTEP 3: Detecting data types...")

    data_types = detect_data_types(file_path)

    print_result(data_types)

    # --------------------------------------------------
    # Step 4: Preprocess dataset
    # --------------------------------------------------

    print("\nSTEP 4: Preprocessing dataset...")

    preprocessing_result = preprocess_data(file_path)

    print_result(preprocessing_result)

    # --------------------------------------------------
    # Step 5: Train and compare models
    # --------------------------------------------------

    print("\nSTEP 5: Training and comparing models...")

    result = train_models(
        file_path,
        target_column,
        problem_type=task,
    )

    # --------------------------------------------------
    # Step 6: Display selected model
    # --------------------------------------------------

    print("\n========== WORKFLOW COMPLETE ==========\n")

    print_result({
        "target": result.get("target_column"),
        "problem_type": result.get("problem_type"),
        "selected_model": result.get("selected_model"),
        "selection_metric": result.get("selection_metric"),
        "selected_cv_mean": result.get("selected_cv_mean"),
        "models": result.get("models"),
        "files_created": {
            "metrics": "model_metrics.json",
            "model": "model_bundle.joblib",
            "test_predictions": "test_predictions.csv",
            "processed_data": "processed_data.csv",
        },
    })


def main():

    print("AI Data Scientist Agent (local ML mode; no LLM API key needed)")
    help_text()

    while True:

        try:
            raw = input("\nagent> ").strip()

        except (EOFError, KeyboardInterrupt):

            print("\nGoodbye!")
            break

        if not raw:
            continue

        try:

            parts = shlex.split(raw)

            command = parts[0].lower()

            # --------------------------------------------------
            # EXIT
            # --------------------------------------------------

            if command in {"exit", "quit"}:

                print("Goodbye!")
                break

            # --------------------------------------------------
            # HELP
            # --------------------------------------------------

            if command in {"help", "?"}:

                help_text()

            # --------------------------------------------------
            # ANALYZE
            # --------------------------------------------------

            elif command == "analyze" and len(parts) == 2:

                print_result(
                    inspect_dataset(parts[1])
                )

            # --------------------------------------------------
            # MISSING VALUES
            # --------------------------------------------------

            elif command == "missing" and len(parts) == 2:

                print_result(
                    check_missing_values(parts[1])
                )

            # --------------------------------------------------
            # DATA TYPES
            # --------------------------------------------------

            elif command == "types" and len(parts) == 2:

                print_result(
                    detect_data_types(parts[1])
                )

            # --------------------------------------------------
            # SUMMARY
            # --------------------------------------------------

            elif command == "summary" and len(parts) == 2:

                print_result(
                    summary_statistics(parts[1])
                )

            # --------------------------------------------------
            # PREPROCESS
            # --------------------------------------------------

            elif command == "preprocess" and len(parts) == 2:

                print_result(
                    preprocess_data(parts[1])
                )

            # --------------------------------------------------
            # DETECT PROBLEM TYPE
            # --------------------------------------------------

            elif command == "detect" and len(parts) == 3:

                print(
                    "Problem type:",
                    detect_problem_type(
                        parts[1],
                        parts[2]
                    )
                )

            # --------------------------------------------------
            # TRAIN
            # --------------------------------------------------

            elif command == "train" and len(parts) in {3, 4}:

                task = (
                    parts[3].lower()
                    if len(parts) == 4
                    else None
                )

                if task and task not in {
                    "regression",
                    "classification"
                }:
                    raise ValueError(
                        "Task must be regression or classification."
                    )

                result = train_models(
                    parts[1],
                    parts[2],
                    task
                )

                print("Training complete.")

                print_result({
                    "target": result.get("target_column"),
                    "problem_type": result.get("problem_type"),
                    "selected_model": result.get("selected_model"),
                    "selection_metric": result.get("selection_metric"),
                    "selected_cv_mean": result.get("selected_cv_mean"),
                    "metrics_file": "model_metrics.json",
                    "model_file": "model_bundle.joblib",
                    "test_predictions_file": "test_predictions.csv",
                })

            # --------------------------------------------------
            # RUN COMPLETE WORKFLOW
            # --------------------------------------------------

            elif command == "run" and len(parts) == 4:

                file_path = parts[1]
                target_column = parts[2]
                task = parts[3].lower()

                if task not in {
                    "regression",
                    "classification"
                }:
                    raise ValueError(
                        "Task must be regression or classification."
                    )

                run_workflow(
                    file_path,
                    target_column,
                    task
                )

            # --------------------------------------------------
            # PREDICT
            # --------------------------------------------------

            elif command == "predict" and len(parts) == 2:

                result = predict_new_data(
                    "model_bundle.joblib",
                    parts[1]
                )

                print_result(result)

            # --------------------------------------------------
            # INVALID COMMAND
            # --------------------------------------------------

            else:

                print(
                    "Command not recognized or wrong number "
                    "of arguments. Type 'help'."
                )

        except Exception as exc:

            print(f"Error: {exc}")


if __name__ == "__main__":
    main()