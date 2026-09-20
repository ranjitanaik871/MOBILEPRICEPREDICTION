"""Predict and evaluate mobile-phone price categories.

Task TA-010 evaluates the trained Task 9 Random Forest model on the unseen
testing dataset using classification metrics and a confusion matrix.
"""

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


BASE_DIR = Path(__file__).resolve().parent
TESTING_DATA_PATH = BASE_DIR / "testing_dataset.csv"
MODEL_PATH = BASE_DIR / "random_forest_model.pkl"
PREDICTIONS_OUTPUT_PATH = BASE_DIR / "price_category_predictions.csv"
RESULTS_OUTPUT_PATH = BASE_DIR / "price_category_evaluation_results.json"
CONFUSION_MATRIX_OUTPUT_PATH = BASE_DIR / "price_category_confusion_matrix.csv"

TARGET_COLUMN = "Price_Category_Encoded"
CLASS_LABELS = {
    0: "Below 15,000",
    1: "15,000-29,999",
    2: "30,000 and above",
}
CLASS_ORDER = sorted(CLASS_LABELS)


def evaluate_model(
    testing_data_path=TESTING_DATA_PATH,
    model_path=MODEL_PATH,
    predictions_output_path=PREDICTIONS_OUTPUT_PATH,
    results_output_path=RESULTS_OUTPUT_PATH,
    confusion_matrix_output_path=CONFUSION_MATRIX_OUTPUT_PATH,
):
    """Generate test predictions, metrics, and a confusion matrix."""
    testing_data = pd.read_csv(testing_data_path)
    if testing_data.empty:
        raise ValueError("The testing dataset is empty.")
    if TARGET_COLUMN not in testing_data.columns:
        raise KeyError(f"Required target column '{TARGET_COLUMN}' was not found.")

    features = testing_data.drop(columns=[TARGET_COLUMN])
    actual = testing_data[TARGET_COLUMN]
    if features.isna().any().any():
        missing_features = features.columns[features.isna().any()].tolist()
        raise ValueError(
            "Missing values found in predictor columns: "
            + ", ".join(missing_features)
        )
    if actual.isna().any():
        raise ValueError(f"The target column '{TARGET_COLUMN}' contains missing values.")

    with Path(model_path).open("rb") as model_file:
        model = pickle.load(model_file)

    expected_features = getattr(model, "feature_names_in_", None)
    if expected_features is not None and list(expected_features) != list(features.columns):
        raise ValueError("Testing features do not match the features used to train the model.")

    predicted = model.predict(features)
    labels = CLASS_ORDER
    matrix = confusion_matrix(actual, predicted, labels=labels)

    metrics = {
        "accuracy": float(accuracy_score(actual, predicted)),
        "precision_weighted": float(
            precision_score(actual, predicted, labels=labels, average="weighted", zero_division=0)
        ),
        "recall_weighted": float(
            recall_score(actual, predicted, labels=labels, average="weighted", zero_division=0)
        ),
        "f1_weighted": float(
            f1_score(actual, predicted, labels=labels, average="weighted", zero_division=0)
        ),
    }
    report = classification_report(
        actual,
        predicted,
        labels=labels,
        target_names=[CLASS_LABELS[label] for label in labels],
        output_dict=True,
        zero_division=0,
    )

    predictions = testing_data.copy()
    predictions["Actual_Price_Category"] = actual.map(CLASS_LABELS)
    predictions["Predicted_Price_Category_Encoded"] = predicted
    predictions["Predicted_Price_Category"] = pd.Series(predicted, index=predictions.index).map(CLASS_LABELS)
    predictions.to_csv(predictions_output_path, index=False)

    matrix_dataframe = pd.DataFrame(
        matrix,
        index=[f"Actual: {CLASS_LABELS[label]}" for label in labels],
        columns=[f"Predicted: {CLASS_LABELS[label]}" for label in labels],
    )
    matrix_dataframe.to_csv(confusion_matrix_output_path)

    results = {
        "task": "TA-010 Price Category Prediction and Evaluation",
        "testing_rows": len(testing_data),
        "feature_count": features.shape[1],
        "target_column": TARGET_COLUMN,
        "class_labels": {str(label): CLASS_LABELS[label] for label in labels},
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "output_files": {
            "predictions": str(Path(predictions_output_path)),
            "results": str(Path(results_output_path)),
            "confusion_matrix": str(Path(confusion_matrix_output_path)),
        },
    }
    Path(results_output_path).write_text(json.dumps(results, indent=2), encoding="utf-8")

    return {
        "testing_rows": len(testing_data),
        "metrics": metrics,
        "confusion_matrix": matrix.tolist(),
        "predictions_output_path": predictions_output_path,
        "results_output_path": results_output_path,
        "confusion_matrix_output_path": confusion_matrix_output_path,
    }


if __name__ == "__main__":
    result = evaluate_model()
    print("Mobile Phone Price Category Prediction and Evaluation")
    print("=" * 54)
    print(f"Testing rows: {result['testing_rows']:,}")
    for metric_name, metric_value in result["metrics"].items():
        print(f"{metric_name.replace('_', ' ').title()}: {metric_value:.4f}")
    print("Confusion matrix:")
    for row in result["confusion_matrix"]:
        print(f"  {row}")
    print(f"Saved predictions: {result['predictions_output_path']}")
    print(f"Saved evaluation results: {result['results_output_path']}")
    print(f"Saved confusion matrix: {result['confusion_matrix_output_path']}")