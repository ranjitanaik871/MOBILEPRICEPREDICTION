"""Train the Random Forest model for mobile-phone price classification.

Task TA-009 trains a multi-class classifier from the prepared training data.
Model evaluation on the held-out testing data belongs to Task TA-010.
"""

import json
import pickle
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier


BASE_DIR = Path(__file__).resolve().parent
TRAINING_DATA_PATH = BASE_DIR / "training_dataset.csv"
MODEL_OUTPUT_PATH = BASE_DIR / "random_forest_model.pkl"
METADATA_OUTPUT_PATH = BASE_DIR / "random_forest_metadata.json"

TARGET_COLUMN = "Price_Category_Encoded"
RANDOM_STATE = 42
N_ESTIMATORS = 300
CLASS_NAMES = {
    "0": "Below 15,000",
    "1": "15,000-29,999",
    "2": "30,000 and above",
}


def train_random_forest(
    training_data_path=TRAINING_DATA_PATH,
    model_output_path=MODEL_OUTPUT_PATH,
    metadata_output_path=METADATA_OUTPUT_PATH,
):
    """Train and save a reproducible Random Forest classification model."""
    dataframe = pd.read_csv(training_data_path)

    if dataframe.empty:
        raise ValueError("The training dataset is empty.")
    if TARGET_COLUMN not in dataframe.columns:
        raise KeyError(f"Required target column '{TARGET_COLUMN}' was not found.")

    features = dataframe.drop(columns=[TARGET_COLUMN])
    target = dataframe[TARGET_COLUMN]

    if features.empty:
        raise ValueError("The training dataset contains no predictor columns.")
    if features.isna().any().any():
        missing_features = features.columns[features.isna().any()].tolist()
        raise ValueError(
            "Missing values found in predictor columns: "
            + ", ".join(missing_features)
        )
    if target.isna().any():
        raise ValueError(f"The target column '{TARGET_COLUMN}' contains missing values.")
    if target.nunique() < 2:
        raise ValueError("At least two target classes are required for classification.")

    classifier = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    classifier.fit(features, target)

    model_output_path = Path(model_output_path)
    metadata_output_path = Path(metadata_output_path)
    with model_output_path.open("wb") as model_file:
        pickle.dump(classifier, model_file)

    metadata = {
        "task": "TA-009 Random Forest Classification",
        "target_column": TARGET_COLUMN,
        "class_names": CLASS_NAMES,
        "feature_names": features.columns.tolist(),
        "training_rows": len(dataframe),
        "feature_count": features.shape[1],
        "class_distribution": {
            str(class_value): int(count)
            for class_value, count in target.value_counts().sort_index().items()
        },
        "model_parameters": {
            "n_estimators": N_ESTIMATORS,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
        },
    }
    metadata_output_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    return {
        "training_rows": len(dataframe),
        "feature_count": features.shape[1],
        "classes": sorted(target.unique().tolist()),
        "model_output_path": model_output_path,
        "metadata_output_path": metadata_output_path,
    }


if __name__ == "__main__":
    result = train_random_forest()
    print("Mobile Phone Random Forest Classification")
    print("=" * 43)
    print(f"Training rows: {result['training_rows']:,}")
    print(f"Predictor features: {result['feature_count']}")
    print(f"Target classes: {result['classes']}")
    print(f"Random Forest trees: {N_ESTIMATORS}")
    print(f"Random state: {RANDOM_STATE}")
    print(f"Saved model: {result['model_output_path']}")
    print(f"Saved metadata: {result['metadata_output_path']}")