"""Split the encoded mobile-phone dataset into training and testing data.

Task TA-008 creates reproducible, stratified datasets for the classification
model. The target column remains in both output files for the next task.
"""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "ml_ready_dataset.csv"
TRAIN_OUTPUT_PATH = BASE_DIR / "training_dataset.csv"
TEST_OUTPUT_PATH = BASE_DIR / "testing_dataset.csv"

TARGET_COLUMN = "Price_Category_Encoded"
TEST_SIZE = 0.20
RANDOM_STATE = 42


def split_dataset(
    input_path=INPUT_PATH,
    train_output_path=TRAIN_OUTPUT_PATH,
    test_output_path=TEST_OUTPUT_PATH,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
):
    """Create and save a stratified train/test split of the model-ready data."""
    dataframe = pd.read_csv(input_path)

    if dataframe.empty:
        raise ValueError("The model-ready dataset is empty.")
    if TARGET_COLUMN not in dataframe.columns:
        raise KeyError(f"Required target column '{TARGET_COLUMN}' was not found.")
    if dataframe[TARGET_COLUMN].isna().any():
        raise ValueError(f"The target column '{TARGET_COLUMN}' contains missing values.")
    if not 0 < test_size < 1:
        raise ValueError("test_size must be between 0 and 1.")
    if dataframe[TARGET_COLUMN].nunique() < 2:
        raise ValueError("At least two target classes are required for classification.")

    features = dataframe.drop(columns=[TARGET_COLUMN])
    target = dataframe[TARGET_COLUMN]
    train_features, test_features, train_target, test_target = train_test_split(
        features,
        target,
        test_size=test_size,
        random_state=random_state,
        stratify=target,
    )

    train_data = train_features.copy()
    train_data[TARGET_COLUMN] = train_target
    test_data = test_features.copy()
    test_data[TARGET_COLUMN] = test_target

    train_data.to_csv(train_output_path, index=False)
    test_data.to_csv(test_output_path, index=False)

    return {
        "total_rows": len(dataframe),
        "feature_count": features.shape[1],
        "training_rows": len(train_data),
        "testing_rows": len(test_data),
        "training_distribution": train_target.value_counts(normalize=True).sort_index().to_dict(),
        "testing_distribution": test_target.value_counts(normalize=True).sort_index().to_dict(),
        "random_state": random_state,
        "train_output_path": train_output_path,
        "test_output_path": test_output_path,
    }


if __name__ == "__main__":
    result = split_dataset()
    print("Mobile Phone Train-Test Split")
    print("=" * 30)
    print(f"Total rows: {result['total_rows']:,}")
    print(f"Features: {result['feature_count']}")
    print(f"Training rows: {result['training_rows']:,}")
    print(f"Testing rows: {result['testing_rows']:,}")
    print(f"Training class distribution: {result['training_distribution']}")
    print(f"Testing class distribution: {result['testing_distribution']}")
    print(f"Random state: {result['random_state']}")
    print(f"Saved training data: {result['train_output_path']}")
    print(f"Saved testing data: {result['test_output_path']}")