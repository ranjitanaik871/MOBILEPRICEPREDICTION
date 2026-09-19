"""Create engineered and encoded features for mobile price classification.

Task TA-007 prepares the cleaned mobile-phone specifications for machine
learning. The source price is never used as an input feature because it is
the source of the target category.
"""

from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "cleaned_dataset.csv"
OUTPUT_PATH = BASE_DIR / "ml_ready_dataset.csv"

TARGET_COLUMN = "Price_Category"
TARGET_MAPPING = {
    "Below 15,000": 0,
    "15,000-29,999": 1,
    "30,000 and above": 2,
}


def add_engineered_features(dataframe):
    """Add features that summarize related mobile-phone specifications."""
    features = dataframe.copy()

    features["Resolution_Megapixels"] = (
        features["Resolution_Width"] * features["Resolution_Height"] / 1_000_000
    )
    features["Total_Camera_MP"] = (
        features["Rear_Camera_MP"] + features["Front_Camera_MP"]
    )
    features["Battery_per_Screen_Inch"] = (
        features["Battery_mAh"] / features["Screen_Size_Inches"].replace(0, pd.NA)
    )
    features["Storage_per_RAM_GB"] = (
        features["Storage_GB"] / features["RAM_GB"].replace(0, pd.NA)
    )
    features["Resolution_Aspect_Ratio"] = (
        features["Resolution_Width"]
        / features["Resolution_Height"].replace(0, pd.NA)
    )

    return features


def prepare_features(input_path=INPUT_PATH, output_path=OUTPUT_PATH):
    """Engineer, encode, and save the model-ready feature table."""
    dataframe = pd.read_csv(input_path)

    required_columns = {
        TARGET_COLUMN,
        "Resolution_Width",
        "Resolution_Height",
        "Rear_Camera_MP",
        "Front_Camera_MP",
        "Battery_mAh",
        "Screen_Size_Inches",
        "Storage_GB",
        "RAM_GB",
    }
    missing_columns = required_columns.difference(dataframe.columns)
    if missing_columns:
        missing = ", ".join(sorted(missing_columns))
        raise KeyError(f"Required columns are missing from the cleaned dataset: {missing}")

    if not set(dataframe[TARGET_COLUMN].dropna()).issubset(TARGET_MAPPING):
        raise ValueError(f"Unexpected values found in {TARGET_COLUMN}.")

    features = add_engineered_features(dataframe.drop(columns=[TARGET_COLUMN]))
    excluded_columns = {"Name", "img", "price"}
    features = features.drop(
        columns=[column for column in excluded_columns if column in features],
        errors="ignore",
    )

    numeric_columns = features.select_dtypes(include="number").columns.tolist()
    categorical_columns = features.select_dtypes(exclude="number").columns.tolist()

    transformers = []
    if numeric_columns:
        transformers.append(
            (
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_columns,
            )
        )
    if categorical_columns:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical_columns,
            )
        )

    transformer = ColumnTransformer(transformers=transformers)
    encoded_features = transformer.fit_transform(features)
    feature_names = transformer.get_feature_names_out()
    model_ready = pd.DataFrame(encoded_features, columns=feature_names, index=dataframe.index)
    model_ready = model_ready.apply(pd.to_numeric, errors="coerce")
    model_ready["Price_Category_Encoded"] = dataframe[TARGET_COLUMN].map(TARGET_MAPPING)

    model_ready.to_csv(output_path, index=False)
    return {
        "input_rows": len(dataframe),
        "input_features": features.shape[1],
        "output_features": model_ready.shape[1] - 1,
        "target_mapping": TARGET_MAPPING,
        "missing_values": int(model_ready.isna().sum().sum()),
        "output_path": output_path,
    }


if __name__ == "__main__":
    result = prepare_features()
    print("Mobile Phone Feature Engineering and Encoding")
    print("=" * 46)
    print(f"Rows processed: {result['input_rows']:,}")
    print(f"Input features after engineering: {result['input_features']}")
    print(f"Encoded model features: {result['output_features']}")
    print(f"Target mapping: {result['target_mapping']}")
    print(f"Missing values after encoding: {result['missing_values']}")
    print(f"Saved dataset: {result['output_path']}")