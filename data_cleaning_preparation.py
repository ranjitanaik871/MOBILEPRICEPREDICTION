"""Clean the raw mobile dataset and create a model-ready feature table."""

import re
from pathlib import Path

import pandas as pd

from target_identification import price_category


BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "mobile_dataset.csv"
CLEANED_DATASET_PATH = BASE_DIR / "cleaned_dataset.csv"


def first_number(value):
    """Return the first number in text, or a missing value."""
    match = re.search(r"\d+(?:\.\d+)?", str(value))
    return float(match.group()) if match else pd.NA


def numbers(value):
    """Return all numbers in text, including values containing thin spaces."""
    return [float(item) for item in re.findall(r"\d+(?:\.\d+)?", str(value))]


def extract_storage(value):
    values = numbers(value)
    return (values[0], values[1]) if len(values) >= 2 else (pd.NA, pd.NA)


def extract_battery(value):
    text = str(value)
    battery = re.search(r"(\d+(?:\.\d+)?)\s*mAh", text, re.IGNORECASE)
    charging = re.search(r"(\d+(?:\.\d+)?)\s*W", text, re.IGNORECASE)
    return (
        float(battery.group(1)) if battery else pd.NA,
        float(charging.group(1)) if charging else pd.NA,
    )


def extract_display(value):
    text = str(value)
    values = numbers(text)
    dimensions = re.search(r"(\d+)\s*[x×]\s*(\d+)", text)
    refresh = re.search(r"(\d+)\s*Hz", text, re.IGNORECASE)
    return (
        values[0] if values else pd.NA,
        float(dimensions.group(1)) if dimensions else pd.NA,
        float(dimensions.group(2)) if dimensions else pd.NA,
        float(refresh.group(1)) if refresh else pd.NA,
    )


def extract_camera(value):
    text = str(value)
    front_text = re.search(r"&\s*(\d+(?:\.\d+)?)\s*MP\s*Front", text, re.IGNORECASE)
    rear_text = re.search(r"(\d+(?:\.\d+)?)\s*MP", text, re.IGNORECASE)
    return (
        float(rear_text.group(1)) if rear_text else pd.NA,
        float(front_text.group(1)) if front_text else pd.NA,
    )


def extract_processor(value):
    text = str(value)
    speed = re.search(r"(\d+(?:\.\d+)?)\s*GHz", text, re.IGNORECASE)
    core = re.search(r"(\d+)\s*Core", text, re.IGNORECASE)
    return (
        float(speed.group(1)) if speed else pd.NA,
        float(core.group(1)) if core else pd.NA,
    )


def has_term(value, term):
    return int(term.lower() in str(value).lower())


def prepare_dataset(input_path=DATASET_PATH, output_path=CLEANED_DATASET_PATH):
    """Clean, transform, impute, and save the model-ready dataset."""
    raw = pd.read_csv(input_path)
    original_rows = len(raw)
    duplicate_rows = int(raw.duplicated().sum())
    cleaned = raw.drop_duplicates().copy()

    storage_values = cleaned["storage"].apply(extract_storage).tolist()
    battery_values = cleaned["battery"].apply(extract_battery).tolist()
    display_values = cleaned["display"].apply(extract_display).tolist()
    camera_values = cleaned["camera"].apply(extract_camera).tolist()
    processor_values = cleaned["processor"].apply(extract_processor).tolist()

    prepared = pd.DataFrame(index=cleaned.index)
    prepared["RAM_GB"] = [value[0] for value in storage_values]
    prepared["Storage_GB"] = [value[1] for value in storage_values]
    prepared["Battery_mAh"] = [value[0] for value in battery_values]
    prepared["Charging_W"] = [value[1] for value in battery_values]
    prepared["Screen_Size_Inches"] = [value[0] for value in display_values]
    prepared["Resolution_Width"] = [value[1] for value in display_values]
    prepared["Resolution_Height"] = [value[2] for value in display_values]
    prepared["Refresh_Rate_Hz"] = [value[3] for value in display_values]
    prepared["Rear_Camera_MP"] = [value[0] for value in camera_values]
    prepared["Front_Camera_MP"] = [value[1] for value in camera_values]
    prepared["Processor_Speed_GHz"] = [value[0] for value in processor_values]
    prepared["Cores"] = [value[1] for value in processor_values]
    for term, column in (
        ("3G", "Has_3G"),
        ("4G", "Has_4G"),
        ("5G", "Has_5G"),
        ("VoLTE", "Has_VoLTE"),
        ("Wi-Fi", "Has_WiFi"),
        ("NFC", "Has_NFC"),
    ):
        prepared[column] = cleaned["sim"].apply(lambda value, item=term: has_term(value, item))
    prepared["External_Memory_Supported"] = cleaned["memoryExternal"].fillna("").str.contains(
        "Memory Card Supported", case=False, regex=False
    ).astype(int)
    prepared["Android_Version"] = cleaned["version"].apply(first_number)
    prepared["FM_Radio"] = cleaned["fm"].fillna("").str.contains(
        "FM Radio", case=False, regex=False
    ).astype(int)

    numeric_columns = list(prepared.columns)
    for column in numeric_columns:
        prepared[column] = pd.to_numeric(prepared[column], errors="coerce")
        prepared[column] = prepared[column].fillna(prepared[column].median())

    prepared["Price_Category"] = cleaned["price"].astype(int).apply(price_category)
    prepared.to_csv(output_path, index=False)

    return {
        "original_rows": original_rows,
        "duplicate_rows_removed": duplicate_rows,
        "cleaned_rows": len(prepared),
        "columns": len(prepared.columns),
        "missing_values": int(prepared.isna().sum().sum()),
        "output_path": output_path,
    }


if __name__ == "__main__":
    result = prepare_dataset()
    print("Mobile Phone Data Cleaning and Preparation")
    print("=" * 43)
    print(f"Original rows: {result['original_rows']:,}")
    print(f"Duplicate rows removed: {result['duplicate_rows_removed']:,}")
    print(f"Cleaned rows: {result['cleaned_rows']:,}")
    print(f"Model-ready columns: {result['columns']}")
    print(f"Missing values after preparation: {result['missing_values']:,}")
    print(f"Saved dataset: {result['output_path']}")