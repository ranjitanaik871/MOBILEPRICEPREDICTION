"""Identify model features from the mobile phone dataset."""

import csv
from pathlib import Path


DATASET_PATH = Path(__file__).resolve().parent / "mobile_dataset.csv"

FEATURE_PLAN = {
    "Numeric features after extraction": [
        "Spec Score",
        "rating",
        "RAM_GB (from storage)",
        "Storage_GB (from storage)",
        "Battery_mAh (from battery)",
        "Charging_W (from battery)",
        "Screen_Size_Inches (from display)",
        "Resolution_Width (from display)",
        "Resolution_Height (from display)",
        "Refresh_Rate_Hz (from display)",
        "Rear_Camera_MP (from camera)",
        "Front_Camera_MP (from camera)",
        "Processor_Speed_GHz (from processor)",
        "Cores (from processor)",
    ],
    "Categorical or binary features after extraction": [
        "tag",
        "3G, 4G, 5G, VoLTE, Wi-Fi, NFC and other connectivity flags (from sim)",
        "External_Memory_Supported (from memoryExternal)",
        "Android_Version (from version)",
        "FM_Radio (from fm)",
    ],
    "Target, not an input feature": [
        "Price_Category created from price using a documented category rule",
    ],
    "Exclude from model inputs": [
        "img: URL and not a phone specification",
        "Name: product identifier; retain for reporting, but avoid using it as a predictive feature",
        "price: source for the target category, so it must not be used as an input feature",
        "Unparsed raw text columns after their useful values have been extracted",
    ],
}


def load_rows(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def print_feature_plan(rows):
    print("Mobile Phone Feature Identification")
    print("=" * 38)
    print(f"Rows reviewed: {len(rows):,}")
    print(f"Columns reviewed: {len(rows[0])}")
    print("\nRecommended feature plan:")
    for group, features in FEATURE_PLAN.items():
        print(f"\n{group}:")
        for feature in features:
            print(f"  - {feature}")

    print("\nRaw columns requiring extraction:")
    for column in ("processor", "storage", "battery", "display", "camera", "sim"):
        present = sum(bool(row[column].strip()) for row in rows)
        print(f"  {column}: usable text in {present:,} of {len(rows):,} rows")

    print("\nDecision about unwanted columns:")
    print("  Do not delete raw columns during feature identification.")
    print("  Exclude img, Name, and price from X during feature preparation.")
    print("  Perform physical deletion or final cleaned-column selection during data cleaning and feature preparation.")


if __name__ == "__main__":
    print_feature_plan(load_rows(DATASET_PATH))
