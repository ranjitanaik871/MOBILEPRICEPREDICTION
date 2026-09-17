"""Identify and document the target variable for price classification."""

import csv
from collections import Counter
from pathlib import Path


DATASET_PATH = Path(__file__).resolve().parent / "mobile_dataset.csv"
TARGET_SOURCE_COLUMN = "price"
TARGET_COLUMN = "Price_Category"


def load_rows(path):
    """Load the mobile phone records from a CSV file."""
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def price_category(price):
    """Convert a phone price into the project's classification category."""
    if price < 15000:
        return "Below 15,000"
    if price < 30000:
        return "15,000-29,999"
    return "30,000 and above"


def identify_target(rows):
    """Print the target-variable decision and category distribution."""
    if not rows:
        raise ValueError("The dataset is empty; a target variable cannot be identified.")

    fields = list(rows[0])
    if TARGET_SOURCE_COLUMN not in fields:
        raise KeyError(f"Required source column '{TARGET_SOURCE_COLUMN}' was not found.")

    prices = [int(row[TARGET_SOURCE_COLUMN]) for row in rows]
    categories = [price_category(price) for price in prices]
    distribution = Counter(categories)

    print("Mobile Phone Target Variable Identification")
    print("=" * 45)
    print(f"Dataset rows reviewed: {len(rows):,}")
    print(f"Source column: {TARGET_SOURCE_COLUMN}")
    print(f"Target column for classification: {TARGET_COLUMN}")
    print("Target type: Categorical (derived from the numeric price column)")
    print("Independent variables: Mobile phone specifications and connectivity features")
    print("Excluded from model inputs: price, because it is used to create the target")

    print("\nPrice-category rule:")
    print("  Below 15,000      -> Below 15,000")
    print("  15,000 to 29,999  -> 15,000-29,999")
    print("  30,000 and above  -> 30,000 and above")

    print("\nTarget-class distribution:")
    for category in ("Below 15,000", "15,000-29,999", "30,000 and above"):
        count = distribution[category]
        print(f"  {category}: {count:,} ({count / len(rows) * 100:.1f}%)")

    return categories


if __name__ == "__main__":
    identify_target(load_rows(DATASET_PATH))