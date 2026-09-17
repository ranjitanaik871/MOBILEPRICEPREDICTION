"""Audit the mobile phone dataset and print a dataset-understanding summary."""

import csv
import re
from collections import Counter, defaultdict
from pathlib import Path


DATASET_PATH = Path(__file__).resolve().parent / "mobile_dataset.csv"


def load_dataset(path):
    with path.open("r", encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def duplicate_groups(rows):
    groups = defaultdict(list)
    for row in rows:
        key = (row["Name"], row["price"], row["storage"])
        groups[key].append(row)
    return [(key, values) for key, values in groups.items() if len(values) > 1]


def price_band(price):
    if price < 15000:
        return "Below 15,000"
    if price < 30000:
        return "15,000-29,999"
    return "30,000 and above"


def analyze(rows):
    fields = list(rows[0])
    missing = {
        field: sum(not row[field].strip() for row in rows)
        for field in fields
    }
    prices = [int(row["price"]) for row in rows]
    scores = [int(row["Spec Score"]) for row in rows]
    bands = Counter(price_band(price) for price in prices)
    tags = Counter(row["tag"] for row in rows)
    duplicates = duplicate_groups(rows)
    numeric_prices = sum(
        bool(re.fullmatch(r"\d+(\.\d+)?", row["price"]))
        for row in rows
    )

    print("Mobile Phone Dataset Understanding Summary")
    print("=" * 44)
    print(f"Records: {len(rows):,}")
    print(f"Columns: {len(fields)}")
    print(f"Fields: {', '.join(fields)}")
    print(f"Numeric price values: {numeric_prices:,} of {len(rows):,}")
    print(f"Price range: Rs. {min(prices):,} to Rs. {max(prices):,}")
    print(f"Average price: Rs. {sum(prices) / len(prices):,.2f}")
    print(f"Spec Score range: {min(scores)} to {max(scores)}")
    print(f"Duplicate groups: {len(duplicates)}")

    print("\nMissing values:")
    for field, count in sorted(missing.items(), key=lambda item: item[1], reverse=True):
        if count:
            print(f"  {field}: {count:,} ({count / len(rows) * 100:.1f}%)")

    print("\nPrice bands:")
    for band in ("Below 15,000", "15,000-29,999", "30,000 and above"):
        print(f"  {band}: {bands[band]:,}")

    print("\nProduct status:")
    for status, count in tags.most_common():
        print(f"  {status}: {count:,}")

    print("\nDuplicate records:")
    for key, values in duplicates:
        print(f"  {key[0]} | Rs. {int(key[1]):,} | {key[2]} | {len(values)} occurrences")


if __name__ == "__main__":
    analyze(load_dataset(DATASET_PATH))
