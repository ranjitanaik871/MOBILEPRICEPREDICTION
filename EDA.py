from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "Visual_Report"
OUTPUT_DIR.mkdir(exist_ok=True)

DATASET_CANDIDATES = [
    BASE_DIR / "train.csv",
    BASE_DIR / "cleaned_dataset.csv",
    BASE_DIR / "mobile_dataset.csv",
]


def find_dataset():
    for candidate in DATASET_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "No dataset found. Expected one of: train.csv, cleaned_dataset.csv, or mobile_dataset.csv"
    )


def load_data():
    path = find_dataset()
    df = pd.read_csv(path)
    print(f"Loaded dataset: {path.name}")
    print(f"Shape: {df.shape}")
    return df


def prepare_target_column(df):
    target_candidates = ["price_range", "Price_Category", "price_category", "target"]
    target_col = next((col for col in target_candidates if col in df.columns), None)

    if target_col is None:
        if "price" in df.columns:
            df["price_range"] = pd.cut(
                df["price"],
                bins=[0, 15000, 30000, float("inf")],
                labels=["Below 15,000", "15,000-29,999", "30,000 and above"],
                right=False,
            )
            target_col = "price_range"
        else:
            raise KeyError("No price category/price target column found in the dataset.")

    return df, target_col


def select_numeric_features(df, target_col):
    excluded = {
        target_col, "Name", "img", "price", "processor", "storage", "battery",
        "display", "camera", "sim", "memoryExternal", "version", "fm"
    }
    numeric_cols = [
        col for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]
    return numeric_cols


def basic_summary(df, target_col):
    print("\nTarget distribution:")
    print(df[target_col].value_counts().sort_index())

    print("\nMissing values per column:")
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        print("No missing values detected.")
    else:
        print(missing)

    print("\nDescriptive statistics for key numeric specifications:")
    numeric_cols = select_numeric_features(df, target_col)
    if numeric_cols:
        print(df[numeric_cols].describe().round(2))


def save_plot(fig, filename):
    path = OUTPUT_DIR / filename
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"Saved plot: {path.name}")


def correlation_heatmap(df, target_col):
    numeric_cols = select_numeric_features(df, target_col)
    if not numeric_cols:
        print("No numeric features available for correlation analysis.")
        return

    corr = df[numeric_cols].corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(14, 10))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, ax=ax)
    ax.set_title("Correlation Heatmap of Mobile Specifications")
    save_plot(fig, "correlation_heatmap.png")
    plt.close(fig)


def feature_vs_price_bar_charts(df, target_col):
    key_features = [
        "RAM_GB",
        "Storage_GB",
        "Battery_mAh",
        "Charging_W",
        "Screen_Size_Inches",
        "Rear_Camera_MP",
        "Front_Camera_MP",
        "Processor_Speed_GHz",
        "Cores",
    ]
    features_to_plot = [feature for feature in key_features if feature in df.columns]

    if not features_to_plot:
        numeric_cols = select_numeric_features(df, target_col)
        features_to_plot = numeric_cols[:5]

    for feature in features_to_plot:
        grouped = df.groupby(target_col)[feature].mean().sort_index()
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(x=grouped.index, y=grouped.values, hue=grouped.index, palette="viridis", dodge=False, ax=ax)
        ax.set_title(f"Average {feature} by {target_col}")
        ax.set_xlabel(target_col)
        ax.set_ylabel(f"Average {feature}")
        if ax.legend_ is not None:
            ax.legend_.remove()
        plt.xticks(rotation=20)
        save_plot(fig, f"{feature}_by_{target_col}.png")
        plt.close(fig)


def distribution_by_category(df, target_col):
    print("\nPrice category statistics by key mobile specs:")
    for feature in ["RAM_GB", "Storage_GB", "Battery_mAh", "Screen_Size_Inches", "Rear_Camera_MP"]:
        if feature in df.columns:
            print(f"\n{feature} mean by {target_col}:")
            print(df.groupby(target_col)[feature].mean().round(2))


def main():
    sns.set_style("whitegrid")
    df = load_data()
    df, target_col = prepare_target_column(df)

    print("\nDataset preview:")
    print(df.head())

    basic_summary(df, target_col)
    distribution_by_category(df, target_col)
    correlation_heatmap(df, target_col)
    feature_vs_price_bar_charts(df, target_col)

    print("\nEDA completed successfully.")


if __name__ == "__main__":
    main()
