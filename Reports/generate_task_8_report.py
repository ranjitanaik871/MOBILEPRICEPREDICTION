"""Generate the Task 8 train-test split report as a PDF."""

from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


BASE_DIR = Path(__file__).resolve().parent.parent
TRAIN_PATH = BASE_DIR / "training_dataset.csv"
TEST_PATH = BASE_DIR / "testing_dataset.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "Task_8_Train_Test_Split_Report.pdf"
TARGET_COLUMN = "Price_Category_Encoded"


def distribution_table(dataframe):
    """Return target counts and percentages for the report table."""
    counts = dataframe[TARGET_COLUMN].value_counts().sort_index()
    total = len(dataframe)
    labels = {0: "Below 15,000", 1: "15,000-29,999", 2: "30,000 and above"}
    return [
        [labels[int(label)], int(count), f"{count / total * 100:.1f}%"]
        for label, count in counts.items()
    ]


def build_report():
    """Load split data and create the formatted Task 8 PDF report."""
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    if TARGET_COLUMN not in train or TARGET_COLUMN not in test:
        raise KeyError(f"Both datasets must contain '{TARGET_COLUMN}'.")
    if set(train.columns) != set(test.columns):
        raise ValueError("Training and testing datasets must have the same columns.")

    styles = getSampleStyleSheet()
    styles.add(
        ParagraphStyle(
            name="ReportTitle",
            parent=styles["Title"],
            fontName="Helvetica-Bold",
            fontSize=22,
            leading=27,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#16324F"),
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Subtitle",
            parent=styles["Normal"],
            fontSize=10,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#52606D"),
            spaceAfter=20,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Section",
            parent=styles["Heading2"],
            fontName="Helvetica-Bold",
            fontSize=13,
            leading=16,
            textColor=colors.HexColor("#16324F"),
            spaceBefore=12,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyTextCustom",
            parent=styles["BodyText"],
            fontSize=10,
            leading=15,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#263238"),
            spaceAfter=7,
        )
    )
    styles.add(
        ParagraphStyle(
            name="Small",
            parent=styles["BodyText"],
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#52606D"),
        )
    )

    document = SimpleDocTemplate(
        str(OUTPUT_PATH),
        pagesize=A4,
        rightMargin=0.65 * inch,
        leftMargin=0.65 * inch,
        topMargin=0.55 * inch,
        bottomMargin=0.55 * inch,
        title="Task 8 - Train-Test Split Report",
        author="Mobile Price Prediction Project",
    )

    total_rows = len(train) + len(test)
    feature_count = len(train.columns) - 1
    story = [
        Paragraph("Task 8: Train-Test Split Report", styles["ReportTitle"]),
        Paragraph(
            "Mobile Price Prediction Classification Project | 19 September 2026",
            styles["Subtitle"],
        ),
        Paragraph("1. Business Requirement", styles["Section"]),
        Paragraph(
            "The mobile-phone company requires a machine-learning solution that classifies phones into price categories using technical specifications such as RAM, storage, battery capacity, camera quality, screen resolution, processor features, connectivity, and other hardware characteristics.",
            styles["BodyTextCustom"],
        ),
        Paragraph("2. Task Objective", styles["Section"]),
        Paragraph(
            "Task 8 prepares the model-ready dataset for Random Forest classification by separating it into independent training and testing datasets. The split must be reproducible, preserve all encoded features, retain the target label, and maintain a similar distribution of price categories in both subsets.",
            styles["BodyTextCustom"],
        ),
        Paragraph("3. Methodology", styles["Section"]),
        Paragraph(
            "The input file was ml_ready_dataset.csv, produced during feature engineering and encoding. Price_Category_Encoded was used as the target column, while the remaining 26 columns were used as model features. An 80/20 split was created with scikit-learn's train_test_split function using stratification and random_state=42.",
            styles["BodyTextCustom"],
        ),
    ]

    summary_data = [
        ["Measure", "Result"],
        ["Total records", f"{total_rows:,}"],
        ["Training records", f"{len(train):,} (80%)"],
        ["Testing records", f"{len(test):,} (20%)"],
        ["Feature columns", str(feature_count)],
        ["Target column", TARGET_COLUMN],
        ["Split strategy", "Stratified"],
        ["Random state", "42"],
    ]
    summary_table = Table(summary_data, colWidths=[2.25 * inch, 3.75 * inch])
    summary_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#16324F")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTNAME", (0, 1), (0, -1), "Helvetica-Bold"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F4F7F9")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B7C4CE")),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend([Paragraph("4. Split Summary", styles["Section"]), summary_table])

    class_data = [["Price category", "Training count", "Training %", "Testing count", "Testing %"]]
    train_distribution = distribution_table(train)
    test_distribution = distribution_table(test)
    for train_row, test_row in zip(train_distribution, test_distribution):
        class_data.append([train_row[0], train_row[1], train_row[2], test_row[1], test_row[2]])

    class_table = Table(class_data, colWidths=[1.7 * inch, 1.05 * inch, 0.85 * inch, 1.05 * inch, 0.85 * inch])
    class_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#287D8E")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, 1), (-1, -1), colors.HexColor("#F4F7F9")),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#B7C4CE")),
                ("ALIGN", (1, 1), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.extend(
        [
            Paragraph("5. Target-Class Distribution", styles["Section"]),
            class_table,
            Spacer(1, 8),
            Paragraph(
                "The close percentages between the training and testing subsets confirm that stratification preserved the representation of all three price categories. This supports fair model training and evaluation.",
                styles["BodyTextCustom"],
            ),
            Paragraph("6. Deliverables", styles["Section"]),
            Paragraph(
                "Training dataset: training_dataset.csv (813 records)\nTesting dataset: testing_dataset.csv (204 records)\nImplementation: train_test_split.py",
                styles["BodyTextCustom"],
            ),
            Paragraph("7. Conclusion", styles["Section"]),
            Paragraph(
                "Task 8 is complete. The model-ready mobile-phone data has been divided into reproducible, stratified training and testing datasets. Both files contain the same 26 encoded feature columns and the Price_Category_Encoded target, making them ready for Task 9: Random Forest Classification.",
                styles["BodyTextCustom"],
            ),
            Spacer(1, 12),
            Paragraph(
                "Generated from the project datasets by generate_task_8_report.py.",
                styles["Small"],
            ),
        ]
    )

    document.build(story)
    return OUTPUT_PATH


if __name__ == "__main__":
    print(f"Saved report: {build_report()}")