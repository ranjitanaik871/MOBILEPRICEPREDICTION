# Mobile Price Prediction

A machine-learning project that classifies mobile phones into price categories from their technical specifications. The project includes data preparation, exploratory analysis, feature engineering, a Random Forest classifier, evaluation outputs, and a Tkinter desktop application for interactive prediction.

The formal requirements document is available in [SRS.md](SRS.md).

## Business Objective

A mobile-phone company needs a repeatable way to classify devices into practical price bands before launch or sale. The system uses hardware, display, camera, processor, operating-system, and connectivity information to predict one of these categories:

| Encoded class | Price category |
| ---: | --- |
| 0 | Below 15,000 |
| 1 | 15,000-29,999 |
| 2 | 30,000 and above |

The category is derived from the source `price` column. The source price is never passed to the model as a predictor, which prevents target leakage.

## Project Status

The implemented pipeline is working and internally consistent:

- Raw dataset: 1,019 rows and 15 columns
- Cleaned dataset: 1,017 rows and 22 columns
- Duplicate rows removed: 2
- Model-ready dataset: 1,017 rows, including 26 predictor features and the encoded target
- Training set: 813 rows
- Testing set: 204 rows
- Missing values after cleaning and feature preparation: 0
- Saved model and metadata feature columns match the testing data

The current evaluation is useful for this project, but it is not a guarantee that every future phone will be classified correctly. The test accuracy is about 85.3%, so predictions should support business decisions rather than replace product or pricing review.

## Model Performance

The saved evaluation in `price_category_evaluation_results.json` reports results on the held-out testing set:

| Metric | Score |
| --- | ---: |
| Accuracy | 85.29% |
| Weighted precision | 85.38% |
| Weighted recall | 85.29% |
| Weighted F1-score | 85.32% |

Per-class F1 scores:

- Below 15,000: 90.60%
- 15,000-29,999: 80.00%
- 30,000 and above: 86.52%

The middle price category is the most difficult class in the current test results. This is expected because phones near the category boundaries can have similar specifications.

## Data and Features

The raw file is `mobile_dataset.csv`. Its text fields are converted into numeric or binary model features by `data_cleaning_preparation.py`.

### Extracted features

- RAM and internal storage in GB
- Battery capacity in mAh
- Charging power in watts
- Screen size in inches
- Display resolution width and height
- Refresh rate in Hz
- Rear and front camera megapixels
- Processor speed in GHz
- Processor core count
- 3G, 4G, 5G, VoLTE, Wi-Fi, NFC, external-memory, and FM-radio flags
- Android version

### Engineered features

The feature engineering step adds:

- `Resolution_Megapixels`
- `Total_Camera_MP`
- `Battery_per_Screen_Inch`
- `Storage_per_RAM_GB`
- `Resolution_Aspect_Ratio`

The final model therefore uses 26 numeric features. Raw text fields, image URLs, product names, and the source price are excluded from the model inputs.

## Processing Workflow

Run the scripts in this order from the project directory:

```text
mobile_dataset.csv
	|
	v
data_cleaning_preparation.py  -> cleaned_dataset.csv
	|
	v
feature_engineering_encoding.py -> ml_ready_dataset.csv
	|
	v
train_test_split.py -> training_dataset.csv, testing_dataset.csv
	|
	v
random_forest_classification.py -> random_forest_model.pkl, random_forest_metadata.json
	|
	v
price_category_prediction_evaluation.py
	-> price_category_predictions.csv
	-> price_category_evaluation_results.json
	-> price_category_confusion_matrix.csv
```

`dataset_understanding.py`, `feature_identification.py`, and `target_identification.py` are analysis and documentation utilities. `EDA.py` creates charts in `Visual_Report/` and can be run after the cleaned dataset has been generated.

## Setup

Use Python 3.10 or later. Install the required packages:

```bash
pip install pandas scikit-learn matplotlib seaborn
```

Tkinter is included with standard Python on Windows. On Linux, install the distribution package that provides Tkinter if it is not already available.

## Run the Project

From `d:\Internship\MOBILEPRICEPREDICTION` on Windows, or from the equivalent project directory on another operating system:

```bash
python dataset_understanding.py
python target_identification.py
python feature_identification.py
python data_cleaning_preparation.py
python EDA.py
python feature_engineering_encoding.py
python train_test_split.py
python random_forest_classification.py
python price_category_prediction_evaluation.py
python app.py
```

For normal use, the existing generated model files allow the application to be started directly:

```bash
python app.py
```

## Desktop Application

The Tkinter application provides:

- Human-readable dropdowns with units such as `8 GB`, `1 TB`, `5,000 mAh`, `65 W`, and `2K / QHD`
- Realistic mobile specification choices
- Checkboxes for supported connectivity and device features
- A prediction action that uses the same engineered features as model training
- A separate billing-summary page
- A summary of the selected configuration
- The predicted price category shown as the final result

The labels shown in the GUI are converted back to numeric values before prediction. For example, `1 TB` is passed to the model as `1024` GB and `2K / QHD` is passed as a 1440-pixel resolution value.

## Important Files

| File | Purpose |
| --- | --- |
| `app.py` | Tkinter interface and interactive prediction logic |
| `mobile_dataset.csv` | Raw source dataset |
| `data_cleaning_preparation.py` | Extracts and cleans specification values |
| `cleaned_dataset.csv` | Cleaned feature table with target category |
| `feature_engineering_encoding.py` | Creates derived features and encoded output |
| `ml_ready_dataset.csv` | Model-ready feature table |
| `train_test_split.py` | Creates reproducible stratified train/test data |
| `training_dataset.csv` | Data used for model training |
| `testing_dataset.csv` | Held-out evaluation data |
| `random_forest_classification.py` | Trains and saves the Random Forest model |
| `random_forest_model.pkl` | Serialized trained model |
| `random_forest_metadata.json` | Feature order, class mapping, and training metadata |
| `price_category_prediction_evaluation.py` | Calculates metrics and confusion matrix |
| `price_category_evaluation_results.json` | Saved evaluation results |
| `price_category_predictions.csv` | Actual and predicted test categories |
| `price_category_confusion_matrix.csv` | Saved confusion matrix |
| `EDA.py` | Creates exploratory charts in `Visual_Report/` |

## Reproducibility

- Train/test split random state: `42`
- Random Forest estimators: `300`
- Random Forest random state: `42`
- Test size: `20%`
- Split strategy: stratified by price category

If the source data or preprocessing code changes, regenerate all derived CSV files, retrain the model, and rerun evaluation before using the GUI. The application loads the saved model metadata to preserve the required feature order.

## Known Limitations

- The classifier predicts categories, not an exact phone price.
- The current test accuracy is 85.29%, not 99%; improvement would require better data quality, more representative current-phone records, and further model validation.
- The dataset contains unusual specification outliers. The GUI intentionally presents practical phone values instead of every raw outlier.
- The saved evaluation JSON may contain absolute paths from the machine where it was generated. Those paths do not affect model prediction; rerunning evaluation refreshes them for the current environment.
- `feature_identification.py` contains a broader planning list than the implemented cleaned pipeline. The actual trained model uses the 26 features recorded in `random_forest_metadata.json`.