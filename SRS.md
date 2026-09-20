# Software Requirements Specification
## Mobile Price Prediction System

**Document version:** 1.0  
**Document status:** Final project requirements specification  
**Application type:** Python desktop machine-learning application  
**Primary interface:** Tkinter GUI  
**Target platform:** Windows desktop

## 1. Introduction

### 1.1 Purpose

This document specifies the requirements for a mobile-phone price-category prediction system. The system analyzes technical specifications of a mobile phone and classifies it into a price category using a trained Random Forest machine-learning model.

The document defines what the system must do, the data it requires, the expected outputs, quality requirements, constraints, and acceptance criteria. It is intended for the project team, reviewers, testers, and future maintainers.

### 1.2 Business Problem

A mobile-phone company needs a consistent way to estimate the likely price category of a device from its specifications. Manual comparison of RAM, storage, battery, display, camera, processor, operating-system, and connectivity features is time-consuming and may be inconsistent.

The proposed system provides a repeatable prediction to support early product analysis and pricing decisions. The result is a price category, not an exact retail price.

### 1.3 Objectives

The system shall:

- Prepare raw mobile-phone data for machine learning.
- Convert technical text fields into usable numeric and binary features.
- Create documented price categories from the source price.
- Train and save a reproducible Random Forest classifier.
- Evaluate the classifier on unseen test data.
- Provide a desktop GUI for entering phone specifications.
- Display the selected configuration and predicted price category clearly.

### 1.4 Intended Audience

- Business stakeholders and product teams
- Project team lead
- Data analysts and machine-learning developers
- Testers and project reviewers
- Future maintainers of the application

### 1.5 Definitions

| Term | Definition |
| --- | --- |
| Price category | The class predicted by the system instead of an exact price |
| Feature | A phone specification used by the model |
| Target | The price category derived from the source price |
| Training data | Data used to fit the Random Forest model |
| Testing data | Held-out data used to measure model performance |
| GUI | Graphical user interface |
| SRS | Software Requirements Specification |

## 2. Scope

### 2.1 In Scope

- Reading the raw mobile-phone CSV dataset.
- Removing duplicate records during preparation.
- Extracting numeric values from raw specification text.
- Imputing missing numeric values with column medians.
- Creating price categories using documented thresholds.
- Engineering derived specification features.
- Creating a stratified 80/20 train/test split.
- Training a Random Forest classifier.
- Saving model metadata and feature order.
- Generating evaluation metrics and a confusion matrix.
- Running predictions through a Tkinter desktop application.
- Showing a billing-style summary of selected phone specifications.

### 2.2 Out of Scope

- Exact price prediction in currency.
- Online marketplace or live price retrieval.
- Automatic collection of current prices from the internet.
- User accounts, authentication, or role-based access.
- Cloud deployment or a web API.
- Database storage of prediction history.
- Automatic model retraining from GUI submissions.
- Guaranteeing a correct prediction for every new device.

## 3. System Overview

The system has two connected parts:

1. **Offline machine-learning pipeline:** prepares data, engineers features, trains the model, and evaluates it.
2. **Desktop prediction application:** collects a phone configuration, rebuilds the same engineered features, loads the saved model, and displays the prediction.

The GUI and training pipeline must use the same feature names, feature calculations, and feature order. The saved metadata file is used to preserve the model input contract.

### 3.1 High-Level Workflow

```text
Raw CSV
  -> Data cleaning and extraction
  -> Feature engineering
  -> Stratified train/test split
  -> Random Forest training
  -> Model evaluation
  -> Tkinter prediction application
```

## 4. Functional Requirements

### 4.1 Data Preparation Requirements

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-001 | The system shall read mobile-phone records from `mobile_dataset.csv`. | High |
| FR-002 | The system shall identify and remove duplicate records during cleaning. | High |
| FR-003 | The system shall extract RAM and storage values from raw storage text. | High |
| FR-004 | The system shall extract battery capacity and charging power from battery text. | High |
| FR-005 | The system shall extract screen size, resolution width, resolution height, and refresh rate from display text. | High |
| FR-006 | The system shall extract rear-camera and front-camera megapixels from camera text. | High |
| FR-007 | The system shall extract processor speed and core count from processor text. | High |
| FR-008 | The system shall convert connectivity and device-support fields into binary values. | High |
| FR-009 | The system shall convert Android version into a numeric value. | Medium |
| FR-010 | The system shall replace missing numeric values with the corresponding column median. | High |
| FR-011 | The system shall save the cleaned data to `cleaned_dataset.csv`. | High |

### 4.2 Target and Feature Requirements

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-012 | The system shall derive the target from the raw `price` column. | High |
| FR-013 | Prices below 15,000 shall map to `Below 15,000`. | High |
| FR-014 | Prices from 15,000 through 29,999 shall map to `15,000-29,999`. | High |
| FR-015 | Prices of 30,000 or more shall map to `30,000 and above`. | High |
| FR-016 | The source price shall not be used as a model predictor. | Critical |
| FR-017 | The system shall calculate resolution megapixels, total camera megapixels, battery per screen inch, storage per RAM, and resolution aspect ratio. | High |
| FR-018 | The final model input shall contain 26 numeric features and one encoded target column in the prepared datasets. | High |

### 4.3 Training and Evaluation Requirements

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-019 | The system shall create an 80/20 train/test split. | High |
| FR-020 | The split shall be stratified by price category. | High |
| FR-021 | The split shall use random state `42` for reproducibility. | Medium |
| FR-022 | The system shall train a three-class Random Forest classifier. | High |
| FR-023 | The classifier shall use 300 estimators and random state `42`. | Medium |
| FR-024 | The system shall save the trained model to `random_forest_model.pkl`. | High |
| FR-025 | The system shall save feature names, class names, training size, and model parameters to `random_forest_metadata.json`. | High |
| FR-026 | The system shall evaluate predictions using accuracy, weighted precision, weighted recall, weighted F1-score, classification report, and confusion matrix. | High |
| FR-027 | The system shall save evaluation predictions and results to CSV and JSON output files. | High |

### 4.4 GUI Requirements

| ID | Requirement | Priority |
| --- | --- | --- |
| FR-028 | The system shall provide a Tkinter desktop interface. | High |
| FR-029 | The GUI shall provide dropdowns for numeric mobile specifications. | High |
| FR-030 | The GUI shall display units and meaningful labels, such as `GB`, `mAh`, `W`, `MP`, `GHz`, and `Hz`. | High |
| FR-031 | The GUI shall provide choices for RAM, storage, battery, charging, display, resolution, refresh rate, cameras, processor, cores, and Android version. | High |
| FR-032 | The GUI shall provide checkboxes for 3G, 4G, 5G, VoLTE, Wi-Fi, NFC, external memory, and FM radio. | High |
| FR-033 | The GUI shall convert display labels back to raw numeric values before prediction. | Critical |
| FR-034 | The GUI shall calculate engineered features in the same way as the training pipeline. | Critical |
| FR-035 | The GUI shall provide a clearly visible prediction action. | High |
| FR-036 | The GUI shall display errors without terminating unexpectedly when prediction cannot be completed. | High |
| FR-037 | The GUI shall show the result on a separate billing-summary page. | High |
| FR-038 | The billing summary shall show the selected configurations and the predicted price category at the end. | High |
| FR-039 | The GUI shall provide a Back action to return from the billing summary to the input form. | Medium |

## 5. Non-Functional Requirements

### 5.1 Usability

- The interface shall use clear labels and visible units.
- Related controls shall be grouped into device specifications and features.
- The prediction action shall be easy to locate.
- The result shall be readable without exposing internal model arrays or debug output.
- The user shall not need to type raw `True` or `False` values.

### 5.2 Performance

- A single GUI prediction should complete within a few seconds on a normal development computer after the model has loaded.
- The GUI shall remain responsive during ordinary form interaction.
- The saved model shall be loaded from the local project directory.

### 5.3 Reliability

- The application shall validate that the model and metadata files exist before prediction.
- The application shall preserve the expected feature order from model metadata.
- Missing engineered columns shall not silently change the intended model schema.
- Training and evaluation scripts shall fail with clear errors when required columns or targets are missing.

### 5.4 Reproducibility

- Data splitting and model training shall use documented random states.
- The model feature names and class mappings shall be saved with the model metadata.
- Regenerating the pipeline from the same source data and code should produce comparable outputs.

### 5.5 Maintainability

- Data preparation, feature engineering, training, evaluation, and GUI logic shall remain in separate modules.
- Paths shall be resolved relative to the project directory.
- The README shall document the execution order and generated files.

## 6. Data Requirements

### 6.1 Source Data

The primary source file is `mobile_dataset.csv`. The source contains raw phone information, including specification text, connectivity information, product information, and price.

### 6.2 Prepared Data

The preparation process currently produces:

- `cleaned_dataset.csv`: extracted numeric and binary features plus `Price_Category`.
- `ml_ready_dataset.csv`: engineered model features plus `Price_Category_Encoded`.
- `training_dataset.csv`: 80% stratified training data.
- `testing_dataset.csv`: 20% stratified testing data.

### 6.3 Data Quality Rules

- Required source columns must be present.
- Numeric extraction failures become missing values and are imputed with medians.
- Duplicate rows are removed during cleaning.
- The target must contain at least two classes for training.
- Training and testing predictors must contain no missing values.
- Evaluation features must match the saved model feature order.

## 7. External Interfaces

### 7.1 User Interface

The primary user interface is the Tkinter desktop application in `app.py`. It has:

- A phone-specification input page.
- Dropdown controls for numeric specifications.
- Checkbox controls for binary features.
- A prediction button.
- A separate billing-summary result page.
- A Back button.

### 7.2 File Interfaces

| File | Interface role |
| --- | --- |
| `mobile_dataset.csv` | Raw input data |
| `random_forest_model.pkl` | Serialized model input/output interface |
| `random_forest_metadata.json` | Feature and class contract |
| `price_category_predictions.csv` | Evaluation prediction output |
| `price_category_evaluation_results.json` | Evaluation metrics output |
| `price_category_confusion_matrix.csv` | Confusion matrix output |

## 8. Use Cases

### UC-001: Generate a Price Category

**Actor:** Product analyst  
**Preconditions:** The model and metadata files exist.  
**Main flow:**

1. The user starts the desktop application.
2. The user selects phone specifications from the dropdowns.
3. The user selects supported connectivity and device features.
4. The user selects Calculate Price.
5. The system converts the selections into the model feature schema.
6. The system loads the Random Forest model.
7. The system generates a price-category prediction.
8. The system displays a billing summary with the selected configuration.
9. The system displays the predicted price category as the final result.

**Alternative flow:** If prediction fails, the system displays a prediction error message and keeps the input form available.

### UC-002: Rebuild the Model Pipeline

**Actor:** Developer or data analyst  
**Preconditions:** Python dependencies and the raw CSV are available.  
**Main flow:**

1. Run data understanding and target identification scripts.
2. Run data cleaning.
3. Run feature engineering.
4. Run the train/test split.
5. Train the Random Forest model.
6. Evaluate the model.
7. Review saved metrics and confusion matrix.

## 9. Acceptance Criteria

The project is acceptable when all of the following are true:

- The raw dataset can be cleaned without an unhandled exception.
- The cleaned dataset contains the documented target column.
- The model-ready dataset contains 26 predictor features and an encoded target.
- The train/test split is stratified and reproducible.
- The model trains and saves successfully.
- Evaluation produces metrics, predictions, and a confusion matrix.
- The testing feature columns match the model metadata.
- The GUI starts successfully after the model files are available.
- The GUI accepts dropdown and checkbox selections.
- The GUI converts formatted values into numeric model inputs.
- The GUI displays a separate billing summary.
- The billing summary includes selected configurations and a final predicted price category.
- The project documentation explains setup, execution order, outputs, and limitations.

## 10. Constraints and Limitations

- The system predicts a category and does not estimate an exact price.
- The verified held-out accuracy is 85.29%; a 99% result must not be claimed without new evidence.
- Model quality depends on the coverage and quality of the source dataset.
- Current GUI dropdowns provide practical phone values rather than every unusual raw dataset outlier.
- The application is a local desktop tool and does not include authentication, remote access, or database persistence.
- Retraining is a developer workflow and is not available from the GUI.

## 11. Traceability Matrix

| Business need | Implemented by | Evidence |
| --- | --- | --- |
| Classify phones by price category | Target creation and Random Forest training | `target_identification.py`, `random_forest_classification.py` |
| Analyze technical specifications | Cleaning and feature engineering | `data_cleaning_preparation.py`, `feature_engineering_encoding.py` |
| Evaluate prediction quality | Test evaluation and confusion matrix | `price_category_prediction_evaluation.py` |
| Provide a usable business tool | Tkinter prediction interface | `app.py` |
| Show the final phone-price decision | Billing summary result page | `app.py` |
| Support repeatable project execution | Fixed seeds and documented scripts | `train_test_split.py`, `random_forest_classification.py`, `README.md` |

## 12. Approval

This SRS is ready to serve as the final requirements document for the implemented Mobile Price Prediction project. Any future change to the dataset, model, GUI inputs, price thresholds, or output behavior should be reviewed against this document and recorded as a new version.
