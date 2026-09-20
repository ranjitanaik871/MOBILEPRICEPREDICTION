"""Web app for mobile phone price-category prediction.

Task TA-010 uses the trained Random Forest model to forecast the price
category from a user-provided set of mobile phone specifications.
The app rebuilds the same engineered features used during model training,
then returns the predicted class label and confidence values.
"""

import json
from pathlib import Path

import pandas as pd
from flask import Flask, jsonify, render_template_string, request
import pickle


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "random_forest_model.pkl"
METADATA_PATH = BASE_DIR / "random_forest_metadata.json"

CLASS_LABELS = {
    0: "Below 15,000",
    1: "15,000-29,999",
    2: "30,000 and above",
}

app = Flask(__name__)


def load_model():
    """Load the trained Random Forest classifier."""
    with MODEL_PATH.open("rb") as model_file:
        return pickle.load(model_file)


def load_expected_feature_columns():
    """Read the model feature names from the saved metadata."""
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return metadata.get("feature_names", [])


def coerce_float(value, default=0.0):
    """Safely convert a form value to float."""
    if value in (None, ""):
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def coerce_bool(value):
    """Convert common string/number input values to 0/1."""
    if value is None:
        return 0
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return 1 if int(value) == 1 else 0
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return 1
    return 0


def build_feature_row(form_data):
    """Convert raw user input into the exact feature layout the model expects."""
    data = {
        "RAM_GB": coerce_float(form_data.get("RAM_GB")),
        "Storage_GB": coerce_float(form_data.get("Storage_GB")),
        "Battery_mAh": coerce_float(form_data.get("Battery_mAh")),
        "Charging_W": coerce_float(form_data.get("Charging_W")),
        "Screen_Size_Inches": coerce_float(form_data.get("Screen_Size_Inches")),
        "Resolution_Width": coerce_float(form_data.get("Resolution_Width")),
        "Resolution_Height": coerce_float(form_data.get("Resolution_Height")),
        "Refresh_Rate_Hz": coerce_float(form_data.get("Refresh_Rate_Hz")),
        "Rear_Camera_MP": coerce_float(form_data.get("Rear_Camera_MP")),
        "Front_Camera_MP": coerce_float(form_data.get("Front_Camera_MP")),
        "Processor_Speed_GHz": coerce_float(form_data.get("Processor_Speed_GHz")),
        "Cores": coerce_float(form_data.get("Cores")),
        "Has_3G": coerce_bool(form_data.get("Has_3G")),
        "Has_4G": coerce_bool(form_data.get("Has_4G")),
        "Has_5G": coerce_bool(form_data.get("Has_5G")),
        "Has_VoLTE": coerce_bool(form_data.get("Has_VoLTE")),
        "Has_WiFi": coerce_bool(form_data.get("Has_WiFi")),
        "Has_NFC": coerce_bool(form_data.get("Has_NFC")),
        "External_Memory_Supported": coerce_bool(form_data.get("External_Memory_Supported")),
        "Android_Version": coerce_float(form_data.get("Android_Version")),
        "FM_Radio": coerce_bool(form_data.get("FM_Radio")),
    }

    engineered = pd.DataFrame([data])
    engineered["Resolution_Megapixels"] = (
        engineered["Resolution_Width"] * engineered["Resolution_Height"] / 1_000_000
    )
    engineered["Total_Camera_MP"] = (
        engineered["Rear_Camera_MP"] + engineered["Front_Camera_MP"]
    )
    engineered["Battery_per_Screen_Inch"] = (
        engineered["Battery_mAh"] / engineered["Screen_Size_Inches"].replace(0, pd.NA)
    )
    engineered["Storage_per_RAM_GB"] = (
        engineered["Storage_GB"] / engineered["RAM_GB"].replace(0, pd.NA)
    )
    engineered["Resolution_Aspect_Ratio"] = (
        engineered["Resolution_Width"] / engineered["Resolution_Height"].replace(0, pd.NA)
    )

    row = {}
    for column_name in [
        "RAM_GB",
        "Storage_GB",
        "Battery_mAh",
        "Charging_W",
        "Screen_Size_Inches",
        "Resolution_Width",
        "Resolution_Height",
        "Refresh_Rate_Hz",
        "Rear_Camera_MP",
        "Front_Camera_MP",
        "Processor_Speed_GHz",
        "Cores",
        "Has_3G",
        "Has_4G",
        "Has_5G",
        "Has_VoLTE",
        "Has_WiFi",
        "Has_NFC",
        "External_Memory_Supported",
        "Android_Version",
        "FM_Radio",
        "Resolution_Megapixels",
        "Total_Camera_MP",
        "Battery_per_Screen_Inch",
        "Storage_per_RAM_GB",
        "Resolution_Aspect_Ratio",
    ]:
        row[f"numeric__{column_name}"] = engineered[column_name].iloc[0]

    return pd.DataFrame([row])


def predict_from_form(form_data):
    """Run prediction for one form submission and return the result payload."""
    model = load_model()
    feature_frame = build_feature_row(form_data)
    expected_columns = load_expected_feature_columns()

    if not expected_columns:
        expected_columns = list(model.feature_names_in_)

    for column_name in expected_columns:
        if column_name not in feature_frame.columns:
            feature_frame[column_name] = 0.0

    feature_frame = feature_frame[expected_columns]
    predicted_code = int(model.predict(feature_frame)[0])
    probabilities = model.predict_proba(feature_frame)[0]

    probability_map = {
        str(index): float(probability)
        for index, probability in enumerate(probabilities)
    }

    return {
        "predicted_price_category": CLASS_LABELS.get(predicted_code, str(predicted_code)),
        "predicted_code": predicted_code,
        "probabilities": probability_map,
    }


@app.route("/", methods=["GET", "POST"])
def index():
    """Render a simple prediction form and handle submissions."""
    if request.method == "POST":
        result = predict_from_form(request.form)
        return render_template_string(
            """
            <!doctype html>
            <html>
            <head>
                <title>Mobile Price Prediction</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 30px; }
                    .card { max-width: 720px; margin: auto; padding: 24px; border: 1px solid #ccc; border-radius: 12px; }
                    .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
                    label { display: block; margin-bottom: 8px; }
                    input, select { width: 100%; padding: 8px; margin-bottom: 12px; box-sizing: border-box; }
                    button { padding: 10px 18px; background: #1f6feb; color: white; border: none; border-radius: 8px; cursor: pointer; }
                    .result { margin-top: 20px; padding: 16px; background: #f2f7ff; border-radius: 8px; }
                </style>
            </head>
            <body>
                <div class="card">
                    <h2>Mobile Price Category Prediction</h2>
                    <div class="result">
                        <p><strong>Prediction:</strong> {{ result['predicted_price_category'] }}</p>
                        <p><strong>Class probabilities:</strong></p>
                        <ul>
                            {% for code, probability in result['probabilities'].items() %}
                                <li>{{ code }} : {{ '%.2f' % (probability * 100) }}%</li>
                            {% endfor %}
                        </ul>
                    </div>
                    <p><a href="/">Back to form</a></p>
                </div>
            </body>
            </html>
            """,
            result=result,
        )

    return render_template_string(
        """
        <!doctype html>
        <html>
        <head>
            <title>Mobile Price Prediction</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 30px; }
                .card { max-width: 900px; margin: auto; padding: 24px; border: 1px solid #ccc; border-radius: 12px; }
                .row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }
                label { display: block; margin-bottom: 8px; }
                input, select { width: 100%; padding: 8px; margin-bottom: 12px; box-sizing: border-box; }
                button { padding: 10px 18px; background: #1f6feb; color: white; border: none; border-radius: 8px; cursor: pointer; }
            </style>
        </head>
        <body>
            <div class="card">
                <h2>Predict Mobile Price Category</h2>
                <form method="post">
                    <div class="row">
                        <div><label>RAM (GB)<input name="RAM_GB" type="number" step="0.1" value="8" required></label></div>
                        <div><label>Storage (GB)<input name="Storage_GB" type="number" step="0.1" value="128" required></label></div>
                        <div><label>Battery (mAh)<input name="Battery_mAh" type="number" step="1" value="4500" required></label></div>
                        <div><label>Charging (W)<input name="Charging_W" type="number" step="0.1" value="45" required></label></div>
                        <div><label>Screen Size (inches)<input name="Screen_Size_Inches" type="number" step="0.1" value="6.7" required></label></div>
                        <div><label>Resolution Width<input name="Resolution_Width" type="number" step="1" value="1080" required></label></div>
                        <div><label>Resolution Height<input name="Resolution_Height" type="number" step="1" value="2412" required></label></div>
                        <div><label>Refresh Rate (Hz)<input name="Refresh_Rate_Hz" type="number" step="1" value="120" required></label></div>
                        <div><label>Rear Camera (MP)<input name="Rear_Camera_MP" type="number" step="0.1" value="50" required></label></div>
                        <div><label>Front Camera (MP)<input name="Front_Camera_MP" type="number" step="0.1" value="32" required></label></div>
                        <div><label>Processor Speed (GHz)<input name="Processor_Speed_GHz" type="number" step="0.1" value="3.2" required></label></div>
                        <div><label>Cores<input name="Cores" type="number" step="1" value="7" required></label></div>
                        <div><label>Has 3G<select name="Has_3G"><option value="1" selected>Yes</option><option value="0">No</option></select></label></div>
                        <div><label>Has 4G<select name="Has_4G"><option value="1" selected>Yes</option><option value="0">No</option></select></label></div>
                        <div><label>Has 5G<select name="Has_5G"><option value="1" selected>Yes</option><option value="0">No</option></select></label></div>
                        <div><label>Has VoLTE<select name="Has_VoLTE"><option value="1" selected>Yes</option><option value="0">No</option></select></label></div>
                        <div><label>Has WiFi<select name="Has_WiFi"><option value="1" selected>Yes</option><option value="0">No</option></select></label></div>
                        <div><label>Has NFC<select name="Has_NFC"><option value="0">No</option><option value="1" selected>Yes</option></select></label></div>
                        <div><label>External Memory Supported<select name="External_Memory_Supported"><option value="0" selected>No</option><option value="1">Yes</option></select></label></div>
                        <div><label>Android Version<input name="Android_Version" type="number" step="0.1" value="13" required></label></div>
                        <div><label>FM Radio<select name="FM_Radio"><option value="1">Yes</option><option value="0" selected>No</option></select></label></div>
                    </div>
                    <button type="submit">Predict Price Category</button>
                </form>
            </div>
        </body>
        </html>
        """
    )


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Return the prediction as JSON for API usage."""
    payload = request.get_json(silent=True) or request.form.to_dict(flat=True)
    result = predict_from_form(payload)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
