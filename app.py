"""Desktop application for mobile phone price-category prediction.

The app uses a trained Random Forest model to predict the price category for a
mobile phone based on technical specifications. It rebuilds the same engineered
features used during model training and displays the prediction result in a
Tkinter window.
"""

import json
import pickle
from pathlib import Path

import pandas as pd
import tkinter as tk
from tkinter import ttk, messagebox


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "random_forest_model.pkl"
METADATA_PATH = BASE_DIR / "random_forest_metadata.json"

CLASS_LABELS = {
    0: "Below 15,000",
    1: "15,000-29,999",
    2: "30,000 and above",
}


def format_choice_for_display(field_name, raw_value):
    """Format a numeric model value as a realistic phone specification."""
    numeric_value = float(raw_value)
    value = str(raw_value)

    if field_name == "RAM_GB":
        return f"{value} GB"
    if field_name == "Storage_GB":
        if numeric_value >= 1024:
            return f"{numeric_value / 1024:g} TB"
        return f"{value} GB"
    if field_name == "Battery_mAh":
        return f"{numeric_value:,.0f} mAh"
    if field_name == "Charging_W":
        return f"{value} W"
    if field_name == "Screen_Size_Inches":
        return f"{value} in"
    if field_name == "Resolution_Width":
        resolution_labels = {
            "1080": "Full HD (1080 px)",
            "1440": "2K / QHD (1440 px)",
            "2160": "4K UHD (2160 px)",
        }
        return resolution_labels.get(value, f"{value} px")
    if field_name == "Resolution_Height":
        return f"{value} px"
    if field_name == "Refresh_Rate_Hz":
        return f"{value} Hz"
    if field_name in {"Rear_Camera_MP", "Front_Camera_MP"}:
        return f"{value} MP"
    if field_name == "Processor_Speed_GHz":
        return f"{value} GHz"
    if field_name == "Cores":
        return f"{value} cores"
    if field_name == "Android_Version":
        return f"Android {value}"
    return value


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
    if text in {"0", "false", "no", "n", "off"}:
        return 0
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


class PredictionApp:
    """Tkinter-based GUI with a form page and a dedicated result page."""

    NUMERIC_FIELDS = [
        ("RAM_GB", "RAM (GB)", ["2", "3", "4", "6", "8", "12", "16", "24", "32"]),
        ("Storage_GB", "Storage (GB)", ["32", "64", "128", "256", "512", "1024"]),
        ("Battery_mAh", "Battery (mAh)", ["2000", "2500", "3000", "3500", "4000", "4500", "5000", "5500", "6000", "6500", "7000", "8000", "10000"]),
        ("Charging_W", "Charging (W)", ["10", "18", "25", "33", "45", "65", "80", "100", "120", "150", "200", "240", "320"]),
        ("Screen_Size_Inches", "Screen Size (inches)", ["4.5", "5.0", "5.5", "6.1", "6.3", "6.5", "6.7", "6.8", "7.0", "7.6"]),
        ("Resolution_Width", "Resolution Width", ["720", "750", "1080", "1170", "1220", "1260", "1440", "1600", "1768", "1920", "2160", "2400", "2560"]),
        ("Resolution_Height", "Resolution Height", ["1280", "1334", "1440", "1560", "1600", "1792", "1920", "2160", "2340", "2400", "2412", "2520", "2640", "3120", "3840"]),
        ("Refresh_Rate_Hz", "Refresh Rate (Hz)", ["60", "90", "120", "144", "165", "185", "240"]),
        ("Rear_Camera_MP", "Rear Camera (MP)", ["2", "5", "8", "12", "16", "32", "48", "50", "64", "108", "200"]),
        ("Front_Camera_MP", "Front Camera (MP)", ["2", "5", "8", "10", "12", "13", "16", "20", "32", "40", "50", "60"]),
        ("Processor_Speed_GHz", "Processor Speed (GHz)", ["1.0", "1.5", "1.8", "2.0", "2.2", "2.4", "2.5", "2.8", "3.0", "3.2", "3.4", "3.6", "4.0", "4.47"]),
        ("Cores", "Cores", ["4", "6", "7", "8", "10", "12"]),
        ("Android_Version", "Android Version", ["8", "9", "10", "11", "12", "13", "14", "15", "16"]),
    ]

    BOOLEAN_FIELDS = [
        ("Has_3G", "Has 3G"),
        ("Has_4G", "Has 4G"),
        ("Has_5G", "Has 5G"),
        ("Has_VoLTE", "Has VoLTE"),
        ("Has_WiFi", "Has WiFi"),
        ("Has_NFC", "Has NFC"),
        ("External_Memory_Supported", "External Memory Supported"),
        ("FM_Radio", "FM Radio"),
    ]

    def __init__(self, root):
        self.root = root
        self.root.title("Mobile Price Prediction System")
        self.root.geometry("1000x680")
        self.root.minsize(900, 600)
        self.root.configure(bg="#f5f1ea")

        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(
            "App.TCombobox",
            fieldbackground="#ffffff",
            background="#ffffff",
            foreground="#172033",
            bordercolor="#cbd5e1",
            lightcolor="#cbd5e1",
            darkcolor="#cbd5e1",
            padding=5,
        )

        self.entries = {}
        self.display_to_raw = {}
        self.current_result = {}
        self.container = tk.Frame(self.root, bg="#f5f1ea")
        self.container.pack(fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        self.frames = {}
        self.build_layout()
        self.show_frame("form")

    def build_layout(self):
        """Create the form and result pages."""
        self.create_form_page()
        self.create_result_page()

    def show_frame(self, frame_name):
        """Display one frame and hide the others."""
        for name, frame in self.frames.items():
            frame.grid_forget()
        self.frames[frame_name].grid(row=0, column=0, sticky="nsew")

    def create_form_page(self):
        """Build the input form page."""
        frame = tk.Frame(self.container, bg="#f5f1ea", padx=22, pady=18)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(2, weight=1)
        self.frames["form"] = frame

        header_bar = tk.Frame(frame, bg="#24323d", padx=18, pady=12)
        header_bar.grid(row=0, column=0, sticky="ew", pady=(0, 14))
        header_bar.grid_columnconfigure(0, weight=1)

        tk.Label(
            header_bar,
            text="MOBILE PRICE PREDICTION",
            font=("Segoe UI", 18, "bold"),
            fg="#ffffff",
            bg="#24323d",
            anchor="w",
        ).grid(row=0, column=0, sticky="w")
        tk.Label(
            header_bar,
            text="Configure a device to generate its price category",
            font=("Segoe UI", 9),
            fg="#c7d2d9",
            bg="#24323d",
            anchor="e",
        ).grid(row=0, column=1, sticky="e", padx=(20, 0))

        body = tk.Frame(frame, bg="#f5f1ea")
        body.grid(row=2, column=0, sticky="nsew")
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(0, weight=3)
        body.grid_columnconfigure(1, weight=1)

        form_card = tk.Frame(body, bg="#fffdf9", padx=16, pady=14, highlightbackground="#d8d0c4", highlightthickness=1)
        form_card.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        form_card.grid_columnconfigure(0, weight=1)
        form_card.grid_rowconfigure(1, weight=1)

        form_inner = tk.Frame(form_card, bg="#fffdf9")
        form_inner.grid(row=1, column=0, sticky="nsew")
        form_columns = 3
        form_inner.grid_columnconfigure(0, weight=1)
        form_inner.grid_columnconfigure(2, weight=1)
        form_inner.grid_columnconfigure(4, weight=1)

        tk.Label(
            form_card,
            text="DEVICE SPECIFICATIONS",
            bg="#fffdf9",
            fg="#0f766e",
            font=("Segoe UI", 10, "bold"),
            anchor="w",
        ).grid(row=0, column=0, sticky="w", pady=(0, 10))

        for index, (field_name, label_text, choices) in enumerate(self.NUMERIC_FIELDS):
            row = 1 + (index // form_columns) * 2
            col = (index % form_columns) * 2

            label = tk.Label(form_inner, text=label_text, bg="#fffdf9", fg="#344054", font=("Segoe UI", 9, "bold"))
            label.grid(row=row, column=col, sticky="w", padx=(0, 12), pady=(3, 2))

            display_choices = [format_choice_for_display(field_name, choice) for choice in choices]
            self.display_to_raw[field_name] = dict(zip(display_choices, choices))
            value_var = tk.StringVar(value=display_choices[0])
            combo = ttk.Combobox(
                form_inner,
                textvariable=value_var,
                values=display_choices,
                state="readonly",
                width=18,
                justify="left",
                style="App.TCombobox",
            )
            combo.grid(row=row + 1, column=col, sticky="ew", padx=(0, 12), pady=(0, 5))
            self.entries[field_name] = value_var

        options_card = tk.Frame(body, bg="#f0f8f5", padx=16, pady=14, highlightbackground="#c9ddd7", highlightthickness=1)
        options_card.grid(row=0, column=1, sticky="nsew")
        options_card.grid_columnconfigure(0, weight=1)

        tk.Label(options_card, text="FEATURES", bg="#f0f8f5", fg="#0f766e", font=("Segoe UI", 10, "bold"), anchor="w").grid(row=0, column=0, sticky="w")
        tk.Label(options_card, text="Select supported features", bg="#f0f8f5", fg="#667085", font=("Segoe UI", 9), anchor="w").grid(row=1, column=0, sticky="w", pady=(2, 12))

        for index, (field_name, label_text) in enumerate(self.BOOLEAN_FIELDS):
            value_var = tk.BooleanVar(value=False)
            check = tk.Checkbutton(
                options_card,
                text=label_text,
                variable=value_var,
                bg="#f0f8f5",
                fg="#344054",
                activebackground="#f0f8f5",
                activeforeground="#0f766e",
                selectcolor="#dbeafe",
                font=("Segoe UI", 9),
                padx=0,
                pady=5,
                anchor="w",
            )
            check.grid(row=index + 2, column=0, sticky="w")
            self.entries[field_name] = value_var

        predict_button = tk.Button(
            options_card,
            text="CALCULATE PRICE",
            command=self.predict,
            bg="#e76f51",
            fg="white",
            font=("Segoe UI", 10, "bold"),
            padx=12,
            pady=10,
            bd=0,
            highlightthickness=0,
            width=18,
            relief="flat",
            activebackground="#c9573d",
            activeforeground="white",
        )
        predict_button.grid(row=len(self.BOOLEAN_FIELDS) + 2, column=0, sticky="ew", pady=(18, 0))

    def create_result_page(self):
        """Build the dedicated prediction result page."""
        frame = tk.Frame(self.container, bg="#f5f1ea", padx=22, pady=18)
        frame.grid(row=0, column=0, sticky="nsew")
        frame.grid_columnconfigure(0, weight=1)
        frame.grid_rowconfigure(1, weight=1)
        self.frames["result"] = frame

        title = tk.Label(
            frame,
            text="Billing Summary",
            font=("Segoe UI", 20, "bold"),
            fg="#24323d",
            bg="#f5f1ea",
            anchor="w",
        )
        title.grid(row=0, column=0, sticky="w", pady=(0, 10))

        result_box = tk.Frame(frame, bg="#24323d", padx=20, pady=18, highlightbackground="#455765", highlightthickness=1)
        result_box.grid(row=1, column=0, sticky="nsew")
        result_box.grid_columnconfigure(0, weight=1)
        result_box.grid_rowconfigure(0, weight=1)

        self.result_text = tk.Text(
            result_box,
            bg="#24323d",
            fg="#e2e8f0",
            height=16,
            width=90,
            wrap="word",
            font=("Segoe UI", 10),
        )
        self.result_text.configure(state="disabled")
        self.result_text.grid(row=0, column=0, sticky="nsew")

        self.result_label = tk.Label(
            result_box,
            text="",
            bg="#24323d",
            fg="#f8fafc",
            font=("Segoe UI", 20, "bold"),
            anchor="w",
            justify="left",
        )
        self.result_label.grid(row=1, column=0, sticky="w", pady=(18, 0))

        back_button = tk.Button(
            frame,
            text="Back",
            command=lambda: self.show_frame("form"),
            bg="#d8ebe5",
            fg="#24323d",
            font=("Segoe UI", 10, "bold"),
            padx=20,
            pady=8,
            bd=0,
            highlightthickness=0,
            relief="flat",
        )
        back_button.grid(row=2, column=0, sticky="w", pady=(12, 0))

    def collect_input(self):
        """Collect the form values into a dictionary for the model."""
        payload = {}
        for field_name, value_var in self.entries.items():
            if isinstance(value_var, tk.StringVar):
                payload[field_name] = self.display_to_raw[field_name].get(
                    value_var.get(), value_var.get()
                )
            elif isinstance(value_var, tk.BooleanVar):
                payload[field_name] = bool(value_var.get())
        return payload

    def predict(self):
        """Run the model and update the dedicated result page."""
        payload = self.collect_input()
        try:
            result = predict_from_form(payload)
        except Exception as exc:  # pragma: no cover - user-facing error handling
            messagebox.showerror("Prediction Error", f"Unable to predict price category: {exc}")
            return

        self.current_result = result
        selected_lines = ["Selected Configurations", "=" * 24]
        for field_name, value_var in self.entries.items():
            value = value_var.get()
            if isinstance(value_var, tk.BooleanVar):
                if value:
                    selected_lines.append(f"{dict(self.BOOLEAN_FIELDS)[field_name]}: Yes")
            else:
                label = next(label for name, label, _ in self.NUMERIC_FIELDS if name == field_name)
                selected_lines.append(f"{label}: {value}")

        self.result_label.config(
            text=f"Predicted Price: {result['predicted_price_category']}"
        )
        self.result_text.configure(state="normal")
        self.result_text.delete("1.0", tk.END)
        self.result_text.insert(tk.END, "\n".join(selected_lines))
        self.result_text.configure(state="disabled")
        self.show_frame("result")


def main():
    """Run the Tkinter application."""
    root = tk.Tk()
    app = PredictionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
