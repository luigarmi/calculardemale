from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib
import pandas as pd
from flask import Flask, jsonify, render_template, request
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from class_info import CLASS_DETAILS, get_class_detail
from feature_catalog import get_feature_schema

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"

MODEL_LABELS = {
    "logistic_regression": "Regresion Logistica",
    "mlp_classifier": "Red Neuronal (MLP)",
}


def load_assets() -> Dict[str, object]:
    if not METRICS_PATH.exists():
        raise FileNotFoundError(
            "Metrics file not found. Run 'python train_models.py' first."
        )

    with METRICS_PATH.open("r", encoding="utf-8") as f:
        metrics_doc = json.load(f)

    feature_order: List[str] = metrics_doc.get("feature_order", [])
    class_labels: List[int] = metrics_doc.get("class_labels", [])

    models: Dict[str, object] = {}
    model_metrics: Dict[str, dict] = {}

    for model_key in MODEL_LABELS:
        model_path = MODELS_DIR / f"{model_key}.joblib"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing model artifact at {model_path}")
        models[model_key] = joblib.load(model_path)
        model_metrics[model_key] = metrics_doc["models"].get(model_key, {})

    feature_schema = get_feature_schema(feature_order)
    class_metadata = []
    if class_labels:
        for class_id in class_labels:
            detail = get_class_detail(class_id)
            class_metadata.append(
                {
                    "id": int(class_id),
                    "label": detail["label"],
                    "description": detail["description"],
                }
            )
    else:
        for class_id, detail in CLASS_DETAILS.items():
            class_metadata.append(
                {
                    "id": int(class_id),
                    "label": detail["label"],
                    "description": detail["description"],
                }
            )

    return {
        "feature_order": feature_order,
        "feature_schema": feature_schema,
        "models": models,
        "model_metrics": model_metrics,
        "class_labels": class_labels,
        "class_metadata": class_metadata,
        "dataset": metrics_doc.get("dataset"),
        "dataset_description": metrics_doc.get("dataset_description"),
        "target_column": metrics_doc.get("target_column"),
    }


ASSETS = load_assets()
TARGET_COL = ASSETS.get("target_column", "outcome")

app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/status")
def api_status():
    payload = {
        "feature_order": ASSETS["feature_order"],
        "feature_schema": ASSETS.get("feature_schema"),
        "models": [
            {"id": key, "label": MODEL_LABELS[key], "metrics": ASSETS["model_metrics"][key]}
            for key in MODEL_LABELS
        ],
        "class_labels": ASSETS.get("class_labels", []),
        "class_metadata": ASSETS.get("class_metadata", []),
        "dataset": {
            "id": ASSETS.get("dataset"),
            "description": ASSETS.get("dataset_description"),
            "target_column": TARGET_COL,
        },
    }
    return jsonify(payload)


def parse_features(payload: dict) -> pd.DataFrame:
    if "features" not in payload:
        raise ValueError("Missing 'features' field in request body.")

    feature_order: List[str] = ASSETS["feature_order"]
    features = payload["features"]
    missing = [f for f in feature_order if f not in features]
    if missing:
        raise ValueError(f"Missing required features: {', '.join(missing)}")

    schema_by_id = ASSETS.get("feature_schema", {}).get("by_id", {})

    def format_bound(value: float | None, feature_type: str) -> str:
        if value is None:
            return "N/A"
        if feature_type in {"binary", "integer"}:
            return str(int(round(value)))
        return f"{value:.2f}"

    def compute_bounds(meta: Dict[str, object]) -> Tuple[float | None, float | None]:
        allowed = meta.get("allowed_values")
        if allowed:
            return float(min(allowed)), float(max(allowed))

        min_raw = meta.get("min")
        max_raw = meta.get("max")
        feature_type = meta.get("type", "number")
        strict_bounds = bool(meta.get("strict_bounds"))

        if min_raw is None and max_raw is None:
            return (0.0 if feature_type in {"integer", "number"} else None, None)

        if strict_bounds:
            min_val = float(min_raw) if min_raw is not None else None
            max_val = float(max_raw) if max_raw is not None else None
            if min_val is not None and min_val < 0 and feature_type in {"integer", "number"}:
                min_val = 0.0
            return min_val, max_val

        if min_raw is not None and max_raw is not None:
            span = max_raw - min_raw
            padding = span * 0.1 if span else max(abs(max_raw) * 0.1, 0.1)
            min_bound = min_raw - padding
            max_bound = max_raw + padding
        else:
            min_bound = min_raw
            max_bound = max_raw

        if min_bound is not None:
            min_bound = max(0.0, min_bound)

        return min_bound, max_bound

    def normalize_binary(raw_value, label: str) -> int:
        if isinstance(raw_value, str):
            token = raw_value.strip().lower()
            if token in {"1", "si", "s", "true", "t", "yes", "y"}:
                return 1
            if token in {"0", "no", "n", "false", "f"}:
                return 0
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"'{label}' solo acepta 0 o 1.") from exc
        coerced = int(round(numeric_value))
        if coerced not in {0, 1}:
            raise ValueError(f"'{label}' solo acepta 0 o 1.")
        return coerced

    def normalize_integer(raw_value, label: str) -> int:
        try:
            numeric_value = float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"'{label}' debe ser un numero entero.") from exc
        coerced = int(round(numeric_value))
        if abs(coerced - numeric_value) > 1e-6:
            raise ValueError(f"'{label}' debe ser un numero entero.")
        return coerced

    def normalize_float(raw_value, label: str) -> float:
        try:
            return float(raw_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"'{label}' debe ser numerico.") from exc

    row: Dict[str, float] = {}
    for feature_id in feature_order:
        raw_value = features[feature_id]
        meta = schema_by_id.get(feature_id, {})
        label = meta.get("label", feature_id)
        feature_type = meta.get("type", "number")

        if feature_type == "binary":
            value = normalize_binary(raw_value, label)
        elif feature_type == "integer":
            value = normalize_integer(raw_value, label)
        else:
            value = normalize_float(raw_value, label)

        min_allowed, max_allowed = compute_bounds(meta)
        if min_allowed is not None and value < min_allowed:
            raise ValueError(
                f"'{label}' esta fuera de rango. Minimo permitido: {format_bound(min_allowed, feature_type)}."
            )
        if max_allowed is not None and value > max_allowed:
            raise ValueError(
                f"'{label}' esta fuera de rango. Maximo permitido: {format_bound(max_allowed, feature_type)}."
            )

        row[feature_id] = value

    return pd.DataFrame([row])


def get_model(model_key: str):
    if model_key not in MODEL_LABELS:
        raise KeyError(f"Model '{model_key}' is not available.")
    return ASSETS["models"][model_key]


@app.post("/api/predict/individual")
def predict_individual():
    payload = request.get_json(silent=True)
    if payload is None:
        return jsonify({"error": "Invalid JSON body."}), 400

    model_key = payload.get("model", "logistic_regression")
    try:
        model = get_model(model_key)
        features_df = parse_features(payload)
    except (KeyError, ValueError) as error:
        return jsonify({"error": str(error)}), 400

    prediction = model.predict(features_df)[0]
    probability_payload = None
    class_labels = list(map(int, getattr(model, "classes_", ASSETS.get("class_labels", []))))
    class_metadata_map = {
        int(item["id"]): {"label": item["label"], "description": item["description"]}
        for item in ASSETS.get("class_metadata", [])
    }
    if hasattr(model, "predict_proba"):
        proba_values = model.predict_proba(features_df)[0]
        probability_payload = [
            {
                "id": int(label),
                "label": class_metadata_map.get(int(label), {}).get("label", str(label)),
                "description": class_metadata_map.get(int(label), {}).get("description", ""),
                "probability": float(prob),
            }
            for label, prob in zip(class_labels, proba_values)
        ]

    prediction_detail = class_metadata_map.get(int(prediction), {"label": str(prediction), "description": ""})

    response = {
        "model": {
            "id": model_key,
            "label": MODEL_LABELS.get(model_key, model_key),
        },
        "prediction": int(prediction),
        "prediction_detail": {
            "id": int(prediction),
            "label": prediction_detail.get("label"),
            "description": prediction_detail.get("description"),
        },
        "probabilities": probability_payload,
        "class_labels": class_labels,
        "class_metadata": ASSETS.get("class_metadata", []),
    }
    return jsonify(response)


def load_uploaded_dataframe(file_storage) -> pd.DataFrame:
    filename = file_storage.filename or ""
    suffix = filename.lower().split(".")[-1]
    buffer = io.BytesIO(file_storage.read())
    buffer.seek(0)
    if suffix == "csv":
        df = pd.read_csv(buffer)
    elif suffix in {"xlsx", "xls"}:
        df = pd.read_excel(buffer)
    else:
        raise ValueError("Unsupported file format. Use .csv or .xlsx files.")
    return df


def generate_confusion_matrix_image(cm, labels: List[int]) -> str:
    fig, ax = plt.subplots(figsize=(4, 4))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap="Blues", ax=ax, colorbar=False)
    plt.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150)
    plt.close(fig)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("ascii")


@app.post("/api/predict/batch")
def predict_batch():
    if "file" not in request.files:
        return jsonify({"error": "Missing file upload named 'file'."}), 400
    model_key = request.form.get("model", "logistic_regression")
    try:
        model = get_model(model_key)
    except KeyError as error:
        return jsonify({"error": str(error)}), 400

    uploaded_file = request.files["file"]

    try:
        df = load_uploaded_dataframe(uploaded_file)
    except ValueError as error:
        return jsonify({"error": str(error)}), 400

    feature_order: List[str] = ASSETS["feature_order"]

    missing_features = [f for f in feature_order if f not in df.columns]
    if missing_features:
        return (
            jsonify(
                {
                    "error": "Missing required feature columns.",
                    "missing_features": missing_features,
                }
            ),
            400,
        )

    if TARGET_COL not in df.columns:
        return (
            jsonify(
                {
                    "error": f"The uploaded file must include the '{TARGET_COL}' column.",
                }
            ),
            400,
        )

    X = df[feature_order]
    y_true = df[TARGET_COL].astype(int)

    predictions = model.predict(X)

    model_labels = list(
        map(int, getattr(model, "classes_", sorted(y_true.unique())))
    )
    cm = confusion_matrix(y_true, predictions, labels=model_labels)
    metrics = {
        "accuracy": accuracy_score(y_true, predictions),
        "precision": precision_score(
            y_true, predictions, zero_division=0, average="weighted"
        ),
        "recall": recall_score(
            y_true, predictions, zero_division=0, average="weighted"
        ),
        "f1": f1_score(y_true, predictions, zero_division=0, average="weighted"),
    }
    report = classification_report(
        y_true, predictions, output_dict=True, zero_division=0
    )
    cm_image = generate_confusion_matrix_image(cm, labels=model_labels)

    preview = df.assign(prediction=predictions).head(10)

    response = {
        "model": {
            "id": model_key,
            "label": MODEL_LABELS.get(model_key, model_key),
        },
        "metrics": metrics,
        "classification_report": report,
        "confusion_matrix": {
            "labels": model_labels,
            "matrix": cm.tolist(),
            "image_png_base64": cm_image,
        },
        "preview": preview.to_dict(orient="records"),
        "total_samples": int(len(df)),
        "class_labels": model_labels,
        "class_metadata": ASSETS.get("class_metadata", []),
    }
    return jsonify(response)


if __name__ == "__main__":
    app.run(debug=True)
