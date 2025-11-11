"""
Utility script to train the classification models used by the final web app.

Supports datasets defined in DATASETS; trains both a logistic regression baseline
and a multilayer perceptron, stores the fitted pipelines under final_project/models/,
and writes a JSON summary with the evaluation metrics captured on a held-out test split.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
METRICS_PATH = MODELS_DIR / "metrics.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2

DATASETS = {
    "covid_hiv": {
        "path": BASE_DIR / "data" / "balanced_normalized_dataset_covid_19_hiv.xlsx",
        "target": "outcome",
        "description": "Dataset balanceado COVID-19 + VIH",
    },
    "demale_hsjm": {
        "path": BASE_DIR / "data" / "DEMALE-HSJM_2025_data_balanced.xlsx",
        "target": "diagnosis",
        "description": "Dataset DEMALE-HSJM 2025 balanceado (3 clases)",
    },
}


@dataclass
class ModelArtifact:
    name: str
    pipeline: Pipeline
    metrics: Dict[str, float]
    report: Dict[str, Dict[str, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrena los modelos para el proyecto final."
    )
    parser.add_argument(
        "--dataset",
        choices=DATASETS.keys(),
        default="demale_hsjm",
        help="Identificador del dataset a utilizar.",
    )
    return parser.parse_args()


def load_dataset(data_path: Path, target_col: str) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    df = pd.read_excel(data_path)
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' is missing from the dataset.")
    return df


def split_features(df: pd.DataFrame, target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    return X, y


def build_logistic_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def build_mlp_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "model",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    solver="adam",
                    alpha=0.001,
                    learning_rate_init=0.01,
                    max_iter=5000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def evaluate_model(
    name: str,
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> ModelArtifact:
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test, y_pred, average="weighted", zero_division=0
        ),
        "recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
        "f1": f1_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    report_dict = classification_report(
        y_test,
        y_pred,
        output_dict=True,
        zero_division=0,
    )
    return ModelArtifact(name=name, pipeline=pipeline, metrics=metrics, report=report_dict)


def train_models(dataset_key: str) -> Tuple[Dict[str, ModelArtifact], List[int]]:
    dataset_cfg = DATASETS[dataset_key]
    df = load_dataset(dataset_cfg["path"], dataset_cfg["target"])
    X, y = split_features(df, dataset_cfg["target"])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    logistic_artifact = evaluate_model(
        "logistic_regression",
        build_logistic_pipeline(),
        X_train,
        X_test,
        y_train,
        y_test,
    )

    mlp_artifact = evaluate_model(
        "mlp_classifier",
        build_mlp_pipeline(),
        X_train,
        X_test,
        y_train,
        y_test,
    )

    class_labels = sorted(np.unique(y).astype(int).tolist())

    artifacts = {
        logistic_artifact.name: logistic_artifact,
        mlp_artifact.name: mlp_artifact,
    }
    return artifacts, class_labels


def persist_artifacts(
    artifacts: Dict[str, ModelArtifact],
    dataset_key: str,
    target_col: str,
    class_labels: List[int],
) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    metrics_payload = {
        "dataset": dataset_key,
        "dataset_description": DATASETS[dataset_key]["description"],
        "target_column": target_col,
        "feature_order": [],
        "class_labels": class_labels,
        "models": {},
    }

    sample_pipeline = next(iter(artifacts.values())).pipeline
    feature_order = sample_pipeline.named_steps["scaler"].feature_names_in_.tolist()
    metrics_payload["feature_order"] = feature_order

    for name, artifact in artifacts.items():
        model_path = MODELS_DIR / f"{name}.joblib"
        joblib.dump(artifact.pipeline, model_path)

        metrics_payload["models"][name] = {
            "metrics": artifact.metrics,
            "classification_report": artifact.report,
        }

    with METRICS_PATH.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)


def main() -> None:
    args = parse_args()
    dataset_cfg = DATASETS[args.dataset]
    artifacts, class_labels = train_models(args.dataset)
    persist_artifacts(
        artifacts,
        dataset_key=args.dataset,
        target_col=dataset_cfg["target"],
        class_labels=class_labels,
    )
    print("Models trained and saved to", MODELS_DIR)
    print(f"Dataset: {args.dataset} -> {dataset_cfg['description']}")
    for name, artifact in artifacts.items():
        accuracy = artifact.metrics["accuracy"]
        print(f"  {name}: accuracy={accuracy:.4f}")


if __name__ == "__main__":
    main()
