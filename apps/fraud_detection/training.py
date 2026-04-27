import json
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
TRAINING_DATA_DIR = DATA_DIR / "training"
MODEL_DIR = BASE_DIR / "ml_models"

DEFAULT_EXTERNAL_DATASET_URL = (
    "https://huggingface.co/datasets/UmairT/scholarships_dataset/resolve/main/scholarships_data.csv"
)
DEFAULT_EXTERNAL_DATASET_PATH = EXTERNAL_DATA_DIR / "scholarships_data.csv"
TRAINING_FRAME_PATH = TRAINING_DATA_DIR / "synthetic_scholarship_fraud_training.csv"
MODEL_ARTIFACT_PATH = MODEL_DIR / "fraud_classifier.pkl"
METRICS_PATH = MODEL_DIR / "training_metrics.json"

FEATURE_COLUMNS = [
    "annual_income",
    "academic_percentage",
    "ip_count",
    "submission_hour",
    "is_weekend",
    "flag_count",
    "duplicate_flag_count",
    "high_severity_flag_count",
    "medium_severity_flag_count",
    "document_count",
    "missing_document_count",
    "verified_document_ratio",
    "avg_tamper_score",
    "max_tamper_score",
    "document_issue_count",
    "applicant_age",
    "has_device_fingerprint",
]
CLASS_LABELS = ["genuine", "suspicious", "fake"]
REQUIRED_DOCUMENT_COUNT = 4


def download_external_dataset(url=DEFAULT_EXTERNAL_DATASET_URL, target_path=DEFAULT_EXTERNAL_DATASET_PATH):
    """Download the external scholarship dataset used to seed training."""
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    urlretrieve(url, target_path)
    return target_path


def _parse_fund_amount(value):
    if pd.isna(value):
        return 0.0

    text = str(value).replace(",", "")
    matches = re.findall(r"\d+(?:\.\d+)?", text)
    if not matches:
        return 0.0

    numeric_values = [float(match) for match in matches]
    if "%" in text:
        return max(numeric_values) * 100.0
    return max(numeric_values)


def _degree_weight(value):
    if pd.isna(value):
        return 0.35

    weights = {
        "course": 0.20,
        "bachelor": 0.42,
        "master": 0.58,
        "phd": 0.78,
    }
    tokens = [token.strip().lower() for token in str(value).split(",") if token.strip()]
    if not tokens:
        return 0.35
    return max(weights.get(token, 0.30) for token in tokens)


def load_external_scholarship_profiles(dataset_path=DEFAULT_EXTERNAL_DATASET_PATH):
    """Load and enrich the open scholarship dataset for training data generation."""
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"External dataset not found: {dataset_path}")

    scholarships = pd.read_csv(dataset_path)
    scholarships["fund_amount"] = scholarships["funds"].map(_parse_fund_amount)
    scholarships["degree_weight"] = scholarships["degrees"].map(_degree_weight)

    max_fund = max(float(scholarships["fund_amount"].max() or 0.0), 1.0)
    scholarships["normalized_fund"] = scholarships["fund_amount"].clip(lower=0.0) / max_fund
    scholarships["competitiveness"] = (
        0.55 * scholarships["degree_weight"] + 0.45 * scholarships["normalized_fund"]
    ).clip(0.0, 1.0)
    return scholarships


def _bounded_normal(rng, mean, stdev, lower, upper):
    return float(np.clip(rng.normal(mean, stdev), lower, upper))


def _choose(rng, values, probabilities):
    return rng.choice(values, p=probabilities).item()


def _compute_rule_risk(row):
    late_night = row["submission_hour"] < 5 or row["submission_hour"] > 22
    income_grade_mismatch = (
        row["annual_income"] > row["income_cap"] and row["academic_percentage"] > row["target_grade"] + 10
    )
    unlikely_age = row["applicant_age"] < 17 or row["applicant_age"] > 35

    risk = (
        0.18 * min(row["ip_count"] / 5.0, 1.0)
        + 0.22 * min(row["duplicate_flag_count"] / 3.0, 1.0)
        + 0.20 * min(row["high_severity_flag_count"] / 2.0, 1.0)
        + 0.12 * min(row["medium_severity_flag_count"] / 2.0, 1.0)
        + 0.18 * row["max_tamper_score"]
        + 0.15 * min(row["document_issue_count"] / 3.0, 1.0)
        + 0.10 * (1.0 - row["verified_document_ratio"])
        + 0.07 * min(row["missing_document_count"] / 2.0, 1.0)
        + 0.06 * row["is_weekend"]
        + (0.08 if late_night else 0.0)
        + (0.08 if income_grade_mismatch else 0.0)
        + (0.06 if unlikely_age else 0.0)
        + (0.05 if not row["has_device_fingerprint"] else 0.0)
        + (0.05 if row["academic_percentage"] < row["target_grade"] - 15 else 0.0)
    )
    return float(np.clip(risk, 0.0, 1.0))


def _assign_label(rng, row):
    risk = _compute_rule_risk(row)
    if risk < 0.38:
        label = "genuine"
    elif risk < 0.68:
        label = "suspicious"
    else:
        label = "fake"

    # Add a small amount of uncertainty so the classifier learns realistic overlap.
    if rng.random() < 0.05:
        alternatives = [item for item in CLASS_LABELS if item != label]
        label = rng.choice(alternatives).item()
    return label


def generate_training_frame(sample_count=8000, seed=42, scholarship_profiles=None):
    """
    Generate a scholarship-domain training set aligned to the current Django app features.

    Public scholarship data seeds the competitiveness baseline, then application-level fraud
    patterns are synthesized around that baseline because public labeled scholarship-fraud
    corpora are not readily available.
    """
    rng = np.random.default_rng(seed)
    scholarship_profiles = scholarship_profiles if scholarship_profiles is not None else pd.DataFrame()
    seeds = rng.choice(CLASS_LABELS, size=sample_count, p=[0.56, 0.29, 0.15])
    rows = []

    if not scholarship_profiles.empty:
        sampled_profiles = scholarship_profiles.sample(n=sample_count, replace=True, random_state=seed)
        sampled_profiles = sampled_profiles.reset_index(drop=True)
    else:
        sampled_profiles = pd.DataFrame([
            {
                "title": "Generic Scholarship",
                "degrees": "Bachelor",
                "location": "Unknown",
                "competitiveness": 0.45,
            }
            for _ in range(sample_count)
        ])

    for index, seed_label in enumerate(seeds):
        profile = sampled_profiles.iloc[index]
        competitiveness = float(profile.get("competitiveness", 0.45))
        target_grade = 60.0 + (competitiveness * 26.0)
        income_cap = max(140000.0, 650000.0 - (competitiveness * 260000.0))

        if seed_label == "genuine":
            applicant_age = int(_bounded_normal(rng, 21.0, 3.0, 16.0, 34.0))
            annual_income = _bounded_normal(rng, income_cap * 0.58, income_cap * 0.22, 50000.0, 900000.0)
            academic_percentage = _bounded_normal(rng, target_grade + 5.0, 9.0, 45.0, 100.0)
            ip_count = int(_choose(rng, [0, 1, 2, 3], [0.58, 0.25, 0.12, 0.05]))
            submission_hour = int(rng.integers(7, 23))
            is_weekend = int(_choose(rng, [0, 1], [0.76, 0.24]))
            duplicate_flag_count = int(_choose(rng, [0, 1], [0.90, 0.10]))
            high_severity_flag_count = int(_choose(rng, [0, 1], [0.90, 0.10]))
            medium_severity_flag_count = int(_choose(rng, [0, 1, 2], [0.70, 0.25, 0.05]))
            document_count = int(_choose(rng, [4, 3, 2], [0.82, 0.13, 0.05]))
            verified_document_ratio = _bounded_normal(rng, 0.88, 0.10, 0.45, 1.0)
            avg_tamper_score = _bounded_normal(rng, 0.07, 0.05, 0.0, 0.25)
            max_tamper_score = _bounded_normal(
                rng,
                avg_tamper_score + 0.05,
                0.04,
                avg_tamper_score,
                0.35,
            )
            document_issue_count = int(_choose(rng, [0, 1, 2], [0.72, 0.22, 0.06]))
            has_device_fingerprint = int(_choose(rng, [1, 0], [0.72, 0.28]))
        elif seed_label == "suspicious":
            applicant_age = int(_bounded_normal(rng, 22.0, 4.0, 16.0, 38.0))
            annual_income = _bounded_normal(rng, income_cap * 0.92, income_cap * 0.32, 50000.0, 1100000.0)
            academic_percentage = _bounded_normal(rng, target_grade - 1.0, 13.0, 30.0, 100.0)
            ip_count = int(_choose(rng, [0, 1, 2, 3, 4, 5], [0.12, 0.22, 0.24, 0.20, 0.14, 0.08]))
            submission_hour = int(rng.integers(0, 24))
            is_weekend = int(_choose(rng, [0, 1], [0.54, 0.46]))
            duplicate_flag_count = int(_choose(rng, [0, 1, 2, 3], [0.35, 0.36, 0.20, 0.09]))
            high_severity_flag_count = int(_choose(rng, [0, 1, 2], [0.42, 0.40, 0.18]))
            medium_severity_flag_count = int(_choose(rng, [0, 1, 2, 3], [0.18, 0.36, 0.30, 0.16]))
            document_count = int(_choose(rng, [4, 3, 2, 1], [0.44, 0.26, 0.20, 0.10]))
            verified_document_ratio = _bounded_normal(rng, 0.60, 0.18, 0.10, 0.95)
            avg_tamper_score = _bounded_normal(rng, 0.17, 0.09, 0.01, 0.45)
            max_tamper_score = _bounded_normal(
                rng,
                avg_tamper_score + 0.10,
                0.07,
                avg_tamper_score,
                0.65,
            )
            document_issue_count = int(_choose(rng, [0, 1, 2, 3, 4], [0.10, 0.28, 0.30, 0.22, 0.10]))
            has_device_fingerprint = int(_choose(rng, [1, 0], [0.48, 0.52]))
        else:
            applicant_age = int(_bounded_normal(rng, 24.0, 6.0, 15.0, 45.0))
            annual_income = _bounded_normal(rng, income_cap * 1.24, income_cap * 0.42, 70000.0, 1600000.0)
            academic_percentage = _bounded_normal(rng, target_grade - 8.0, 18.0, 10.0, 100.0)
            if rng.random() < 0.18:
                academic_percentage = _bounded_normal(rng, 97.0, 2.0, 92.0, 100.0)
            ip_count = int(_choose(rng, [0, 1, 2, 3, 4, 5, 6, 7, 8], [0.02, 0.05, 0.10, 0.16, 0.18, 0.18, 0.14, 0.10, 0.07]))
            submission_hour = int(rng.integers(0, 24))
            is_weekend = int(_choose(rng, [0, 1], [0.34, 0.66]))
            duplicate_flag_count = int(_choose(rng, [0, 1, 2, 3, 4], [0.05, 0.18, 0.28, 0.28, 0.21]))
            high_severity_flag_count = int(_choose(rng, [0, 1, 2, 3], [0.08, 0.22, 0.44, 0.26]))
            medium_severity_flag_count = int(_choose(rng, [0, 1, 2, 3], [0.06, 0.24, 0.42, 0.28]))
            document_count = int(_choose(rng, [4, 3, 2, 1, 0], [0.12, 0.18, 0.26, 0.24, 0.20]))
            verified_document_ratio = _bounded_normal(rng, 0.30, 0.18, 0.0, 0.8)
            avg_tamper_score = _bounded_normal(rng, 0.28, 0.14, 0.02, 0.85)
            max_tamper_score = _bounded_normal(
                rng,
                avg_tamper_score + 0.18,
                0.10,
                avg_tamper_score,
                1.0,
            )
            document_issue_count = int(_choose(rng, [0, 1, 2, 3, 4], [0.03, 0.12, 0.23, 0.34, 0.28]))
            has_device_fingerprint = int(_choose(rng, [1, 0], [0.22, 0.78]))

        document_count = min(document_count, REQUIRED_DOCUMENT_COUNT)
        missing_document_count = REQUIRED_DOCUMENT_COUNT - document_count
        flag_count = (
            duplicate_flag_count
            + high_severity_flag_count
            + medium_severity_flag_count
            + document_issue_count
            + max(ip_count - 1, 0) // 2
        )

        row = {
            "annual_income": round(annual_income, 2),
            "academic_percentage": round(academic_percentage, 2),
            "ip_count": ip_count,
            "submission_hour": submission_hour,
            "is_weekend": is_weekend,
            "flag_count": int(flag_count),
            "duplicate_flag_count": duplicate_flag_count,
            "high_severity_flag_count": high_severity_flag_count,
            "medium_severity_flag_count": medium_severity_flag_count,
            "document_count": document_count,
            "missing_document_count": missing_document_count,
            "verified_document_ratio": round(verified_document_ratio, 4),
            "avg_tamper_score": round(avg_tamper_score, 4),
            "max_tamper_score": round(max_tamper_score, 4),
            "document_issue_count": document_issue_count,
            "applicant_age": applicant_age,
            "has_device_fingerprint": has_device_fingerprint,
            "scholarship_title": profile.get("title", "Generic Scholarship"),
            "scholarship_location": profile.get("location", "Unknown"),
            "scholarship_degrees": profile.get("degrees", "Bachelor"),
            "scholarship_competitiveness": round(competitiveness, 4),
            "target_grade": round(target_grade, 2),
            "income_cap": round(income_cap, 2),
        }
        row["label"] = _assign_label(rng, row)
        rows.append(row)

    return pd.DataFrame(rows)


def train_supervised_model(training_frame, random_state=42):
    """Train and evaluate the multiclass scholarship fraud classifier."""
    features = training_frame[FEATURE_COLUMNS]
    labels = training_frame["label"]

    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=0.2,
        random_state=random_state,
        stratify=labels,
    )

    model = RandomForestClassifier(
        n_estimators=350,
        random_state=random_state,
        class_weight="balanced_subsample",
        min_samples_leaf=3,
        max_depth=16,
    )
    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
    feature_importance = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    sorted_importance = feature_importance.sort_values(ascending=False)

    matrix = confusion_matrix(y_test, predictions, labels=CLASS_LABELS)
    confusion = {}
    for actual_index, actual_label in enumerate(CLASS_LABELS):
        confusion[actual_label] = {}
        for predicted_index, predicted_label in enumerate(CLASS_LABELS):
            confusion[actual_label][predicted_label] = int(matrix[actual_index][predicted_index])

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "macro_f1": round(float(f1_score(y_test, predictions, average="macro")), 4),
        "weighted_f1": round(float(f1_score(y_test, predictions, average="weighted")), 4),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "class_distribution": {
            label: round(float(value), 4)
            for label, value in labels.value_counts(normalize=True).sort_index().items()
        },
        "confusion_matrix": confusion,
        "classification_report": report,
        "top_feature_importance": {
            key: round(float(value), 4)
            for key, value in sorted_importance.head(8).items()
        },
    }
    return model, metrics


def train_and_save_model(
    sample_count=8000,
    seed=42,
    external_dataset_path=DEFAULT_EXTERNAL_DATASET_PATH,
    external_dataset_url=DEFAULT_EXTERNAL_DATASET_URL,
    download_if_missing=True,
):
    """Download external data when needed, then generate, train, evaluate, and save artifacts."""
    external_dataset_path = Path(external_dataset_path)
    if not external_dataset_path.exists():
        if not download_if_missing:
            raise FileNotFoundError(f"External dataset not found: {external_dataset_path}")
        download_external_dataset(url=external_dataset_url, target_path=external_dataset_path)

    scholarship_profiles = load_external_scholarship_profiles(external_dataset_path)
    training_frame = generate_training_frame(
        sample_count=sample_count,
        seed=seed,
        scholarship_profiles=scholarship_profiles,
    )
    model, metrics = train_supervised_model(training_frame, random_state=seed)

    TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    training_frame.to_csv(TRAINING_FRAME_PATH, index=False)

    artifact = {
        "model": model,
        "feature_names": FEATURE_COLUMNS,
        "class_labels": CLASS_LABELS,
        "required_document_count": REQUIRED_DOCUMENT_COUNT,
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "training_rows": int(len(training_frame)),
        "external_dataset_path": str(external_dataset_path),
        "external_dataset_rows": int(len(scholarship_profiles)),
        "metrics": metrics,
    }

    with open(MODEL_ARTIFACT_PATH, "wb") as artifact_file:
        pickle.dump(artifact, artifact_file)

    with open(METRICS_PATH, "w", encoding="utf-8") as metrics_file:
        json.dump(
            {
                **metrics,
                "training_rows": int(len(training_frame)),
                "trained_at": artifact["trained_at"],
                "external_dataset_path": str(external_dataset_path),
                "external_dataset_rows": int(len(scholarship_profiles)),
                "training_frame_path": str(TRAINING_FRAME_PATH),
                "model_artifact_path": str(MODEL_ARTIFACT_PATH),
            },
            metrics_file,
            indent=2,
        )

    return artifact, metrics, TRAINING_FRAME_PATH
