import pickle

import numpy as np
from django.db.models import Avg, Max
from django.utils.dateparse import parse_date

from apps.fraud_detection.training import CLASS_LABELS, FEATURE_COLUMNS, MODEL_ARTIFACT_PATH


def _load_ml_backend():
    try:
        from sklearn.ensemble import RandomForestClassifier
    except Exception:
        return None
    return RandomForestClassifier


def extract_feature_dict(application):
    """Convert an application into the numeric feature set used by the classifier."""
    from apps.applications.models import Application

    docs = application.documents.all()
    flags = application.flags.exclude(flag_type="ml_anomaly")
    duplicate_flag_types = [
        "duplicate_phone",
        "duplicate_email",
        "duplicate_aadhaar",
        "duplicate_bank",
        "bulk_submission",
        "ip_reuse",
    ]

    ip_count = Application.objects.filter(
        ip_address=application.ip_address
    ).exclude(id=application.id).count()
    hour = application.submitted_at.hour
    is_weekend = 1 if application.submitted_at.weekday() >= 5 else 0
    doc_count = docs.count()
    verified_count = docs.filter(is_verified=True).count()
    verified_ratio = verified_count / doc_count if doc_count else 0.0
    avg_tamper_score = docs.aggregate(avg=Avg("tamper_score"))["avg"] or 0.0
    max_tamper_score = docs.aggregate(max=Max("tamper_score"))["max"] or 0.0

    dob = application.dob
    if isinstance(dob, str):
        dob = parse_date(dob)

    submitted_date = application.submitted_at.date()
    age = submitted_date.year - dob.year
    before_birthday = (submitted_date.month, submitted_date.day) < (dob.month, dob.day)
    applicant_age = age - (1 if before_birthday else 0)

    return {
        "annual_income": float(application.annual_income),
        "academic_percentage": float(application.academic_percentage),
        "ip_count": ip_count,
        "submission_hour": hour,
        "is_weekend": is_weekend,
        "flag_count": flags.count(),
        "duplicate_flag_count": flags.filter(flag_type__in=duplicate_flag_types).count(),
        "high_severity_flag_count": flags.filter(severity="high").count(),
        "medium_severity_flag_count": flags.filter(severity="medium").count(),
        "document_count": doc_count,
        "missing_document_count": len(application.missing_document_types()),
        "verified_document_ratio": round(verified_ratio, 4),
        "avg_tamper_score": round(float(avg_tamper_score), 4),
        "max_tamper_score": round(float(max_tamper_score), 4),
        "document_issue_count": flags.filter(flag_type="doc_tamper").count(),
        "applicant_age": applicant_age,
        "has_device_fingerprint": 1 if application.device_fingerprint.strip() else 0,
    }


def extract_features(application):
    """Return the model feature vector in the trained column order."""
    feature_dict = extract_feature_dict(application)
    return np.array([feature_dict[column] for column in FEATURE_COLUMNS], dtype=float)


def train_model(applications=None, **kwargs):
    """
    Backwards-compatible entry point for training the fraud classifier.

    The current implementation delegates to the supervised training pipeline that uses
    external scholarship reference data plus synthetic application-level fraud examples.
    """
    from apps.fraud_detection.training import train_and_save_model

    if applications is not None and "sample_count" not in kwargs:
        try:
            kwargs["sample_count"] = max(len(applications), 8000)
        except TypeError:
            pass

    return train_and_save_model(**kwargs)


def _load_model_artifact():
    if _load_ml_backend() is None:
        return None

    try:
        with open(MODEL_ARTIFACT_PATH, "rb") as model_file:
            artifact = pickle.load(model_file)
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        return None

    if not isinstance(artifact, dict):
        return None
    if "model" not in artifact or "feature_names" not in artifact:
        return None
    if artifact["feature_names"] != FEATURE_COLUMNS:
        return None
    return artifact


def predict_application(application):
    """Return a tuple of (predicted_label, fraud_score, probability_map) when a model exists."""
    artifact = _load_model_artifact()
    if artifact is None:
        return None

    model = artifact["model"]
    features = extract_features(application).reshape(1, -1)
    predicted_label = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]

    probability_map = {}
    for index, label in enumerate(model.classes_):
        probability_map[label] = float(probabilities[index])

    # Weighted risk score: fake contributes full weight, suspicious contributes half.
    fraud_score = probability_map.get("fake", 0.0) + (0.5 * probability_map.get("suspicious", 0.0))
    fraud_score = float(np.clip(fraud_score, 0.0, 1.0))

    # Guard against missing classes in very small retraining runs.
    for label in CLASS_LABELS:
        probability_map.setdefault(label, 0.0)

    return predicted_label, fraud_score, probability_map


def score_application(application):
    """Return a fraud score between 0.0 (safe) and 1.0 (fraudulent)."""
    prediction = predict_application(application)
    if prediction is None:
        return rule_based_score(application)
    return prediction[1]


def rule_based_score(application):
    """Fallback scoring when no trained ML model exists."""
    features = extract_feature_dict(application)
    late_night = features["submission_hour"] < 5 or features["submission_hour"] > 22
    income_grade_mismatch = (
        features["annual_income"] > 700000 and features["academic_percentage"] > 94
    )
    unlikely_age = features["applicant_age"] < 17 or features["applicant_age"] > 35

    score = (
        0.18 * min(features["ip_count"] / 5.0, 1.0)
        + 0.22 * min(features["duplicate_flag_count"] / 3.0, 1.0)
        + 0.20 * min(features["high_severity_flag_count"] / 2.0, 1.0)
        + 0.12 * min(features["medium_severity_flag_count"] / 2.0, 1.0)
        + 0.18 * features["max_tamper_score"]
        + 0.15 * min(features["document_issue_count"] / 3.0, 1.0)
        + 0.10 * (1.0 - features["verified_document_ratio"])
        + 0.07 * min(features["missing_document_count"] / 2.0, 1.0)
        + 0.06 * features["is_weekend"]
        + (0.08 if late_night else 0.0)
        + (0.08 if income_grade_mismatch else 0.0)
        + (0.06 if unlikely_age else 0.0)
        + (0.05 if not features["has_device_fingerprint"] else 0.0)
    )
    return float(np.clip(score, 0.0, 1.0))


def _sync_ml_flag(application, probabilities):
    from apps.fraud_detection.models import FraudFlag

    FraudFlag.objects.filter(application=application, flag_type="ml_anomaly").delete()

    if application.status == "genuine" and application.fraud_score < 0.45:
        return

    severity = "medium" if application.status == "suspicious" else "high"
    description = (
        f"Classifier predicted {application.status} with fraud score {application.fraud_score:.2f}."
    )
    if probabilities:
        description += (
            " Probabilities:"
            f" genuine={probabilities.get('genuine', 0.0):.2f},"
            f" suspicious={probabilities.get('suspicious', 0.0):.2f},"
            f" fake={probabilities.get('fake', 0.0):.2f}."
        )

    FraudFlag.objects.create(
        application=application,
        flag_type="ml_anomaly",
        description=description,
        severity=severity,
    )


def classify_application(application):
    """Classify and update the application status based on fraud score."""
    prediction = predict_application(application)
    probabilities = None

    if prediction is not None:
        predicted_label, score, probabilities = prediction
        application.status = predicted_label
    else:
        score = rule_based_score(application)
        if score < 0.3:
            application.status = "genuine"
        elif score < 0.65:
            application.status = "suspicious"
        else:
            application.status = "fake"

    application.fraud_score = score
    application.save(update_fields=["fraud_score", "status"])
    _sync_ml_flag(application, probabilities)
    return score
