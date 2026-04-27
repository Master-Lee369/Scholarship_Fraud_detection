from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings

from apps.applications.models import Application, Document
from apps.fraud_detection.duplicate_checker import check_duplicates
from apps.fraud_detection.ml_scorer import classify_application, extract_feature_dict, rule_based_score
from apps.fraud_detection.models import FraudFlag


@override_settings(MEDIA_ROOT="test_media")
class FraudDetectionTests(TestCase):
    def setUp(self):
        self.user_model = get_user_model()
        self.primary_user = self.user_model.objects.create_user(
            username="primary",
            password="test-pass-123",
        )
        self.secondary_user = self.user_model.objects.create_user(
            username="secondary",
            password="test-pass-123",
        )

    def _create_application(self, user, **overrides):
        payload = {
            "user": user,
            "full_name": "Ravi Kumar",
            "dob": "2003-06-12",
            "email": "ravi@example.com",
            "phone": "9998887776",
            "aadhaar_number": "111122223333",
            "bank_account": "998877665544",
            "address": "Bengaluru",
            "annual_income": "300000.00",
            "academic_percentage": 88.0,
        }
        payload.update(overrides)
        return Application.objects.create(**payload)

    def test_duplicate_checks_are_idempotent(self):
        self._create_application(self.primary_user)
        target = self._create_application(
            self.secondary_user,
            email="ravi@example.com",
            phone="9998887776",
            aadhaar_number="111122223333",
            bank_account="998877665544",
        )

        check_duplicates(target)
        check_duplicates(target)

        self.assertEqual(FraudFlag.objects.filter(application=target).count(), 4)

    @patch("apps.fraud_detection.ml_scorer._load_ml_backend", return_value=None)
    def test_classifier_falls_back_when_ml_backend_is_unavailable(self, _load_backend):
        application = self._create_application(self.primary_user)

        score = classify_application(application)
        application.refresh_from_db()

        self.assertGreaterEqual(score, 0.0)
        self.assertLess(score, 0.3)
        self.assertEqual(application.status, "genuine")

    def test_extract_feature_dict_reflects_documents_and_flags(self):
        duplicate = self._create_application(
            self.secondary_user,
            email="other@example.com",
            phone="1234512345",
            aadhaar_number="999988887777",
            bank_account="111100009999",
            ip_address="10.0.0.1",
        )
        application = self._create_application(
            self.primary_user,
            ip_address="10.0.0.1",
            device_fingerprint="browser-abc",
        )

        Document.objects.create(
            application=application,
            doc_type="identity",
            file=SimpleUploadedFile("identity.png", b"img", content_type="image/png"),
            extracted_text="Ravi Kumar 12/06/2003",
            is_verified=True,
            tamper_score=0.05,
        )
        Document.objects.create(
            application=application,
            doc_type="income",
            file=SimpleUploadedFile("income.png", b"img", content_type="image/png"),
            extracted_text="Income certificate",
            is_verified=False,
            tamper_score=0.28,
        )
        FraudFlag.objects.create(
            application=application,
            flag_type="duplicate_email",
            description="Email reused",
            severity="high",
        )
        FraudFlag.objects.create(
            application=application,
            flag_type="doc_tamper",
            description="DOB mismatch",
            severity="medium",
        )
        self.assertIsNotNone(duplicate)

        features = extract_feature_dict(application)

        self.assertEqual(features["ip_count"], 1)
        self.assertEqual(features["document_count"], 2)
        self.assertEqual(features["missing_document_count"], 2)
        self.assertEqual(features["duplicate_flag_count"], 1)
        self.assertEqual(features["document_issue_count"], 1)
        self.assertEqual(features["high_severity_flag_count"], 1)
        self.assertEqual(features["medium_severity_flag_count"], 1)
        self.assertAlmostEqual(features["verified_document_ratio"], 0.5, places=2)
        self.assertAlmostEqual(features["max_tamper_score"], 0.28, places=2)
        self.assertEqual(features["has_device_fingerprint"], 1)

    def test_rule_based_score_rises_for_risky_application(self):
        application = self._create_application(
            self.primary_user,
            annual_income="900000.00",
            academic_percentage=97.0,
            ip_address="192.168.0.10",
        )
        self._create_application(
            self.secondary_user,
            email="other@example.com",
            phone="1234512345",
            aadhaar_number="999988887777",
            bank_account="111100009999",
            ip_address="192.168.0.10",
        )
        Document.objects.create(
            application=application,
            doc_type="identity",
            file=SimpleUploadedFile("identity.png", b"img", content_type="image/png"),
            extracted_text="Ravi Kumar",
            is_verified=False,
            tamper_score=0.62,
        )
        FraudFlag.objects.create(
            application=application,
            flag_type="duplicate_aadhaar",
            description="Aadhaar reused",
            severity="high",
        )
        FraudFlag.objects.create(
            application=application,
            flag_type="doc_tamper",
            description="DOB mismatch",
            severity="medium",
        )

        score = rule_based_score(application)

        self.assertGreater(score, 0.5)

    @patch(
        "apps.fraud_detection.ml_scorer.predict_application",
        return_value=("fake", 0.91, {"genuine": 0.02, "suspicious": 0.08, "fake": 0.90}),
    )
    def test_classifier_creates_ml_flag_for_high_risk_prediction(self, _prediction):
        application = self._create_application(self.primary_user)

        score = classify_application(application)
        application.refresh_from_db()
        ml_flag = FraudFlag.objects.get(application=application, flag_type="ml_anomaly")

        self.assertEqual(score, 0.91)
        self.assertEqual(application.status, "fake")
        self.assertEqual(ml_flag.severity, "high")
        self.assertIn("Classifier predicted fake", ml_flag.description)
