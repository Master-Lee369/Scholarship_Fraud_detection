from unittest.mock import patch

from django.contrib.auth import get_user_model
from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase, override_settings
from django.urls import reverse

from apps.applications.models import Application


@override_settings(MEDIA_ROOT="test_media")
class ApplicationFlowTests(TestCase):
    def setUp(self):
        self.user = get_user_model().objects.create_user(
            username="applicant",
            password="test-pass-123",
        )
        self.client.force_login(self.user)
        self.application = Application.objects.create(
            user=self.user,
            full_name="Asha Kumar",
            dob="2004-01-10",
            email="asha@example.com",
            phone="9876543210",
            aadhaar_number="123456789012",
            bank_account="1234567890",
            address="Chennai",
            annual_income="250000.00",
            academic_percentage=91.5,
        )

    @patch("apps.applications.views.classify_application")
    @patch("apps.applications.views.cross_verify_documents", return_value=[])
    @patch("apps.applications.views.check_duplicates")
    @patch("apps.applications.views.detect_tampering", return_value=0.05)
    @patch("apps.applications.views.extract_text", return_value="Asha Kumar 10/01/2004")
    def test_duplicate_document_type_is_rejected(
        self,
        _extract_text,
        _detect_tampering,
        _check_duplicates,
        _cross_verify_documents,
        _classify_application,
    ):
        first_upload = SimpleUploadedFile(
            "identity.png",
            b"fake-image-data",
            content_type="image/png",
        )
        second_upload = SimpleUploadedFile(
            "identity-2.png",
            b"fake-image-data",
            content_type="image/png",
        )

        first_response = self.client.post(
            reverse("upload_documents", kwargs={"pk": self.application.pk}),
            {"doc_type": "identity", "file": first_upload},
            follow=True,
        )
        second_response = self.client.post(
            reverse("upload_documents", kwargs={"pk": self.application.pk}),
            {"doc_type": "identity", "file": second_upload},
            follow=True,
        )

        self.assertEqual(first_response.status_code, 200)
        self.assertEqual(second_response.status_code, 200)
        self.assertContains(second_response, "already been uploaded")
        self.assertEqual(self.application.documents.filter(doc_type="identity").count(), 1)

    def test_status_page_loads_without_existing_application(self):
        self.application.delete()

        response = self.client.get(reverse("application_status"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "No application submitted yet")
