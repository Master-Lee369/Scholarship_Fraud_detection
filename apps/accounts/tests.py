from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse


class AccountFlowTests(TestCase):
    def setUp(self):
        self.user_model = get_user_model()

    def test_home_page_loads(self):
        response = self.client.get(reverse("home"))

        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "ScholarGuard AI")

    def test_applicant_login_redirects_to_application_status(self):
        self.user_model.objects.create_user(
            username="applicant",
            password="test-pass-123",
            role="applicant",
        )

        response = self.client.post(
            reverse("login"),
            {"username": "applicant", "password": "test-pass-123"},
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.request["PATH_INFO"], reverse("application_status"))

    def test_staff_login_redirects_to_dashboard(self):
        self.user_model.objects.create_user(
            username="reviewer",
            password="test-pass-123",
            role="admin",
            is_staff=True,
        )

        response = self.client.post(
            reverse("login"),
            {"username": "reviewer", "password": "test-pass-123"},
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.request["PATH_INFO"], reverse("admin_dashboard"))

    def test_registration_creates_user_and_logs_them_in(self):
        response = self.client.post(
            reverse("register"),
            {
                "username": "newapplicant",
                "email": "new@applicant.com",
                "phone": "9876543210",
                "password1": "Strong-pass-123",
                "password2": "Strong-pass-123",
            },
            follow=True,
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.request["PATH_INFO"], reverse("application_status"))
        self.assertTrue(self.user_model.objects.filter(username="newapplicant").exists())
        created_user = self.user_model.objects.get(username="newapplicant")
        self.assertEqual(created_user.role, "applicant")
        self.assertFalse(created_user.is_staff)

    def test_admin_role_users_are_saved_as_staff(self):
        user = self.user_model.objects.create_user(
            username="coordinator",
            password="test-pass-123",
            role="admin",
        )

        self.assertTrue(user.is_staff)
        self.assertTrue(user.is_admin())

    def test_staff_users_are_saved_with_admin_role(self):
        user = self.user_model.objects.create_user(
            username="staffer",
            password="test-pass-123",
            is_staff=True,
        )

        self.assertEqual(user.role, "admin")
        self.assertTrue(user.is_admin())
