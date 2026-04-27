from django.contrib.auth.models import AbstractUser
from django.db import models


class CustomUser(AbstractUser):
    ROLE_CHOICES = [('admin', 'Admin'), ('applicant', 'Applicant')]
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='applicant')
    phone = models.CharField(max_length=15, blank=True)

    def is_admin(self):
        return self.role == 'admin' or self.is_staff

    def save(self, *args, **kwargs):
        # Keep the app-specific role aligned with Django's staff permissions.
        if self.is_superuser:
            self.role = 'admin'
            self.is_staff = True
        elif self.role == 'admin':
            self.is_staff = True
        elif self.is_staff:
            self.role = 'admin'

        super().save(*args, **kwargs)
