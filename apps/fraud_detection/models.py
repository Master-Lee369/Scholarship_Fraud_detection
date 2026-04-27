from django.db import models
from apps.applications.models import Application

class FraudFlag(models.Model):
    FLAG_TYPES = [
        ('duplicate_phone', 'Duplicate Phone'),
        ('duplicate_email', 'Duplicate Email'),
        ('duplicate_aadhaar', 'Duplicate Aadhaar'),
        ('duplicate_bank', 'Duplicate Bank Account'),
        ('bulk_submission', 'Bulk Submission'),
        ('ip_reuse', 'IP Address Reuse'),
        ('doc_tamper', 'Document Tampering'),
        ('ml_anomaly', 'ML Anomaly Detected'),
    ]

    application = models.ForeignKey(Application, on_delete=models.CASCADE, related_name='flags')
    flag_type = models.CharField(max_length=50, choices=FLAG_TYPES)
    description = models.TextField()
    severity = models.CharField(max_length=10, choices=[('low','Low'),('medium','Medium'),('high','High')])
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.flag_type} - {self.application.full_name}"