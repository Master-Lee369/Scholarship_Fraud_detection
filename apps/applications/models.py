from django.db import models
from django.conf import settings
from django.utils import timezone

class Application(models.Model):
    STATUS_CHOICES = [
        ('pending', 'Pending'),
        ('genuine', 'Genuine'),
        ('suspicious', 'Suspicious'),
        ('fake', 'Fake'),
        ('approved', 'Approved'),
        ('rejected', 'Rejected'),
    ]

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    full_name = models.CharField(max_length=200)
    dob = models.DateField()
    email = models.EmailField()
    phone = models.CharField(max_length=15)
    aadhaar_number = models.CharField(max_length=12)
    bank_account = models.CharField(max_length=20)
    address = models.TextField()
    annual_income = models.DecimalField(max_digits=10, decimal_places=2)
    academic_percentage = models.FloatField()
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    device_fingerprint = models.CharField(max_length=255, blank=True)
    submitted_at = models.DateTimeField(default=timezone.now)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='pending')
    fraud_score = models.FloatField(default=0.0)
    admin_notes = models.TextField(blank=True)

    class Meta:
        ordering = ['-submitted_at']

    def __str__(self):
        return f"{self.full_name} - {self.status}"

    def uploaded_document_types(self):
        return set(self.documents.values_list('doc_type', flat=True))

    def missing_document_types(self):
        required = {doc_type for doc_type, _label in Document.DOC_TYPES}
        return sorted(required - self.uploaded_document_types())

    def has_all_required_documents(self):
        return not self.missing_document_types()


class Document(models.Model):
    DOC_TYPES = [
        ('identity', 'Identity Proof'),
        ('income', 'Income Certificate'),
        ('academic', 'Academic Certificate'),
        ('bank', 'Bank Passbook'),
    ]
    application = models.ForeignKey(Application, on_delete=models.CASCADE, related_name='documents')
    doc_type = models.CharField(max_length=50, choices=DOC_TYPES)
    file = models.FileField(upload_to='documents/%Y/%m/')
    extracted_text = models.TextField(blank=True)
    is_verified = models.BooleanField(default=False)
    tamper_score = models.FloatField(default=0.0)
    uploaded_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.doc_type} - {self.application.full_name}"
